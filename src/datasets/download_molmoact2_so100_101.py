#!/usr/bin/env python3
"""Download source repositories listed in repo_list.json.

The current manifest uses entries like:

    lerobot:00ri/so100_battery

Those are LeRobot datasets hosted as Hugging Face dataset repositories.  This
script downloads each dataset under ``<output-dir>/<owner>/<repo>`` so the
layout mirrors ``language_annotations/<owner>/<repo>``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse


DEFAULT_REPO_LIST = "repo_list.json"
DEFAULT_OUTPUT_DIR = "source_data"
REPORT_NAME = "download_report.jsonl"
NEXT_DATASET_DELAY_SECONDS = 60
DELAYED_RESULT_STATUSES = {"skipped", "verified", "incomplete", "repaired"}


@dataclass(frozen=True)
class RepoSpec:
    raw: str
    kind: str
    repo_id: str
    local_parts: tuple[str, ...]
    clone_url: str | None = None

    @property
    def local_path(self) -> Path:
        return Path(*self.local_parts)


@dataclass
class DownloadResult:
    raw: str
    dest: str
    status: str
    seconds: float
    message: str = ""


@dataclass
class VerificationResult:
    checked: int
    missing: list[str]
    size_mismatches: list[tuple[str, int, int]]

    @property
    def complete(self) -> bool:
        return not self.missing and not self.size_mismatches

    def message(self) -> str:
        parts = [f"checked={self.checked}"]
        if self.missing:
            parts.append(
                "missing="
                + format_problem_sample(
                    self.missing,
                    lambda path: path,
                )
            )
        if self.size_mismatches:
            parts.append(
                "size_mismatch="
                + format_problem_sample(
                    self.size_mismatches,
                    lambda item: (
                        f"{item[0]} (local={item[1]}, remote={item[2]})"
                    ),
                )
            )
        return "; ".join(parts)


def format_problem_sample(items, formatter, max_items: int = 5) -> str:
    rendered = [formatter(item) for item in items[:max_items]]
    suffix = (
        ""
        if len(items) <= max_items
        else f", ... +{len(items) - max_items} more"
    )
    return f"{len(items)} [{', '.join(rendered)}{suffix}]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download all source LeRobot/Hugging Face dataset repositories "
            "listed in repo_list.json."
        )
    )
    parser.add_argument(
        "--repo-list",
        default=DEFAULT_REPO_LIST,
        help=f"path to repo list JSON (default: {DEFAULT_REPO_LIST})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "where repositories are downloaded (default: source_data). "
            "Use --output-dir . to place owner/repo folders at the dataset root."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="number of repositories to download concurrently (default: 4)",
    )
    parser.add_argument(
        "--hf-workers",
        type=int,
        default=8,
        help="per-repository Hugging Face download workers (default: 8)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token; defaults to HF_TOKEN when set",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="optional revision/branch/commit to download from each repository",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="download only the first N entries; useful for smoke tests",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="start index in repo_list.json (default: 0)",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        help=(
            "glob pattern for files to include; may be passed multiple times. "
            "Forwarded to Hugging Face snapshot_download."
        ),
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help=(
            "glob pattern for files to exclude; may be passed multiple times. "
            "Forwarded to Hugging Face snapshot_download."
        ),
    )
    parser.add_argument(
        "--redownload-existing",
        action="store_true",
        help="download even when the destination directory already has files",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help=(
            "force Hugging Face files to be downloaded again; implies "
            "--redownload-existing"
        ),
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help=(
            "for existing Hugging Face datasets, compare the remote file list "
            "and file sizes before skipping; incomplete datasets are repaired "
            "unless --verify-only is set"
        ),
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "verify existing Hugging Face datasets and report incomplete ones "
            "without downloading or modifying files"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print planned downloads without touching the network",
    )
    return parser.parse_args()


def load_repo_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError(f"{path} must contain a JSON array of strings")
    return data


def clean_github_repo_id(value: str) -> str:
    value = value.strip().removesuffix(".git").strip("/")
    parts = value.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"invalid GitHub repo id: {value!r}")
    return value


def parse_repo(raw: str) -> RepoSpec:
    raw = raw.strip()
    if not raw:
        raise ValueError("empty repository entry")

    if raw.startswith(("http://", "https://", "git@")):
        return parse_url_repo(raw)

    if ":" in raw:
        prefix, value = raw.split(":", 1)
        prefix = prefix.lower()
        value = value.strip()
        if prefix in {"lerobot", "hf", "huggingface"}:
            return parse_hf_repo(raw, value)
        if prefix == "github":
            repo_id = clean_github_repo_id(value)
            owner, name = repo_id.split("/", 1)
            return RepoSpec(
                raw=raw,
                kind="github",
                repo_id=repo_id,
                local_parts=(owner, name),
                clone_url=f"https://github.com/{repo_id}.git",
            )
        raise ValueError(f"unsupported repository prefix {prefix!r} in {raw!r}")

    # Current manifests are explicit, but treating owner/repo as a HF dataset is
    # a useful fallback for manually created small lists.
    return parse_hf_repo(raw, raw)


def parse_hf_repo(raw: str, repo_id: str) -> RepoSpec:
    repo_id = repo_id.strip().strip("/")
    parts = repo_id.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"invalid Hugging Face dataset id: {repo_id!r}")
    owner, name = parts
    return RepoSpec(
        raw=raw,
        kind="hf_dataset",
        repo_id=repo_id,
        local_parts=(owner, name),
    )


def parse_url_repo(raw: str) -> RepoSpec:
    if raw.startswith("git@github.com:"):
        repo_id = clean_github_repo_id(raw.split(":", 1)[1])
        owner, name = repo_id.split("/", 1)
        return RepoSpec(raw, "github", repo_id, (owner, name), raw)

    parsed = urlparse(raw)
    host = parsed.netloc.lower()
    path = parsed.path.strip("/")

    if host == "github.com":
        repo_id = clean_github_repo_id(path)
        owner, name = repo_id.split("/", 1)
        return RepoSpec(raw, "github", repo_id, (owner, name), raw)

    if host == "huggingface.co":
        parts = path.split("/")
        if len(parts) >= 3 and parts[0] == "datasets":
            repo_id = "/".join(parts[1:3]).removesuffix(".git")
            return parse_hf_repo(raw, repo_id)

    raise ValueError(f"unsupported repository URL: {raw!r}")


def has_existing_files(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def verify_hf_dataset(
    spec: RepoSpec,
    dest: Path,
    *,
    token: str | None,
    revision: str | None,
    allow_patterns: list[str] | None,
    ignore_patterns: list[str] | None,
) -> VerificationResult:
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.hf_api import RepoFile
        from huggingface_hub.utils import filter_repo_objects
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to verify Hugging Face datasets. "
            "Install it with: python3 -m pip install huggingface_hub"
        ) from exc

    api = HfApi()
    repo_files = (
        item
        for item in api.list_repo_tree(
            repo_id=spec.repo_id,
            repo_type="dataset",
            revision=revision,
            recursive=True,
            expand=True,
            token=token,
        )
        if isinstance(item, RepoFile)
    )
    filtered_files = list(
        filter_repo_objects(
            repo_files,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            key=lambda item: item.path,
        )
    )

    missing: list[str] = []
    size_mismatches: list[tuple[str, int, int]] = []
    for repo_file in filtered_files:
        local_path = dest / repo_file.path
        if not local_path.is_file():
            missing.append(repo_file.path)
            continue
        local_size = local_path.stat().st_size
        if local_size != repo_file.size:
            size_mismatches.append((repo_file.path, local_size, repo_file.size))

    return VerificationResult(
        checked=len(filtered_files),
        missing=missing,
        size_mismatches=size_mismatches,
    )


def remove_mismatched_files(dest: Path, verification: VerificationResult) -> None:
    for path, _local_size, _remote_size in verification.size_mismatches:
        (dest / path).unlink(missing_ok=True)


def download_hf_dataset(
    spec: RepoSpec,
    dest: Path,
    *,
    token: str | None,
    revision: str | None,
    hf_workers: int,
    allow_patterns: list[str] | None,
    ignore_patterns: list[str] | None,
    force_download: bool,
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for lerobot/Hugging Face datasets. "
            "Install it with: python3 -m pip install huggingface_hub"
        ) from exc

    kwargs = {
        "repo_id": spec.repo_id,
        "repo_type": "dataset",
        "local_dir": str(dest),
        "max_workers": hf_workers,
        "force_download": force_download,
    }
    if token:
        kwargs["token"] = token
    if revision:
        kwargs["revision"] = revision
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns
    if ignore_patterns:
        kwargs["ignore_patterns"] = ignore_patterns

    snapshot_download(**kwargs)


def download_git_repo(spec: RepoSpec, dest: Path) -> None:
    if spec.clone_url is None:
        raise ValueError(f"missing clone URL for {spec.raw!r}")
    if shutil.which("git") is None:
        raise RuntimeError("git is required to clone GitHub repositories")
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", spec.clone_url, str(dest)],
        check=True,
    )


def download_one(
    spec: RepoSpec,
    output_dir: Path,
    args: argparse.Namespace,
) -> DownloadResult:
    dest = output_dir / spec.local_path
    start = time.monotonic()

    try:
        existing = has_existing_files(dest)

        if args.verify_only:
            if spec.kind != "hf_dataset":
                return DownloadResult(
                    raw=spec.raw,
                    dest=str(dest),
                    status="skipped",
                    seconds=time.monotonic() - start,
                    message="verify-only is implemented for Hugging Face datasets only",
                )
            verification = verify_hf_dataset(
                spec,
                dest,
                token=args.token,
                revision=args.revision,
                allow_patterns=args.include,
                ignore_patterns=args.exclude,
            )
            return DownloadResult(
                raw=spec.raw,
                dest=str(dest),
                status="verified" if verification.complete else "incomplete",
                seconds=time.monotonic() - start,
                message=verification.message(),
            )

        if existing and not args.redownload_existing:
            if args.verify_existing and spec.kind == "hf_dataset":
                verification = verify_hf_dataset(
                    spec,
                    dest,
                    token=args.token,
                    revision=args.revision,
                    allow_patterns=args.include,
                    ignore_patterns=args.exclude,
                )
                if verification.complete:
                    return DownloadResult(
                        raw=spec.raw,
                        dest=str(dest),
                        status="verified",
                        seconds=time.monotonic() - start,
                        message=verification.message(),
                    )
                remove_mismatched_files(dest, verification)
                download_hf_dataset(
                    spec,
                    dest,
                    token=args.token,
                    revision=args.revision,
                    hf_workers=args.hf_workers,
                    allow_patterns=args.include,
                    ignore_patterns=args.exclude,
                    force_download=args.force_download,
                )
                return DownloadResult(
                    raw=spec.raw,
                    dest=str(dest),
                    status="repaired",
                    seconds=time.monotonic() - start,
                    message=verification.message(),
                )

            return DownloadResult(
                raw=spec.raw,
                dest=str(dest),
                status="skipped",
                seconds=time.monotonic() - start,
                message="destination already exists; pass --redownload-existing to refresh",
            )

        if args.dry_run:
            return DownloadResult(
                raw=spec.raw,
                dest=str(dest),
                status="planned",
                seconds=time.monotonic() - start,
            )

        dest.mkdir(parents=True, exist_ok=True)
        if spec.kind == "hf_dataset":
            download_hf_dataset(
                spec,
                dest,
                token=args.token,
                revision=args.revision,
                hf_workers=args.hf_workers,
                allow_patterns=args.include,
                ignore_patterns=args.exclude,
                force_download=args.force_download,
            )
        elif spec.kind == "github":
            if has_existing_files(dest):
                raise RuntimeError(
                    "destination exists; git clone requires an empty target"
                )
            download_git_repo(spec, dest)
        else:
            raise ValueError(f"unsupported repository kind: {spec.kind}")

        return DownloadResult(
            raw=spec.raw,
            dest=str(dest),
            status="downloaded",
            seconds=time.monotonic() - start,
        )
    except Exception as exc:  # noqa: BLE001 - keep batch downloads moving
        return DownloadResult(
            raw=spec.raw,
            dest=str(dest),
            status="failed",
            seconds=time.monotonic() - start,
            message=str(exc),
        )


def write_report_line(report_path: Path, result: DownloadResult) -> None:
    with report_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "repo": result.raw,
                    "dest": result.dest,
                    "status": result.status,
                    "seconds": round(result.seconds, 3),
                    "message": result.message,
                },
                ensure_ascii=True,
            )
            + "\n"
        )


def select_repos(repos: Iterable[str], start: int, limit: int | None) -> list[str]:
    selected = list(repos)[start:]
    if limit is not None:
        selected = selected[:limit]
    return selected


def main() -> int:
    args = parse_args()
    if args.verify_only and (args.redownload_existing or args.force_download):
        raise SystemExit(
            "--verify-only cannot be combined with --redownload-existing "
            "or --force-download"
        )
    if args.dry_run and (args.verify_existing or args.verify_only):
        raise SystemExit(
            "--dry-run cannot be combined with verification because "
            "verification requires network access"
        )
    if args.force_download:
        args.redownload_existing = True

    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")
    if args.hf_workers < 1:
        raise SystemExit("--hf-workers must be >= 1")
    if args.start < 0:
        raise SystemExit("--start must be >= 0")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")

    repo_list_path = Path(args.repo_list)
    output_dir = Path(args.output_dir)
    report_path = output_dir / REPORT_NAME

    raw_repos = select_repos(load_repo_list(repo_list_path), args.start, args.limit)
    specs = [parse_repo(raw) for raw in raw_repos]

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    if report_path.exists() and not args.dry_run:
        report_path.unlink()

    effective_jobs = 1 if args.verify_existing or args.verify_only else args.jobs

    print(f"Repository list: {repo_list_path}")
    print(f"Output dir:      {output_dir}")
    print(f"Report:          {report_path}")
    print(f"Repositories:    {len(specs)}")
    print(f"Jobs:            {effective_jobs}")
    if effective_jobs != args.jobs:
        print(f"Requested jobs:  {args.jobs} (verification is serialized)")
    if args.dry_run:
        print("Dry run:         yes")

    counters = {
        "downloaded": 0,
        "verified": 0,
        "repaired": 0,
        "skipped": 0,
        "incomplete": 0,
        "failed": 0,
        "planned": 0,
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=effective_jobs) as executor:
        future_to_spec = {}
        next_spec_index = 0

        def submit_next() -> bool:
            nonlocal next_spec_index
            if next_spec_index >= len(specs):
                return False
            spec = specs[next_spec_index]
            next_spec_index += 1
            future_to_spec[executor.submit(download_one, spec, output_dir, args)] = spec
            return True

        for _ in range(min(effective_jobs, len(specs))):
            submit_next()

        completed = 0
        while future_to_spec:
            done, _ = concurrent.futures.wait(
                future_to_spec,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                future_to_spec.pop(future)
                completed += 1
                result = future.result()
                counters[result.status] = counters.get(result.status, 0) + 1
                if not args.dry_run:
                    write_report_line(report_path, result)

                msg = (
                    f"[{completed}/{len(specs)}] "
                    f"{result.status:10} {result.raw} -> {result.dest}"
                )
                if result.message:
                    msg += f" ({result.message})"
                print(msg, flush=True)

                if (
                    result.status in DELAYED_RESULT_STATUSES
                    and next_spec_index < len(specs)
                ):
                    print(
                        "Waiting "
                        f"{NEXT_DATASET_DELAY_SECONDS}s after "
                        f"{result.status} dataset before starting next dataset...",
                        flush=True,
                    )
                    time.sleep(NEXT_DATASET_DELAY_SECONDS)

                submit_next()

    print(
        "Summary: "
        + ", ".join(f"{key}={value}" for key, value in sorted(counters.items()))
    )
    return 1 if counters.get("failed", 0) or counters.get("incomplete", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
