import numpy as np
import torch
from torch.utils.data import IterableDataset


class MixedLeRobotAct(IterableDataset):
    """Randomly mix prebatched LeRobot datasets at their natural sample ratio."""

    def __init__(self, sources, batch_size, drop_last=True, seed=0):
        super().__init__()
        self.sources = list(sources)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self._epoch = 0

        if len(self.sources) != 2:
            raise ValueError("MixedLeRobotAct expects exactly two data sources.")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        names = [name for name, _ in self.sources]
        if len(set(names)) != len(names):
            raise ValueError(f"MixedLeRobotAct source names must be unique, got {names}")

        self.source_num_records = {}
        for name, dataset in self.sources:
            if not callable(getattr(dataset, "_build_epoch_records", None)):
                raise TypeError(f"Source '{name}' does not provide _build_epoch_records().")
            if not callable(getattr(dataset, "_build_batch", None)):
                raise TypeError(f"Source '{name}' does not provide _build_batch().")
            num_records = int(getattr(dataset, "_num_records_per_epoch", 0))
            if num_records <= 0:
                raise ValueError(f"Source '{name}' has no training records.")
            self.source_num_records[name] = num_records

        self._num_records_per_epoch = sum(self.source_num_records.values())

    @staticmethod
    def _get_shard_info():
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        return rank, world_size, worker_id, num_workers

    @staticmethod
    def _shard_records(records, rank, world_size, worker_id, num_workers):
        total_shards = world_size * num_workers
        shard_index = rank * num_workers + worker_id
        return records[shard_index::total_shards]

    def _build_source_records(self, epoch, shard_info):
        rank, world_size, worker_id, num_workers = shard_info
        return [
            self._shard_records(
                dataset._build_epoch_records(epoch),
                rank,
                world_size,
                worker_id,
                num_workers,
            )
            for _, dataset in self.sources
        ]

    @staticmethod
    def _draw_source_counts(remaining, batch_size, rng):
        first_count = int(
            rng.hypergeometric(
                ngood=int(remaining[0]),
                nbad=int(remaining[1]),
                nsample=int(batch_size),
            )
        )
        return first_count, int(batch_size) - first_count

    def __iter__(self):
        epoch = self._epoch
        self._epoch += 1

        shard_info = self._get_shard_info()
        rank, _, worker_id, num_workers = shard_info
        shard_index = rank * num_workers + worker_id
        records_by_source = self._build_source_records(epoch, shard_info)
        positions = [0, 0]
        remaining = [len(records) for records in records_by_source]
        rng = np.random.default_rng(self.seed + epoch * 1_000_003 + shard_index)

        while sum(remaining) > 0:
            current_batch_size = min(self.batch_size, sum(remaining))
            if current_batch_size < self.batch_size and self.drop_last:
                break

            source_counts = self._draw_source_counts(remaining, current_batch_size, rng)
            batch_samples = []
            for source_index, count in enumerate(source_counts):
                if count <= 0:
                    continue
                start = positions[source_index]
                end = start + count
                source_records = records_by_source[source_index][start:end]
                source_samples = self.sources[source_index][1]._build_batch(source_records)
                if len(source_samples) != count:
                    source_name = self.sources[source_index][0]
                    raise RuntimeError(
                        f"Source '{source_name}' built {len(source_samples)} samples "
                        f"for {count} records."
                    )
                batch_samples.extend(source_samples)
                positions[source_index] = end
                remaining[source_index] -= count

            order = rng.permutation(len(batch_samples))
            yield [batch_samples[int(index)] for index in order]

    def __len__(self):
        if self.drop_last:
            return self._num_records_per_epoch // self.batch_size
        return int(np.ceil(self._num_records_per_epoch / self.batch_size))
