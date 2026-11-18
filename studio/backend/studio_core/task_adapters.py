"""Internal task contracts; no plugin discovery or alternate training loop."""
from typing import Protocol, TypedDict


class Batch(TypedDict):
    features: dict
    targets: object
    sample_ids: list[str]


class BatchStats(TypedDict):
    loss_sum: float
    correct: int
    count: int


class TaskAdapter(Protocol):
    id: str
    version: int

    def validate_contract(self, resolved, shapes, manifest): ...
    def bind_batch(self, raw_batch, device) -> Batch: ...
    def loss(self, outputs, batch): ...
    def summarize(self, outputs, batch, loss) -> BatchStats: ...
    def finalize(self, totals: BatchStats) -> dict: ...


def get_task_adapter(identifier, version):
    if identifier != 'classification' or type(version) is not int or version != 1:
        raise ValueError('Unknown task adapter or version')
    from .classification_task import ClassificationTask
    return ClassificationTask
