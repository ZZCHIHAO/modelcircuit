"""Internal task contracts; no plugin discovery or alternate training loop."""
from typing import Protocol, TypedDict


class Batch(TypedDict):
    features: dict
    targets: object
    sample_ids: list[str]


class TaskAdapter(Protocol):
    id: str
    version: int
    paired: bool
    output_name: str
    prediction_fields: tuple[str, ...]

    def validate_contract(self, resolved, shapes, manifest): ...
    def bind_batch(self, raw_batch, device) -> Batch: ...
    def loss(self, outputs, batch): ...
    def validate_outputs(self, outputs, batch, manifest): ...
    def summarize(self, outputs, batch, loss) -> dict: ...
    def merge(self, totals, batch) -> dict: ...
    def finalize(self, totals) -> dict: ...
    def progress_metrics(self, metrics) -> dict: ...
    def prediction_rows(self, outputs, batch, manifest) -> list: ...
    def dataset_details(self, manifest) -> dict: ...


def get_task_adapter(identifier, version):
    if type(version) is int and version == 1:
        if identifier == 'classification':
            from .classification_task import ClassificationTask
            return ClassificationTask
        if identifier == 'regression':
            from .regression_task import RegressionTask
            return RegressionTask
    raise ValueError('Unknown task adapter or version')
