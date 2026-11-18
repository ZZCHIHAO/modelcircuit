"""Child-process entry point. Never reads live UI state."""
import os
import traceback


def experiment_worker(graph, config, messages, stop, artifact_directory=None):
    try:
        if stop.is_set():
            raise InterruptedError("Experiment cancelled before preparation")
        messages.put({"type": "status", "status": "preparing"})
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        import torch
        torch.set_num_threads(1)
        messages.put({"type": "status", "status": "running"})
        options = {"progress": lambda event: messages.put({"type": "progress", **event}),
                   "cancelled": stop.is_set}
        if config.get("profile") == "classification-v2":
            if artifact_directory is None:
                raise ValueError("Classification worker requires a managed artifact directory")
            from studio_core.classification_runtime import run_classification
            result = run_classification(graph, config, artifact_directory, **options)
        else:
            from studio_core.runtime import run_experiment
            result = run_experiment(graph, config, **options)
        messages.put({"type": "result", "result": result})
    except InterruptedError:
        messages.put({"type": "cancelled"})
    except Exception as error:
        messages.put({"type": "error", "error": str(error), "traceback": traceback.format_exc()})

