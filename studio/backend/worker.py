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
        options = {"progress": lambda event: messages.put({"type": "progress", **event}),
                   "cancelled": stop.is_set}
        from studio_core.classical_task import contains_classical
        from studio_core.inference_spec import PROFILE as INFERENCE_PROFILE
        from studio_core.project_runner import PROFILE as PROJECT_PROFILE
        from studio_core.debug_spec import PROFILE as DEBUG_PROFILE
        if config.get('profile') == DEBUG_PROFILE:
            import torch
            from studio_core.debug_runtime import run_debug
            torch.set_num_threads(1)
            messages.put({'type': 'status', 'status': 'running'})
            result = run_debug(graph, config, **options)
        elif config.get('profile') == PROJECT_PROFILE:
            if artifact_directory is None:
                raise ValueError('Project training requires a managed artifact directory')
            import torch
            from studio_core.project_runner import train_project
            torch.set_num_threads(1)
            messages.put({'type': 'status', 'status': 'running'})
            result = train_project(config['project_path'], config['data_path'], artifact_directory,
                                   source_root=config['source_root'], device=config['device'],
                                   expected_model_digest=config['expected_model_digest'], defer_report=True, **options)
        elif config.get('profile') == INFERENCE_PROFILE:
            if artifact_directory is None:
                raise ValueError('Segmentation inference requires a managed artifact directory')
            import torch
            from studio_core.segmentation_inference import run_segmentation_inference
            torch.set_num_threads(1)
            messages.put({'type': 'status', 'status': 'running'})
            result = run_segmentation_inference(config, artifact_directory, defer_report=True, **options)
        elif contains_classical(graph):
            if artifact_directory is None:
                raise ValueError("Classical worker requires a managed artifact directory")
            from studio_core.classical_task import run_classical
            messages.put({"type": "status", "status": "running"})
            result = run_classical(graph, config, artifact_directory, **options)
        else:
            import torch
            from studio_core.experiment_spec import SUPERVISED_PROFILES
            torch.set_num_threads(1)
            messages.put({"type": "status", "status": "running"})
            if config.get("profile") in SUPERVISED_PROFILES:
                if artifact_directory is None:
                    raise ValueError("Supervised worker requires a managed artifact directory")
                if config['profile'] == 'fitnets-v1':
                    from studio_core.fitnets import run_fitnets
                    result = run_fitnets(graph, config, artifact_directory, **options)
                else:
                    from studio_core.classification_runtime import run_supervised
                    result = run_supervised(graph, config, artifact_directory, **options)
            else:
                from studio_core.runtime import run_experiment
                result = run_experiment(graph, config, **options)
        messages.put({"type": "result", "result": result})
    except InterruptedError:
        messages.put({"type": "cancelled"})
    except Exception as error:
        messages.put({"type": "error", "error": str(error), "traceback": traceback.format_exc()})

