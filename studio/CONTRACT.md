# S1 shared contract

This contract is shared by backend, code bridge and frontend. Only root may change it during parallel implementation. All examples and events are real synthetic CPU experiments, not medical research results.

## Project graph

JSON document:

```json
{
  "schema_version": 1,
  "id": "shared_classifier",
  "name": "共享双路分类器",
  "revision": 1,
  "inputs": [{"id":"a","name":"输入 A","shape":[null,4],"dtype":"float32"},{"id":"b","name":"输入 B","shape":[null,4],"dtype":"float32"}],
  "instances": [
    {"id":"encoder","kind":"chip","definition_id":"encoder_chip","name":"共享编码器","params":{"in_features":4,"hidden_features":8},"frozen":false},
    {"id":"fusion","kind":"Add","name":"相加融合","params":{}},
    {"id":"head","kind":"Linear","name":"分类头","params":{"in_features":8,"out_features":2},"frozen":false}
  ],
  "calls": [
    {"id":"encode_a","instance_id":"encoder","inputs":{"x":"input:a"},"position":{"x":250,"y":90}},
    {"id":"encode_b","instance_id":"encoder","inputs":{"x":"input:b"},"position":{"x":250,"y":300}},
    {"id":"fuse","instance_id":"fusion","inputs":{"a":"call:encode_a","b":"call:encode_b"},"position":{"x":540,"y":190}},
    {"id":"classify","instance_id":"head","inputs":{"x":"call:fuse"},"position":{"x":800,"y":190}}
  ],
  "outputs": [{"id":"logits","source":"call:classify"},{"id":"features","source":"call:fuse"}],
  "definitions": [{
    "id":"encoder_chip","name":"编码芯片","version":"0.1.0",
    "public_parameters":[{"id":"in_features","name":"输入维度","default":4},{"id":"hidden_features","name":"特征维度","default":8}],
    "bindings":{"in_features":["dense.in_features"],"hidden_features":["dense.out_features"]},
    "graph":{
      "inputs":[{"id":"x","name":"特征输入","shape":[null,4],"dtype":"float32"}],
      "instances":[{"id":"dense","kind":"Linear","name":"特征投影","params":{"in_features":4,"out_features":8}},{"id":"activation","kind":"ReLU","name":"激活","params":{}}],
      "calls":[{"id":"project","instance_id":"dense","inputs":{"x":"input:x"},"position":{"x":220,"y":180}},{"id":"activate","instance_id":"activation","inputs":{"x":"call:project"},"position":{"x":520,"y":180}}],
      "outputs":[{"id":"out","source":"call:activate"}]
    }
  }]
}
```

- IDs match `[A-Za-z_][A-Za-z0-9_]*` and are unique within their scope. Root graph/definition IDs differ from view IDs.
- `input:<id>` / `call:<id>` denote one tensor value; calls have named arguments. Root has named multiple outputs. A chip in this first profile has one tensor output.
- Supported primitive kinds: Linear(x; in_features/out_features), ReLU(x), GELU(x), Add(a,b), Concat(a,b; dim default -1), LayerNorm(x; normalized_shape), Dropout(x; p).
- Root `definitions` registry applies to nested graphs; nested graph may also contain chip instances. Reject recursive definitions.
- Bindings override target primitive parameters in a resolved instance copy. Public input dimension overrides first chip input's last dimension when `in_features` is provided.
- No automatic shape coercion or parameter rewriting. Invalid contracts return errors; unknown/missing sources/cycles cannot run.
- Node movement modifies only `position`. Semantic edits increment root `revision`.

## Backend functions

Package `studio_core` in `studio/backend`. Tests use unittest and are run with `studio/backend` in PYTHONPATH. Do not require GUI libraries.

`examples.example_graph() -> dict` returns a fresh copy of the above model.

`schema.validate_graph(graph) -> {valid:bool, errors:[{message,node_id?,field?}], warnings:[...], shapes:{"input:a":[null,4],"call:fuse":[null,8],...}, parameter_count:int|null}`. Return diagnostics for unsupported/invalid syntax and values, never silent defaults replacing an algorithm.

`runtime.build_model(graph) -> nn.Module`: `.forward(**inputs)` returns a dict keyed by root output IDs. `.components` is an nn.ModuleDict keyed by instance ID; inside chip module `.components` likewise. It must keep one module instance for two calls to the same instance ID.

`runtime.run_experiment(graph, config, progress=None, cancelled=None) -> result`. Config: `steps` 1..200 default20, `lr` 0.001..1 default0.05, `seed` 0..2^31-1 default42, `batch_size` 2..64 default16. Synthetic float inputs per root input, integer labels from input values for two classes; only this demo training task is enabled. CPU only. Progress callback(dict) optional; cancelled() bool optional. For cancellation raise `InterruptedError`.

Result envelope:

```json
{
 "status":"succeeded", "device":"cpu", "seed":42, "steps":20,
 "data_source":"synthetic_s1", "loss_before":0.8, "loss_after":0.6,
 "accuracy":0.7, "history":[{"step":1,"loss":0.8}],
 "probes":{"call:fuse":{"shape":[16,8],"dtype":"float32","count":128,"sample_size":128,"finite_count":128,"nan_count":0,"inf_count":0,"min":-1.0,"max":1.0,"mean":0.0,"std":0.5,"values":[[0.0]],"source":"measured"}},
 "parameters":[{"name":"components.encoder.components.dense.weight","shape":[8,4],"trainable":true,"grad_norm":0.1,"update_norm":0.01}],
 "parameter_count":58,"shared_instances":[{"instance_id":"encoder","call_ids":["encode_a","encode_b"]}],
 "outputs":{"logits":[[0.1,0.2]]},
 "output_metadata":{"logits":{"shape":[16,2],"count":32,"preview_count":32,"truncated":false,"source":"measured","tensor_source":"call:classify"}},
 "warnings":[]
}
```

Numeric values above are schema illustrations only. Actual results must be computed. `values` can be a bounded sample; report sample_size honestly. Report frozen gradients as unavailable or zero according to actual state. Use names/dtypes consistently with frontend.

`outputs` is a transport preview: at most 256 scalars per output and 4096 scalars across outputs. `output_metadata` gives full shape/count and the actual preview count/truncation; the illustrative `outputs` list above is abbreviated. `build_model(...).forward(...)` returns full tensors. Nonfinite outputs fail with their source location. Probe statistics use the full measured tensor, while `values` is sampled. Front/back loss and accuracy are evaluated on the same synthetic batch in eval mode; history and final gradients come from the training steps. This is not an independent test-set evaluation.

`python_io.export_python(graph) -> str`, `python_io.import_python(code) -> dict` (the reconstructed graph). Import never executes user code. Exports standalone ordinary PyTorch classes using `.components` ModuleDict so state keys match runtime. Shared instances constructed once, called multiple times. Include chip definitions/classes and direct explicit `forward` computations. Preserve original IDs via class/variable mapping or nonauthoritative metadata; reparse constructors and forward source to reflect edits. Unsupported syntax returns ValueError with readable message. Metadata must not hide changes. No metadata still reconstructs structure up to ID renaming. GUI may state R1/R2 profile limits rather than claim arbitrary import.

## HTTP

Coordinator on `127.0.0.1:8765`; client Vite on `127.0.0.1:4173` proxies `/api` to coordinator. Client uses relative requests only.

- GET `/api/health`: `{ok:true,version:"0.1.0",torch_available:bool,torch_version:string|null,capabilities:[...]}`.
- GET `/api/example`: `{graph:...}`.
- POST `/api/validate`: `{graph}` -> validation report.
- POST `/api/export`: `{graph}` -> `{code,profile:"s1-pytorch-v1"}`.
- POST `/api/import`: `{code}` -> `{graph,profile:"s1-pytorch-v1"}`.
- POST `/api/runs`: `{graph,config}` -> `{run_id,status:"queued",revision}`.
- GET `/api/runs/<id>`: `{run_id,status,revision,progress:{step,total,loss},result:null|result,error:null|string,events:[...]}`.
- POST `/api/runs/<id>/cancel`: `{}` -> current status `cancel_requested` until process exit confirms cancelled.
- Error HTTP400/422/500: `{error:"readable reason",details?:...}`. No stack traces or untrusted code execution.

Frontend starts with GET example, can persist drafts in localStorage, graph JSON open/save and downloadable Python. Current graph revision and run revision are distinct. Empty probes say 尚未采集. No fake measured plots. Unknown features are omitted or honestly disabled.

Strict finite JSON and separate byte budgets apply: request body 1,000,000 bytes; worker result 2,000,000 bytes; persisted run record and API response 8,000,000 bytes. Oversize payloads fail explicitly. At most two active runs, each with a 120-second timeout and a two-second cancellation grace period. These are bounded S1 policies, not guarantees for arbitrary algorithms or large models.

## Ownership

Core agent: schema.py, examples.py, runtime.py, __init__.py, tests/test_core.py.
Code bridge agent: python_io.py, tests/test_python_io.py only.
UI agent: frontend src files and frontend-only tests only; no package/runtime edits without root coordination.
Root: bootstrap/dependencies, coordinator/worker, API tests, docs/launch scripts, integration and QA.
