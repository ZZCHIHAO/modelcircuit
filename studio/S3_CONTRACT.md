# S3 native chips and paired-input classification contract

2026-11-15. User approved the proposed next work and explicitly said “推进”. This is the first implementation slice of the approved roadmap: editable reusable chips and paired image/tabular input. Keep native DearPyGui, CPU/single-CUDA execution, original ml_forge sources, S1/S2 behavior and archives. No Use Computer/OS/browser input automation, no clinical/model download or long training, no GitHub upload. Existing design screenshots/code remain the visual target. Approval is already present; routine choices do not need repeated approval. Git has no HEAD/identity/remote; do not invent commits or create worktrees from an absent commit.

## Scope and ownership

- Model implementer: backend studio_core/{schema,runtime,python_io}.py, optional ports.py, focused tests for named chip outputs and static Python roundtrip. Do not edit native document/UI.
- Data implementer: experiment_spec.py, classification_data.py, classification_runtime.py, new paired_data.py and focused tests. Keep classification-v2 worker dispatch/server interfaces unchanged. Do not edit model or native.
- UI implementer: native ui.py/data_panel.py/experiment_panel.py, new chip_panel.py, focused UI tests. Do not edit document/workspace_state or backend. Compose small controls rather than make ui.py indefinitely larger.
- Root: native document.py, chip_library.py, workspace_state.py; specs/plan, integration/acceptance, main CLI if needed, docs/build/archive. Shared contract amendments are communicated before dependents change.

## Named outputs in graph, runtime, Python

- Existing schema_version=1 remains readable. Add scope-local edge `call:<call_id>:<output_id>`. Input edge remains `input:<input_id>`.
- Primitive named output is `out`. A chip's output IDs are its graph.outputs IDs. Bare `call:<id>` remains valid for a primitive or exactly one-output chip; bare references to multi-output chips are rejected as ambiguous, never implicitly select the first output.
- Shape validation checks each declared output; equivalent source aliases in one call do not duplicate tensor/clone costs. Preserve S2's conservative cumulative per-scope working budget across nested calls, rather than changing it into a global unique-storage/memory measurement. Report shapes includes qualified output references and legacy aliases for single-output calls. Invalid/nonexistent ports, cycles, recursive definitions and incompatible shapes remain errors.
- Runtime routes every named chip output, retains instance/call separation, freezes/shares the same parameter objects, keeps state_dict keys and capture/device/resource limits. Probe identities use qualified refs; existing single-output S1 probe names remain available. Internal probe paths keep `call:outer/...` semantics.
- Export multi-output chip forward as a dict; downstream code explicitly indexes the selected key, with no duplicate module invocation. Static importer recognizes only supported local dict-return/indexing patterns; unknown dynamic indexing/code stays rejected. Current source overrides stale metadata.
- Python profile `ports-pytorch-v3` is used for new named/multi-output structures; old S1/CNN profiles remain unchanged when no new structure is present. Importer accepts all three. Export/import preserves multi-output hierarchy, shared invocation objects, parameters, calls and connection ports, including when visual metadata is removed.
- Definitions may declare `output_mode:'named'` to preserve a named dict return even with one output. Absent mode means legacy tensor for exactly one output, named for multiple outputs. New editor-created/encapsulated chips use named mode; expanding an old chip sets named mode. Import infers the mode from actual return syntax. Export also upgrades legacy single-output IDs other than `out` to named returns, so meaningful names cannot disappear when metadata is removed; ordinary S1 out/tensor chips remain unchanged.
- Old public in_features input-shape convenience may remain only for single-input chips. Multi-input chips require explicit shapes and never silently edit their first input.

## Native Document commands (root provides)

All mutations are atomic on errors and preserve editable incomplete graph drafts. Public IDs are valid schema identifiers. Returned values are ordinary Python data, with no DPG/torch dependency.

```
Document.output_ports(call_id) -> list[str]
Document.source_options() -> list[str]   # scope input refs + unambiguous call output refs
Document.add_input(identifier, shape, name=None, dtype='float32')
Document.rename_input(old_id, new_id)
Document.remove_input(identifier)
Document.set_input_shape(identifier, shape)
Document.add_output(identifier, source=None, name=None)
Document.rename_output(old_id, new_id)
Document.remove_output(identifier)
Document.set_public_parameters(definition_id, parameters, bindings)
Document.encapsulate(call_ids, name, definition_id, output_call_id=None, outputs=None)
Document.undo() -> bool; Document.redo() -> bool
Document.can_undo / can_redo -> bool
Document.import_chip(path) -> definition_id
Document.save_chip(definition_id, version, library_directory) -> Path
Document.replace_graph(graph)
```

- Inputs/outputs operate on current scope, including root. Renaming an internal input updates internal sources and every call's argument key for that definition. Renaming an internal output updates qualified caller references across scopes. Adding a second output first qualifies existing bare references using the former unique output ID. Removing an input/output still consumed internally/externally raises a clear error; the user can disconnect first.
- New output source may be None as an editable draft and is connected through existing output-node controls. Root/definition interfaces must retain at least one input and output; multi-output chips are supported by core.
- `parameters` is the existing list of `{id,name?,type,default}` declarations; `bindings` maps public ID to existing local `instance_id.parameter` paths. UI edits these through rows, with JSON used only for a parameter value/shape or a list of binding paths. Invalid paths/types/dangling bindings are rejected atomically; dimension changes can leave visible downstream validation errors, consistent with existing parameter editing.
- Declared parameter type is auto/int/float/bool/shape; missing type means auto for legacy definitions. int/bool use exact Python JSON types; float allows finite int/float excluding bool; shape allows a positive int or nonempty positive-int list. replace_graph participates in semantic undo/redo (used by starter/workspace replacement).
- Encapsulation works in current scope; `outputs` maps exposed output ID to a selected call's source (`call:id` or `call:id:port`). Existing four-argument single-output form stays compatible. Boundary inputs are derived from selected calls' external sources. Copy selected instances and preserve sharing *within* the new chip; original graph is retained and the new instance is independent. No silent replacement or cross-boundary alias capture. Recursion/cycles are rejected.
- Undo/redo applies to semantic document commands (including interfaces/public parameters/import/encapsulation); bounded history <=32MiB across undo+redo, max32 states. Movement/selection/navigation are not recorded in this slice. Undo/redo uses a new monotonically increasing revision and returns to root when needed. Existing revision semantics for other commands stay compatible.

## Local chip library

`chip_library.py` defines an independent finite JSON format `{chip_format_version:1,definition_id,definitions}` containing the selected definition and all transitive definitions. No live weights/external object IDs are hidden in it. Version is explicit semantic version (e.g.1.0.0); nested definitions retain their own version.

Library file path is managed below user data/chips, with versioned immutable files. Same ID+version+same contents is idempotent; different contents require another version. Imports validate syntax/closure/limits and reject recursion. Existing identical definitions are reused; conflicting IDs are renamed consistently in all nested references, preserving the current project. Atomic save; malformed/colliding imports leave document/history unchanged.

## Paired image/tabular classification

Keep `profile:'classification-v2'`, add `data.kind:'paired_csv'` and explicit modality-to-graph-port mapping `input_ports:{image:'image',tabular:'clinical'}`. Existing `input_port:'x'` remains for old kinds; `input_ports` is None/absent for old kinds and must not change old checkpoint signatures. No fixed implicit tensor duplication.

Default paired data shape:

```json
{"kind":"paired_csv","path":"","image_root":"","image_column":"image_path","id_column":"sample_id",
 "target_column":"label","feature_columns":[],"group_column":"patient_id","split_column":null,
 "standardize":true,"image_size":[32,32],"channels":3,"validation_fraction":0.2,"test_fraction":0.2}
```

- Each CSV row defines exactly one sample, one image and numeric feature vector. sample_id is nonempty/unique; group_column is mandatory for this adapter. Explicit split values train/val/test or deterministic group-aware split. Every split contains all labels; no missing-modality skipping, cross-group split, image-path/content leakage or duplicate sample IDs. Identical numeric-only features across patients are allowed: repeated age/sex etc is not by itself a leak. Full paired inputs, group, labels and split bind dataset identity.
- image_root empty means CSV parent; resolve image paths under that explicit root, reject outside-root/symlink escape. Bounded source bytes/counts/pixels, single-frame images only; current source identity/cancellation/finite float32 checks carry over. Numeric scaler fits train only. Deterministic resize/channel/scale uses existing 2D behavior. Source edits after inspection are rejected.
- Paired manifest adds `input_shapes:{image:[null,C,H,W],tabular:[null,F]}`, and includes class_names/counts/class_counts/dataset_digest/records/preprocessing/auditable locators. Existing legacy manifest version/fields stay unchanged. Paired Dataset item is `({'image':float32_tensor,'tabular':float32_tensor},label_int,sample_id)`; one DataLoader jointly shuffles/collates both modalities.
- Config input_ports maps every modality exactly once to distinct valid graph root inputs, with exact shapes/dtypes. Runtime handles dict batches and validates batch sizes against labels, moves all tensors to selected device, then calls model(**mapped_inputs). Existing tensor batch path remains compatible. Checkpoint signatures bind mapping for paired data; S2 signatures do not change.
- True train/val/test, val-only best, independent final test, weights_only checkpoint, epoch resume, finite probes/artifact/message budgets/cancel/device behavior remain in force. Probe provenance includes both actual named inputs and the same sample IDs. No new loss/optimizer graph in this slice.

## Workspace/UI seams

- Root extends editable experiment validation for paired kind and `input_ports`, retaining empty paths/features as drafts while rejecting malformed types/ranges before DPG.
- `workspace_state.compatibility_issue(document,manifest,input_port='x',input_ports=None)` handles both paths.
- `starter_graph(manifest)` paired starter has image CNN branch + clinical MLP branch -> Concat -> logits, root inputs image and clinical. Clear two/three-row layout; no node overlap. Existing starter output/legacy tests retained.
- UI adds paired task with CSV image/id/group column selectors, image root, numeric features, size/channels and two input-port fields. Async inspection/column/device paths remain; starter replacement confirmed. Incomplete edits do not crash autosave/restore.
- UI draws every chip output as a named socket. Primitive/single-output old socket references remain compatible. Inspector output sources use source_options and visible selected ports. Clicking a multi-output chip offers explicit output probe selection.
- Add semantic undo/redo, current-scope interface editor, public-parameter/binding editor and local versioned chip save/import, accessible in existing layout. No arbitrary Python execution or new frontend stack.

## Acceptance

1. Existing backend/native regressions pass; old S1/S2 projects/checkpoints/probe names and Python profiles remain usable.
2. Two-output nested chip shares one instance across two calls; named routing/forward/gradients/update match an independently handwritten PyTorch reference. Missing/ambiguous ports fail. Generated code, edited current code and metadata-free import preserve semantics.
3. Native callbacks add/rename/remove ports, set public bindings, save/reimport versioned chips and undo/redo; failure is atomic and existing graph preserved. Reopen workspace preserves paired config. DPG-owned framebuffer only, no computer control.
4. Generated paired CSV/PNG fixtures exercise group/image leakage, source mutation, train-only scaling, sample alignment/shuffle, mapping errors, exact CPU resume, best inference and GPU single-device training. Test fixtures are never scientific performance evidence.
5. End-to-end: paired image+tabular starter -> multi-output custom chip -> share/connect -> inspect probes -> CPU/cuda:0 train -> model Python export/import -> save/reopen. Frozen CPU and CUDA EXEs execute the same actual path. Preserve S2 ZIPs, make S3 archives and fresh evidence records; no upload.

Deferred: configurable loss/optimizer graph, interactive backward/optimizer breakpoints, arbitrary Python opaque components, full training-engineering export, 3D/segmentation, LLM, DDP/remote. These remain approved roadmap items but are not claimed by this first slice.
