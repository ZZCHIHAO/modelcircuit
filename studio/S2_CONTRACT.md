# S2 native classification contract

2026-11-14. User authorized continued development of the native DearPyGui workbench. No Use Computer/OS input automation; GitHub remains deferred. Preserve original `ml_forge/` business sources and S1 regression behavior. This phase adds real CSV and 2D image classification, CPU/single CUDA device, train/validation/test, checkpoint/resume. Multi-GPU execution, 3D, segmentation, LLM and arbitrary Python are not S2 promises.

## Ownership

- Model implementer: `studio_core/{schema,runtime,python_io}.py`, optional pure shape helper, focused model tests, primitive defaults in `native/document.py` only.
- Data/training implementer: `studio_core/{classification_data,classification_runtime}.py`, focused data/training tests only.
- UI implementer: native UI/new panels/workspace state and native tests; do not modify backend or primitive defaults in document.py.
- Root: `studio_core/{experiment_spec,devices}.py`, server/worker/main/packaging, integration, acceptance, docs.

## Real training config

```json
{
  "profile":"classification-v2",
  "data":{
    "kind":"csv", "path":"D:/data/features.csv",
    "target_column":"label", "feature_columns":["f1","f2"],
    "group_column":null, "split_column":null, "standardize":true,
    "validation_fraction":0.2, "test_fraction":0.2
  },
  "input_port":"x", "epochs":10, "batch_size":16,
  "lr":0.001, "optimizer":"adamw", "seed":42,
  "device":"cpu", "max_runtime_seconds":3600,
  "resume_checkpoint":null, "expected_dataset_digest":null
}
```

`device` is exactly `cpu` or `cuda:<nonnegative ordinal>`. Only one device per run. No auto fallback. Epochs1..1000, batch1..64, lr1e-6..1, optimizer sgd/adamw, seed0..2^31-1, max_runtime_seconds10..604800. Root's `experiment_spec.validate_classification_config(config)->dict` fills defaults/rejects unknown keys and invalid finite scalars, without importing torch/scanning data. Worker revalidates hardware. S1 config remains the original four scalars, without profile.

ImageFolder data replaces CSV fields with `{kind:"image_folder",path,split_mode:"stratified"|"directories",image_size:[H,W],channels:1|3,validation_fraction:0.2,test_fraction:0.2}`. `directories` means root/train/<class>, root/val/<class>, root/test/<class>. `stratified` means root/<class>/<image>. Supported common raster images only; no DICOM/NIfTI in this phase. Image size8..512 each, subject to graph/tensor budgets. CSV is numeric features + string class labels; missing/nonfinite values fail. At least two classes and a usable train/val/test partition. Explicit split_column accepts train/val/test; group_column, when supplied, must be disjoint across splits. Fraction splitting is seeded/stratified; group-aware splitting takes precedence and must report actual counts/fail missing classes.

## Data module (no torch on import or inspection)

`classification_data.inspect_dataset(data_spec, seed=42, cancelled=None)->manifest`:

- checks source availability/limits, finite CSV features or image decoding;
- creates stable sample IDs, class_names/label indices, records with split/group and source identity;
- computes dataset_digest from actual contents, effective data spec, labels and split;
- fits numeric standardization on train only (constant std ->1);
- rejects cross-split path/ID/content leakage; reports the exact offending source;
- uses bounded work (initial ceiling20k samples/512 CSV features/128 classes, image source size limits), no silent skip;
- does not require CUDA or GUI.

Manifest required keys: `version:2, kind, data_spec, seed, dataset_digest, class_names, input_shape:[null,F]|[null,C,H,W], counts:{train,val,test}, class_counts, records, preprocessing`. Records minimally `{id,split,label,group?,...source locator and digest...}`. Effective shapes, finite preprocessing and relative/absolute source mapping must be unambiguous. The implementation can add fields. Manifest/result JSON must be finite. Large per-sample data stays in artifact files rather than control messages.

`classification_data.build_datasets(manifest)->{train:Dataset,val:Dataset,test:Dataset}` lazily imports torch. Each item returns `(features_float32_tensor, class_index_int, sample_id_str)`; DataLoader uses num_workers=0. It validates the recorded source contents, rather than silently training changed files. CSV can preload its small bounded matrix. Image loading uses deterministic resize/channel conversion/scale-to-[0,1], no implicit clinical normalization or shared random augmentation.

## Training module

`classification_runtime.run_classification(graph, config, artifact_directory, progress=None, cancelled=None)->result`:

1. validate config; inspect/freeze manifest; check expected digest if provided;
2. require one mapped graph root input and logits `[batch,K]`, K=class count; graph validation must not silently adapt user connections/parameters;
3. resolve CPU/cuda:N via root `devices.resolve_device(device)->torch.device` (lazy torch);
4. build model on that device, loader workers0, SGD or AdamW, CrossEntropy;
5. train all train minibatches per epoch; evaluate val separately; choose best by val loss only;
6. atomically write `last.pt`, `best.pt`, manifest.json, run_spec.json and metrics artifacts; resume at completed-epoch boundaries with graph/data/preprocessing/optimizer identity checks and RNG/shuffle state;
7. load best, evaluate untouched test once, save predictions.csv with sample IDs/labels/predictions;
8. capture a bounded actual validation batch's tensor probes, gradients from the selected checkpoint epoch, and parameter update norms with explicit provenance.

Check cancellation during inspection/batches/evaluation; preserve last complete checkpoint. Do not evaluate test to select epochs. No silent CPU fallback, batch reduction, bad-file skipping or invented metrics. Avoid per-node full tensor cloning during every train minibatch: use model capture flag, enable it for the final actual probe forward. Progress uses `{phase,epoch,epochs,batch,batches,step,total,loss?,train_loss?,val_loss?,accuracy?}` and scalar metadata only.

Result keeps common `status,device,probes,parameters,parameter_count,shared_instances,outputs,output_metadata,warnings` so existing instruments can adapt, and adds:

```text
profile: classification-v2
data_source: csv | image_folder
history: [{epoch,train_loss,train_accuracy,val_loss,val_accuracy,step,loss}]
metrics: {train:{loss,accuracy,count},val:{...},test:{...}}
dataset: {dataset_digest,class_names,counts,manifest_path}
best_epoch, completed_epochs, global_step
artifacts: {last_checkpoint,best_checkpoint,predictions,manifest,run_spec}
probe_context: {split:"val",sample_ids,epoch,mode:"eval",device,gradient_source}
loss_before/loss_after/accuracy must not masquerade as independent test results; UI uses metrics/profile explicitly.
```

Result summaries stay below 2MB. Checkpoints are trusted local artifacts, loaded with weights_only=True and explicit map_location; no arbitrary serialized Python modules. Exact CPU epoch-boundary resume vs continuous run is tested.

## Model extension

Add PyTorch-named Conv2d, MaxPool2d, AdaptiveAvgPool2d, Flatten to schema/runtime/Python bridge/native defaults. Strict params/defaults and actual rank/channel/shape arithmetic; no tuple-output pooling. Preserve sharing and nested chips. Python tuple/list normalization and current-source edits must round-trip; unrecognized Python remains rejected. Export/import CNN profile is versioned, while S1 outputs stay compatible.

`runtime.build_model(graph, device="cpu", capture=True)->nn.Module`; default S1 behavior preserved. Device anchor buffer must be nonpersistent so exported state keys still match. `model.set_capture(enabled)` applies recursively. All primitive outputs are budget-checked before allocation, including conv/pool; disabling probes must not disable execution budget checks.

Optional graph `resource_profile:"classification-v2"` enables an explicit larger bounded profile: max tensor16,777,216 elements, cumulative134,217,728 elements, parameters20,000,000; S1 default limits remain unchanged. Null batch still budgeted at64. Propagate root limits through nested graphs; never use mutable module-global limits. Common image224 starter can use strided Conv to stay within limits. Generated/imported CNN graphs must preserve/select the appropriate resource profile without overriding edited Python semantics.

## Root integration

RunManager accepts the new profile, injects a managed artifact directory via worker keyword argument (not arbitrary user output path), freezes requested config/graph and allows long explicit per-run limits. Only one active job may reserve a given CUDA device. Original S1 test workers/signatures remain supported. Single worker+num_workers0 avoids daemon child-process problems; multi-GPU/DDP is deferred.

`devices.device_capabilities()->dict`: separate current torch runtime CUDA support from NVIDIA hardware found by nvidia-smi. UI capabilities query must run outside DPG rendering callbacks. `resolve_device` fails on unavailable/invalid CUDA; records actual device in result. No GPU probe should be advertised as real training.

## Native workspace/UI

Preserve original native layout and model/chip/Python interactions. Add Data settings modal and an Experiment panel with task, dataset summary/inspection, split/count/class information, epochs/optimizer, device refresh and selection. No new visual ideation is needed: existing native screenshot/code is the target.

Persist workspace envelope `{workspace_version:2,model:<graph>,experiment:<config>}` in explicit JSON and automatic workspace.json; accept legacy graph JSON. Python changes the model only and revalidates data compatibility. S1 remains an explicitly labeled synthetic mode. Composite run identity includes model semantics, effective experiment settings and inspected dataset digest; data/optimizer/device changes invalidate old measurements; node movement alone does not.

Provide a deliberate starter-model action from inspected data: CSV MLP or small CNN with correct input/class dimensions. Confirm before replacing an existing graph; do not silently duplicate one data tensor into the old default model's two inputs. Expose unsupported/no CUDA reason and available GPU names; do not offer a fake multi-GPU count control.

Curve/result panels show train/val separately, held-out test only at final best-checkpoint evaluation, explicit probe provenance and saved artifacts. Missing results remain empty. UI must stay responsive while scanning/training; all DPG calls on main thread.
