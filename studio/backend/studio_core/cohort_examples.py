"""Small synthetic examples for the desktop project picker and S14 checks."""

import csv
from pathlib import Path

EXAMPLES = {
    '随机森林分类': 'table-classification',
    '随机森林回归': 'table-regression',
    'CT 与表格融合': 'ct-tabular',
    'MRI 与表格融合': 'mri-tabular',
    '二维切片与三维体融合': 'ct-slice-volume',
}


def _write_rows(path, columns, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def write_demo_ct(folder, image):
    """Write a small uncompressed synthetic series in known patient space."""
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    folder.mkdir()
    series_uid, frame_uid = generate_uid(), generate_uid()
    depth = image.shape[-1]
    for z in range(depth):
        path = folder / f'slice-{depth - z}.dcm'
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        item = FileDataset(path, {}, file_meta=meta, preamble=b'\0' * 128)
        item.SOPClassUID = CTImageStorage
        item.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        item.Modality, item.SeriesInstanceUID, item.FrameOfReferenceUID = 'CT', series_uid, frame_uid
        item.ImageType = ['ORIGINAL', 'PRIMARY', 'AXIAL']
        item.Rows, item.Columns = image.shape[1], image.shape[0]
        item.SamplesPerPixel, item.PhotometricInterpretation = 1, 'MONOCHROME2'
        item.PixelRepresentation, item.BitsAllocated, item.BitsStored, item.HighBit = 1, 16, 16, 15
        item.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        item.ImagePositionPatient = [12., -3., 40. + z * 2.5]
        item.PixelSpacing, item.SliceThickness = [1.25, .75], 2.5
        item.RescaleSlope, item.RescaleIntercept = 2, -1000
        item.PixelData = image[:, :, z].T.astype('<i2').tobytes()
        item.save_as(path, enforce_file_format=True)


def table_examples(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'table.csv'
    rows = []
    for patient in range(10):
        split = 'test' if patient >= 8 else 'val' if patient >= 6 else 'train'
        for visit in range(2):
            category = 'test_only' if split == 'test' else ('site_a', 'site_b')[patient % 2]
            rows.append([f'visit-{patient}-{visit}', f'group-{patient}', split,
                         ('low', 'high')[patient % 2], patient * 1.5 + visit,
                         '' if (patient + visit) % 5 == 0 else 31 + patient * 2 + visit,
                         0.9 + 0.2 * visit, category])
    _write_rows(path, ('sample_id', 'patient_id', 'split', 'label', 'target', 'age', 'score', 'site'), rows)
    return path


def volume_examples(directory, *, mri=False, cancelled=None):
    import numpy as np

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'cases.csv'
    rows = []
    for index in range(6):
        if cancelled and cancelled():
            raise InterruptedError('Synthetic example creation cancelled')
        split = ('train', 'val', 'test')[index // 2]
        label = ('control', 'case')[index % 2]
        image = np.random.default_rng(410 + index).integers(20, 100, (13, 11, 5), dtype=np.int16)
        image[2:7, 3:8, 1:4] += 120 + (index % 2) * 250
        source = f'volume-{index}'
        sequences = []
        if mri:
            import nibabel as nib

            def save(name, pixels):
                affine = np.diag([.75, 1.25, 2.5, 1.])
                affine[:3, 3] = [12., -3., 40.]
                volume = nib.Nifti1Image(pixels.astype(np.float32), affine)
                volume.header.set_xyzt_units('mm')
                nib.save(volume, directory / name)

            source += '.nii.gz'
            sequences = [f'flair-{index}.nii.gz']
            save(source, image)
            save(sequences[0], image.astype(np.float32) ** 1.5)
        else:
            write_demo_ct(directory / source, image)
        rows.append([f'case-{index}', f'patient-{index}', split, label, source,
                     26 + index * 4, 1.5 + (index % 2), *sequences])
    _write_rows(path, ('case_id', 'patient_id', 'split', 'label', 'volume_path', 'age', 'score',
                       *(['flair'] if mri else [])), rows)
    return path


def create_cohort_example(destination, scenario, *, runtime_source=None, cancelled=None):
    from .classification_data import inspect_dataset
    from .classical_task import estimator_graph
    from .experiment_spec import default_classification_config, default_regression_config, multimodal_ports
    from .multimodal_templates import multimodal_graph
    from .research_project import export_project

    if scenario not in EXAMPLES.values():
        raise ValueError('Unknown cohort example')
    destination = Path(destination).expanduser().resolve()
    destination.mkdir()
    source = destination / 'example-data'
    if scenario.startswith('table-'):
        path = table_examples(source)
        regression = scenario == 'table-regression'
        config = default_regression_config('table_csv') if regression else default_classification_config('table_csv')
        config['data'].update(path=str(path), split_column='split', folds=3, target_column='target' if regression else 'label',
                              numeric_columns=['age', 'score'], categorical_columns=['site'])
        manifest = inspect_dataset(config['data'], config['seed'], cancelled)
        graph = estimator_graph(manifest)
    else:
        mri = scenario == 'mri-tabular'
        derived = scenario == 'ct-slice-volume'
        path = volume_examples(source, mri=mri, cancelled=cancelled)
        config = default_classification_config('multimodal_csv')
        config.update(epochs=1, batch_size=2)
        config['data'].update(path=str(path), split_column='split', modalities=['volume', 'image' if derived else 'tabular'],
                              volume_format='mri_nifti' if mri else 'dicom_ct', ct_window=None if mri else [-1000., 400.],
                              sequence_columns=['flair'] if mri else [], registered=mri, spacing_mm=[2., 1., 1.],
                              feature_columns=[] if derived else ['age', 'score'], volume_shape=[4, 8, 8],
                              image_source='center_slice' if derived else 'raster', image_size=[8, 8])
        config['input_ports'] = multimodal_ports(config['data'])
        manifest = inspect_dataset(config['data'], config['seed'], cancelled)
        graph = multimodal_graph(manifest, config['input_ports'])
    project = export_project(destination / 'project', graph, config, runtime_source=runtime_source)
    return project, path, None
