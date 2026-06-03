"""
scan DICOMs in the test split and report any that fail to decode.
run interactively: python find_corrupt_dicoms.py
"""

import sys
import numpy as np
import pandas as pd
import pydicom
from multiprocessing import Pool
from pathlib import Path
from dotenv import load_dotenv
import os

sys.path.insert(0, os.path.expanduser('~/thesis-xai'))
load_dotenv()

KAGGLE_CACHE   = Path(os.getenv('KAGGLE_CACHE'))
RSNA_TRAIN_DIR = KAGGLE_CACHE / 'competitions/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train'
LABELS_CSV     = KAGGLE_CACHE / 'competitions/stage_2_train.csv'
INDEX_CSV      = Path.home() / 'rsna_dicom_index_2.csv'

SEED      = 20260605
N_WORKERS = max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', 4)) - 1)

label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

# reproduce the same split as 0_preprocess_rsna.py
from sklearn.model_selection import GroupShuffleSplit

dicom_index = pd.read_csv(INDEX_CSV)
dicom_index = dicom_index[dicom_index['split'] == 'train']

labels_df = pd.read_csv(LABELS_CSV)
labels_df['SubType'] = labels_df['ID'].str.rsplit('_', n=1).str[-1]
labels_df['SliceID'] = labels_df['ID'].str.rsplit('_', n=1).str[0]
labels_df = labels_df.drop_duplicates(subset=['SliceID', 'SubType'], keep='first')
labels_wide = labels_df.pivot(index='SliceID', columns='SubType', values='Label').reset_index()
labels_wide = labels_wide[['SliceID'] + label_cols + ['any']].drop(columns=['any'])
labels_wide['filename'] = labels_wide['SliceID'] + '.dcm'
labels_with_patient = labels_wide.merge(
    dicom_index[['filename', 'PatientID']], on='filename', how='left'
).dropna(subset=['PatientID'])

gss1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
dev_idx, test_idx = next(gss1.split(labels_with_patient, groups=labels_with_patient['PatientID']))
test_df = labels_with_patient.iloc[test_idx]

print(f"scanning {len(test_df)} test DICOMs with {N_WORKERS} workers...", flush=True)


def check_file(filename):
    path = RSNA_TRAIN_DIR / filename
    if not path.exists():
        return filename, 'missing'
    try:
        ds = pydicom.dcmread(str(path))
        _ = ds.pixel_array
        return filename, 'ok'
    except Exception as e:
        return filename, str(e)


with Pool(processes=N_WORKERS) as pool:
    results = pool.map(check_file, test_df['filename'].tolist(), chunksize=256)

corrupt = [(f, e) for f, e in results if e != 'ok']
print(f"\nfound {len(corrupt)} problem file(s):")
for f, e in corrupt:
    print(f"  {f}: {e}")

if not corrupt:
    print("all DICOMs decoded successfully - the original error may have been transient.")
