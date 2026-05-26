"""
RSNA Intracranial Hemorrhage — Preprocessing & Cache Script
Converts DICOMs --> HU-windowed NumPy arrays and saves to .npz cache
Run via sbatch: sbatch 0_preprocess_rsna.sh
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
import pydicom
from skimage.transform import resize
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path
from dotenv import load_dotenv

#Config

SEED = 20260605
IMG_SIZE         = 256
N_TRAIN_PATIENTS = 3000
N_TEST_PATIENTS  = 750

# If a smaller cache already exists and you want to extend it, set these.
# Set both to None to always do a fresh load.
OLD_TRAIN_PATIENTS = 800   # existing cache size is 800/200
OLD_TEST_PATIENTS  = 200

label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

#Paths
sys.path.insert(0, os.path.expanduser('~/thesis-xai'))
load_dotenv()

os.environ['KAGGLEHUB_CACHE'] = os.getenv('KAGGLE_CACHE')
KAGGLE_CACHE   = Path(os.environ['KAGGLEHUB_CACHE'])
CACHE_DIR      = KAGGLE_CACHE / 'preprocessed'
RSNA_TRAIN_DIR = KAGGLE_CACHE / 'competitions/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train'
LABELS_CSV     = KAGGLE_CACHE / 'competitions/stage_2_train.csv'
INDEX_CSV      = Path.home() / 'rsna_dicom_index_2.csv'

CACHE_DIR.mkdir(parents=True, exist_ok=True)

#DICOM loader
def load_dicom_as_array(filepath, img_size):
    ds           = pydicom.dcmread(str(filepath))
    pixel_array  = ds.pixel_array.astype(np.float32)
    hu           = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
    hu           = np.clip(hu, 0, 80)           # brain window [0, 80]
    hu           = hu / 80.0                    # normalise to [0, 1]
    hu           = resize(hu, (img_size, img_size), anti_aliasing=True)
    return hu

def load_from_df(df, label_cols):
    images, labels, missing = [], [], 0
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 5000 == 0:
            print(f"  [{datetime.datetime.now().strftime('%H:%M:%S')}] {i}/{total} slices...", flush=True)
        dcm_path = RSNA_TRAIN_DIR / row['filename']
        if dcm_path.exists():
            img = load_dicom_as_array(dcm_path, img_size=IMG_SIZE)
            images.append(img)
            labels.append(row[label_cols].values.astype(np.float32))
        else:
            missing += 1
    print(f"  Loaded: {len(images)}, Missing: {missing}", flush=True)
    return np.array(images)[..., np.newaxis], np.array(labels)

#Main
print(f"\n{'='*60}")
print(f"Preprocessing started at {datetime.datetime.now()}")
print(f"Target: {N_TRAIN_PATIENTS} train patients, {N_TEST_PATIENTS} test patients")
print(f"{'='*60}\n", flush=True)

cache_file = CACHE_DIR / f'rsna_train{N_TRAIN_PATIENTS}_test{N_TEST_PATIENTS}_{IMG_SIZE}.npz'

if cache_file.exists():
    print(f"Cache already exists: {cache_file}")
    print("Nothing to do. Delete the cache file if you want to reprocess.")
    data = np.load(cache_file)
    x_train, y_train = data['x_train'], data['y_train']
    x_test,  y_test  = data['x_test'],  data['y_test']

else:
    # Load DICOM index (metadata only — filename, PatientID, split)
    print(f"Loading DICOM index from: {INDEX_CSV}", flush=True)
    dicom_index = pd.read_csv(INDEX_CSV)
    dicom_index = dicom_index[dicom_index['split'] == 'train']   # RSNA train set only

    # Load and parse labels from stage_2_train.csv (long format → wide format)
    # Mirrors notebook cells 65–70
    print(f"Loading labels from: {LABELS_CSV}", flush=True)
    labels_rsna_df = pd.read_csv(LABELS_CSV)

    labels_rsna_df['SubType'] = labels_rsna_df['ID'].str.rsplit('_', n=1).str[-1]
    labels_rsna_df['SliceID'] = labels_rsna_df['ID'].str.rsplit('_', n=1).str[0]
    labels_rsna_df = labels_rsna_df.drop_duplicates(subset=['SliceID', 'SubType'], keep='first')

    labels_wide = labels_rsna_df.pivot(index='SliceID', columns='SubType', values='Label').reset_index()
    labels_wide = labels_wide[['SliceID'] + label_cols + ['any']]
    labels_wide = labels_wide.drop(columns=['any'])  # drop 'any' — derived label, causes leakage

    # Merge with DICOM index to get PatientID per slice (notebook cell 74)
    labels_wide['filename'] = labels_wide['SliceID'] + '.dcm'
    labels_with_patient = labels_wide.merge(
        dicom_index[['filename', 'PatientID']], on='filename', how='left'
    )
    labels_with_patient = labels_with_patient.dropna(subset=['PatientID'])

    # Patient-level split — never slice-level, prevents data leakage (notebook cell 74)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(gss.split(labels_with_patient, groups=labels_with_patient['PatientID']))

    train_df = labels_with_patient.iloc[train_idx]
    test_df  = labels_with_patient.iloc[test_idx]

    train_patients_all = train_df['PatientID'].unique()
    test_patients_all  = test_df['PatientID'].unique()

    overlap = set(train_patients_all) & set(test_patients_all)
    print(f"Train-Test patient overlap: {len(overlap)} (should be 0)", flush=True)
    print(f"Available — train patients: {len(train_patients_all)}, test patients: {len(test_patients_all)}", flush=True)

    # Select N patients
    train_patients = train_patients_all[:N_TRAIN_PATIENTS]
    test_patients  = test_patients_all[:N_TEST_PATIENTS]

    train_df = train_df[train_df['PatientID'].isin(train_patients)]
    test_df  = test_df[test_df['PatientID'].isin(test_patients)]

    print(f"Selected — train slices: {len(train_df)}, test slices: {len(test_df)}", flush=True)

    # Try to extend existing smaller cache
    old_cache = CACHE_DIR / f'rsna_train{OLD_TRAIN_PATIENTS}_test{OLD_TEST_PATIENTS}_{IMG_SIZE}.npz'

    if old_cache.exists() and OLD_TRAIN_PATIENTS < N_TRAIN_PATIENTS:
        print(f"\nExtending from existing cache: {old_cache}", flush=True)
        old_data     = np.load(old_cache)
        x_train_old  = old_data['x_train']
        y_train_old  = old_data['y_train']
        x_test_old   = old_data['x_test']
        y_test_old   = old_data['y_test']

        # Patients already in old cache (first OLD_N from same split)
        old_train_patients = set(train_patients_all[:OLD_TRAIN_PATIENTS])
        old_test_patients  = set(test_patients_all[:OLD_TEST_PATIENTS])

        # Only load the NEW patients
        new_train_df = train_df[~train_df['PatientID'].isin(old_train_patients)]
        new_test_df  = test_df[~test_df['PatientID'].isin(old_test_patients)]

        print(f"New train slices to process: {len(new_train_df)}", flush=True)
        print(f"New test  slices to process: {len(new_test_df)}", flush=True)

        x_train_new, y_train_new = load_from_df(new_train_df, label_cols)
        x_test_new,  y_test_new  = load_from_df(new_test_df,  label_cols)

        x_train = np.concatenate([x_train_old, x_train_new])
        y_train = np.concatenate([y_train_old, y_train_new])
        x_test  = np.concatenate([x_test_old,  x_test_new])
        y_test  = np.concatenate([y_test_old,   y_test_new])

    else:
        # Fresh load from scratch
        print("\nNo existing cache to extend — processing DICOMs from scratch...", flush=True)
        print("Loading train set...", flush=True)
        x_train, y_train = load_from_df(train_df, label_cols)

        print("Loading test set...", flush=True)
        x_test, y_test   = load_from_df(test_df,  label_cols)

    print(f"\nSaving to cache: {cache_file}", flush=True)
    np.savez(cache_file,
             x_train=x_train, y_train=y_train,
             x_test=x_test,   y_test=y_test)
    print("Saved.", flush=True)

print(f"\nx_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"x_test:  {x_test.shape},  y_test:  {y_test.shape}")
print(f"\nPreprocessing finished at {datetime.datetime.now()}")
print("Done!")