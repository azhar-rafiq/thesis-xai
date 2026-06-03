"""
for each patient with multiple studies, check if any subtype label
differs across studies. conflicting patients have inconsistent ICH
labels and may need to be excluded from training.

run: python check_multi_study_conflicts.py
outputs:
  multi_study_conflicts.csv  - full table of conflicting patients
  multi_study_summary.txt    - summary stats
"""

import sys
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, os.path.expanduser('~/thesis-xai'))
load_dotenv()

KAGGLE_CACHE = Path(os.getenv('KAGGLE_CACHE'))
LABELS_CSV   = KAGGLE_CACHE / 'competitions/stage_2_train.csv'
INDEX_CSV    = Path.home() / 'rsna_dicom_index_2.csv'
OUT_DIR      = Path(__file__).parent

label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

dicom_index = pd.read_csv(INDEX_CSV)
labels_df   = pd.read_csv(LABELS_CSV)

# pivot labels to wide format (one row per slice)
labels_df['SubType'] = labels_df['ID'].str.rsplit('_', n=1).str[-1]
labels_df['SliceID'] = labels_df['ID'].str.rsplit('_', n=1).str[0]
labels_df = labels_df.drop_duplicates(subset=['SliceID', 'SubType'], keep='first')
labels_wide = labels_df.pivot(index='SliceID', columns='SubType', values='Label').reset_index()
labels_wide['filename'] = labels_wide['SliceID'] + '.dcm'

# attach PatientID and StudyInstanceUID
merged = labels_wide.merge(
    dicom_index[['filename', 'PatientID', 'StudyInstanceUID']], on='filename', how='left'
).dropna(subset=['PatientID'])

# study-level labels: positive if any slice in the study is positive
study_labels = (merged
    .groupby(['PatientID', 'StudyInstanceUID'])[label_cols]
    .max()
    .reset_index())

n_patients = merged['PatientID'].nunique()
n_studies  = study_labels['StudyInstanceUID'].nunique()

# patients with more than one study
multi_study = study_labels.groupby('PatientID').filter(lambda x: len(x) > 1)
n_multi = multi_study['PatientID'].nunique()

# flag patients where any label differs across their studies
conflicts = multi_study.groupby('PatientID').filter(
    lambda g: (g[label_cols].nunique() > 1).any()
)
n_conflicts = conflicts['PatientID'].nunique()

# write summary
summary_lines = [
    f"total patients:                          {n_patients}",
    f"total studies:                           {n_studies}",
    f"avg studies per patient:                 {n_studies / n_patients:.3f}",
    f"patients with >1 study:                  {n_multi}",
    f"patients with conflicting labels:        {n_conflicts}",
]
summary_path = OUT_DIR / 'multi_study_summary.txt'
summary_path.write_text('\n'.join(summary_lines) + '\n')

# write full conflict table
conflicts_path = OUT_DIR / 'multi_study_conflicts.csv'
conflicts.sort_values(['PatientID', 'StudyInstanceUID']).to_csv(conflicts_path, index=False)

for line in summary_lines:
    print(line)
print(f"\nconflicts table -> {conflicts_path}")
print(f"summary         -> {summary_path}")
