from pathlib import Path
import pydicom
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

#library already installed in jobindexdicom.sh

RSNA_TRAIN_DIR = Path('/rds/projects/k/karwatha-karwath-hds-pg-research/axr1222/data/competitions/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train')
RSNA_TEST_DIR  = Path('/rds/projects/k/karwatha-karwath-hds-pg-research/axr1222/data/competitions/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_test')

def read_header(args):
    filepath, split = args
    try:
        ds = pydicom.dcmread(str(filepath), stop_before_pixels=True)
        #get all ID and UID to find any group related analysis
        return {
            'filename':            filepath.name,
            'split':               split,
            'PatientID':           str(ds.PatientID),
            'StudyInstanceUID':    str(ds.StudyInstanceUID),
            'SeriesInstanceUID':   str(ds.SeriesInstanceUID),
            'SOPInstanceUID':      str(ds.SOPInstanceUID),
            'ImagePositionZ':      float(ds.ImagePositionPatient[2]),  #z only
            'InstanceNumber':      int(ds.InstanceNumber) if hasattr(ds, 'InstanceNumber') else None,
        }
    except Exception as e:
        return {'filename': filepath.name, 'split': split, 'error': str(e)}

if __name__ == '__main__':
    train_files = [(f, 'train') for f in RSNA_TRAIN_DIR.glob('*.dcm')]
    test_files  = [(f, 'test')  for f in RSNA_TEST_DIR.glob('*.dcm')]
    all_files   = train_files + test_files
    
    print(f"Train: {len(train_files)}, Test: {len(test_files)}, Total: {len(all_files)}")

    print(f"Found {len(all_files)} files")

    with Pool(processes=8) as pool: #using 8 cpus
        results = list(tqdm(pool.imap(read_header, all_files),
            total=len(all_files),
            desc="Reading DICOM headers"
        ))

    df = pd.DataFrame(results)

    #save to csv
    df.to_csv('rsna_dicom_index_2.csv', index=False)
    print(f"Done. Saved {len(df)} rows.")

    #check head
    print(df['PatientID'].value_counts().head(10))