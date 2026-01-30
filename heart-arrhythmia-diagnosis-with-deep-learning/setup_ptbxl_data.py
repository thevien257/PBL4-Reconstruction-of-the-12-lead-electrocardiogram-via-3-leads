"""
PTB-XL Dataset Setup Script

This script converts PTB-XL dataset to the format expected by the ECG reconstruction codebase.
It creates:
1. MongoDB-like data structure (stored in memory or pickle)
2. Feature map pickle files for train/valid/test splits

Usage in Colab:
    python setup_ptbxl_data.py --ptbxl_path /content/data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1

Author: Adapted for PTB-XL dataset
"""

import os
import pickle
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split


def load_ptbxl_database(ptbxl_path):
    """Load PTB-XL metadata database."""
    csv_path = os.path.join(ptbxl_path, 'ptbxl_database.csv')
    df = pd.read_csv(csv_path, index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(lambda x: eval(x))  # Convert string dict to actual dict
    return df


def load_scp_statements(ptbxl_path):
    """Load SCP statements mapping."""
    csv_path = os.path.join(ptbxl_path, 'scp_statements.csv')
    return pd.read_csv(csv_path, index_col=0)


def load_ecg_signal(ptbxl_path, filename, sampling_rate=500):
    """Load a single ECG signal from WFDB format."""
    if sampling_rate == 100:
        record_path = os.path.join(ptbxl_path, 'records100', filename)
    else:
        record_path = os.path.join(ptbxl_path, 'records500', filename)

    # Remove .dat/.hea extension if present
    record_path = record_path.replace('_lr', '').replace('_hr', '')
    if record_path.endswith('.dat') or record_path.endswith('.hea'):
        record_path = record_path[:-4]

    try:
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal  # Shape: (samples, 12)
        return signal, record.sig_name
    except Exception as e:
        print(f"Error loading {record_path}: {e}")
        return None, None


def get_diagnostic_class(scp_codes, scp_df):
    """
    Classify ECG based on SCP codes.
    Returns: 'MI' (myocardial infarction), 'STTC' (ST/T changes), 'NORM' (normal), 'OTHER'
    """
    # MI-related codes
    mi_codes = ['IMI', 'AMI', 'PMI', 'ALMI', 'ASMI', 'ILMI', 'IPMI', 'INJAL', 'INJAS',
                'INJIN', 'INJLA', 'INJIL', 'IPLMI', 'IPLI', 'STEMI']

    # ST elevation codes
    ste_codes = ['STE', 'STTC', 'STD', 'NST_', 'ISC_', 'ISCA', 'ISCI']

    for code in scp_codes.keys():
        if code in mi_codes or 'MI' in code or 'INFARCT' in code.upper():
            return 'st_elevation_or_infarct'

    for code in scp_codes.keys():
        if code in ste_codes:
            return 'st_elevation_or_infarct'

    # Check if normal
    if 'NORM' in scp_codes:
        return 'non_st_elevation_or_infarct'

    return 'non_st_elevation_or_infarct'


def convert_to_mongodb_format(df, ptbxl_path, sampling_rate=500):
    """
    Convert PTB-XL data to MongoDB-like format expected by the codebase.

    Expected format per element:
    {
        'ElementID': str,
        'lead': {
            'I': pd.Series, 'II': pd.Series, 'III': pd.Series,
            'aVL': pd.Series, 'aVR': pd.Series, 'aVF': pd.Series,
            'V1': pd.Series, 'V2': pd.Series, 'V3': pd.Series,
            'V4': pd.Series, 'V5': pd.Series, 'V6': pd.Series
        },
        'RestingECG': {
            'PatientDemographics': {'PatientID': str, 'PatientAge': int},
            'Diagnosis': {'DiagnosisStatement': str}
        },
        'diagnostic_class': str
    }
    """
    # Standard 12-lead names in WFDB PTB-XL format
    ptbxl_lead_names = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    target_lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    # Map WFDB names to target names
    lead_name_map = {
        'I': 'I', 'II': 'II', 'III': 'III',
        'AVR': 'aVR', 'AVL': 'aVL', 'AVF': 'aVF',
        'V1': 'V1', 'V2': 'V2', 'V3': 'V3',
        'V4': 'V4', 'V5': 'V5', 'V6': 'V6',
        'aVR': 'aVR', 'aVL': 'aVL', 'aVF': 'aVF'  # Already correct format
    }

    scp_df = load_scp_statements(ptbxl_path)

    data_collection = {}
    element_ids = []
    diagnostic_classes = {'st_elevation_or_infarct': [], 'non_st_elevation_or_infarct': []}

    print(f"Converting {len(df)} ECG records...")

    for idx, (ecg_id, row) in enumerate(df.iterrows()):
        if idx % 1000 == 0:
            print(f"  Processing {idx}/{len(df)}...")

        # Load ECG signal
        filename = row['filename_hr'] if sampling_rate == 500 else row['filename_lr']
        signal, sig_names = load_ecg_signal(ptbxl_path, filename, sampling_rate)

        if signal is None:
            continue

        # Create element ID
        element_id = f"ptbxl_{ecg_id}"

        # Convert signal to lead dictionary (multiply by 1000 as code divides by 1000)
        lead_dict = {}
        for i, name in enumerate(sig_names):
            mapped_name = lead_name_map.get(name, name)
            # Store as pandas Series, multiply by 1000 (code will divide by 1000)
            lead_dict[mapped_name] = pd.Series(signal[:, i] * 1000)

        # Get diagnostic class
        diag_class = get_diagnostic_class(row['scp_codes'], scp_df)

        # Create element in MongoDB format
        element = {
            'ElementID': element_id,
            'lead': lead_dict,
            'RestingECG': {
                'PatientDemographics': {
                    'PatientID': str(row['patient_id']),
                    'PatientAge': int(row['age']) if pd.notna(row['age']) else 0
                },
                'Diagnosis': {
                    'DiagnosisStatement': str(row['scp_codes'])
                }
            },
            'diagnostic_class': diag_class
        }

        data_collection[element_id] = element
        element_ids.append(element_id)
        diagnostic_classes[diag_class].append(element_id)

    print(f"Converted {len(data_collection)} ECG records successfully!")
    print(f"  ST elevation/infarct: {len(diagnostic_classes['st_elevation_or_infarct'])}")
    print(f"  Non-ST elevation/infarct: {len(diagnostic_classes['non_st_elevation_or_infarct'])}")

    return data_collection, element_ids, diagnostic_classes


def create_train_valid_test_split(element_ids, df, valid_ratio=0.15, test_ratio=0.15):
    """
    Create train/valid/test splits.
    Uses patient-based splitting to avoid data leakage.
    """
    # Get patient IDs for each element
    element_to_patient = {}
    for ecg_id, row in df.iterrows():
        element_id = f"ptbxl_{ecg_id}"
        if element_id in element_ids:
            element_to_patient[element_id] = row['patient_id']

    # Get unique patients
    patients = list(set(element_to_patient.values()))

    # Split patients
    train_patients, temp_patients = train_test_split(
        patients, test_size=(valid_ratio + test_ratio), random_state=42
    )
    valid_patients, test_patients = train_test_split(
        temp_patients, test_size=test_ratio/(valid_ratio + test_ratio), random_state=42
    )

    train_patients = set(train_patients)
    valid_patients = set(valid_patients)
    test_patients = set(test_patients)

    # Assign elements to splits
    train_ids = [eid for eid in element_ids if element_to_patient.get(eid) in train_patients]
    valid_ids = [eid for eid in element_ids if element_to_patient.get(eid) in valid_patients]
    test_ids = [eid for eid in element_ids if element_to_patient.get(eid) in test_patients]

    print(f"\nData split:")
    print(f"  Train: {len(train_ids)} ({len(train_ids)/len(element_ids)*100:.1f}%)")
    print(f"  Valid: {len(valid_ids)} ({len(valid_ids)/len(element_ids)*100:.1f}%)")
    print(f"  Test:  {len(test_ids)} ({len(test_ids)/len(element_ids)*100:.1f}%)")

    return train_ids, valid_ids, test_ids


def save_feature_maps(output_path, element_ids, train_ids, valid_ids, test_ids, diagnostic_classes):
    """Save pickle files in the expected Feature_map structure."""

    # Create directory structure
    dataset_path = os.path.join(output_path, 'Feature_map', 'Dataset')
    os.makedirs(dataset_path, exist_ok=True)

    # Save main maps
    with open(os.path.join(dataset_path, 'map.pkl'), 'wb') as f:
        pickle.dump(element_ids, f)

    with open(os.path.join(dataset_path, 'clean_map.pkl'), 'wb') as f:
        pickle.dump(element_ids, f)  # All are clean

    with open(os.path.join(dataset_path, 'corrupted_map.pkl'), 'wb') as f:
        pickle.dump([], f)  # No corrupted

    with open(os.path.join(dataset_path, 'train_map.pkl'), 'wb') as f:
        pickle.dump(train_ids, f)

    with open(os.path.join(dataset_path, 'valid_map.pkl'), 'wb') as f:
        pickle.dump(valid_ids, f)

    with open(os.path.join(dataset_path, 'test_map.pkl'), 'wb') as f:
        pickle.dump(test_ids, f)

    # Create patient map (element_id -> patient_id mapping not needed for basic run)
    with open(os.path.join(dataset_path, 'patient_map.pkl'), 'wb') as f:
        pickle.dump({}, f)

    # Save diagnostic class maps
    for class_name, class_ids in diagnostic_classes.items():
        class_path = os.path.join(output_path, 'Feature_map', 'Dataclass', class_name)
        os.makedirs(class_path, exist_ok=True)

        with open(os.path.join(class_path, 'map.pkl'), 'wb') as f:
            pickle.dump(class_ids, f)

        with open(os.path.join(class_path, 'clean_map.pkl'), 'wb') as f:
            pickle.dump(class_ids, f)

        # Split class IDs
        class_train = [eid for eid in class_ids if eid in train_ids]
        class_valid = [eid for eid in class_ids if eid in valid_ids]
        class_test = [eid for eid in class_ids if eid in test_ids]

        with open(os.path.join(class_path, 'train_map.pkl'), 'wb') as f:
            pickle.dump(class_train, f)

        with open(os.path.join(class_path, 'valid_map.pkl'), 'wb') as f:
            pickle.dump(class_valid, f)

        with open(os.path.join(class_path, 'test_map.pkl'), 'wb') as f:
            pickle.dump(class_test, f)

    print(f"\nFeature maps saved to: {output_path}/Feature_map/")


def save_data_collection(output_path, data_collection):
    """Save the data collection as pickle (replaces MongoDB)."""
    collection_path = os.path.join(output_path, 'data_collection.pkl')

    print(f"Saving data collection to {collection_path}...")
    with open(collection_path, 'wb') as f:
        pickle.dump(data_collection, f)

    print(f"Data collection saved! ({len(data_collection)} records)")
    return collection_path


def main():
    parser = argparse.ArgumentParser(description='Setup PTB-XL data for ECG reconstruction')
    parser.add_argument('--ptbxl_path', type=str, required=True,
                        help='Path to PTB-XL dataset folder')
    parser.add_argument('--output_path', type=str, default='./../Data',
                        help='Output path for processed data (default: ./../Data)')
    parser.add_argument('--sampling_rate', type=int, default=500, choices=[100, 500],
                        help='Sampling rate to use (default: 500)')
    parser.add_argument('--valid_ratio', type=float, default=0.15,
                        help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test set ratio (default: 0.15)')

    args = parser.parse_args()

    print("=" * 60)
    print("PTB-XL Dataset Setup for ECG Reconstruction")
    print("=" * 60)
    print(f"PTB-XL path: {args.ptbxl_path}")
    print(f"Output path: {args.output_path}")
    print(f"Sampling rate: {args.sampling_rate} Hz")
    print("=" * 60)

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Load PTB-XL database
    print("\n1. Loading PTB-XL database...")
    df = load_ptbxl_database(args.ptbxl_path)
    print(f"   Found {len(df)} ECG records")

    # Convert to MongoDB format
    print("\n2. Converting to MongoDB format...")
    data_collection, element_ids, diagnostic_classes = convert_to_mongodb_format(
        df, args.ptbxl_path, args.sampling_rate
    )

    # Create train/valid/test split
    print("\n3. Creating train/valid/test splits...")
    train_ids, valid_ids, test_ids = create_train_valid_test_split(
        element_ids, df, args.valid_ratio, args.test_ratio
    )

    # Save feature maps
    print("\n4. Saving feature maps...")
    save_feature_maps(args.output_path, element_ids, train_ids, valid_ids, test_ids, diagnostic_classes)

    # Save data collection
    print("\n5. Saving data collection...")
    save_data_collection(args.output_path, data_collection)

    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. The data is ready at: {args.output_path}")
    print(f"2. Run training with the existing codebase")
    print("=" * 60)


if __name__ == '__main__':
    main()
