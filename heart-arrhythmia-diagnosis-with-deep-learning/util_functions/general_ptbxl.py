"""
PTB-XL compatible general utilities.
Replaces MongoDB collection with pickle-based data loading.
"""

import json
import shutil
import pickle
import os


# Global data collection cache
_data_collection = None
_data_collection_path = None


def get_parent_folder():
    parent_folder = "./../Data/"
    return parent_folder


def remove_dir(folder: str):
    try:
        shutil.rmtree(folder)
    except:
        pass


def get_collection(database_params_file=None):
    """
    Returns a pickle-based collection that mimics MongoDB interface.
    """
    global _data_collection, _data_collection_path

    parent_folder = get_parent_folder()
    collection_path = os.path.join(parent_folder, 'data_collection.pkl')

    # Load if not cached or path changed
    if _data_collection is None or _data_collection_path != collection_path:
        print(f"Loading data collection from {collection_path}...")
        with open(collection_path, 'rb') as f:
            _data_collection = pickle.load(f)
        _data_collection_path = collection_path
        print(f"Loaded {len(_data_collection)} records")

    return PickleCollection(_data_collection)


class PickleCollection:
    """
    A class that mimics MongoDB collection interface but uses pickle data.
    """

    def __init__(self, data_dict):
        self.data = data_dict

    def find_one(self, query):
        """Find a single document matching the query."""
        if "_id" in query:
            element_id = query["_id"]
            return self.data.get(element_id)
        elif "ElementID" in query:
            element_id = query["ElementID"]
            return self.data.get(element_id)
        return None

    def find(self, query):
        """Find documents matching the query."""
        if "ElementID" in query and "$in" in query["ElementID"]:
            element_ids = query["ElementID"]["$in"]
            results = []
            for eid in element_ids:
                if eid in self.data:
                    results.append(self.data[eid])
            return results
        elif "_id" in query and "$in" in query["_id"]:
            element_ids = query["_id"]["$in"]
            results = []
            for eid in element_ids:
                if eid in self.data:
                    results.append(self.data[eid])
            return results
        return []


def get_twelve_keys():
    return ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def get_lead_keys(leads: str):

    if leads == 'limb':
        keys = ['I', 'II']

    elif leads == 'limb+comb(v3+v4)':
        keys = ['I', 'II', ['V3', 'V4']]

    elif leads == 'limb+v2+v4':
        keys = ['I', 'II', 'V2', 'V4']

    elif leads == 'full_limb':
        keys = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF']

    elif leads == 'limb+v1':
        keys = ['I', 'II', 'V1']

    elif leads == 'limb+v2':
        keys = ['I', 'II', 'V2']

    elif leads == 'limb+v3':
        keys = ['I', 'II', 'V3']

    elif leads == 'limb+v4':
        keys = ['I', 'II', 'V4']

    elif leads == 'limb+v5':
        keys = ['I', 'II', 'V5']

    elif leads == 'limb+v6':
        keys = ['I', 'II', 'V6']

    elif leads == 'precordial':
        keys = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    elif leads == 'full':
        keys = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    else:
        raise ValueError

    return keys


def get_data_classes(dataset: str):

    if dataset == 'infarct+other':
        data_classes = ['st_elevation_or_infarct', 'other']
    elif dataset == 'infarct+noninfarct':
        data_classes = ['st_elevation_or_infarct', 'non_st_elevation_or_infarct']
    else:
        data_classes = [None]

    return data_classes


def get_detect_classes(detect_class: str):
    detect_classes = [detect_class]
    return detect_classes


def get_value_range():
    min_value = -2.5
    amplitude = 5.0
    wave_sample = 2500
    return min_value, amplitude, wave_sample
