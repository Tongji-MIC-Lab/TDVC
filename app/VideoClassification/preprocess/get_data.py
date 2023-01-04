from pathlib import Path
import json


def get_database(data, subset):
    video_ids = []
    video_paths = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            video_ids.append(key)

    return video_ids, video_paths, annotations


def get_data(annotation_path):
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    video_ids, video_paths, annotations = get_database(data, 'validation')
    return video_ids
