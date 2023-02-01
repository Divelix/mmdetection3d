import numpy as np
from concurrent import futures as futures
from pathlib import Path


def get_image_index_str(img_idx):
    return '{:06d}'.format(img_idx)


def get_lidar_info_path(idx,
                         prefix,
                         dir_name='cloud',
                         file_ext='.bin',
                         training=True,
                         relative_path=True,
                         exist_check=True):
    img_idx_str = get_image_index_str(idx)
    img_idx_str += file_ext
    prefix = Path(prefix)
    if training:
        file_path = Path('training') / dir_name / img_idx_str
    else:
        file_path = Path('testing') / dir_name / img_idx_str
    if exist_check and not (prefix / file_path).exists():
        raise ValueError('file not exist: {}'.format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)


def get_velodyne_path(idx,
                      prefix,
                      training=True,
                      relative_path=True,
                      exist_check=True):
    return get_lidar_info_path(idx, prefix, 'clouds', '.bin', training, relative_path, exist_check)


def get_label_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True):
    return get_lidar_info_path(idx, prefix, 'labels', '.txt', training, relative_path, exist_check)


def get_label_anno(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip().split(' ') for line in lines]
    annotations = {
        'name': np.array([line[0] for line in lines]),
        'location': np.array([[float(info) for info in line[1:4]] for line in lines]).reshape(-1, 3),
        'dimensions': np.array([[float(info) for info in line[4:7]] for line in lines]).reshape(-1, 3),
        'rotation_z': np.array([float(line[7]) for line in lines]).reshape(-1)
    }
    return annotations


def get_lidar_info(path,
                    training=True,
                    label_info=True,
                    sample_ids=[],
                    num_worker=8,
                    relative_path=True):
    """
    LIDAR annotation format:
    {
        point_cloud: {
            num_features: 4
            sample_idx: ...
            velodyne_path: ...
        }
        annos: {
            name: [num_gt] ground truth name array
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_z: [num_gt] angle array
        }
    }
    """
    root_path = Path(path)
    def map_func(idx):
        info = {}
        info['point_cloud'] = {
            'num_features': 4, 
            'sample_idx': idx,
            'velodyne_path': get_velodyne_path(idx, path, training, relative_path)
        }

        annotations = None
        if label_info:
            label_path = get_label_path(idx, path, training, relative_path)
            if relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path)

        if annotations is not None:
            info['annos'] = annotations
        return info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, sample_ids)

    return list(image_infos)