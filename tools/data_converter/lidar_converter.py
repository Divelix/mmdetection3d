from pathlib import Path
from concurrent import futures as futures

import mmcv
import numpy as np

from mmdet3d.core.bbox import box_np_ops
from .lidar_data_utils import get_lidar_info
from .kitti_converter import _read_imageset_file


def _calculate_num_points_in_gt(data_path,
                                infos,
                                relative_path,
                                num_features=4):
    for info in mmcv.track_iter_progress(infos):
        # print("------------------- info -------------------")
        # import pprint 
        # pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(info)
        
        pc_info = info['point_cloud']
        if relative_path:
            v_path = str(Path(data_path) / pc_info['velodyne_path'])
        else:
            v_path = pc_info['velodyne_path']
        points_v = np.fromfile(v_path, dtype=np.float32, count=-1).reshape([-1, num_features])
        annos = info['annos']
        dims = annos['dimensions']
        loc = annos['location']
        rots = annos['rotation_z']
        gt_boxes_lidar = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
        indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)
        num_points_in_gt = indices.sum(0)
        print(f" | num_points_in_gt: {num_points_in_gt}")
        # num_ignored = len(annos['dimensions']) - num_obj
        # num_points_in_gt = np.concatenate([num_points_in_gt, -np.ones([num_ignored])])
        annos['num_points_in_gt'] = num_points_in_gt.astype(np.int32)


def create_lidar_info_file(data_path,
                           pkl_prefix='lidar',
                           save_path=None,
                           relative_path=True):
    """Create info file of LIDAR dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        data_path (str): Path of the data root.
        pkl_prefix (str, optional): Prefix of the info file to be generated.
            Default: 'kitti'.
        save_path (str, optional): Path to save the info file.
            Default: None.
        relative_path (bool, optional): Whether to use relative path.
            Default: True.
    """
    imageset_folder = Path(data_path) / 'ImageSets'
    train_img_ids = _read_imageset_file(str(imageset_folder / 'train.txt'))
    val_img_ids = _read_imageset_file(str(imageset_folder / 'val.txt'))
    test_img_ids = _read_imageset_file(str(imageset_folder / 'test.txt'))

    print('Generate info. this may take several minutes.')
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)

    lidar_infos_train = get_lidar_info(
        data_path,
        training=True,
        sample_ids=train_img_ids,
        relative_path=relative_path)
    _calculate_num_points_in_gt(data_path, lidar_infos_train, relative_path)
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'Kitti info train file is saved to {filename}')
    mmcv.dump(lidar_infos_train, filename)

    lidar_infos_val = get_lidar_info(
        data_path,
        training=True,
        sample_ids=val_img_ids,
        relative_path=relative_path)
    _calculate_num_points_in_gt(data_path, lidar_infos_val, relative_path)
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'Kitti info val file is saved to {filename}')
    mmcv.dump(lidar_infos_val, filename)

    filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
    print(f'Kitti info trainval file is saved to {filename}')
    mmcv.dump(lidar_infos_train + lidar_infos_val, filename)

    lidar_infos_test = get_lidar_info(
        data_path,
        training=False,
        label_info=False,
        sample_ids=test_img_ids,
        relative_path=relative_path)
    filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    print(f'Kitti info test file is saved to {filename}')
    mmcv.dump(lidar_infos_test, filename)
