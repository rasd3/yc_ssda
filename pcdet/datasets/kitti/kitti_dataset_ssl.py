import os
import copy
import time
import json
import pickle
from collections import defaultdict

import numpy as np
from skimage import io
from pathlib import Path

from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.eval.detection.data_classes import DetectionBox, DetectionConfig, DetectionMetrics, DetectionBox, DetectionMetricDataList
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from typing import Optional, Tuple, Dict, Any
from nuscenes import NuScenes

from pcdet.datasets.augmentor.augmentor_utils import *
from .kitti_utils import *
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..dataset import DatasetTemplate


class KittiDatasetSSL(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        map_name_to_kitti = {
            'car': 'Car',
            'pedestrian': 'Pedestrian',
            'truck': 'Truck',
            'bus': 'Bus',
            'motorcycle': 'Motorcycle',
            'bicycle': 'Cyclist'
        }
        self.class_names_k = copy.deepcopy(self.class_names)
        for idx in range(len(self.class_names_k)):
            if self.class_names_k[idx] in map_name_to_kitti:
                self.class_names_k[idx] = map_name_to_kitti[self.class_names_k[idx]]
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.shift_coor = self.dataset_cfg.get('SHIFT_COOR', None)
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
        self.repeat = self.dataset_cfg.REPEAT

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        # self.test = self.split == 'test' or self.split == 'val'

        if not self.training:
            self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None
        else:
            self.sample_id_list = [x.strip().split(' ')[0] for x in
                               open(split_dir).readlines()] if split_dir.exists() else None
            self.sample_index_list = [x.strip().split(' ')[1] for x in
                                  open(split_dir).readlines()] if split_dir.exists() else None

        self.kitti_infos = []
        self.include_kitti_data(self.mode)

        if self.training:
            all_train = len(self.kitti_infos)
            sample_index_list_i = [int(a) for a in self.sample_index_list]
            self.unlabeled_index_list = list(set(list(range(all_train))) - set(sample_index_list_i))  # float()!!!
            # print(self.unlabeled_index_list)
            self.unlabeled_kitti_infos = []

            temp = []
            for i in self.sample_index_list:
                temp.append(self.kitti_infos[int(i)])
            if len(self.sample_index_list) < 3712: # not 100%
                for i in self.unlabeled_index_list:
                    self.unlabeled_kitti_infos.append(self.kitti_infos[int(i)])
            else:
                self.unlabeled_index_list = list(range(len(self.sample_index_list)))
                for i in self.sample_index_list:
                    self.unlabeled_kitti_infos.append(self.kitti_infos[int(i)])
                print("full set", len(self.unlabeled_kitti_infos))
            self.kitti_infos = temp
            assert len(self.kitti_infos) == len(self.sample_id_list)
            self.labeled_infos = self.sample_index_list
            self.unlabeled_infos = self.unlabeled_kitti_infos
        self.labeled_infos = self.kitti_infos

        if self.logger is not None:
            self.logger.info('Total samples for KITTI dataset: %d' % (len(self.kitti_infos)))

    def include_kitti_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading KITTI SSL dataset')
        kitti_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)

        self.kitti_infos.extend(kitti_infos)

        # if self.logger is not None:
        #     self.logger.info('Total samples for KITTI dataset: %d' % (len(kitti_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names_k, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_image_shape(self, idx):
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists()
        return object3d_kitti.get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    def get_road_plane(self, idx):
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            info['image'] = image_info
            calib = self.get_calib(sample_idx)

            P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}

            info['calib'] = calib_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                loc_lidar = calib.rect_to_lidar(loc)
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                loc_lidar[:, 2] += h[:, 0] / 2
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar

                info['annos'] = annotations

                if count_inside_pts:
                    points = self.get_lidar(sample_idx)
                    calib = self.get_calib(sample_idx)
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])

                    fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                    pts_fov = points[fov_flag]
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('kitti_dbinfos_%s_%d.pkl' % (split, len(self.sample_index_list)))

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        # with open(info_path, 'rb') as f:
        #     infos = pickle.load(f)
        infos = self.kitti_infos

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict, shift_coor):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            if self.shift_coor:
                pred_boxes[:, :3] -= np.array(self.shift_coor, dtype=np.float32)

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index]
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict, self.shift_coor)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def kitti_eval(self, eval_det_annos, class_names=None, **kwargs):
        from .kitti_object_eval_python import eval as kitti_eval
        map_name_to_kitti = {
            'car': 'Car',
            'truck': 'Truck',
            'bus': 'Bus',
            'motorcycle': 'Motorcycle',
            'pedestrian': 'Pedestrian',
            'bicycle': 'Cyclist',
        }
        for anno in eval_det_annos:
            for k in range(anno['name'].shape[0]):
                if anno['name'][k] in map_name_to_kitti:
                    anno['name'][k] = map_name_to_kitti[anno['name'][k]]
                else:
                    anno['name'][k] = 'Person_sitting'
        for idx in range(len(class_names)):
            if class_names[idx] in map_name_to_kitti:
                class_names[idx] = map_name_to_kitti[class_names[idx]]

        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

    def nuscene_eval(self, eval_det_annos, class_names=None, **kwargs):
        import json
        from nuscenes.nuscenes import NuScenes
        from . import kitti_utils
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]

        map_name_to_kitti = {
            'Car': 'car',
            'Pedestrian': 'pedestrian',
            'Cyclist': 'bicycle',
        }
        for anno in eval_det_annos:
            for k in range(anno['name'].shape[0]):
                if anno['name'][k] in map_name_to_kitti:
                    anno['name'][k] = map_name_to_kitti[anno['name'][k]]
        for anno in eval_gt_annos:
            for k in range(anno['name'].shape[0]):
                if anno['name'][k] in map_name_to_kitti:
                    anno['name'][k] = map_name_to_kitti[anno['name'][k]]

        for idx in range(len(class_names)):
            if class_names[idx] in map_name_to_kitti:
                class_names[idx] = map_name_to_kitti[class_names[idx]]

        nusc = NuScenes(version='for_nusc_eval',
                        dataroot=str(self.root_path),
                        verbose=True)
        nusc_annos = kitti_utils.transform_det_annos_to_nusc_annos(
            eval_det_annos)
        nusc_gt_annos = kitti_utils.transform_det_annos_to_nusc_gt_annos(
            eval_gt_annos[:len(eval_det_annos)], eval_det_annos)

        nusc_annos['meta'] = {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        }

        nusc_gt_annos['meta'] = {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        }

        # output_path = Path(kwargs['output_path'])
        output_path = Path("./")
        output_path.mkdir(exist_ok=True, parents=True)
        res_path = str(output_path / 'results_pred_nusc.json')
        with open(res_path, 'w') as f:
            json.dump(nusc_annos, f)
        
        res_gt_path = str(output_path / 'results_gt_nusc.json')
        with open(res_gt_path, 'w') as f:
            json.dump(nusc_gt_annos, f)

        self.logger.info(
            f'The predictions of NuScenes have been saved to {res_path}')

        # if self.dataset_cfg.VERSION == 'waymo_processed_data_v0_5_0':
        #     return 'No ground-truth annotations for evaluation', {}

        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.detection.evaluate import NuScenesEval

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
            'waymo_processed_data_v0_5_0': 'val',
            'v1.0-test': 'test'
        }
        try:
            eval_version = 'detection_cvpr_2019'
            eval_config = config_factory(eval_version)
        except:
            eval_version = 'cvpr_2019'
            eval_config = config_factory(eval_version)

        nusc_eval = NuScenesWaymoEval(
            nusc,
            config=eval_config,
            result_path=res_path,
            gt_path=res_gt_path,
            eval_set='val',
            output_dir=str(output_path),
            verbose=True,
        )
        metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)

        with open(output_path / 'metrics_summary.json', 'r') as f:
            metrics = json.load(f)

        result_str, result_dict = kitti_utils.format_kitti_results(
            metrics, self.class_names, version=eval_version)
        return result_str, result_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.kitti_infos[0].keys():
            return None, {}

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = self.kitti_eval(eval_det_annos, class_names)
        elif kwargs['eval_metric'] == 'nuscenes':
            ap_result_str, ap_dict = self.nuscene_eval(eval_det_annos, class_names)

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs

        if self.training:
            return len(self.kitti_infos) * self.repeat
        else:
            return len(self.kitti_infos)


    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        index = index % len(self.kitti_infos)

        info = copy.deepcopy(self.kitti_infos[index])
        data_dict_labeled = self.get_item_single(info)
        if self.shift_coor:
            data_dict_labeled['points'][:, :3] += np.array(self.shift_coor, dtype=np.float32)
            data_dict_labeled['gt_boxes'][:, :3] += np.array(self.shift_coor, dtype=np.float32)

        if self.training:
            #  index_unlabeled = np.random.choice(self.unlabeled_index_list, 1)[0]
            index_unlabeled = np.random.randint(len(self.unlabeled_kitti_infos))
            info_unlabeled = copy.deepcopy(self.unlabeled_kitti_infos[index_unlabeled])

            data_dict_unlabeled = self.get_item_single(info_unlabeled, no_db_sample=True)
            if self.shift_coor:
                data_dict_unlabeled['points'][:, :3] += np.array(self.shift_coor, dtype=np.float32)
                data_dict_unlabeled['gt_boxes'][:, :3] += np.array(self.shift_coor, dtype=np.float32)
            return [data_dict_labeled, data_dict_unlabeled]
        else:
            return data_dict_labeled

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        # print(batch_list)
        if isinstance(batch_list[0], list):
            for cur_sample in batch_list:
                for key, val in cur_sample[0].items():
                    data_dict[key].append(val)
                data_dict['mask'].append(np.ones([len(batch_list)]))
                for key, val in cur_sample[1].items():
                    data_dict[key].append(val)
                data_dict['mask'].append(np.zeros([len(batch_list)]))
            batch_size = len(batch_list) * 2
        else:
            for cur_sample in batch_list:
                for key, val in cur_sample.items():
                    data_dict[key].append(val)
            batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points', 'voxels_ema', 'voxel_num_points_ema']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords', 'points_ema', 'voxel_coords_ema']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes', 'gt_boxes_ema']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret

    def get_item_single(self, info, no_db_sample=False):
        sample_idx = info['point_cloud']['lidar_idx']

        points = self.get_lidar(sample_idx)
        calib = self.get_calib(sample_idx)

        img_shape = info['image']['image_shape']
        if self.dataset_cfg.FOV_POINTS_ONLY:
            pts_rect = calib.lidar_to_rect(points[:, 0:3])
            fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
            points = points[fov_flag]

        input_dict = {
            'points': points,
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        data_dict = self.prepare_data(data_dict=input_dict, no_db_sample=no_db_sample)
        # if isinstance(data_dict, list):
        #     print(data_dict)
        data_dict['image_shape'] = img_shape
        return data_dict


    def prepare_data(self, data_dict, no_db_sample=False):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array([n in self.class_names_k for n in data_dict['gt_names']], dtype=np.bool_)

            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                },
                no_db_sample=no_db_sample
            )
            # print(data_dict)
            points_ema = data_dict['points'].copy()
            gt_boxes_ema = data_dict['gt_boxes'].copy()
            gt_boxes_ema, points_ema, _ = global_scaling(gt_boxes_ema, points_ema, [0, 2],
                                                         scale_=1/data_dict['scale'])
            gt_boxes_ema, points_ema, _ = global_rotation(gt_boxes_ema, points_ema, [-1, 1],
                                                          rot_angle_=-data_dict['rot_angle'])
            gt_boxes_ema, points_ema, _ = random_flip_along_x(gt_boxes_ema, points_ema, enable_=data_dict['flip_x'])
            gt_boxes_ema, points_ema, _ = random_flip_along_y(gt_boxes_ema, points_ema, enable_=data_dict['flip_y'])
            data_dict['points_ema'] = points_ema
            data_dict['gt_boxes_ema'] = gt_boxes_ema

        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names_k)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            
            if self.training:
                data_dict['gt_boxes_ema'] = data_dict['gt_boxes_ema'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([self.class_names_k.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes
            if self.training:
                gt_boxes_ema = np.concatenate((data_dict['gt_boxes_ema'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
                data_dict['gt_boxes_ema'] = gt_boxes_ema

        # print((data_dict['points'] ** 2).sum(), (data_dict['points_ema'] ** 2).sum()*(data_dict['scale']**2))
        if self.training:
            points = data_dict['points'].copy()
            gt_boxes = data_dict['gt_boxes'].copy()
            # points_ema = data_dict['points_ema'].copy()
            data_dict['points'] = data_dict['points_ema']
            data_dict['gt_boxes'] = data_dict['gt_boxes_ema']
        data_dict = self.point_feature_encoder.forward(data_dict)
        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        if self.training:
            data_dict['points_ema'] = data_dict['points']
            data_dict['gt_boxes_ema'] = data_dict['gt_boxes']
            data_dict['voxels_ema'] = data_dict['voxels']
            data_dict['voxel_coords_ema'] = data_dict['voxel_coords']
            data_dict['voxel_num_points_ema'] = data_dict['voxel_num_points']

            data_dict['points'] = points
            data_dict['gt_boxes'] = gt_boxes
            data_dict.pop('voxels', None)
            data_dict.pop('voxel_coords', None)
            data_dict.pop('voxel_num_points', None)
            data_dict = self.point_feature_encoder.forward(data_dict)
            data_dict = self.data_processor.forward(
                data_dict=data_dict
            )

        '''if self.training:
            if no_db_sample:
                data_dict['gt_boxes_ema'].fill(0)
                data_dict['gt_boxes'].fill(0)'''

        # TODO: currently commented out to prevent bug
        # if self.training and len(data_dict['gt_boxes']) == 0:
        #    new_index = np.random.randint(self.__len__())
        #    return self.__getitem__(new_index)

        data_dict.pop('gt_names', None)

        return data_dict

def category_to_detection_name_waymo(category_name: str) -> Optional[str]:
    """
    Default label mapping from nuScenes to nuScenes detection classes.
    Note that pedestrian does not include personal_mobility, stroller and wheelchair.
    :param category_name: Generic nuScenes class.
    :return: nuScenes detection class.
    """
    detection_mapping = {
        'Vehicle': 'car',
        'Pedestrian': 'pedestrian',
        'Cyclist': 'motorcycle',
    }

    if category_name in detection_mapping:
        return detection_mapping[category_name]
    else:
        return None

class DetectionWaymoEval(DetectionEval):
    """
    dumy class
    """

    def __init__(self,
                 nusc: NuScenes,
                 config: DetectionConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str = None,
                 use_smAP: bool = False,
                 verbose: bool = True):
        pass

class NuScenesWaymoEval(DetectionWaymoEval):
    def __init__(self,
                 nusc: NuScenes,
                 config: DetectionConfig,
                 result_path: str,
                 gt_path: str,
                 eval_set: str,
                 output_dir: str = None,
                 verbose: bool = True):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        self.nusc = nusc
        self.result_path = result_path
        self.gt_path=gt_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')
        self.pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, DetectionBox,
                                                     verbose=verbose)
        # self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox, verbose=verbose)
        self.gt_boxes, self.meta = load_prediction(self.gt_path, self.cfg.max_boxes_per_sample, DetectionBox,
                                                     verbose=verbose)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        # Add center distances.
        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)

        self.sample_tokens = self.gt_boxes.sample_tokens

    def evaluate(self) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = DetectionMetricDataList()
        self.cfg.class_names = ['car', 'pedestrian', 'bicycle']
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md = accumulate(self.gt_boxes, self.pred_boxes, class_name, self.cfg.dist_fcn_callable, dist_th)
                metric_data_list.set(class_name, dist_th, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            # Compute APs.
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

    def render(self, metrics: DetectionMetrics, md_list: DetectionMetricDataList) -> None:
        """
        Renders various PR and TP curves.
        :param metrics: DetectionMetrics instance.
        :param md_list: DetectionMetricDataList instance.
        """
        if self.verbose:
            print('Rendering PR and TP curves')

    def main(self,
             plot_examples: int = 0,
             render_curves: bool = True) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param plot_examples: How many example visualizations to write to disk.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: A dict that stores the high-level metrics and meta data.
        """
        # Run evaluation.
        metrics, metric_data_list = self.evaluate()

        # Render PR and TP curves.
        if render_curves:
            self.render(metrics, metric_data_list)

        # Dump the metric data, meta and metrics to disk.
        if self.verbose:
            print('Saving metrics to: %s' % self.output_dir)
        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = self.meta.copy()
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # Print high-level metrics.
        print('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
            'attr_err': 'mAAE'
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        print('NDS: %.4f' % (metrics_summary['nd_score']))
        print('Eval time: %.1fs' % metrics_summary['eval_time'])

        # Print per-class metrics.
        print()
        print('Per-class results:')
        print('Object Class\tAP\tATE\tASE\tAOE\tAVE\tAAE')
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
                  % (class_name, class_aps[class_name],
                     class_tps[class_name]['trans_err'],
                     class_tps[class_name]['scale_err'],
                     class_tps[class_name]['orient_err'],
                     class_tps[class_name]['vel_err'],
                     class_tps[class_name]['attr_err']))

        return metrics_summary

def add_center_dist(nusc: NuScenes,
                    eval_boxes: EvalBoxes):
    """
    Adds the cylindrical (xy) center distance from ego vehicle to each box.
    :param nusc: The NuScenes instance.
    :param eval_boxes: A set of boxes, either GT or predictions.
    :return: eval_boxes augmented with center distances.
    """
    for sample_token in eval_boxes.sample_tokens:
        # sample_rec = nusc.get('sample', sample_token)
        # sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        # pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

        for box in eval_boxes[sample_token]:
            # Both boxes and ego pose are given in global coord system, so distance can be calculated directly.
            # Note that the z component of the ego pose is 0.
            ego_translation = (box.translation[0],
                               box.translation[1],
                               box.translation[2])
            if isinstance(box, DetectionBox) or isinstance(box, TrackingBox):
                box.ego_translation = ego_translation
            else:
                raise NotImplementedError

    return eval_boxes

def filter_eval_boxes(nusc: NuScenes,
                      eval_boxes: EvalBoxes,
                      max_dist: Dict[str, float],
                      verbose: bool = False) -> EvalBoxes:
    """
    Applies filtering to boxes. Distance, bike-racks and points per box.
    :param nusc: An instance of the NuScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param max_dist: Maps the detection name to the eval distance threshold for that class.
    :param verbose: Whether to print to stdout.
    """
    # Retrieve box type for detectipn/tracking boxes.
    class_field = _get_box_class_field(eval_boxes)

    # Accumulators for number of filtered boxes.
    total, dist_filter, point_filter, bike_rack_filter = 0, 0, 0, 0
    for ind, sample_token in enumerate(eval_boxes.sample_tokens):

        # Filter on distance first.
        total += len(eval_boxes[sample_token])
        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if
                                          box.ego_dist < max_dist[box.__getattribute__(class_field)]]
        dist_filter += len(eval_boxes[sample_token])

        # Then remove boxes with zero points in them. Eval boxes have -1 points by default.
        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if not box.num_pts == 0]
        point_filter += len(eval_boxes[sample_token])

        # Perform bike-rack filtering.
        # sample_anns = nusc.get('sample', sample_token)['anns']
        # bikerack_recs = [nusc.get('sample_annotation', ann) for ann in sample_anns if
        #                  nusc.get('sample_annotation', ann)['category_name'] == 'static_object.bicycle_rack']
        # bikerack_boxes = [Box(rec['translation'], rec['size'], Quaternion(rec['rotation'])) for rec in bikerack_recs]
        filtered_boxes = []
        for box in eval_boxes[sample_token]:
            # if box.__getattribute__(class_field) in ['bicycle', 'motorcycle']:
            #     in_a_bikerack = True
            #     # for bikerack_box in bikerack_boxes:
            #     #     if np.sum(points_in_box(bikerack_box, np.expand_dims(np.array(box.translation), axis=1))) > 0:
            #     #         in_a_bikerack = True
            #     if not in_a_bikerack:
            #         filtered_boxes.append(box)
            # else:
            #     filtered_boxes.append(box)
            filtered_boxes.append(box)

        eval_boxes.boxes[sample_token] = filtered_boxes
        bike_rack_filter += len(eval_boxes.boxes[sample_token])

    if verbose:
        print("=> Original number of boxes: %d" % total)
        print("=> After distance based filtering: %d" % dist_filter)
        print("=> After LIDAR and RADAR points based filtering: %d" % point_filter)
        print("=> After bike rack filtering: %d" % bike_rack_filter)

    return eval_boxes

def load_prediction(result_path: str, max_boxes_per_sample: int, box_cls, verbose: bool = False) \
        -> Tuple[EvalBoxes, Dict]:
    """
    Loads object predictions from file.
    :param result_path: Path to the .json result file provided by the user.
    :param max_boxes_per_sample: Maximim number of boxes allowed per sample.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The deserialized results and meta data.
    """

    # Load from file and check that the format is correct.
    with open(result_path) as f:
        data = json.load(f)
    assert 'results' in data, 'Error: No field `results` in result file. Please note that the result format changed.' \
                              'See https://www.nuscenes.org/object-detection for more information.'

    # Deserialize results and get meta data.
    all_results = EvalBoxes.deserialize(data['results'], box_cls)
    meta = data['meta']
    if verbose:
        print("Loaded results from {}. Found detections for {} samples."
              .format(result_path, len(all_results.sample_tokens)))

    # Check that each sample has no more than x predicted boxes.
    for sample_token in all_results.sample_tokens:
        assert len(all_results.boxes[sample_token]) <= max_boxes_per_sample, \
            "Error: Only <= %d boxes per sample allowed!" % max_boxes_per_sample

    return all_results, meta

def _get_box_class_field(eval_boxes: EvalBoxes) -> str:
    """
    Retrieve the name of the class field in the boxes.
    This parses through all boxes until it finds a valid box.
    If there are no valid boxes, this function throws an exception.
    :param eval_boxes: The EvalBoxes used for evaluation.
    :return: The name of the class field in the boxes, e.g. detection_name or tracking_name.
    """
    assert len(eval_boxes.boxes) > 0
    box = None
    for val in eval_boxes.boxes.values():
        if len(val) > 0:
            box = val[0]
            break
    if isinstance(box, DetectionBox):
        class_field = 'detection_name'
    elif isinstance(box, TrackingBox):
        class_field = 'tracking_name'
    else:
        raise Exception('Error: Invalid box type: %s' % box)

    return class_field

def create_kitti_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = KittiDatasetSSL(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('kitti_infos_%s.pkl' % train_split)
    val_filename = save_path / ('kitti_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'kitti_infos_trainval.pkl'
    test_filename = save_path / 'kitti_infos_test.pkl'

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    kitti_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    print('Kitti info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    kitti_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)
    print('Kitti info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(kitti_infos_train + kitti_infos_val, f)
    print('Kitti info trainval file is saved to %s' % trainval_filename)

    dataset.set_split('test')
    kitti_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(kitti_infos_test, f)
    print('Kitti info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_kitti_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_kitti_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'kitti',
            save_path=ROOT_DIR / 'data' / 'kitti'
        )
