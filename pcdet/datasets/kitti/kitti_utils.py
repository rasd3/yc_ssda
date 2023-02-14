import numpy as np
import operator

from ...utils import box_utils
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from nuscenes.eval.common.data_classes import EvalBoxes
from pathlib import Path
from nuscenes import NuScenes

cls_attr_dist = {
    'barrier': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'bicycle': {
        'cycle.with_rider': 2791,
        'cycle.without_rider': 8946,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'bus': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 9092,
        'vehicle.parked': 3294,
        'vehicle.stopped': 3881,
    },
    'car': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 114304,
        'vehicle.parked': 330133,
        'vehicle.stopped': 46898,
    },
    'construction_vehicle': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 882,
        'vehicle.parked': 11549,
        'vehicle.stopped': 2102,
    },
    'ignore': {
        'cycle.with_rider': 307,
        'cycle.without_rider': 73,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 165,
        'vehicle.parked': 400,
        'vehicle.stopped': 102,
    },
    'motorcycle': {
        'cycle.with_rider': 4233,
        'cycle.without_rider': 8326,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'pedestrian': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 157444,
        'pedestrian.sitting_lying_down': 13939,
        'pedestrian.standing': 46530,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'traffic_cone': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'trailer': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 3421,
        'vehicle.parked': 19224,
        'vehicle.stopped': 1895,
    },
    'truck': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 21339,
        'vehicle.parked': 55626,
        'vehicle.stopped': 11097,
    },
}
cls_label_map = {'car' : 'car', 
                 'bicycle': 'bicycle', 
                 'pedestrian': 'pedestrian',
                 'Car': 'car',
                 'Cyclist': 'bicycle', 
                 'Pedestrian': 'pedestrian'
                 }

def transform_annotations_to_kitti_format(annos, map_name_to_kitti=None, info_with_fakelidar=False):
    """
    Args:
        annos:
        map_name_to_kitti: dict, map name to KITTI names (Car, Pedestrian, Cyclist)
        info_with_fakelidar:
    Returns:

    """
    for anno in annos:
        for k in range(anno['name'].shape[0]):
            anno['name'][k] = map_name_to_kitti[anno['name'][k]]

        anno['bbox'] = np.zeros((len(anno['name']), 4))
        anno['bbox'][:, 2:4] = 50  # [0, 0, 50, 50]
        anno['truncated'] = np.zeros(len(anno['name']))
        anno['occluded'] = np.zeros(len(anno['name']))
        if 'boxes_lidar' in anno:
            gt_boxes_lidar = anno['boxes_lidar'].copy()
        else:
            gt_boxes_lidar = anno['gt_boxes_lidar'].copy()

        if len(gt_boxes_lidar) > 0:
            if info_with_fakelidar:
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(gt_boxes_lidar)

            gt_boxes_lidar[:, 2] -= gt_boxes_lidar[:, 5] / 2
            anno['location'] = np.zeros((gt_boxes_lidar.shape[0], 3))
            anno['location'][:, 0] = -gt_boxes_lidar[:, 1]  # x = -y_lidar
            anno['location'][:, 1] = -gt_boxes_lidar[:, 2]  # y = -z_lidar
            anno['location'][:, 2] = gt_boxes_lidar[:, 0]  # z = x_lidar
            dxdydz = gt_boxes_lidar[:, 3:6]
            anno['dimensions'] = dxdydz[:, [0, 2, 1]]  # lwh ==> lhw
            anno['rotation_y'] = -gt_boxes_lidar[:, 6] - np.pi / 2.0
            anno['alpha'] = -np.arctan2(-gt_boxes_lidar[:, 1], gt_boxes_lidar[:, 0]) + anno['rotation_y']
        else:
            anno['location'] = anno['dimensions'] = np.zeros((0, 3))
            anno['rotation_y'] = anno['alpha'] = np.zeros(0)

    return annos

def transform_det_annos_to_nusc_annos(det_annos):
    nusc_annos = {
        'results': {},
        'meta': None,
    }

    for det in det_annos:
        annos = []
        box_list = boxes_lidar_to_nusenes(det)

        for k, box in enumerate(box_list):
            name = det['name'][k]
            attr = None
            attr = attr if attr is not None else max(
                cls_attr_dist[name].items(), key=operator.itemgetter(1))[0]
            nusc_anno = {
                'sample_token': det['frame_id'],
                'translation': box.center.tolist(),
                'size': box.wlh.tolist(),
                'rotation': box.orientation.elements.tolist(),
                'velocity': box.velocity[:2].tolist(),
                'detection_name': cls_label_map[name],
                'detection_score': box.score,
                'attribute_name': attr
            }
            annos.append(nusc_anno)

        nusc_annos['results'].update({det['frame_id']: annos})

    return nusc_annos

def transform_det_annos_to_nusc_gt_annos(gt_annos, det_annos):
    nusc_annos = {
        'results': {},
        'meta': None,
    }

    cls_gt_label_map = {'car' : 'car', 
                        'bicycle': 'bicycle', 
                        'pedestrian': 'pedestrian',
                        'Car': 'car',
                        'Cyclist': 'bicycle', 
                        'Pedestrian': 'pedestrian'
                        }
    for idx, det in enumerate(gt_annos):
        annos = []
        box_list = gt_boxes_lidar_to_nusenes(det)

        for k, box in enumerate(box_list):
            name = det['name'][k]
            attr = None
            if name in cls_gt_label_map:
                attr = attr if attr is not None else max(
                    cls_attr_dist[cls_gt_label_map[name]].items(), key=operator.itemgetter(1))[0]
                nusc_anno = {
                    'sample_token': det_annos[idx]['frame_id'],
                    'translation': box.center.tolist(),
                    'size': box.wlh.tolist(),
                    'rotation': box.orientation.elements.tolist(),
                    'velocity': box.velocity[:2].tolist(),
                    'detection_name': cls_gt_label_map[name],
                    'detection_score': box.score,
                    'attribute_name': attr
                }
                annos.append(nusc_anno)

        # nusc_annos['results'].update({det["metadata"]["context_name"]: annos})
        nusc_annos['results'].update({det_annos[idx]['frame_id']: annos})
        #nusc_annos['results'].update({'sample_token': annos})

    return nusc_annos

def format_kitti_results(metrics, class_names, version='default'):
    result = '----------------Nuscene %s results-----------------\n' % version
    for name in class_names:
        name = cls_label_map[name]
        threshs = ', '.join(list(metrics['label_aps'][name].keys()))
        ap_list = list(metrics['label_aps'][name].values())

        err_name = ', '.join([
            x.split('_')[0]
            for x in list(metrics['label_tp_errors'][name].keys())
        ])
        error_list = list(metrics['label_tp_errors'][name].values())

        result += f'***{name} error@{err_name} | AP@{threshs}\n'
        result += ', '.join(['%.2f' % x for x in error_list]) + ' | '
        result += ', '.join(['%.2f' % (x * 100) for x in ap_list])
        result += f" | mean AP: {metrics['mean_dist_aps'][name]}"
        result += '\n'

    result += '--------------average performance-------------\n'
    details = {}
    for key, val in metrics['tp_errors'].items():
        result += '%s:\t %.4f\n' % (key, val)
        details[key] = val

    result += 'mAP:\t %.4f\n' % metrics['mean_ap']
    result += 'NDS:\t %.4f\n' % metrics['nd_score']

    details.update({
        'mAP': metrics['mean_ap'],
        'NDS': metrics['nd_score'],
    })

    return result, details

def boxes_lidar_to_nusenes(det_info):
    box_num = det_info['score'].shape[0]
    if box_num:
        boxes3d = det_info['boxes_lidar']
        scores = det_info['score']
        labels = det_info['pred_labels']

    box_list = []
    for k in range(box_num):
        quat = Quaternion(axis=[0, 0, 1], radians=boxes3d[k, 6])
        velocity = (*boxes3d[k, 7:9],
                    0.0) if boxes3d.shape[1] == 9 else (0.0, 0.0, 0.0)
        box = Box(
            boxes3d[k, :3],
            boxes3d[k, [4, 3, 5]],  # wlh
            quat,
            label=labels[k],
            score=scores[k],
            velocity=velocity,
        )
        box_list.append(box)
    return box_list

def gt_boxes_lidar_to_nusenes(det_info):
    boxes3d = det_info['gt_boxes_lidar']
    scores = np.ones(det_info['name'].shape[0])
    label_map = {'Car':0, "Cyclist":1, "Pedestrian":2}

    box_list = []
    for k in range(boxes3d.shape[0]):
        label_k = label_map[det_info['name'][k]] if det_info['name'][k] in label_map else -1
        quat = Quaternion(axis=[0, 0, 1], radians=boxes3d[k, 6])
        velocity = (*boxes3d[k, 7:9],
                    0.0) if boxes3d.shape[1] == 9 else (0.0, 0.0, 0.0)
        box = Box(
            boxes3d[k, :3],
            boxes3d[k, [4, 3, 5]],  # wlh
            quat,
            label=label_k,
            score=scores[k],
            velocity=velocity,
        )
        box_list.append(box)
    return box_list

