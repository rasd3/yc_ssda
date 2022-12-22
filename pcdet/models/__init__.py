from collections import namedtuple

import numpy as np
import torch

from .detectors import build_detector


def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        if key in ['frame_id', 'metadata', 'calib', 'image_shape']:
            continue
        batch_dict[key] = torch.from_numpy(val).float().cuda()


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])
    ModelReturnDLA = namedtuple('ModelReturnDLA', ['loss', 'tb_dict', 'disp_dict', 'dla_feat'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        use_local_alignment = batch_dict['use_local_alignment'] if 'use_local_alignment' in batch_dict else False
        if use_local_alignment:
            ret_dict, tb_dict, disp_dict, dla_feat = model(batch_dict)
        else:
            ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        if use_local_alignment:
            return ModelReturnDLA(loss, tb_dict, disp_dict, dla_feat)
        else:
            return ModelReturn(loss, tb_dict, disp_dict)

    return model_func
