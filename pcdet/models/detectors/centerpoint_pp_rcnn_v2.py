from pcdet.models.dense_heads.bev_feature_extractor_v2 import BEVFeatureExtractorV2
from .detector3d_template_v2 import Detector3DTemplateV2
import numpy as np
import torch

class CenterPoint_PointPillar_RCNNV2(Detector3DTemplateV2):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.use_local_alignment = self.model_cfg.get('USE_LOCAL_ALIGNMENT', False)
        self.use_adaptive_thres = model_cfg.get('USE_ADAPTIVE_THRES', False)
        if self.use_local_alignment:
            self.dla_cfg = self.model_cfg.get('DLA_CONFIG', None)
            self.dla_model = Net_MDA()

    def forward(self, batch_dict):
        batch_dict['spatial_features_stride'] = 1
        only_domain_loss = batch_dict.get('domain_target', False)
        for idx, cur_module in enumerate(self.module_list):
            if only_domain_loss and cur_module.__class__.__name__ == 'CenterHeadRCNNV2':
                break
            if str(cur_module) == "BEVFeatureExtractorV2()" or str(cur_module) == "PVRCNNHead()":
                pred_dicts, recall_dicts = self.post_processing_for_refine(batch_dict)
                if not self.training:
                    break
                rois, roi_scores, roi_labels = self.reorder_rois_for_refining(batch_dict['batch_size'], pred_dicts)
                batch_dict['rois'] = rois
                batch_dict['roi_scores'] = roi_scores
                batch_dict['roi_labels'] = roi_labels
                batch_dict['has_class_labels'] = True
                #batch_dict['final_predict'] = pred_dicts
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(only_domain_loss)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            if False:
                pred_dicts = self.post_process(batch_dict)
                rois, roi_scores, roi_labels = self.reorder_rois_for_refining(batch_dict['batch_size'], pred_dicts)
                batch_dict['rois'] = rois
                batch_dict['roi_labels'] = roi_labels
                batch_dict['has_class_labels'] = True
                pred_dicts, recall_dicts = self.post_processing_for_roi__(batch_dict)

            return pred_dicts, recall_dicts

    def get_training_loss(self, only_domain_loss=False):
        disp_dict, tb_dict = {}, {}
        loss = torch.tensor(0.).cuda().item()

        if not only_domain_loss:
            loss_rpn, tb_dict = self.dense_head.get_loss()
            loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
            loss = loss_rpn + loss_rcnn

        if self.backbone_2d.use_domain_cls:
            loss_domain, tb_dict = self.backbone_2d.get_loss(tb_dict)
            loss = loss + loss_domain

        return loss, tb_dict, disp_dict
