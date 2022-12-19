import torch
import cv2

from .detector3d_template import Detector3DTemplate
from pcdet.utils.simplevis import nuscene_vis

from ...utils import box_coder_utils, common_utils, loss_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ..model_utils import mmd
from .dan_model import Net_MDA

class PVRCNN(Detector3DTemplate):

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg,
                         num_class=num_class,
                         dataset=dataset)
        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'backbone_2d', 'pfe',
            'dense_head', 'point_head', 'roi_head'
        ]
        self.module_list = self.build_networks()
        # Domain Local Alignment 
        self.use_local_alignment = self.model_cfg.get('USE_LOCAL_ALIGNMENT', False)
        if self.use_local_alignment:
            self.dla_cfg = self.model_cfg.get('DLA_CONFIG', None)
            self.dla_model = Net_MDA()

    def forward(self, batch_dict):
        if False:
            import cv2
            b_size = batch_dict['gt_boxes'].shape[0]
            for b in range(b_size):
                points = batch_dict['points'][batch_dict['points'][:, 0] ==
                                              b][:, 1:4].detach().cpu().numpy()
                gt_boxes = batch_dict['gt_boxes'][b].detach().cpu().numpy()
                gt_boxes[:, 6] = -gt_boxes[:, 6].copy()
                det = nuscene_vis(points, gt_boxes)
                cv2.imwrite('test_%02d.png' % b, det)

        only_domain_loss = batch_dict.get('domain_target', False)
        for idx, cur_module in enumerate(self.module_list):
            if only_domain_loss: # and self.module_topology[idx] == 'pfe':
                try:
                    batch_dict = cur_module(batch_dict, disable_gt_roi_when_pseudo_labeling=True)
                except:
                    batch_dict_ema = cur_module(batch_dict)
            else:
                batch_dict = cur_module(batch_dict)


        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(only_domain_loss)

            ret_dict = {'loss': loss}
            if self.use_local_alignment and batch_dict['use_local_alignment']:
                pred_dicts, recall_dicts = self.post_processing(batch_dict)
                batch_dict['pred_dicts'] = pred_dicts
                if False:
                    import cv2
                    b_size = len(batch_dict['frame_id'])
                    for b in range(b_size):
                        points = batch_dict['points'][batch_dict['points'][:, 0] ==
                                                      b][:, 1:4].detach().cpu().numpy()
                        if 'gt_boxes' in batch_dict:
                            gt_boxes = batch_dict['gt_boxes'][b].detach().cpu().numpy()
                            gt_boxes[:, 6] = -gt_boxes[:, 6].copy()
                            det = nuscene_vis(points, gt_boxes)
                            cv2.imwrite('test_%02d_gt.png' % b, det)
                        pred_boxes = pred_dicts[b]['pred_boxes'].detach().cpu().numpy()
                        pred_boxes[:, 6] = -pred_boxes[:, 6].copy()
                        det = nuscene_vis(points, pred_boxes)
                        cv2.imwrite('test_%02d_pred.png' % b, det)
                dla_feat = self.get_dla_features(batch_dict)
                #  dla_feat = None
                return ret_dict, tb_dict, disp_dict, dla_feat
            else:
                return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts, {}

    def get_training_loss(self, only_domain_loss=False):
        disp_dict, tb_dict = {}, {}
        loss = torch.tensor(0.).cuda()
        if not only_domain_loss:
            loss_rpn, tb_dict = self.dense_head.get_loss()
            loss_point, tb_dict = self.point_head.get_loss(tb_dict)
            loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
            loss = loss + loss_rpn + loss_point + loss_rcnn

        if self.backbone_2d.use_domain_cls:
            loss_domain, tb_dict = self.backbone_2d.get_loss(tb_dict)
            loss = loss + loss_domain

        return loss, tb_dict, disp_dict

    def get_dla_features(self, batch_dict):
        rel_thres, abs_thres = self.dla_cfg.REL_THRES, self.dla_cfg.ABS_THRES
        pred_dicts = batch_dict['pred_dicts']

        b_features = []
        if False:
            for i in range(batch_dict['gt_boxes'].shape[0]):
                mask = batch_dict['points'][:, 0] == i
                points = batch_dict['points'][mask][:, 1:]
                gt_boxes = batch_dict['gt_boxes'][i].cpu().numpy()
                gt_boxes[:, 6] = -gt_boxes[:, 6].copy()
                map1 = nuscene_vis(points.cpu().numpy(),
                                             gt_boxes)
                cv2.imwrite('test_%d.png' % i, map1)
        for b in range(len(pred_dicts)):
            pred_boxes = pred_dicts[b]['pred_boxes']
            pred_scores = pred_dicts[b]['pred_scores']
            # filter top K box according to pred_boxes
            if False:
                num_over_thres = (pred_scores < abs_thres).nonzero()[0][0]
                if num_over_thres < rel_thres:
                    breakpoint()
                    b_features.append(torch.tensor([]).cuda())
                    continue
            pred_boxes = pred_boxes[:rel_thres]
            pred_scores = pred_scores[:rel_thres]

            mask = batch_dict['points'][:, 0] == b
            points = batch_dict['points'][mask][:, 1:4]
            if False:
                pred_boxess = pred_boxes.detach().cpu().numpy()
                pred_boxess[:, 6] = -pred_boxess[:, 6].copy()
                map1 = nuscene_vis(points.cpu().numpy(),
                                             pred_boxess)
                cv2.imwrite('pred_%d.png' % b, map1)
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(points.unsqueeze(0), 
                                                                        pred_boxes.unsqueeze(0))[0]
            s_points = []
            for b_idx in range(rel_thres):
                b_mask = box_idxs_of_pts == b_idx
                b_points = points[b_mask]
                b_points_samp = common_utils.sample_points(b_points, self.dla_cfg.SAMP_METHOD, 256)
                s_points.append(b_points_samp.unsqueeze(0))
            s_points = torch.cat(s_points, dim=0).permute(0, 2, 1).unsqueeze(3)
            if batch_dict['domain_target']:
                s_features = self.dla_model(s_points, node_adaptation_t=True)
            else:
                s_features = self.dla_model(s_points, node_adaptation_s=True)
            b_features.append(s_features.unsqueeze(0))
        b_features = torch.cat(b_features, dim=0)
        return b_features
