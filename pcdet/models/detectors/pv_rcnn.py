import torch

from .detector3d_template import Detector3DTemplate
from pcdet.utils.simplevis import nuscene_vis


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

    def forward(self, batch_dict):
        if False:
            import cv2
            b_size = batch_dict['gt_boxes'].shape[0]
            for b in range(b_size):
                points = batch_dict['points'][batch_dict['points'][:, 0] ==
                                              b][:, 1:4].cpu().numpy()
                gt_boxes = batch_dict['gt_boxes'][b].cpu().numpy()
                gt_boxes[:, 6] = -gt_boxes[:, 6].copy()
                det = nuscene_vis(points, gt_boxes)
                cv2.imwrite('test_%02d.png' % b, det)
            breakpoint()

        only_domain_loss = batch_dict.get('domain_target', False)
        for idx, cur_module in enumerate(self.module_list):
            if only_domain_loss and self.module_topology[idx] == 'pfe':
                break
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(only_domain_loss)

            ret_dict = {'loss': loss}
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            if False:
                import cv2
                b_size = batch_dict['gt_boxes'].shape[0]
                for b in range(b_size):
                    points = batch_dict['points'][batch_dict['points'][:, 0] ==
                                                  b][:, 1:4].cpu().numpy()
                    gt_boxes = batch_dict['gt_boxes'][b].cpu().numpy()
                    gt_boxes[:, 6] = -gt_boxes[:, 6].copy()
                    det = nuscene_vis(points, gt_boxes)
                    cv2.imwrite('test_%02d_gt.png' % b, det)
                    pred_boxes = pred_dicts[b]['pred_boxes'].cpu().numpy()
                    pred_boxes[:, 6] = -pred_boxes[:, 6].copy()
                    det = nuscene_vis(points, pred_boxes)
                    cv2.imwrite('test_%02d_pred.png' % b, det)
                breakpoint()
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
