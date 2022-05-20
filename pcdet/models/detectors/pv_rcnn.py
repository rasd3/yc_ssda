from .detector3d_template import Detector3DTemplate
from pcdet.utils.simplevis import nuscene_vis


class PVRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        if False:
            import cv2
            b_size = batch_dict['gt_boxes'].shape[0]
            for b in range(b_size):
                points = batch_dict['points'][batch_dict['points'][:, 0] == b][:, 1:4].cpu().numpy()
                gt_boxes = batch_dict['gt_boxes'][b].cpu().numpy()
                gt_boxes[:, 6] = -gt_boxes[:, 6].copy()
                det = nuscene_vis(points, gt_boxes)
                cv2.imwrite('test_%02d.png' % b, det)
            breakpoint()

        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            #  print(self.training)
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts, {}

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict
