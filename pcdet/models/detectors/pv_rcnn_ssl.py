import os

import torch
import copy

from pcdet.datasets.augmentor.augmentor_utils import *
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from .detector3d_template import Detector3DTemplate
from .pv_rcnn import PVRCNN
from pcdet.utils.simplevis import nuscene_vis


class PVRCNN_SSL(Detector3DTemplate):

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg,
                         num_class=num_class,
                         dataset=dataset)
        # something changes so need deep copy
        model_cfg_copy = copy.deepcopy(model_cfg)
        dataset_copy = copy.deepcopy(dataset)
        self.pv_rcnn = PVRCNN(model_cfg=model_cfg,
                              num_class=num_class,
                              dataset=dataset)

        self.pv_rcnn_ema = PVRCNN(model_cfg=model_cfg_copy,
                                  num_class=num_class,
                                  dataset=dataset_copy)
        for param in self.pv_rcnn_ema.parameters():
            param.detach_()
        self.add_module('pv_rcnn', self.pv_rcnn)
        self.add_module('pv_rcnn_ema', self.pv_rcnn_ema)

        # self.module_list = self.build_networks()
        # self.module_list_ema = self.build_networks()
        self.thresh = model_cfg.THRESH
        self.num_class = num_class
        self.sem_thresh = model_cfg.SEM_THRESH
        self.unlabeled_supervise_cls = model_cfg.UNLABELED_SUPERVISE_CLS
        self.unlabeled_supervise_refine = model_cfg.UNLABELED_SUPERVISE_REFINE
        self.unlabeled_supervise_box = True
        self.unlabeled_weight = model_cfg.UNLABELED_WEIGHT
        self.no_nms = model_cfg.NO_NMS
        self.supervise_mode = model_cfg.SUPERVISE_MODE
        self.no_data = model_cfg.get('NO_DATA', None)
        self.use_adaptive_thres = model_cfg.get('USE_ADAPTIVE_THRES', False)
        self.adaptive_thres = model_cfg.get('ADAPTIVE_THRES', False)

    def forward(self, batch_dict):
        if False:
            import cv2
            b_size = batch_dict['gt_boxes'].shape[0]
            for b in [1]: #range(b_size):
                points = batch_dict['points'][batch_dict['points'][:, 0] ==
                                              b][:, 1:4].cpu().numpy()
                gt_boxes = batch_dict['gt_boxes'][b].cpu().numpy().copy()
                gt_boxes[:, 6] = -gt_boxes[:, 6]
                det = nuscene_vis(points, gt_boxes)
                cv2.imwrite('test_%02d.png' % b, det)
            breakpoint()

        disp_dict = {}
        if self.training:
            mask = batch_dict['mask'].view(-1)

            labeled_mask = torch.nonzero(mask).squeeze(1).long()
            unlabeled_mask = torch.nonzero(1 - mask).squeeze(1).long()
            #
            if self.no_data is not None:
                if self.no_data == 'TL':
                    labeled_mask = labeled_mask[:1]
                elif self.no_data == 'TU':
                    unlabeled_mask = unlabeled_mask[:0]
                    self.unlabeled_supervise_refine = False
                    self.unlabeled_supervise_cls = False
                    self.unlabeled_supervise_box = False
                elif self.no_data == 'SL':
                    labeled_mask = labeled_mask[1:]
                else:
                    NotImplementedError('No timplement self.no_data')
            #
            batch_dict_ema = {}
            keys = list(batch_dict.keys())
            for k in keys:
                if k + '_ema' in keys:
                    continue
                if k.endswith('_ema'):
                    batch_dict_ema[k[:-4]] = batch_dict[k]
                else:
                    batch_dict_ema[k] = batch_dict[k]

            with torch.no_grad():
                # self.pv_rcnn_ema.eval()  # Important! must be in train mode
                for cur_module in self.pv_rcnn_ema.module_list:
                    try:
                        batch_dict_ema = cur_module(
                            batch_dict_ema,
                            disable_gt_roi_when_pseudo_labeling=True)
                    except:
                        batch_dict_ema = cur_module(batch_dict_ema)
                pred_dicts, recall_dicts = self.pv_rcnn_ema.post_processing(
                    batch_dict_ema,
                    no_recall_dict=True,
                    override_thresh=0.0,
                    no_nms=self.no_nms)
                pseudo_boxes = []
                pseudo_scores = []
                pseudo_labels = []
                max_box_num = batch_dict['gt_boxes'].shape[1]
                max_pseudo_box_num = 0
                if self.use_adaptive_thres:
                    # calculate if pred boxes match with gt boxes in labeled data
                    C_THRES = torch.tensor([0.7, 0.7, 0.7, 0.5, 0.5]).cuda()
                    ind = unlabeled_mask[0] - 1
                    tl_num_gt = batch_dict_ema['gt_boxes'][ind].sum(1).nonzero().shape[0]
                    tl_gt_boxes = batch_dict_ema['gt_boxes'][ind][:tl_num_gt]

                    t_pred_boxes = pred_dicts[ind]['pred_boxes'].clone()
                    t_pred_scores = pred_dicts[ind]['pred_scores'].clone()
                    t_pred_labels = pred_dicts[ind]['pred_labels'].clone()
                    
                    if t_pred_boxes.shape[0] and tl_num_gt:
                        pred_iou = iou3d_nms_utils.boxes_iou3d_gpu(t_pred_boxes[:, :7], 
                                                                   tl_gt_boxes[:, :7])
                        pred_thres = C_THRES[t_pred_labels - 1]
                        pred_match = pred_iou.max(1)[0] >= pred_thres
                        pred_res = []
                        for cls in range(1, self.num_class + 1):
                            c_inds = t_pred_labels == cls
                            tc_boxes = t_pred_boxes[c_inds]
                            tc_scores = t_pred_scores[c_inds]
                            tc_labels = t_pred_labels[c_inds]
                            tc_match = pred_match[c_inds]
                            pred_res.append(torch.stack([tc_scores, tc_match]))
                        disp_dict['ad_cls_pred'] = pred_res

                for ind in unlabeled_mask:
                    pseudo_score = pred_dicts[ind]['pred_scores']
                    pseudo_box = pred_dicts[ind]['pred_boxes']
                    pseudo_label = pred_dicts[ind]['pred_labels']
                    pseudo_sem_score = pred_dicts[ind]['pred_sem_scores']

                    if len(pseudo_label) == 0:
                        pseudo_boxes.append(
                            pseudo_label.new_zeros((0, 8)).float())
                        continue

                    conf_thresh = torch.tensor(
                        self.thresh,
                        device=pseudo_label.device).unsqueeze(0).repeat(
                            len(pseudo_label),
                            1).gather(dim=1,
                                      index=(pseudo_label - 1).unsqueeze(-1))

                    valid_inds = pseudo_score > conf_thresh.squeeze()

                    valid_inds = valid_inds * (pseudo_sem_score >
                                               self.sem_thresh[0])

                    pseudo_sem_score = pseudo_sem_score[valid_inds]
                    pseudo_box = pseudo_box[valid_inds]
                    pseudo_label = pseudo_label[valid_inds]

                    #  if len(valid_inds) > max_box_num:
                    #  _, inds = torch.sort(pseudo_score, descending=True)
                    #  inds = inds[:max_box_num]
                    #  pseudo_box = pseudo_box[inds]
                    #  pseudo_label = pseudo_label[inds]

                    pseudo_boxes.append(
                        torch.cat(
                            [pseudo_box,
                             pseudo_label.view(-1, 1).float()],
                            dim=1))
                    if pseudo_box.shape[0] > max_pseudo_box_num:
                        max_pseudo_box_num = pseudo_box.shape[0]
                    # pseudo_scores.append(pseudo_score)
                    # pseudo_labels.append(pseudo_label)

                max_box_num = batch_dict['gt_boxes'].shape[1]

                # assert max_box_num >= max_pseudo_box_num
                ori_unlabeled_boxes = batch_dict['gt_boxes'][unlabeled_mask,
                                                             ...]

                if max_box_num >= max_pseudo_box_num:
                    for i, pseudo_box in enumerate(pseudo_boxes):
                        diff = max_box_num - pseudo_box.shape[0]
                        if diff > 0:
                            pseudo_box = torch.cat([
                                pseudo_box,
                                torch.zeros(
                                    (diff, 8), device=pseudo_box.device)
                            ],
                                                   dim=0)
                        batch_dict['gt_boxes'][unlabeled_mask[i]] = pseudo_box
                else:
                    ori_boxes = batch_dict['gt_boxes']
                    new_boxes = torch.zeros(
                        (ori_boxes.shape[0], max_pseudo_box_num,
                         ori_boxes.shape[2]),
                        device=ori_boxes.device)
                    for i, inds in enumerate(labeled_mask):
                        diff = max_pseudo_box_num - ori_boxes[inds].shape[0]
                        new_box = torch.cat([
                            ori_boxes[inds],
                            torch.zeros(
                                (diff, 8), device=ori_boxes[inds].device)
                        ],
                                            dim=0)
                        new_boxes[inds] = new_box
                    for i, pseudo_box in enumerate(pseudo_boxes):

                        diff = max_pseudo_box_num - pseudo_box.shape[0]
                        if diff > 0:
                            pseudo_box = torch.cat([
                                pseudo_box,
                                torch.zeros(
                                    (diff, 8), device=pseudo_box.device)
                            ],
                                                   dim=0)
                        new_boxes[unlabeled_mask[i]] = pseudo_box
                    batch_dict['gt_boxes'] = new_boxes

                batch_dict['gt_boxes'][
                    unlabeled_mask, ...] = random_flip_along_x_bbox(
                        batch_dict['gt_boxes'][unlabeled_mask, ...],
                        batch_dict['flip_x'][unlabeled_mask, ...])

                batch_dict['gt_boxes'][
                    unlabeled_mask, ...] = random_flip_along_y_bbox(
                        batch_dict['gt_boxes'][unlabeled_mask, ...],
                        batch_dict['flip_y'][unlabeled_mask, ...])

                batch_dict['gt_boxes'][
                    unlabeled_mask, ...] = global_rotation_bbox(
                        batch_dict['gt_boxes'][unlabeled_mask, ...],
                        batch_dict['rot_angle'][unlabeled_mask, ...])

                batch_dict['gt_boxes'][
                    unlabeled_mask, ...] = global_scaling_bbox(
                        batch_dict['gt_boxes'][unlabeled_mask, ...],
                        batch_dict['scale'][unlabeled_mask, ...])

                pseudo_ious = []
                pseudo_accs = []
                pseudo_fgs = []
                for i, ind in enumerate(unlabeled_mask):
                    'statistics'
                    anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(
                        batch_dict['gt_boxes'][ind, ...][:, 0:7],
                        ori_unlabeled_boxes[i, :, 0:7])
                    cls_pseudo = batch_dict['gt_boxes'][ind, ...][:, 7]
                    unzero_inds = torch.nonzero(cls_pseudo).squeeze(1).long()
                    cls_pseudo = cls_pseudo[unzero_inds]
                    if len(unzero_inds) > 0:
                        iou_max, asgn = anchor_by_gt_overlap[
                            unzero_inds, :].max(dim=1)
                        pseudo_ious.append(iou_max.unsqueeze(0))
                        acc = (ori_unlabeled_boxes[i][:, 7].gather(
                            dim=0, index=asgn) == cls_pseudo).float().mean()
                        pseudo_accs.append(acc.unsqueeze(0))
                        fg = (iou_max > 0.5).float().sum(
                            dim=0, keepdim=True) / len(unzero_inds)

                        sem_score_fg = (pseudo_sem_score[unzero_inds] * (iou_max > 0.5).float()).sum(dim=0, keepdim=True) \
                                       / torch.clamp((iou_max > 0.5).float().sum(dim=0, keepdim=True), min=1.0)
                        sem_score_bg = (pseudo_sem_score[unzero_inds] * (iou_max < 0.5).float()).sum(dim=0, keepdim=True) \
                                       / torch.clamp((iou_max < 0.5).float().sum(dim=0, keepdim=True), min=1.0)
                        pseudo_fgs.append(fg)
                        #  only for 100% label
                        if self.supervise_mode >= 1:
                            filter = iou_max > 0.3
                            asgn = asgn[filter]
                            batch_dict['gt_boxes'][
                                ind, ...][:] = torch.zeros_like(
                                    batch_dict['gt_boxes'][ind, ...][:])
                            batch_dict['gt_boxes'][
                                ind, ...][:len(asgn)] = ori_unlabeled_boxes[
                                    i, :].gather(
                                        dim=0,
                                        index=asgn.unsqueeze(-1).repeat(1, 8))

                            if self.supervise_mode == 2:
                                batch_dict['gt_boxes'][ind, ...][:len(asgn), 0:3] += 0.1 * torch.randn((len(asgn), 3), device=iou_max.device) * \
                                                                                     batch_dict['gt_boxes'][ind, ...][
                                                                                     :len(asgn), 3:6]
                                batch_dict['gt_boxes'][ind, ...][:len(asgn), 3:6] += 0.1 * torch.randn((len(asgn), 3), device=iou_max.device) * \
                                                                                     batch_dict['gt_boxes'][ind, ...][
                                                                                     :len(asgn), 3:6]
                    else:
                        ones = torch.ones((1), device=unlabeled_mask.device)
                        sem_score_fg = ones
                        sem_score_bg = ones
                        pseudo_ious.append(ones)
                        pseudo_accs.append(ones)
                        pseudo_fgs.append(ones)

            for cur_module in self.pv_rcnn.module_list:
                batch_dict = cur_module(batch_dict)

            loss_rpn_cls, loss_rpn_box, tb_dict = self.pv_rcnn.dense_head.get_loss(
                scalar=False)
            loss_point, tb_dict = self.pv_rcnn.point_head.get_loss(
                tb_dict, scalar=False)
            loss_rcnn_cls, loss_rcnn_box, tb_dict = self.pv_rcnn.roi_head.get_loss(
                tb_dict, scalar=False)

            if not self.unlabeled_supervise_cls:
                loss_rpn_cls = loss_rpn_cls[labeled_mask, ...].sum()
            else:
                loss_rpn_cls = loss_rpn_cls[labeled_mask, ...].sum(
                ) + loss_rpn_cls[unlabeled_mask,
                                 ...].sum() * self.unlabeled_weight

            if not self.unlabeled_supervise_box:
                loss_rpn_box = loss_rpn_box[labeled_mask, ...].sum()
            else:
                loss_rpn_box = loss_rpn_box[labeled_mask, ...].sum(
                ) + loss_rpn_box[unlabeled_mask, ...].sum() * self.unlabeled_weight

            loss_point = loss_point[labeled_mask, ...].sum()
            loss_rcnn_cls = loss_rcnn_cls[labeled_mask, ...].sum()

            if not self.unlabeled_supervise_refine:
                loss_rcnn_box = loss_rcnn_box[labeled_mask, ...].sum()
            else:
                loss_rcnn_box = loss_rcnn_box[labeled_mask, ...].sum(
                ) + loss_rcnn_box[unlabeled_mask,
                                  ...].sum() * self.unlabeled_weight

            loss = loss_rpn_cls + loss_rpn_box + loss_point + loss_rcnn_cls + loss_rcnn_box
            tb_dict_ = {}
            for key in tb_dict.keys():
                if 'loss' in key:
                    tb_dict_[key + "_labeled"] = tb_dict[key][labeled_mask,
                                                              ...].sum()
                    tb_dict_[key + "_unlabeled"] = tb_dict[key][unlabeled_mask,
                                                                ...].sum() * self.unlabeled_weight
                elif 'acc' in key:
                    tb_dict_[key + "_labeled"] = tb_dict[key][labeled_mask,
                                                              ...].sum()
                    tb_dict_[key + "_unlabeled"] = tb_dict[key][unlabeled_mask,
                                                                ...].sum() * self.unlabeled_weight
                elif 'point_pos_num' in key:
                    tb_dict_[key + "_labeled"] = tb_dict[key][labeled_mask,
                                                              ...].sum()
                    tb_dict_[key + "_unlabeled"] = tb_dict[key][unlabeled_mask,
                                                                ...].sum() * self.unlabeled_weight
                else:
                    tb_dict_[key] = tb_dict[key]

            if len(unlabeled_mask):
                tb_dict_['pseudo_ious'] = torch.cat(pseudo_ious, dim=0).mean()
                tb_dict_['pseudo_accs'] = torch.cat(pseudo_accs, dim=0).mean()
                tb_dict_['max_pseudo_box_num'] = max_pseudo_box_num

                tb_dict_['sem_score_fg'] = sem_score_fg.mean()
                tb_dict_['sem_score_bg'] = sem_score_bg.mean()
                tb_dict_['max_box_num'] = max_box_num

            ret_dict = {'loss': loss}
            return ret_dict, tb_dict_, disp_dict

        else:
            for cur_module in self.pv_rcnn.module_list:
                batch_dict = cur_module(batch_dict)

            pred_dicts, recall_dicts = self.pv_rcnn.post_processing(batch_dict)

            return pred_dicts, recall_dicts, {}

    def get_supervised_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict

    def update_global_step(self):
        self.global_step += 1
        alpha = 0.999
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (self.global_step + 1), alpha)
        for ema_param, param in zip(self.pv_rcnn_ema.parameters(),
                                    self.pv_rcnn.parameters()):
            ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' %
                    (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' %
                        checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
            new_key = 'pv_rcnn.' + key
            if new_key in self.state_dict() and self.state_dict(
            )[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))
            new_key = 'pv_rcnn_ema.' + key
            if new_key in self.state_dict() and self.state_dict(
            )[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val
            new_key = key
            if new_key in self.state_dict() and self.state_dict(
            )[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' %
                            (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' %
                    (len(update_model_state), len(self.state_dict())))
