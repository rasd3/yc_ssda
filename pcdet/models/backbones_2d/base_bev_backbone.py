import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from .domain_classifier import model_dict as dc_model_dict


class BaseBEVBackbone(nn.Module):

    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(
                self.model_cfg.LAYER_STRIDES) == len(
                    self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(
                self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(c_in_list[idx],
                          num_filters[idx],
                          kernel_size=3,
                          stride=layer_strides[idx],
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx],
                              num_filters[idx],
                              kernel_size=3,
                              padding=1,
                              bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(
                        nn.Sequential(
                            nn.ConvTranspose2d(num_filters[idx],
                                               num_upsample_filters[idx],
                                               upsample_strides[idx],
                                               stride=upsample_strides[idx],
                                               bias=False),
                            nn.BatchNorm2d(num_upsample_filters[idx],
                                           eps=1e-3,
                                           momentum=0.01), nn.ReLU()))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(
                        nn.Sequential(
                            nn.Conv2d(num_filters[idx],
                                      num_upsample_filters[idx],
                                      stride,
                                      stride=stride,
                                      bias=False),
                            nn.BatchNorm2d(num_upsample_filters[idx],
                                           eps=1e-3,
                                           momentum=0.01), nn.ReLU()))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(c_in,
                                       c_in,
                                       upsample_strides[-1],
                                       stride=upsample_strides[-1],
                                       bias=False),
                    nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                ))

        self.num_bev_features = c_in
        # for domain classifier
        self.batch_size = -1
        self.use_domain_cls = self.model_cfg.get('USE_DOMAIN_CLASSIFIER',
                                                 False)
        self.dc_version = self.model_cfg.get('DC_VERSION', None)
        self.mgfa = self.model_cfg.get('MGFA', False)
        if self.use_domain_cls:
            self.domain_data_split = False
            self.domain_data_trg = False
            self.domain_cls = dc_model_dict[self.dc_version]()
            self.domain_loss_cfg = self.model_cfg.get('LOSS_CONFIG', None)
            if self.domain_loss_cfg.LOSS_DC == 'NLL':
                self.domain_cls_loss = torch.nn.NLLLoss().cuda()
            elif self.domain_loss_cfg.LOSS_DC == 'BCE':
                self.domain_cls_loss = F.binary_cross_entropy_with_logits
            else:
                raise NotImplementedError('specify implemented loss name')

            self.forward_ret_dict = {}
        if self.mgfa:
            self.mgfa_feats_scale = self.model_cfg.get('MGFA_FEATS_SCALE', [0, 1, 2])
            self.mgfa_proj = []
            self.mgfa_pool = []
            ch_in = num_filters + [c_in]
            for ch in ch_in:
                self.mgfa_pool.append(nn.AdaptiveAvgPool2d(1).cuda())
                m_seq = nn.Sequential(nn.Linear(ch, ch*2),
                                      nn.BatchNorm1d(ch*2),
                                      nn.ReLU(),
                                      nn.Linear(ch*2, ch))
                self.mgfa_proj.append(m_seq.cuda())


    def get_loss(self, tb_dict=None):
        batch_size = self.batch_size
        tb_dict = {} if tb_dict is None else tb_dict
        domain_output = self.forward_ret_dict['domain_output']
        trg_idx = torch.tensor([i*2+1 for i in range(batch_size // 2)])
        if 'POOL' in self.dc_version:
            domain_label = torch.zeros(batch_size, dtype=torch.long).cuda()
            domain_label[trg_idx] = 1
        elif 'CONV' in self.dc_version:
            domain_label = torch.FloatTensor(domain_output.shape).cuda()
            domain_label[trg_idx - 1] = 0.
            domain_label[trg_idx] = 1.
        if self.domain_data_split:
            for b in range(batch_size):
                domain_label[b].fill_(self.forward_ret_dict['domain_target'][b])

        dc_loss_weight = self.domain_loss_cfg.LOSS_WEIGHTS.dc_weight
        domain_cls_loss = self.domain_cls_loss(domain_output, domain_label)
        domain_cls_loss *= dc_loss_weight

        tb_dict['domain_cls_loss'] = domain_cls_loss.item()
        return domain_cls_loss, tb_dict

    def remove_trg_data(self, data_dict):
        batch_size = data_dict['batch_size']
        b_mask = torch.tensor([True, False] * (batch_size // 2), dtype=torch.bool)
        data_dict['spatial_features'] = data_dict['spatial_features'][b_mask]
        data_dict['spatial_features_2d'] = data_dict['spatial_features_2d'][b_mask]
        data_dict['batch_size'] = batch_size // 2
        data_dict['gt_boxes'] = data_dict['gt_boxes'][b_mask]
        pt_mask = data_dict['points'][:, 0] % 2 == 0
        data_dict['points'] = data_dict['points'][pt_mask]
        data_dict['points'][:, 0] = data_dict['points'][:, 0] // 2
        vox_mask = data_dict['voxel_coords'][:, 0] % 2 == 0
        data_dict['voxel_coords'] = data_dict['voxel_coords'][vox_mask]
        data_dict['voxel_coords'][:, 0] = data_dict['voxel_coords'][:, 0] // 2
        data_dict['voxels'] = data_dict['voxels'][vox_mask]
        data_dict['voxel_features'] = data_dict['voxel_features'][vox_mask]
        multi_scale_3d_features = data_dict['multi_scale_3d_features']
        for key in multi_scale_3d_features.keys():
            indices = multi_scale_3d_features[key].indices
            ms_mask = indices[:, 0] % 2 == 0
            multi_scale_3d_features[key].indices = multi_scale_3d_features[
                key].indices[ms_mask]
            multi_scale_3d_features[key].indices[:, 0] = multi_scale_3d_features[key].indices[:, 0] // 2
            multi_scale_3d_features[key].features = multi_scale_3d_features[key].features[ms_mask]

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """

        self.batch_size = data_dict['batch_size']
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                y = self.deblocks[i](x)
                ups.append(y)
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x
        ret_dict['spatial_features_2d'] = x
        if self.use_domain_cls and 'cur_train_meta' in data_dict:
            # from DANN
            iter_meta = data_dict['cur_train_meta']
            p = float(
                i + iter_meta['cur_epoch'] * iter_meta['total_it_each_epoch']
            ) / iter_meta['total_epochs'] / iter_meta['total_it_each_epoch']
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            domain_out = self.domain_cls(x, alpha)

            self.forward_ret_dict['domain_output'] = domain_out
            self.forward_ret_dict['domain_target'] = data_dict['domain']
            # remove target data
            if not data_dict['cur_train_meta'].get('data_split', False):
                self.remove_trg_data(data_dict)
            else:
                self.domain_data_split = True
        if not self.use_domain_cls and self.dc_version and 'cur_train_meta' in data_dict:
            if not data_dict['cur_train_meta'].get('data_split', False):
                self.remove_trg_data(data_dict)
            else:
                self.domain_data_split = True
        if self.mgfa and 'cur_train_meta' in data_dict:
            feat_names = ['spatial_features_1x', 'spatial_features_2x', 'spatial_features_2d']
            mgfa_proj_feats = []
            for idx, feat_name in enumerate(feat_names):
                if idx in self.mgfa_feats_scale:
                    mgfa_feat = ret_dict[feat_name]
                    mgfa_feat = self.mgfa_pool[idx](mgfa_feat).view(mgfa_feat.size(0), -1)
                    mgfa_feat = self.mgfa_proj[idx](mgfa_feat)
                    mgfa_proj_feats.append(mgfa_feat)
            data_dict['mgfa_feats'] = mgfa_proj_feats

        return data_dict
