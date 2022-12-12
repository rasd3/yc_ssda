import torch
from torch import nn
import numpy as np

def torch_to_np_dtype(ttype):
    type_map = {
        torch.float16: np.dtype(np.float16),
        torch.float32: np.dtype(np.float32),
        torch.float16: np.dtype(np.float64),
        torch.int32: np.dtype(np.int32),
        torch.int64: np.dtype(np.int64),
        torch.uint8: np.dtype(np.uint8),
    }
    return type_map[ttype]

def rotation_2d_pillar(points, angles):
    """rotation 2d points based on origin point clockwise when angle positive.
    
    Args:
        points (float array, shape=[N, point_size, 2]): points to be rotated.
        angles (float array, shape=[N]): rotation angle.
    Returns:
        float array: same shape as points
    """
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    rot_mat_T = torch.stack([torch.stack([rot_cos, -rot_sin]), torch.stack([rot_sin, rot_cos])])
    return torch.einsum("aij,jka->aik", (points, rot_mat_T))

def center_to_corner_box2d_pillar(centers, dims, angles=None, origin=0.5):
    """convert kitti locations, dimensions and angles to corners.
    format: center(xy), dims(xy), angles(clockwise when positive)
    
    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
    
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd_pillar(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d_pillar(corners, angles)
    corners += centers.reshape([-1, 1, 2])
    return corners

def corners_nd_pillar(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point. 
    
    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.
    
    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners. 
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    dtype = torch_to_np_dtype(dims.dtype)
    if isinstance(origin, float):
        origin = [origin] * ndim
    corners_norm = np.stack(np.unravel_index(np.arange(2**ndim), [2] * ndim),
                            axis=1).astype(dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dtype)
    corners_norm = torch.from_numpy(corners_norm).type_as(dims)
    corners = dims.view([-1, 1, ndim]) * corners_norm.view(1, 2**ndim, ndim)
    return corners

def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans

class BEVFeatureExtractorV2(nn.Module): 
    def __init__(self, model_cfg, pc_start, 
            voxel_size, out_stride, num_point):
        super().__init__()
        self.pc_start = pc_start 
        self.voxel_size = voxel_size
        self.out_stride = out_stride
        self.num_point = num_point #아마 spatial feature수인거 같은데 한번더 확인 ㄱㄱ

    def absl_to_relative(self, absolute):
        a1 = (absolute[..., 0] - self.pc_start[0]) / self.voxel_size[0] / self.out_stride 
        a2 = (absolute[..., 1] - self.pc_start[1]) / self.voxel_size[1] / self.out_stride 

        return a1, a2
    
    def get_box_center(self, batch_dict):
        # box [List]
        # boxes = batch_dict['final_box_dicts']
        boxes = batch_dict['rois']
        #boxes = batch_dict['batch_box_preds']
        centers = [] 
        for box in boxes:            
            if self.num_point == 1 or len(box) == 0:
                centers.append(box[:, :3])
                
            elif self.num_point == 5:
                center2d = box[:, :2]
                height = box[:, 2:3]
                dim2d = box[:, 3:5]
                rotation_y = box[:, -1]

                corners = center_to_corner_box2d_pillar(center2d, dim2d, rotation_y)
                #corners = center_to_corner_box2d(center2d, dim2d, rotation_y)

                front_middle = torch.cat([(corners[:, 0] + corners[:, 1])/2, height], dim=-1)
                back_middle = torch.cat([(corners[:, 2] + corners[:, 3])/2, height], dim=-1)
                left_middle = torch.cat([(corners[:, 0] + corners[:, 3])/2, height], dim=-1)
                right_middle = torch.cat([(corners[:, 1] + corners[:, 2])/2, height], dim=-1) 

                points = torch.cat([box[:, :3], front_middle, back_middle, left_middle, \
                    right_middle], dim=0)

                centers.append(points)
            else:
                raise NotImplementedError()

        return centers

    def forward(self, batch_dict):
        batch_size = len(batch_dict['spatial_features_2d'])
        bev_feature = batch_dict['spatial_features_2d'].permute(0, 2, 3, 1) #.contiguous() # pred_dicts[bs_idx]['bev_feature'] = bev_feature[bs_idx].permute(0, 2, 3, 1).contiguous()
        ret_maps = [] 

        # batch_centers = batch_dict['rois'][..., :3]
        batch_centers = self.get_box_center(batch_dict)

        for batch_idx in range(batch_size):
            #batch_centers = self.get_box_center(batch_dict['final_predict'][batch_idx]['pred_boxes'])
            xs, ys = self.absl_to_relative(batch_centers[batch_idx])
            #xs, ys = self.absl_to_relative(batch_centers[0])
            
            # N x C 
            feature_map = bilinear_interpolate_torch(bev_feature[batch_idx],
             xs, ys)

            if self.num_point >1:
                section_size = len(feature_map) // self.num_point
                feature_map = torch.cat([feature_map[i*section_size: (i+1)*section_size] for i in range(self.num_point)], dim=1)

            ret_maps.append(feature_map)

        batch_dict['roi_features'] = ret_maps #ret_maps[0].shape = [NMX_MAX_size, -]
        return batch_dict
