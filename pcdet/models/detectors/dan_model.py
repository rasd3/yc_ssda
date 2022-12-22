import os

import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, activation='relu'):
        super(conv_2d, self).__init__()
        if activation == 'relu':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=False)
            )
        elif activation == 'tanh':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel),
                nn.BatchNorm2d(out_ch),
                nn.Tanh()
            )
        elif activation == 'leakyrelu':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU()
            )


    def forward(self, x):
        x = self.conv(x)
        return x

class fc_layer(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, activation='leakyrelu'):
        super(fc_layer, self).__init__()
        if activation == 'relu':
            self.ac = nn.ReLU(inplace=False)
        elif activation == 'leakyrelu':
            self.ac = nn.LeakyReLU()
        if bn:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch),
                nn.BatchNorm1d(out_ch),
                self.ac
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch),
                self.ac
            )

    def forward(self, x):
        x = self.fc(x)
        return x

class transform_net(nn.Module):
    def __init__(self, in_ch, K=3):
        super(transform_net, self).__init__()
        self.K = K
        self.conv2d1 = conv_2d(in_ch, 64, 1)
        self.conv2d2 = conv_2d(64, 128, 1)
        self.conv2d3 = conv_2d(128, 1024, 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(512, 1))
        self.fc1 = fc_layer(1024, 512)
        self.fc2 = fc_layer(512, 256)
        self.fc3 = nn.Linear(256, K*K)
    


    def forward(self, x):
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        x = self.conv2d3(x)
        x, _ = torch.max(x, dim=2, keepdim=False)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        iden = torch.eye(self.K).view(1,self.K * self. K).repeat(x.size(0),1)
        iden = iden.to(device='cuda') 
        x = x + iden
        x = x.view(x.size(0), self.K, self.K)
        return x

class adapt_layer_off(nn.Module):
    def __init__(self, num_node=64, offset_dim=3, trans_dim_in=64, trans_dim_out=64, fc_dim=64):
        super(adapt_layer_off, self).__init__()
        self.num_node = num_node
        self.offset_dim = offset_dim
        self.trans = conv_2d(trans_dim_in, trans_dim_out, 1)
        self.pred_offset = nn.Sequential(
            nn.Conv2d(trans_dim_out, offset_dim, kernel_size=1, bias=False),
            nn.Tanh())
        self.residual = conv_2d(trans_dim_in, fc_dim, 1)

    def forward(self, input_fea, input_loc):
        # Initialize node
        fpoint_idx = farthest_point_sample(input_loc, self.num_node)  # (B, num_node)
        fpoint_loc = index_points(input_loc, fpoint_idx)  # (B, 3, num_node)
        fpoint_fea = index_points(input_fea, fpoint_idx)  # (B, C, num_node)
        group_idx = query_ball_point(0.3, 64, input_loc, fpoint_loc)   # (B, num_node, 64)
        group_fea = index_points(input_fea, group_idx)  # (B, C, num_node, 64)
        group_fea = group_fea - fpoint_fea.unsqueeze(3).expand(-1, -1, -1, self.num_node)

        # Learn node offset
        seman_trans = self.pred_offset(group_fea)  # (B, 3, num_node, 64)
        group_loc = index_points(input_loc, group_idx)   # (B, 3, num_node, 64)
        group_loc = group_loc - fpoint_loc.unsqueeze(3).expand(-1, -1, -1, self.num_node)
        node_offset = (seman_trans*group_loc).mean(dim=-1)

        # Update node and get node feature
        node_loc = fpoint_loc+node_offset.squeeze(-1)  # (B,3,num_node)
        group_idx = query_ball_point(None, 64, input_loc, node_loc)
        residual_fea = self.residual(input_fea)
        group_fea = index_points(residual_fea, group_idx)
        node_fea, _ = torch.max(group_fea, dim=-1, keepdim=True)

        # Interpolated back to original point
        output_fea = upsample_inter(input_loc, node_loc, input_fea, node_fea, k=3).unsqueeze(3)

        return output_fea, node_fea, node_offset


# Channel Attention
class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CALayer, self).__init__()
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=False),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        self.bn = nn.BatchNorm1d(4096)

    def forward(self, x):
        y = self.conv_du(x)
        y = x * y + x
        y = y.view(y.shape[0], -1)
        y = self.bn(y)

        return y

# Grad Reversal
class GradReverse(torch.autograd.Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)

def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)

# Generator
class Pointnet_g(nn.Module):
    def __init__(self):
        super(Pointnet_g, self).__init__()
        self.trans_net1 = transform_net(3, 3)
        self.trans_net2 = transform_net(64, 64)
        self.conv1 = conv_2d(3, 64, 1)
        self.conv2 = conv_2d(64, 64, 1)
        # SA Node Module
        self.conv3 = adapt_layer_off()  # (64->128)
        self.conv4 = conv_2d(128, 128, 1)
        self.conv5 = conv_2d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(1024)

    def forward(self, x, node = False):
        x_loc = x.squeeze(-1)

        transform = self.trans_net1(x)
        x = x.transpose(2, 1)

        x = x.squeeze(-1)
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        transform = self.trans_net2(x)
        x = x.transpose(2, 1)

        x = x.squeeze(-1)
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)

        x, node_fea, node_off = self.conv3(x, x_loc)  # x = [B, dim, num_node, 1]/[64, 64, 1024, 1]; x_loc = [B, xyz, num_node] / [64, 3, 1024]
        x = self.conv4(x)
        x = self.conv5(x)

        x, _ = torch.max(x, dim=2, keepdim=False)

        x = x.squeeze(-1)
  
        x = self.bn1(x)

        if node == True:
            return x, node_fea, node_off
        else:
            return x, node_fea

# Classifier
class Pointnet_c(nn.Module):
    def __init__(self, num_class=10):
        super(Pointnet_c, self).__init__()
        self.fc = nn.Linear(1024, num_class)
        
    def forward(self, x):
        x = self.fc(x)
        return x
        
class Net_MDA(nn.Module):
    def __init__(self, model_name='Pointnet'):
        super(Net_MDA, self).__init__()
        if model_name == 'Pointnet':
            self.g = Pointnet_g() 
            self.attention_s = CALayer(64*64)
            self.attention_t = CALayer(64*64)
            self.c1 = Pointnet_c()  
            self.c2 = Pointnet_c() 
            
    def forward(self, x, constant=1, adaptation=False, node_vis=False, mid_feat=False, node_adaptation_s=False, node_adaptation_t=False):
        x, feat_ori, node_idx = self.g(x, node=True)
        batch_size = feat_ori.size(0)

        # sa node visualization
        if node_vis ==True:
            return node_idx

        # collect mid-level feat
        if mid_feat == True:
            return x, feat_ori
        
        if node_adaptation_s == True:
            # source domain sa node feat
            feat_node = feat_ori.view(batch_size, -1)
            feat_node_s = self.attention_s(feat_node.unsqueeze(2).unsqueeze(3))
            return feat_node_s
        elif node_adaptation_t == True:
            # target domain sa node feat
            feat_node = feat_ori.view(batch_size, -1)
            feat_node_t = self.attention_t(feat_node.unsqueeze(2).unsqueeze(3))
            return feat_node_t

        if adaptation == True:
            x = grad_reverse(x, constant)

        y1 = self.c1(x)
        y2 = self.c2(x)
        return y1, y2

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, C, N]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, C, N = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, :, farthest].view(B, 3, 1)
        dist = torch.sum((xyz - centroid) ** 2, 1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, C, N]/[B,C,N,1]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, C, S]
    """
    if len(points.shape) == 4:
        points = points.squeeze()
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    points = points.permute(0,2,1) #(B,N,C)
    new_points = points[batch_indices, idx, :]
    if len(new_points.shape)==3:
        new_points = new_points.permute(0,2,1)
    elif len(new_points.shape) == 4:
        new_points = new_points.permute(0,3,1,2)
    return new_points

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, C, N]
        new_xyz: query points, [B, C, S]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, C, N = xyz.shape
    _, _, S = new_xyz.shape
    sqrdists = square_distance(new_xyz, xyz)
    if radius is not None:
        group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
        group_idx[sqrdists > radius ** 2] = N
        group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
        mask = group_idx == N
        group_idx[mask] = group_first[mask]
    else:
        group_idx = torch.sort(sqrdists, dim=-1)[1][:,:,:nsample]
    return group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, C, N]
        dst: target points, [B, C, M]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, _, N = src.shape
    _, _, M = dst.shape
    dist = -2 * torch.matmul(src.permute(0, 2, 1), dst)
    dist += torch.sum(src ** 2, 1).view(B, N, 1)
    dist += torch.sum(dst ** 2, 1).view(B, 1, M)
    return dist

def upsample_inter(xyz1, xyz2, points1, points2, k):
    """
    Input:
        xyz1: input points position data, [B, C, N]
        xyz2: sampled input points position data, [B, C, S]
        points1: input points data, [B, D, N]/[B,D,N,1]
        points2: input points data, [B, D, S]/[B,D,S,1]
        k:
    Return:
        new_points: upsampled points data, [B, D+D, N]
    """
    if points1 is not None:
        if len(points1.shape) == 4:
            points1 = points1.squeeze()
    if len(points2.shape) == 4:
        points2 = points2.squeeze()
    B, C, N = xyz1.size()
    _, _, S = xyz2.size()

    dists = square_distance(xyz1, xyz2) #(B, N, S)
    dists, idx = dists.sort(dim=-1)
    dists, idx = dists[:, :, :k], idx[:, :, :k]  # [B, N, 3]
    dists[dists < 1e-10] = 1e-10
    weight = 1.0 / dists  # [B, N, 3]
    weight = weight / torch.sum(weight, dim=-1).view(B, N, 1)  # [B, N, 3]; weight = [64, 1024, 3]
    interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, 1, N, k), dim=3) #(B,D,N); idx = [64, 1024, 3]; points2 = [64, 64, 64];
    if points1 is not None:
        new_points = torch.cat([points1, interpolated_points], dim=1)  # points1 = [64, 64, 1024];
        return new_points
    else:
        return interpolated_points



def pairwise_distance(x):
    batch_size = x.size(0)
    point_cloud = torch.squeeze(x)
    if batch_size == 1:
        point_cloud = torch.unsqueeze(point_cloud, 0)
    point_cloud_transpose = torch.transpose(point_cloud, dim0=1, dim1=2)
    point_cloud_inner = torch.matmul(point_cloud_transpose, point_cloud)
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = torch.sum(point_cloud ** 2, dim=1, keepdim=True)
    point_cloud_square_transpose = torch.transpose(point_cloud_square, dim0=1, dim1=2)
    return point_cloud_square + point_cloud_inner + point_cloud_square_transpose


def gather_neighbor(x, nn_idx, n_neighbor):
    x = torch.squeeze(x)
    batch_size = x.size()[0]
    num_dim = x.size()[1]
    num_point = x.size()[2]
    point_expand = x.unsqueeze(2).expand(batch_size, num_dim, num_point, num_point)
    nn_idx_expand = nn_idx.unsqueeze(1).expand(batch_size, num_dim, num_point, n_neighbor)
    pc_n = torch.gather(point_expand, -1, nn_idx_expand)
    return pc_n

def get_neighbor_feature(x, n_point, n_neighbor):
    if len(x.size()) == 3:
        x = x.unsqueeze()
    adj_matrix = pairwise_distance(x)
    _, nn_idx = torch.topk(adj_matrix, n_neighbor, dim=2, largest=False)
    nn_idx = nn_idx[:, :n_point, :]
    batch_size = x.size()[0]
    num_dim = x.size()[1]
    num_point = x.size()[2]
    point_expand = x[:, :, :n_point, :].expand(-1, -1, -1, num_point)
    nn_idx_expand = nn_idx.unsqueeze(1).expand(batch_size, num_dim, n_point, n_neighbor)
    pc_n = torch.gather(point_expand, -1, nn_idx_expand)
    return pc_n


def get_edge_feature(x, n_neighbor):
    if len(x.size()) == 3:
        x = x.unsqueeze(3)
    adj_matrix = pairwise_distance(x)
    _, nn_idx = torch.topk(adj_matrix, n_neighbor, dim=2, largest=False)
    point_cloud_neighbors = gather_neighbor(x, nn_idx, n_neighbor)
    point_cloud_center = x.expand(-1, -1, -1, n_neighbor)
    edge_feature = torch.cat((point_cloud_center, point_cloud_neighbors-point_cloud_center), dim=1)
    return edge_feature

