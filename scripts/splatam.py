import argparse
import os
import shutil
import sys
import time
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from datasets.gradslam_datasets import (
    load_dataset_config,
    ICLDataset,
    ReplicaDataset,
    ReplicaV2Dataset,
    AzureKinectDataset,
    ScannetDataset,
    Ai2thorDataset,
    Record3DDataset,
    RealsenseDataset,
    TUMDataset,
    ScannetPPDataset,
    NeRFCaptureDataset,
    MatrixCityDataset
)
from utils.common_utils import seed_everything, save_params_ckpt, save_params
from utils.eval_helpers import report_loss, report_progress, eval
from utils.keyframe_selection import keyframe_selection_overlap
from utils.recon_helpers import setup_camera
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transform_to_frame, l1_loss_v1, matrix_to_quaternion
)
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify

from diff_gaussian_rasterization import GaussianRasterizer as Renderer

'''
    函数目的：获取各种数据集的接口函数，具体的获取方法需要在datasets/gradslam_datasets里的py文件中提供
'''
def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["matrixcitysmallaerial"]:
        return MatrixCityDataset(config_dict, basedir, sequence, **kwargs)
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replicav2"]:
        return ReplicaV2Dataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["tum"]:
        return TUMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannetpp"]:
        return ScannetPPDataset(basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["nerfcapture"]:
        return NeRFCaptureDataset(basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")

"""
    函数目的：获取三维点云的位置，大小，颜色信息
    输入：color、深度depth、相机内参instrinsics、世界到相机的转换矩阵w2c
    输出：point_cld：三维高斯的位置，每行都是一个点，四列，前三列xyz，最后一列color
         mean3_sq_dist：三维高斯的尺度，一列，均方距离
    调用位置：初始化第一帧的高斯initialize_first_timestep
            密集化高斯add_new_gaussians
"""
def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True, 
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
    '''
        A.内参解析，内参矩阵：
        [CX 0 FX]
        [0 CY FY]
        [0  0  1]  
    '''
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]
    '''
        B.将像素点从像素坐标系转化到相机坐标系，从而得到点云
    '''
    width, height = color.shape[2], color.shape[1] # 获取图像的长宽像素数
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), 
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')# 像素网格生成，Compute indices of pixels，根据图像的长宽生成像素矩阵，输出x(y)_grid的shape为(width,height)
    xx = (x_grid - CX)/FX # 根据u = FX/Z*X+CX, (u-CX)/FX = X/Z = xx, u为参考为像素坐标系，X参考为相机坐标系
    yy = (y_grid - CY)/FY# 根据u = FY/Z*Y+CY, (u-CY)/FY = Y/Z = yy
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    # 输出的shape为(1,height * width)例如width = 6,height=3
    # xx = [0 1 2 3 4 5  0 1 2 3 4 5  0 1 2 3 4 5]表示行index
    # yy = [0 0 0 0 0 0  1 1 1 1 1 1  2 2 2 2 2 2]表示列index
    depth_z = depth[0].reshape(-1) # 即Z，实际的深度值
    '''
        C.点云初始化，将每个像素点从像素坐标系转化到世界坐标系，也就是高斯的位置
    '''
    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1) # xx*Z = X，输出的shape为(height * width, 3)，例如
    # [0 0 z1] 
    # [1 0 z2]
    # [2 0 z3]
    # [0 1 z4] 
    # [1 1 z5]
    # [2 1 z6]
    if transform_pts: #将每个像素点从像素坐标系转化到世界坐标系
        pix_ones = torch.ones(height * width, 1).cuda().float() # shape为(height * width, 1)
        pts4 = torch.cat((pts_cam, pix_ones), dim=1) # shape为(height * width, 4)最后一列都是1，例如
        # [0 0 z1 1] 
        # [1 0 z2 1]
        # [2 0 z3 1]
        # [0 1 z4 1] 
        # [1 1 z5 1]
        # [2 1 z6 1]
        c2w = torch.inverse(w2c) 
        pts = (c2w @ pts4.T).T[:, :3] #将每个像素点从像素坐标系转化到世界坐标系
        # print(pts.shape)
        # w2c为4x4:
        # [R3x3 t]
        # [0 0 0 1]
        # pts4.T shape为(4, height * width)最后一行都是1:
        # [0 1 2 0 1 2] x
        # [0 0 0 1 1 1] y
        # [z1...     z6] z
        # [1 1 1 1 1 1] 1
        # T[:, :3]取最后三列
    else:
        pts = pts_cam
    '''
        D.点云（高斯）尺度：计算每个新点云的方差（均方距离），用于初始化高斯分布的尺度（半径）参数
    '''
    # Optional: Compute mean squared distance for initializing the scale of the Gaussians
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY)/2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
    '''
        E.点云着色，前三列xyz最后一列color
    '''
    # Colorize point cloud
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
    point_cld = torch.cat((pts, cols), -1) #四列，前三列xyz最后一列color
    '''
        F.掩码
    '''
    # Optional: Select points based on mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld

"""
    函数目的：初始化高斯参数和变量
    高斯参数：
        1）输入点云的位置，颜色，旋转矩阵，不透明度，大小 
        2）相机参数：旋转矩阵和平移矩阵，num_frames维度，用于衡量当前点云在不同帧中的旋转平移
    高斯变量：
        1）max_2D_radius：所有高斯点的最大二维半径
        2）means2D_gradient_accum：所有高斯点的二维累计梯度
        3）denom：未知
        4）timestep：当前时间
"""
def initialize_params(init_pt_cld, num_frames, mean3_sq_dist):
    '''
        A1.高斯参数params初始化：1）输入点云的位置，颜色，旋转矩阵（未旋转），不透明度（0），大小
    '''
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3] # [num_gaussians, 3]点云的xyz坐标，即高斯的位置
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # 将[1, 0, 0, 0]作为元素沿着x轴复制一次，沿着y轴复制num_pts次，得到形状为 (num_pts, 4) 的二维数组，其中每一行都是 [1, 0, 0, 0] 
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda") 
    # 3D Gaussian待优化的参数
    params = {
        'means3D': means3D, # 中心
        'rgb_colors': init_pt_cld[:, 3:6], # 颜色 
        'unnorm_rotations': unnorm_rots, # 未标准化的旋转矩阵，用于表示3D高斯的旋转。由四元数表示。unnorm_rotations形状为 (num_pts, 4) 的数组。初始化每一行都是[1, 0, 0, 0]，表示没有应用任何旋转。
        'logit_opacities': logit_opacities, # 不透明度，初始化为0
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1)), # 对均方距离开平方得到标准差，然后求对数，增加一个维度，在前两维度都复制一遍（因为初始化是各项同性高斯）
    }
    '''
        A2.高斯参数params初始化：2）相机参数设置：旋转矩阵（未旋转），平移矩阵（未平移），num_frames维度，用于衡量当前点云在不同帧中的旋转平移
    '''
    # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
    cam_rots = np.tile([1, 0, 0, 0], (1, 1)) # 得到二维数组[[1, 0, 0, 0]] 
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames)) 
    #对cam_rots 进行了修改。cam_rots[:, :, None] 增加了一个新的维度，得到：
    #[[[1]
    #[0]
    #[0]
    #[0]]]
    # 然后，np.tile(..., (1, 1, num_frames)) 将以上数组在第一和第二维度上各复制一次，在第三维度上复制 num_frames 次，例如num_frames=10，得到：
    # [[[1 1 1 1 1 1 1 1 1 1]
    # [0 0 0 0 0 0 0 0 0 0]
    # [0 0 0 0 0 0 0 0 0 0]
    # [0 0 0 0 0 0 0 0 0 0]]]
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))
    '''
        A3.参数params转换为PyTorch张量
    '''
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))
    '''
        B.高斯变量variables初始化，都初始化为零
            max_2D_radius：所有高斯点的最大二维半径
            means2D_gradient_accum：所有高斯点的二维累计梯度
            denom：未知
            timestep：当前时间
    '''
    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float()}

    return params, variables

def initialize_optimizer(params, lrs_dict, tracking):
    lrs = lrs_dict
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    if tracking:
        return torch.optim.Adam(param_groups)
    else:
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def initialize_first_timestep(dataset, num_frames, scene_radius_depth_ratio, 
                              mean_sq_dist_method, densify_dataset=None, gaussian_distribution=None):
    '''
    函数目的：对第一帧初始化高斯
    输入：
        dataset：包含了color, depth, intrinsics, pose的数据
        num_frames：帧数，即数据的数目
        scene_radius_depth_ratio：最大第一帧深度与场景半径的比率，剪枝和致密化用的，config文件提供
        mean_sq_dist_method：计算均方距离的方法，实际只有projective一个选项
        densify_dataset=None：第一次初始化没有densify
        gaussian_distribution=None 高斯分布是各向同性还是异性，实际上最后一层函数没将其作为输入，默认同性
    输出：
        高斯参数params
        高斯变量variables
        相机内参intrinsics
        第一帧的相机外参w2c
        相机模型cam
    '''
    '''
        A.数据获取及预处理
        从数据集中获取第一帧RGB-D数据（颜色、深度）、相机内参和相机位姿c2w(4x4)
        Get RGB-D Data & Camera Parameters
    '''
    color, depth, intrinsics, pose = dataset[0]
    # B.数据处理、格式处理、各项设置
    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)

    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3] # 取左上角3x3元素
    w2c = torch.linalg.inv(pose) # 求逆,pose是c2w
    # Setup Camera
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())

    '''
        B.密集化处理:如果传参提供了密集化数据集，则做相应处理
        一般没有提供，因此初始化的时候不用处理直接进入else
    '''
    if densify_dataset is not None:
        # Get Densification RGB-D Data & Camera Parameters
        color, depth, densify_intrinsics, _ = densify_dataset[0]
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        densify_intrinsics = densify_intrinsics[:3, :3]
        densify_cam = setup_camera(color.shape[2], color.shape[1], densify_intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())
    else:
        densify_intrinsics = intrinsics

    '''
        D.初始化点云和高斯
    '''
    mask = (depth > 0) # Mask out invalid depth values，大于则true否则false
    mask = mask.reshape(-1)
    '''
        D1.点云初始化
        init_pt_cld：世界坐标系下的三维点云位置xyz、颜色c，
        mean3_sq_dist：均方距离
    '''
    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, densify_intrinsics, w2c, 
                                                mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method)
    '''
        D2.高斯初始化，包括高斯参数params和高斯变量variables
        高斯参数：
            1）输入点云的位置，颜色，旋转矩阵，不透明度，大小 
            2）相机参数：旋转矩阵和平移矩阵，num_frames维度，用于衡量当前点云在不同帧中的旋转平移
        高斯变量：除了scene_radius，其他的都初始化为0
            1）max_2D_radius：所有高斯点的最大二维半径
            2）means2D_gradient_accum：所有高斯点的二维累计梯度
            3）denom：未知
            4）timestep：当前时间
            5）scene_radius：场景半径，下面加的，不是在initialize_params里赋值的
    '''
    # Initialize Parameters
    # params, variables = initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution)
    params, variables = initialize_params(init_pt_cld, num_frames, mean3_sq_dist)
    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    # 初始化高斯半径r = D_{GT}/focal length，根据投影关系Z/f = X/x，相当于表示图像平面上（这里似乎等被同于像素平面了）一个像素对应的3d长度
    # 论文原话为 a radius equal to having a one-pixel radius upon projection into the 2D image given by dividing the depth by the focal length
    variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio #scene_radius_depth_ratio是最大第一帧深度与场景半径的比率

    if densify_dataset is not None:
        return params, variables, intrinsics, w2c, cam, densify_intrinsics, densify_cam
    else:
        return params, variables, intrinsics, w2c, cam


def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss,
             sil_thres, use_l1, ignore_outlier_depth_loss, tracking=False, 
             mapping=False, do_ba=False, plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None):
    '''
    函数目的：在Tracking、Mapping的过程中计算当前帧的loss
    输入：相机参数 params
         当前数据 curr_data
         中间变量 variables
         迭代的时间索引 iter_time_idx
         损失权重 loss_weights
         是否使用深度图用于损失计算 use_sil_for_loss
         阈值 sil_thres 等等
    '''
    # Initialize Loss Dictionary
    losses = {}
    '''
    Step 1:
    根据输入的参数和当前迭代的时间步，调用 transform_to_frame 函数将世界坐标系中的点转换为相机坐标系中的高斯分布中心点，并考虑是否需要计算梯度。
    不同的模式（tracking、mapping）会影响对哪些参数计算梯度。
        tracking的时候camera pose需要计算梯度
        mapping的时候BA优化,则高斯和pose的梯度都要优化
        单纯的mapping则只需要优化高斯的梯度
    '''
    if tracking:
        # Get current frame Gaussians, where only the camera pose gets gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=True)
    elif mapping:
        # mapping的时候BA优化,则高斯和pose的梯度都要优化
        if do_ba:# 但do_ba一直是False，不执行
            # Get current frame Gaussians, where both camera pose and Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=True)
        else:# 单纯的mapping则只需要优化高斯的梯度
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=False)
    else:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                             gaussians_grad=True,
                                             camera_grad=False)

    '''
    Step 2:
    Initialize Render Variables
    将输入的参数 params 转换成一个包含渲染相关变量的字典rendervar与depth_sil_rendervar

    '''
    # 下面两行代码用于获取渲染rgb渲染量以及深度渲染量
    rendervar = transformed_params2rendervar(params, transformed_gaussians)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_gaussians)
    '''
    Step 3:
    Rendering RGB, Depth and Silhouette

    '''
    rendervar['means2D'].retain_grad() #在进行RGB渲染时，保留其梯度信息(means2D)。
    # 使用渲染器 Renderer 对当前帧进行RGB渲染，得到RGB图像 im、半径信息 radius。im.shape (3, height,width)
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar) #这里的Renderer是import from diff_gaussian_rasterization,也就是高斯光栅化的渲染
    # 将 means2D 的梯度累积到 variables 中，这是为了在颜色渲染过程中进行密集化（densification）。
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    # Depth & Silhouette Rendering
    # 使用渲染器 Renderer 对当前帧进行深度和轮廓渲染，得到深度轮廓图 depth_sil (3, height,width)
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    # 从深度轮廓图中提取深度信息 depth，轮廓信息 silhouette，以及深度的平方 depth_sq。
    depth = depth_sil[0, :, :].unsqueeze(0) # (1, height,width)
    silhouette = depth_sil[1, :, :] # (height,width)
    presence_sil_mask = (silhouette > sil_thres)
    depth_sq = depth_sil[2, :, :].unsqueeze(0)
    # 计算深度的不确定性，即深度平方的差值，然后将其分离出来并进行 detach 操作(不计算梯度)。
    uncertainty = depth_sq - depth**2
    uncertainty = uncertainty.detach()

    '''
    Step 4: 生成mask，用来选择需要计算loss的点
    nan_mask考虑深度和不确定度的有效性
    depth_error考虑误差异常
    presence_sil_mask考虑在tracking计算loss的时候使用轮廓图的存在性
    '''

    # Mask with valid depth values (accounts for outlier depth values)
    # 建一个 nan_mask，用于标记深度和不确定性的有效值，避免处理异常值。
    nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
    if ignore_outlier_depth_loss: # （Ignore, Matrixcity为false）如果开启了 ignore_outlier_depth_loss，则基于深度误差生成一个新的掩码 mask，并且该掩码会剔除深度值异常的区域。
        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 10*depth_error.median())
        mask = mask & (curr_data['depth'] > 0)
    else: #如果没有开启 ignore_outlier_depth_loss，则直接使用深度大于零的区域作为 mask。
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask
    # Mask with presence silhouette mask (accounts for empty space)
    # 如果在跟踪模式下且开启了使用轮廓图进行损失计算 (use_sil_for_loss)，则将 mask 与轮廓图的存在性掩码 presence_sil_mask 相与。
    if tracking and use_sil_for_loss: # Matrixcity的use_sil_for_loss为true
        mask = mask & presence_sil_mask

    '''
    至此,生成RGB图像、深度图、并根据需要进行掩码处理，以便后续在计算损失时使用
    Step 5：计算depth and RGB loss
    '''

    # Depth loss(计算深度的loss)
    if use_l1: #如果使用L1损失 (use_l1)，则将 mask 进行 detach 操作，即不计算其梯度。
        mask = mask.detach()
        if tracking: #如果在跟踪模式下 (tracking)，计算深度损失 (losses['depth']) 为当前深度图与渲染深度图之间差值的绝对值之和（只考虑掩码内的区域）。
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
        else: #如果不在跟踪模式下，计算深度损失为当前深度图与渲染深度图之间差值的绝对值的平均值（只考虑掩码内的区域）。上下一模一样
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()

    # RGB Loss(计算RGB的loss)
    # 如果在跟踪模式下 (tracking) 并且使用轮廓图进行损失计算 (use_sil_for_loss) 或者忽略异常深度值 (ignore_outlier_depth_loss)，计算RGB损失 (losses['im']) 为当前图像与渲染图像之间差值的绝对值之和（只考虑掩码内的区域）。
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss): # Matrixcity的use_sil_for_loss为true
        color_mask = torch.tile(mask, (3, 1, 1)) # 形状与 mask 相同，维度为 (3, H, W)。它的值是通过将 mask 沿着通道维度重复 3 次而得到的。这通常用于将二进制掩码转换为彩色掩码，以便在图像上显示不同的区域
        color_mask = color_mask.detach()
        losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
    elif tracking: #如果在跟踪模式下，但没有使用轮廓图进行损失计算，计算RGB损失为当前图像与渲染图像之间差值的绝对值之和。
        losses['im'] = torch.abs(curr_data['im'] - im).sum()
    else: #如果不在跟踪模式下，计算RGB损失为L1损失和结构相似性损失的加权和，其中 l1_loss_v1 是L1损失的计算函数，calc_ssim 是结构相似性损失的计算函数。
        losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))

    # (Ignore) Visualize the Diff Images
    if tracking and visualize_tracking_loss: # MatrixCity 的visualize_tracking_loss是False
        fig, ax = plt.subplots(2, 4, figsize=(12, 6))
        weighted_render_im = im * color_mask
        weighted_im = curr_data['im'] * color_mask
        weighted_render_depth = depth * mask
        weighted_depth = curr_data['depth'] * mask
        diff_rgb = torch.abs(weighted_render_im - weighted_im).mean(dim=0).detach().cpu()
        diff_depth = torch.abs(weighted_render_depth - weighted_depth).mean(dim=0).detach().cpu()
        viz_img = torch.clip(weighted_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[0, 0].imshow(viz_img)
        ax[0, 0].set_title("Weighted GT RGB")
        viz_render_img = torch.clip(weighted_render_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[1, 0].imshow(viz_render_img)
        ax[1, 0].set_title("Weighted Rendered RGB")
        ax[0, 1].imshow(weighted_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[0, 1].set_title("Weighted GT Depth")
        ax[1, 1].imshow(weighted_render_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[1, 1].set_title("Weighted Rendered Depth")
        ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.8)
        ax[0, 2].set_title(f"Diff RGB, Loss: {torch.round(losses['im'])}")
        ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
        ax[1, 2].set_title(f"Diff Depth, Loss: {torch.round(losses['depth'])}")
        ax[0, 3].imshow(presence_sil_mask.detach().cpu(), cmap="gray")
        ax[0, 3].set_title("Silhouette Mask")
        ax[1, 3].imshow(mask[0].detach().cpu(), cmap="gray")
        ax[1, 3].set_title("Loss Mask")
        # Turn off axis
        for i in range(2):
            for j in range(4):
                ax[i, j].axis('off')
        # Set Title
        fig.suptitle(f"Tracking Iteration: {tracking_iteration}", fontsize=16)
        # Figure Tight Layout
        fig.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"tmp.png"), bbox_inches='tight')
        plt.close()
        plot_img = cv2.imread(os.path.join(plot_dir, f"tmp.png"))
        cv2.imshow('Diff Images', plot_img)
        cv2.waitKey(1)
        ## Save Tracking Loss Viz
        # save_plot_dir = os.path.join(plot_dir, f"tracking_%04d" % iter_time_idx)
        # os.makedirs(save_plot_dir, exist_ok=True)
        # plt.savefig(os.path.join(save_plot_dir, f"%04d.png" % tracking_iteration), bbox_inches='tight')
        # plt.close()
    
    # 下面代码进行了损失的加权和最终的损失值计算
    # 对每个损失项按照其权重进行加权，得到 weighted_losses 字典，其中 k 是损失项的名称，v 是对应的损失值，loss_weights 是各个损失项的权重。
    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()} # loss_weights是输入的权重
    print('weighted_loss', weighted_losses)
    # 最终损失值 loss 是加权损失项的和。
    loss = sum(weighted_losses.values()) 
    print('loss',loss)

    seen = radius > 0 #创建一个布尔掩码 seen，其中对应的位置为 True 表示在当前迭代中观察到了某个点。
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen]) #更新 variables['max_2D_radius'] 中已观察到的点的最大半径。
    variables['seen'] = seen #将 seen 存储在 variables 字典中。
    weighted_losses['loss'] = loss #最终，将总损失值存储在 weighted_losses 字典中的 'loss' 键下。

    return loss, variables, weighted_losses


def initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    if gaussian_distribution == "isotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == "anisotropic":
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    return params


def add_new_gaussians(params, variables, curr_data, sil_thres, time_idx, mean_sq_dist_method):
    '''
    函数目的：在建图过程中根据当前帧的数据进行高斯分布的密集化
    '''
    '''
    Step1: 根据论文的公式9确定哪些像素需要增加高斯non_presence_mask
    '''
    # Silhouette Rendering（轮廓渲染）
    transformed_pts = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)#将高斯模型转换到frame坐标系下（返回的transformed_pts就是在相机坐标系下的高斯中心点）
    # 注意，此处的params（如下定义，实际上就是高斯函数，同时也包含pose等）
    # params = {
    #     'means3D': means3D,
    #     'rgb_colors': init_pt_cld[:, 3:6],
    #     'unnorm_rotations': unnorm_rots,
    #     'logit_opacities': logit_opacities,
    #     'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1)),
    # }

    #获取深度的渲染变量（#所谓的深度轮廓其实就是相机坐标系下的（深度值，1，深度的平方））
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_pts) 
    
    # 通过渲染器 Renderer 得到深度图和轮廓图，其中 depth_sil 包含了深度信息和轮廓信息。
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    # non_presence_sil_mask代表当前帧中未出现的区域？
    non_presence_sil_mask = (silhouette < sil_thres) #通过设置阈值 sil_thres（输入参数为0.5），创建一个轮廓图的非存在掩码 # 对应paper的公式9

    # Check for new foreground objects by using GT depth
    # 利用当前深度图和渲染后的深度图，通过 depth_error 计算深度误差，并生成深度非存在掩码 non_presence_depth_mask。
    gt_depth = curr_data['depth'][0, :, :] #获取真值的深度图
    render_depth = depth_sil[0, :, :] #获取渲染的深度图
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0) #计算深度误差
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median()) # 对应paper的公式9

    # Determine non-presence mask
    # 将轮廓图非存在掩码和深度非存在掩码合并生成整体的非存在掩码 non_presence_mask。
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    # 检测到非存在掩码中有未出现的点时，根据当前帧的数据生成新的高斯分布参数，并将这些参数添加到原有的高斯分布参数中
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        # 获取当前相机的旋转和平移信息:
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach()) #获取当前帧的相机未归一化旋转信息。
        curr_cam_tran = params['cam_trans'][..., time_idx].detach() #对旋转信息进行归一化。
        # 构建当前帧相机到世界坐标系的变换矩阵:
        curr_w2c = torch.eye(4).cuda().float() #创建一个单位矩阵
        # 利用归一化后的旋转信息和当前帧的相机平移信息，更新变换矩阵的旋转和平移部分。
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        # 生成有效深度掩码:
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0) #生成当前帧的有效深度掩码 valid_depth_mask。
        # 更新非存在掩码:
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1) #将 non_presence_mask 和 valid_depth_mask 进行逐元素与操作，得到更新后的非存在掩码。
        # 获取新的点云和平均平方距离:
        #利用 get_pointcloud 函数，传入当前帧的图像、深度图、内参、变换矩阵和非存在掩码，生成新的点云 new_pt_cld。同时计算这些新点云到已存在高斯分布的平均平方距离 mean3_sq_dist。
        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method) #参数文件中定义mean_sq_dist_method为projective
        # 初始化新的高斯分布参数:
        # 利用新的点云和平均平方距离，调用 initialize_new_params 函数生成新的高斯分布参数 new_params。
        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist)
        # 将新的高斯分布参数添加到原有参数中:
        for k, v in new_params.items(): #对于每个键值对 (k, v)，其中 k 是高斯分布参数的键，v 是对应的值，在 params 中将其与新参数 v 拼接，并转换为可梯度的 torch.nn.Parameter 对象。
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
        # (更新相关的统计信息)初始化一些统计信息，如梯度累积、分母、最大2D半径等。
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        # (更新时间步信息)将新的点云对应的时间步信息 new_timestep（都是当前帧的时间步）拼接到原有的时间步信息中。
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'],new_timestep),dim=0)

    # 将更新后的模型参数 params 和相关的统计信息 variables 返回。
    return params, variables



def initialize_camera_pose(params, curr_time_idx, forward_prop):
    '''
        函数作用：tracking的迭代优化阶段用于初始化相机姿态：根据当前时间(使用的是curr_time_idx索引)初始化相机的旋转(cam_unnorm_rots)和平移参数(cam_trans)
    '''
    with torch.no_grad():  # 用来确保在这个上下文中,没有梯度计算
        if curr_time_idx > 1 and forward_prop:  # 如果当前帧数大于1且为前向传播，则基于恒定速度模型constant velocity model初始化当前帧的相机姿态
            # 旋转
            prev_rot1 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-1].detach())  # 获取前一帧的旋转
            prev_rot2 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-2].detach())  # 获取前两帧的旋转
            new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))  # 计算新的旋转
            params['cam_unnorm_rots'][..., curr_time_idx] = new_rot.detach()  # 更新当前帧的旋转
            # 平移
            prev_tran1 = params['cam_trans'][..., curr_time_idx-1].detach()  # 获取前一帧的平移
            prev_tran2 = params['cam_trans'][..., curr_time_idx-2].detach()  # 获取前两帧的平移
            new_tran = prev_tran1 + (prev_tran1 - prev_tran2)  # 计算新的平移
            params['cam_trans'][..., curr_time_idx] = new_tran.detach()  # 更新当前帧的平移
        else: #否则，直接复制前一帧的相机姿态到当前帧
            params['cam_unnorm_rots'][..., curr_time_idx] = params['cam_unnorm_rots'][..., curr_time_idx-1].detach()  # 使用前一帧的旋转初始化当前帧的旋转
            params['cam_trans'][..., curr_time_idx] = params['cam_trans'][..., curr_time_idx-1].detach()  # 使用前一帧的平移初始化当前帧的平移
    return params  # 返回参数


def convert_params_to_store(params):#(Ignore)没引用到
    '''
        (Ignore)没引用到
    '''
    params_to_store = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_to_store[k] = v.detach().clone()
        else:
            params_to_store[k] = v
    return params_to_store


def rgbd_slam(config: dict):
    """
        SLAM主函数
        输入：splatam.py模块中的config字典
    """
    '''
        1.1 Print Config
    '''
    print("Loaded Config:")
    if "use_depth_loss_thres" not in config['tracking']:
        # not in则不使用深度误差的阈值来决定tracking是否停止，matrixcity是not in并且直接用迭代最大值来决定是否停止
        config['tracking']['use_depth_loss_thres'] = False
        config['tracking']['depth_loss_thres'] = 100000 # 该参数在matrixcity中实际没用到
    if "visualize_tracking_loss" not in config['tracking']:
        # MatrixCity是not in
        config['tracking']['visualize_tracking_loss'] = False
    if "gaussian_distribution" not in config:
        config['gaussian_distribution'] = "isotropic"
    print(f"{config}")
    '''
        1.2 输出文件的保存路径Create Output Directories
        workdir： "./experiments/{group_name}", 例如：gruop_name: "MatrixCity"，workdir： "./experiments/MatrixCity"
        run_name： "{scene_name}_{seed}", 例如: scene_name = scenes[0], scenes = ["smallaerial"], seed = 0, run_name = "smallaerial_0"
    '''
    output_dir = os.path.join(config["workdir"], config["run_name"])
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    '''
        1.3 Init WandB
    '''
    if config['use_wandb']:
        wandb_time_step = 0
        wandb_tracking_step = 0
        wandb_mapping_step = 0
        wandb_run = wandb.init(project=config['wandb']['project'],
                               entity=config['wandb']['entity'],
                               group=config['wandb']['group'],
                               name=config['wandb']['name'],
                               config=config)
    '''
        2 加载设备和数据集（相关代码较长，中间涉及到几个环节的Init seperate dataloader）
    '''
    # Get Device
    device = torch.device(config["primary_device"])
    # Load Dataset
    print("Loading Dataset ...")
    dataset_config = config["data"]
    if "gradslam_data_cfg" not in dataset_config: 
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:#一般数据集是in
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
    if "ignore_bad" not in dataset_config: 
        dataset_config["ignore_bad"] = False #一般数据集是not in
    if "use_train_split" not in dataset_config: 
        dataset_config["use_train_split"] = True #一般数据集是not in
    if "densification_image_height" not in dataset_config: 
        #如果not in，则在Densification的时候一次加载当前帧和之前所有帧的数据，Matrixcity是not in
        dataset_config["densification_image_height"] = dataset_config["desired_image_height"]
        dataset_config["densification_image_width"] = dataset_config["desired_image_width"]
        seperate_densification_res = False 
    else:
        if dataset_config["densification_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["densification_image_width"] != dataset_config["desired_image_width"]:
            seperate_densification_res = True
        else:
            seperate_densification_res = False
    if "tracking_image_height" not in dataset_config:
        dataset_config["tracking_image_height"] = dataset_config["desired_image_height"]
        dataset_config["tracking_image_width"] = dataset_config["desired_image_width"]
        seperate_tracking_res = False #一般数据集这个是not in
    else:
        if dataset_config["tracking_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["tracking_image_width"] != dataset_config["desired_image_width"]:
            seperate_tracking_res = True
        else:
            seperate_tracking_res = False
    # Poses are relative to the first frame
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
    )
    num_frames = dataset_config["num_frames"]
    if num_frames == -1: #一般是-1，除了iphone采集的数据集
        num_frames = len(dataset) #数据集帧总数
    '''
        3 第一帧,初始化高斯点云
    '''
    # (Ignore) Init seperate dataloader for densification if required 
    if seperate_densification_res:
        densify_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["densification_image_height"],
            desired_width=dataset_config["densification_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
        )
        # Initialize Parameters, Canonical & Densification Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam, \
            densify_intrinsics, densify_cam = initialize_first_timestep(dataset, num_frames,
                                                                        config['scene_radius_depth_ratio'],
                                                                        config['mean_sq_dist_method'],
                                                                        densify_dataset=densify_dataset,
                                                                        gaussian_distribution=config['gaussian_distribution'])                                                                                                                  
    else: 
        # Initialize Parameters & Canoncial Camera parameters    
        '''
            输入：数据集，帧数
                scene_radius_depth_ratio：第一帧的最大深度与场景半径的比例
                mean_sq_dist_method：计算均方距离的方法，默认projective
                gaussian_distribution：高斯分布是各向同性还是异性，默认同性
            输出：
                params
                variables
                intrinsics
                first_frame_w2c
                cam
        '''
        params, variables, intrinsics, first_frame_w2c, cam = initialize_first_timestep(dataset, num_frames, 
                                                                                        config['scene_radius_depth_ratio'],
                                                                                        config['mean_sq_dist_method'],
                                                                                        gaussian_distribution=config['gaussian_distribution'])
    '''
        (Ignore) Init seperate dataloader for tracking if required
        一般用不到，直接跳过
    '''
    if seperate_tracking_res:
        tracking_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["tracking_image_height"],
            desired_width=dataset_config["tracking_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
        )
        tracking_color, _, tracking_intrinsics, _ = tracking_dataset[0]
        tracking_color = tracking_color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        tracking_intrinsics = tracking_intrinsics[:3, :3]
        tracking_cam = setup_camera(tracking_color.shape[2], tracking_color.shape[1], 
                                    tracking_intrinsics.cpu().numpy(), first_frame_w2c.detach().cpu().numpy())
    
    '''
        4 初始化用于记录整个slam的变量
            1. 带时间戳的关键帧列表
            2. 迭代过的真实位姿
            3. mapping和tracking的计时变量
    '''
    keyframe_list = []# Initialize list to keep track of Keyframes
    keyframe_time_indices = []
    # Init Variables to keep track of ground truth poses
    gt_w2c_all_frames = []
    # tracking的计时
    tracking_iter_time_sum = 0 #SLAM-Tracking的总迭代时间之和
    tracking_iter_time_count = 0 #SLAM-Tracking的总迭代数
    tracking_frame_time_sum = 0 #SLAM-Tracking的帧处理总时间之和
    tracking_frame_time_count = 0 #SLAM-Tracking的总帧数
    # mapping的计时
    mapping_iter_time_sum = 0
    mapping_iter_time_count = 0
    mapping_frame_time_sum = 0
    mapping_frame_time_count = 0

    '''
        (Ignore) Load Checkpoint
        一般用不到，直接到else
    '''
    if config['load_checkpoint']:
        checkpoint_time_idx = config['checkpoint_time_idx']
        print(f"Loading Checkpoint for Frame {checkpoint_time_idx}")
        ckpt_path = os.path.join(config['workdir'], config['run_name'], f"params{checkpoint_time_idx}.npz")
        params = dict(np.load(ckpt_path, allow_pickle=True))
        params = {k: torch.tensor(params[k]).cuda().float().requires_grad_(True) for k in params.keys()}
        variables['max_2D_radius'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['means2D_gradient_accum'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['denom'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['timestep'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        # Load the keyframe time idx list
        keyframe_time_indices = np.load(os.path.join(config['workdir'], config['run_name'], f"keyframe_time_indices{checkpoint_time_idx}.npy"))
        keyframe_time_indices = keyframe_time_indices.tolist()
        # Update the ground truth poses list
        for time_idx in range(checkpoint_time_idx):
            # Load RGBD frames incrementally instead of all frames
            color, depth, _, gt_pose = dataset[time_idx]
            # Process poses
            gt_w2c = torch.linalg.inv(gt_pose)
            gt_w2c_all_frames.append(gt_w2c)
            # Initialize Keyframe List
            if time_idx in keyframe_time_indices:
                # Get the estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                color = color.permute(2, 0, 1) / 255
                depth = depth.permute(2, 0, 1)
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
    else:
        checkpoint_time_idx = 0 
    
    '''
        5 SLAM：以时间顺序处理RGB-D帧，进行跟踪（Tracking）和建图（Mapping）
    '''
    # Iterate over Scan
    for time_idx in tqdm(range(checkpoint_time_idx, num_frames)): #循环迭代处理 RGB-D 帧，循环的起始索引是 checkpoint_time_idx（也就是是否从某帧开始，一般都是0开始），终止索引是 num_frames
        '''
            5.1 获取并处理当前time_idx下的rgbd,w2c
        '''
        color, depth, _, gt_pose = dataset[time_idx] #从数据集 dataset 中incrementally地加载 RGB-D 帧的颜色、深度、姿态等信息
        gt_w2c = torch.linalg.inv(gt_pose) # 对姿态信息进行处理，计算gt_pose的逆，也就是世界到相机的变换矩阵 gt_w2c
        
        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255 # 颜色归一化
        depth = depth.permute(2, 0, 1)
        
        gt_w2c_all_frames.append(gt_w2c)
        curr_gt_w2c = gt_w2c_all_frames

        iter_time_idx = time_idx # Optimize only current time step for tracking

        '''
            5.2 初始化用于mapping和tracking的数据，包括：相机模型，当前time_idx的rgbd，相机内参，第一帧外参，迭代过程至今的外参curr_gt_w2c
        '''
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': iter_time_idx, 'intrinsics': intrinsics, 
                     'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}# 初始化用于mapping（选择frame）的数据curr_data
        num_iters_mapping = config['mapping']['num_iters']# 设置建图迭代次数

        if seperate_tracking_res:#ignore
            tracking_color, tracking_depth, _, _ = tracking_dataset[time_idx]
            tracking_color = tracking_color.permute(2, 0, 1) / 255
            tracking_depth = tracking_depth.permute(2, 0, 1)
            tracking_curr_data = {'cam': tracking_cam, 'im': tracking_color, 'depth': tracking_depth, 'id': iter_time_idx,
                                  'intrinsics': tracking_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
        else:#初始化用于tracking的数据tracking_curr_data
            tracking_curr_data = curr_data

        if time_idx > 0:# 初始化当前帧的相机位姿 Initialize the camera pose for the current frame
            params = initialize_camera_pose(params, time_idx, forward_prop=config['tracking']['forward_prop']) # 在configs/replica/splatam.py中，forward_prop是True

        '''
            5.3 Tracking
        '''
        tracking_start_time = time.time()
        if time_idx > 0 and not config['tracking']['use_gt_poses']: # 如果当前时间索引 time_idx 大于 0 且不使用真实姿态
            # ** Sec 1.2 多个变量的重置、初始化和各项设置 **
            '''
                Tracking.Step1 初始化：
                    1. 优化器optimizer
                    2. 候选位姿candidate_cam_unnorm_rot，candidate_cam_tran
                    3. 最小loss
                    4. 针对一帧Tracking的迭代计数器iter，do_continue_slam，最大迭代数
                    5. 进度条：progress_bar，显示当前是第几帧
            '''
            optimizer = initialize_optimizer(params, config['tracking']['lrs'], tracking=True)# 重置优化器和学习率（Reset Optimizer & Learning Rates for tracking）
            candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()# 初始化变量candidate_cam_unnorm_rot以跟踪最佳的相机旋转（Keep Track of Best Candidate Rotation & Translation）
            candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()# 初始化变量candidate_cam_tran以跟踪最佳的相机平移
            current_min_loss = float(1e20)# 初始化变量 current_min_loss 用于跟踪当前迭代中的最小损失
            
            iter = 0 #循环迭代次数记数，每一帧都会初始化一次
            do_continue_slam = False #matrixcity中实际上没用到
            num_iters_tracking = config['tracking']['num_iters'] #针对每一帧Tracking的最大迭代次数
            
            progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")# 使用 tqdm 创建一个进度条，显示当前跟踪迭代的进度 
            
            '''
                Tracking.Step2 迭代优化位姿
            '''
            while True:
                iter_start_time = time.time() # 计算迭代开始的时间
                '''
                    Tracking.Step2.A 计算loss，backward，Optimization
                '''
                loss, variables, losses = get_loss(params, tracking_curr_data, variables, iter_time_idx, config['tracking']['loss_weights'],
                                                   config['tracking']['use_sil_for_loss'], config['tracking']['sil_thres'],
                                                   config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'], tracking=True, 
                                                   plot_dir=eval_dir, visualize_tracking_loss=config['tracking']['visualize_tracking_loss'],
                                                   tracking_iteration=iter)#计算当前帧的损失
                if config['use_wandb']:# Report Loss
                    wandb_tracking_step = report_loss(losses, wandb_run, wandb_tracking_step, tracking=True)
                
                loss.backward()# Backprop 将loss进行反向传播,计算梯度
                optimizer.step()# Optimizer Update
                optimizer.zero_grad(set_to_none=True)

                '''
                    Tracking.Step2.B 更新当前最优loss对应的位姿
                '''
                with torch.no_grad():#梯度参数不变情况下
                    # 如果当前损失小于 current_min_loss，更新最小损失对应的相机旋转和平移 Save the best candidate rotation & translation  
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
                        candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
                    
                    # Report Progress
                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True,
                                            wandb_run=wandb_run, wandb_step=wandb_tracking_step, wandb_save_qual=config['wandb']['save_qual'])
                        else:
                            report_progress(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
                    else:
                        progress_bar.update(1)
                
                '''
                    Tracking.Step2.C 更新迭代计数，判断是否结束迭代
                    Matrixcity达到最大迭代次数直接stop
                '''
                #更新:SLAM-Tracking总循环迭代次数、总循环迭代的运行时间、针对当前帧tracking的循环迭代次数iter
                #频率：每一次迭代
                iter_end_time = time.time()
                tracking_iter_time_sum += iter_end_time - iter_start_time
                tracking_iter_time_count += 1
                
                iter += 1 #每一次迭代更新当前帧的迭代次数
                if iter == num_iters_tracking:
                    '''
                        判断是否要结束当前帧的Tracking：检查是否最大迭代次数，满足则终止计算 Check if we should stop tracking 
                        Matrixcity达到最大迭代次数直接stop
                    '''
                    if losses['depth'] < config['tracking']['depth_loss_thres'] and config['tracking']['use_depth_loss_thres']:
                        # (Ignore) matrixcitydataset默认：config['tracking']['depth_loss_thres'] = 100000满足；config['tracking']['use_depth_loss_thres'] = False不满足
                        break 
                    elif config['tracking']['use_depth_loss_thres'] and not do_continue_slam: 
                        # (Ignore)每一帧do_continue_slam初始化是false，满足；但config['tracking']['use_depth_loss_thres'] = False不满足
                        do_continue_slam = True
                        progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
                        num_iters_tracking = 2*num_iters_tracking
                        if config['use_wandb']:
                            wandb_run.log({"Tracking/Extra Tracking Iters Frames": time_idx,
                                        "Tracking/step": wandb_time_step})
                    else:#Matrixcity达到最大迭代次数直接stop
                        break

            '''
                Tracking.Step3 Tracking迭代优化位姿结束，更新最优位姿
            '''
            progress_bar.close() 
            with torch.no_grad():# 从while循环出来了,更新最佳位姿，Copy over the best candidate rotation & translation
                params['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
                params['cam_trans'][..., time_idx] = candidate_cam_tran
        
        # （Ignore）另一个分支:如果当前时间索引 time_idx 大于 0 且使用真实姿态，但matrixcity中config['tracking']['use_gt_poses']=false不满足
        elif time_idx > 0 and config['tracking']['use_gt_poses']:
            with torch.no_grad():
                # Get the ground truth pose relative to frame 0
                rel_w2c = curr_gt_w2c[-1]
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                rel_w2c_tran = rel_w2c[:3, 3].detach()
                # Update the camera parameters
                params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                params['cam_trans'][..., time_idx] = rel_w2c_tran
        
        '''
            5.4 更新Tracking的计时器，报告跟踪进度
        '''
        # 更新：SLAM-Tracking帧的总运行时间和帧数
        # 频率：每一帧Tracking完更新一次
        tracking_end_time = time.time()
        tracking_frame_time_sum += tracking_end_time - tracking_start_time
        tracking_frame_time_count += 1

        # 报告跟踪进度,可视化进度条并自动保存参数
        if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
            try:
                # Report Final Tracking Progress
                progress_bar = tqdm(range(1), desc=f"Tracking Result Time Step: {time_idx}")
                with torch.no_grad():
                    if config['use_wandb']:
                        report_progress(params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True,
                                        wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'], global_logging=True)
                    else:
                        report_progress(params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
                progress_bar.close()
            except:
                ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                save_params_ckpt(params, ckpt_output_dir, time_idx)
                print('Failed to evaluate trajectory.')


        '''
            5.5 Densification & Keyframe Selection & Mapping
        '''
        if time_idx == 0 or (time_idx+1) % config['map_every'] == 0: #第一帧开始Densification，然后每map_every帧一次，例如map_every=3，则致密化的是：第1帧，第3帧...
            '''
                5.5.1 Densification
            '''
            if config['mapping']['add_new_gaussians'] and time_idx > 0: #第二帧开始
                '''
                    Densification.Step1 设置Densification的数据源
                    if 满足用当前帧，else用当前帧和之前所有帧，matrixcity是后者
                '''
                if seperate_densification_res:# 如果if判断成立，逐个加载RGB-D帧，而不是一次性加载所有帧
                    densify_color, densify_depth, _, _ = densify_dataset[time_idx]
                    densify_color = densify_color.permute(2, 0, 1) / 255 #permute 是 PyTorch 中的一个函数，它用于改变张量（tensor）的维度顺序12。这个函数的参数是一个或多个整数，代表新的维度顺序1。例如，如果我们有一个形状为 (2, 3, 5) 的张量 x，我们可以使用 x.permute(2, 0, 1) 来改变维度的顺序。这将返回一个新的张量，其形状为 (5, 2, 3)
                    densify_depth = densify_depth.permute(2, 0, 1)
                    densify_curr_data = {'cam': densify_cam, 'im': densify_color, 'depth': densify_depth, 'id': time_idx, 
                                 'intrinsics': densify_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
                else: # 否则使用包含当前帧和之前帧的数据 curr_data,matrixcity是这个
                    densify_curr_data = curr_data
                '''
                    Densification.Step2 添加新的gaussians
                    Add new Gaussians to the scene based on the Silhouette
                '''
                params, variables = add_new_gaussians(params, variables, densify_curr_data, 
                                                      config['mapping']['sil_thres'], time_idx,
                                                      config['mean_sq_dist_method'],"isotropic")
                post_num_pts = params['means3D'].shape[0]# 记录添加新的高斯后，post_num_pts是高斯分布数量
                if config['use_wandb']:
                    wandb_run.log({"Mapping/Number of Gaussians": post_num_pts,
                                   "Mapping/step": wandb_time_step})
            
            '''
                5.5.2 KeyFrame Selection
            '''
            with torch.no_grad(): # 在此代码块内部进行的计算不会涉及梯度计算
                '''
                    KeyFrame Selection.Step1 从时间索引提取当前帧相机位姿并做坐标系转换
                    Get the current estimated rotation & translation
                '''
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                '''
                    KeyFrame Selection.Step2 根据重合度从关键帧列表（除去最后一帧关键帧）选择k-2个关键帧+当前帧+最后一帧关键帧
                    根据配置中的 mapping_window_size，计算需要选择的关键帧数量 num_keyframes
                '''
                num_keyframes = config['mapping_window_size']-2 #这里的减去2对应论文的原文，对应着"k-2个先前关键帧"的由来，在参数传入的时候就做好了k-2的限制
                # selected_keyframes是一个列表，列表的元素frame_idx对应keyframe_list里的index，每次调用keyframe_selection_overlap都更新一次，最小值0，最大值为keyframe_list-2或num_keyframes-1
                selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, intrinsics, keyframe_list[:-1], num_keyframes)#根据重叠程度进行关键帧选择，selected_keyframes是id
                # selected_time_idx是时间戳列表，里面的元素代表第几帧，例如0代表第一帧
                selected_time_idx = [keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes] 
                # 添加keyframe最后一帧的时间戳和keyframe index
                if len(keyframe_list) > 0: 
                    selected_time_idx.append(keyframe_list[-1]['id'])
                    selected_keyframes.append(len(keyframe_list)-1)
                # 添加当前帧的时间戳和keyframe index -1表示倒数第一个
                selected_time_idx.append(time_idx)
                selected_keyframes.append(-1) #
                print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}")# Print the selected keyframes

            # 执行Mapping的优化前，初始化优化器Reset Optimizer & Learning Rates for Full Map Optimization
            optimizer = initialize_optimizer(params, config['mapping']['lrs'], tracking=False) 

            '''
                5.5.3 Mapping
            '''
            mapping_start_time = time.time()
            if num_iters_mapping > 0:
                progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
            for iter in range(num_iters_mapping):
                iter_start_time = time.time()
                # Randomly select a frame until current time step amongst keyframes
                rand_idx = np.random.randint(0, len(selected_keyframes))
                selected_rand_keyframe_idx = selected_keyframes[rand_idx]
                if selected_rand_keyframe_idx == -1: # selected_keyframes.append(-1)只有第一帧，就用当前帧
                    iter_time_idx = time_idx
                    iter_color = color
                    iter_depth = depth
                else:# Use Keyframe Data
                    iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                    iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
                    iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
                iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
                iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx, 
                             'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
                # 重点函数：计算当前帧的损失
                loss, variables, losses = get_loss(params, iter_data, variables, iter_time_idx, config['mapping']['loss_weights'],
                                                config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'], mapping=True)
                if config['use_wandb']: # Report Loss
                    wandb_mapping_step = report_loss(losses, wandb_run, wandb_mapping_step, mapping=True)
                loss.backward() # Backprop
                with torch.no_grad():
                    # Prune Gaussians
                    if config['mapping']['prune_gaussians']: # replica和Matrixcity都是True，执行
                        params, variables = prune_gaussians(params, variables, optimizer, iter, config['mapping']['pruning_dict'])
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Pruning": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                    # Gaussian-Splatting's Gradient-based Densification
                    if config['mapping']['use_gaussian_splatting_densification']: # replica和Matrixcity都是False，不执行
                        params, variables = densify(params, variables, optimizer, iter, config['mapping']['densify_dict'])
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Densification": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                    # Optimizer Update
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # Report Progress
                    if config['report_iter_progress']:
                        if config['use_wandb']:
                            report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_mapping_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, online_time_idx=time_idx)
                        else:
                            report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            mapping=True, online_time_idx=time_idx)
                    else:
                        progress_bar.update(1)
                # Update the runtime numbers
                iter_end_time = time.time()
                mapping_iter_time_sum += iter_end_time - iter_start_time
                mapping_iter_time_count += 1
            if num_iters_mapping > 0:
                progress_bar.close()
            # Update the runtime numbers
            mapping_end_time = time.time()
            mapping_frame_time_sum += mapping_end_time - mapping_start_time
            mapping_frame_time_count += 1

            if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
                try:
                    # Report Mapping Progress
                    progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {time_idx}")
                    with torch.no_grad():
                        if config['use_wandb']:
                            report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, online_time_idx=time_idx, global_logging=True)
                        else:
                            report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            mapping=True, online_time_idx=time_idx)
                    progress_bar.close()
                except:
                    ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                    save_params_ckpt(params, ckpt_output_dir, time_idx)
                    print('Failed to evaluate trajectory.')
        
        # Add frame to keyframe list    
        if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or \
                    (time_idx == num_frames-2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any()):
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
                keyframe_time_indices.append(time_idx)
        
        # (Ignore) Checkpoint every iteration, MatrixCity没有
        if time_idx % config["checkpoint_interval"] == 0 and config['save_checkpoints']:
            ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
            save_params_ckpt(params, ckpt_output_dir, time_idx)
            np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"), np.array(keyframe_time_indices))
        
        # Increment WandB Time Step
        if config['use_wandb']:
            wandb_time_step += 1

        torch.cuda.empty_cache()

    # Compute Average Runtimes
    if tracking_iter_time_count == 0:
        tracking_iter_time_count = 1
        tracking_frame_time_count = 1
    if mapping_iter_time_count == 0:
        mapping_iter_time_count = 1
        mapping_frame_time_count = 1
    tracking_iter_time_avg = tracking_iter_time_sum / tracking_iter_time_count
    tracking_frame_time_avg = tracking_frame_time_sum / tracking_frame_time_count
    mapping_iter_time_avg = mapping_iter_time_sum / mapping_iter_time_count
    mapping_frame_time_avg = mapping_frame_time_sum / mapping_frame_time_count
    print(f"\nAverage Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms")
    print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
    print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
    print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")
    if config['use_wandb']:
        wandb_run.log({"Final Stats/Average Tracking Iteration Time (ms)": tracking_iter_time_avg*1000,
                       "Final Stats/Average Tracking Frame Time (s)": tracking_frame_time_avg,
                       "Final Stats/Average Mapping Iteration Time (ms)": mapping_iter_time_avg*1000,
                       "Final Stats/Average Mapping Frame Time (s)": mapping_frame_time_avg,
                       "Final Stats/step": 1})
    
    # Evaluate Final Parameters
    with torch.no_grad():
        if config['use_wandb']:
            print('start evel')
            eval(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                 wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                 mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                 eval_every=config['eval_every'])
            print('end evel')
        else:
            eval(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                 mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                 eval_every=config['eval_every'])

    # Add Camera Parameters to Save them
    params['timestep'] = variables['timestep']
    params['intrinsics'] = intrinsics.detach().cpu().numpy()
    params['w2c'] = first_frame_w2c.detach().cpu().numpy()
    params['org_width'] = dataset_config["desired_image_width"]
    params['org_height'] = dataset_config["desired_image_height"]
    params['gt_w2c_all_frames'] = []
    for gt_w2c_tensor in gt_w2c_all_frames:
        params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
    params['gt_w2c_all_frames'] = np.stack(params['gt_w2c_all_frames'], axis=0)
    params['keyframe_time_indices'] = np.array(keyframe_time_indices)
    
    # Save Parameters
    save_params(params, output_dir)

    # Close WandB Run
    if config['use_wandb']:
        wandb.finish()




if __name__ == "__main__":
    '''
        1 从命令行加载要用来做实验的数据集的配置文件module到experiment变量中
    '''
    parser = argparse.ArgumentParser() # 创建了一个新的命令行输入解析对象
    parser.add_argument("experiment", type=str, help="Path to experiment file") # 添加了一个命令行参数，这个参数名字是 “experiment”，它的类型是字符串，它的帮助信息是 “Path to experiment file”
    args = parser.parse_args() # 解析命令行输入，并将结果存储在 args.experiment 中，例：python scripts/splatam.py configs/replica/splatam.py
    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module() # 加载了一个 Python 模块，这个模块的路径是从命令行输入中获取的

    '''
        2 设置实验因子，确保每次实验结果可复现
    '''
    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])
    
    '''
        3 根据experiment模块的配置，设置文件保存路径
    '''
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    ) # 创建了一个结果目录的路径，这个路径是由工作目录和运行名称拼接而成的，这两个值都是从加载的模块的配置中获取的

    '''
        4 如果没有checkpoint，则根据路径创建文件保存文件夹
    '''
    if not experiment.config['load_checkpoint']: # 检查配置中的 ‘load_checkpoint’ 选项。如果这个选项为 False，那么就会创建结果目录，并将配置文件复制到结果目录中
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py")) # 例如会将configs/replica/splatam.py复制到experiments/replica/room0_0/下，命名为config.py
    
    '''
        5 开始SLAM,输入experiment的配置字典config
    '''
    rgbd_slam(experiment.config)