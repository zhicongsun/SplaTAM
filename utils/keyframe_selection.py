"""
Code for Keyframe Selection based on re-projection of points from 
the current frame to the keyframes.
"""

import torch
import numpy as np


def get_pointcloud(depth, intrinsics, w2c, sampled_indices):
    '''
        函数目的：根据需要采样的2d indices获取对应的三维点云并剔除重复点
    '''
    '''
        Step1:根据二维采样点的像素坐标sampled_indices获取实际三维点云的坐标
    '''
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of sampled pixels
    xx = (sampled_indices[:, 1] - CX)/FX
    yy = (sampled_indices[:, 0] - CY)/FY
    depth_z = depth[0, sampled_indices[:, 0], sampled_indices[:, 1]]

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)# xx*Z = X，输出的shape为(采样点数, 3)
    # print('pts_cam:',pts_cam)
    # print('torch.ones_like(pts_cam[:, :1]):',torch.ones_like(pts_cam[:, :1]))

    pts4 = torch.cat([pts_cam, torch.ones_like(pts_cam[:, :1])], dim=1)#加了一列1，torch.ones_like:Returns a tensor filled with the scalar value 1, with the same size as input
    # print('pts4:',pts4)
    c2w = torch.inverse(w2c)
    pts = (c2w @ pts4.T).T[:, :3]
    # print('pts:',pts)

    '''
        Step2:剔除坐标重复的三维点
    '''
    # Remove points at camera origin
    A = torch.abs(torch.round(pts, decimals=4))# torch.round: Rounds elements of input to the nearest integer(decimals: Number of decimal places to round to), then torch.abs: obtain abs numb
    # print('A:',A)
    B = torch.zeros((1, 3)).cuda().float()
    # print('B',B)
    _, idx, counts = torch.cat([A, B], dim=0).unique(
        dim=0, return_inverse=True, return_counts=True)
    #_是所有的不同三维点
    #idx表明torch.cat([A, B]每个点对应_中的哪个
    # print('idx',idx)
    # print('counts',counts.shape)
    # print('torch.where(counts.gt(1))',torch.where(counts.gt(1)))
    # print('counts.gt',torch.where(counts.gt(1))[0])
    # print('counts.gt',torch.where(counts.gt(1)))
    mask = torch.isin(idx, torch.where(counts.gt(1))[0])
    #torch.where(counts.gt(1))[0]返回count大于1的点对应在_的类型
    #torch.isin(idx, torch.where(counts.gt(1))[0])返回false true列表，idx中有上述对应类型的设置true
    # print('mask',counts[torch.where(counts.gt(1))[0]])
    invalid_pt_idx = mask[:len(A)]#相当于把0那一行去掉了，不懂为什么要加0
    valid_pt_idx = ~invalid_pt_idx#false 和true倒转
    pts = pts[valid_pt_idx]#取有效的点

    return pts


def keyframe_selection_overlap(gt_depth, w2c, intrinsics, keyframe_list, k, pixels=1600):
    """
    Select overlapping keyframes to the current camera observation.

    Args:
        gt_depth (tensor): ground truth depth image of the current frame.
        w2c (tensor): world to camera matrix (4 x 4).
        keyframe_list (list): a list containing info for each keyframe.
        k (int): number of overlapping keyframes to select.
        pixels (int, optional): number of pixels to sparsely sample 
            from the image of the current camera. Defaults to 1600.
        超参数
            pixels决定当前帧的采样数
            K决定在候选的keyframe list里随机抽样得到的最终keyframe list里的帧数
    Returns:
        selected_keyframe_list (list): list of selected keyframe id.
    """

    '''
        Step1: 采样pixels个点，获取对应的三维坐标，除去重复点得到pts
    '''
    # Radomly Sample Pixel Indices from valid depth pixels
    width, height = gt_depth.shape[2], gt_depth.shape[1]
    valid_depth_indices = torch.where(gt_depth[0] > 0)
    valid_depth_indices = torch.stack(valid_depth_indices, dim=1) #通过这两步得到depth>0的点的坐标tensor，shape[0]对应depth>0的点个数,每一行是对应的坐标
    indices = torch.randint(valid_depth_indices.shape[0], (pixels,))#valid_depth_indices.shape[0]为index的范围，采样pixels个点，每个点的值在这个index范围内取
    sampled_indices = valid_depth_indices[indices]
    # Back Project the selected pixels to 3D Pointcloud
    pts = get_pointcloud(gt_depth, intrinsics, w2c, sampled_indices)

    list_keyframe = []
    for keyframeid, keyframe in enumerate(keyframe_list):
        '''
            Step2: 遍历每个关键帧，将采样点投影到该关键帧的2d空间，计算这些2d投影点在该关键帧边框内的比例percent_inside
        '''
        # Get the estimated world2cam of the keyframe
        est_w2c = keyframe['est_w2c']
        # Transform the 3D pointcloud to the keyframe's camera space
        pts4 = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
        transformed_pts = (est_w2c @ pts4.T).T[:, :3]
        # Project the 3D pointcloud to the keyframe's image space
        points_2d = torch.matmul(intrinsics, transformed_pts.transpose(0, 1))#transpose转制，matul矩阵相乘
        points_2d = points_2d.transpose(0, 1)
        points_z = points_2d[:, 2:] + 1e-5
        points_2d = points_2d / points_z
        projected_pts = points_2d[:, :2]
        # Filter out the points that are outside the image
        edge = 20
        mask = (projected_pts[:, 0] < width-edge)*(projected_pts[:, 0] > edge) * \
            (projected_pts[:, 1] < height-edge)*(projected_pts[:, 1] > edge)
        mask = mask & (points_z[:, 0] > 0)
        # Compute the percentage of points that are inside the image
        percent_inside = mask.sum()/projected_pts.shape[0]
        list_keyframe.append(
            {'id': keyframeid, 'percent_inside': percent_inside})

    '''
        Step3: 基于percent_inside从达到小对关键帧排序，选取percent_inside大于0的，之后随机抽样k个形成最终的关键帧列表selected_keyframe_list
    '''
    # Sort the keyframes based on the percentage of points that are inside the image
    list_keyframe = sorted(
        list_keyframe, key=lambda i: i['percent_inside'], reverse=True)#list_keyframe列表中的元素按照'percent_inside'的值从大到小排列。
    # Select the keyframes with percentage of points inside the image > 0
    selected_keyframe_list = [keyframe_dict['id']
                                for keyframe_dict in list_keyframe if keyframe_dict['percent_inside'] > 0.0]
    selected_keyframe_list = list(np.random.permutation(
        np.array(selected_keyframe_list))[:k])
    # 将selected_keyframe_list列表随机排列，然后选取前k个元素，最后将结果转换为列表。
    # 这样selected_keyframe_list列表中的元素就会被随机选取的k个元素替换。这是一种常见的随机抽样方法。
    # 如果k小于selected_keyframe_list的长度，那么结果列表的长度就会是k。
    # 如果k大于selected_keyframe_list的长度，那么结果列表的长度就会是selected_keyframe_list的长度。
    # 如果k等于selected_keyframe_list的长度，那么结果列表就是selected_keyframe_list的一个随机排列。

    return selected_keyframe_list