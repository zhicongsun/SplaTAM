import os
from os.path import join as p_join

scenes = ["smallaerial"] 

primary_device="cuda:0"
seed = 2
scene_name = scenes[0]


group_name = "MatrixCity"
run_name = f"{scene_name}_{seed}"

map_every = 1 # 多少帧进行一次densification
keyframe_every = 5 # adding keyframe every nth frame
mapping_window_size = 24 #关键帧的数目，超参
tracking_iters = 40 #每一帧的tracking的循环迭代的次数
mapping_iters = 60



config = dict(
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    seed=seed,
    primary_device=primary_device,
    map_every=map_every, # Mapping every nth frame
    keyframe_every=keyframe_every, # Keyframe every nth frame
    mapping_window_size=mapping_window_size, # Mapping window size
    report_global_progress_every=500, # 多久报告跟踪进度,可视化进度条并自动保存参数Report Global Progress every nth frame
    eval_every=5, # Evaluate every nth frame (at end of SLAM)
    scene_radius_depth_ratio=3, # 第一帧最大深度与场景半径的比率 Max First Frame Depth to Scene Radius Ratio (For Pruning/Densification) 
    # 对应论文中的focal length，用于初始化高斯半径r = D_{GT}/focal length，代码为variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio
    mean_sq_dist_method="projective", # 计算均方距离的方法["projective", "knn"] (Type of Mean Squared Distance Calculation for Scale of Gaussians)
    gaussian_distribution="isotropic", # 各向同性还是异性["isotropic", "anisotropic"] (Isotropic -> Spherical Covariance, Anisotropic -> Ellipsoidal Covariance)
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False, # Save Checkpoints
    checkpoint_interval=100, # Checkpoint Interval
    use_wandb=True,
    wandb=dict(
        entity="zeeco",
        project="SplaTAM",
        group=group_name,
        name=run_name,
        save_qual=False,
        eval_save_qual=True,
    ),
    data=dict(
        basedir="./data/MatrixCitySmallAerial", #done
        gradslam_data_cfg="./configs/data/matrixcitysmallaerial.yaml", #not done
        sequence=scene_name,
        desired_image_height=1080,#done
        desired_image_width=1920,#done
        start=0,
        end=-1,
        stride=1,
        num_frames=-1, #默认
    ),
    tracking=dict(
        use_gt_poses=False, # Use GT Poses for Tracking
        forward_prop=True, # Forward Propagate Poses
        num_iters=tracking_iters,
        use_sil_for_loss=True,#是否使用轮廓图的存在性生成mask
        sil_thres=0.99,#判断轮廓图存在性的阈值
        use_l1=True,
        ignore_outlier_depth_loss=False,#生成mask的时候是否忽略异常深度的点
        loss_weights=dict(
            im=0.5,
            depth=1.0,
        ),# loss的权重
        lrs=dict(
            means3D=0.0,
            rgb_colors=0.0,
            unnorm_rotations=0.0,
            logit_opacities=0.0,
            log_scales=0.0,
            cam_unnorm_rots=0.0004,
            cam_trans=0.002,
        ),
    ),
    mapping=dict(
        num_iters=mapping_iters,
        add_new_gaussians=True,
        sil_thres=0.5, # For Addition of new Gaussians
        use_l1=True,
        use_sil_for_loss=False,
        ignore_outlier_depth_loss=False,
        use_gaussian_splatting_densification = False,#自己加的
        loss_weights=dict(
            im=0.5,
            depth=1.0,
        ), # loss的权重
        lrs=dict(
            means3D=0.0001,
            rgb_colors=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.001,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
        ),
        prune_gaussians=True, # Prune Gaussians during Mapping
        pruning_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=0, #开始修剪的迭代次数
            remove_big_after=0, #开始检查高斯分布的尺度是否太大的迭代次数
            stop_after=20, #停止迭代次数
            prune_every=20,#修剪的周期
            densify_every=100,
            grad_thresh=0.0002,
            num_to_split_into=2,
            removal_opacity_threshold=0.005, #高斯不透明度的阈值
            final_removal_opacity_threshold=0.005, #最后一次迭代高斯不透明度的阈值
            reset_opacities_every=3000, # Doesn't consider iter 0 重置所有高斯不透明度的迭代次数
        ),
    ),
    viz=dict(
        render_mode='color', # ['color', 'depth' or 'centers']
        offset_first_viz_cam=True, # Offsets the view camera back by 0.5 units along the view direction (For Final Recon Viz)
        show_sil=False, # Show Silhouette instead of RGB
        visualize_cams=True, # Visualize Camera Frustums and Trajectory
        viz_w=960, viz_h=540,
        viz_near=0.01, viz_far=100.0,
        view_scale=2,
        viz_fps=5, # FPS for Online Recon Viz
        enter_interactive_post_online=True, # Enter Interactive Mode after Online Recon Viz
    ),
)