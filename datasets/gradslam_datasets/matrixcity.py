import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from natsort import natsorted

from .basedataset import GradSLAMDataset

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"#使能cv2打开exr的功能

import json

class MatrixCityDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 1080,
        desired_width: Optional[int] = 1920,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "transforms_10.json") #外参
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/block1_10/*.png")) #rgb
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/block1_10/*.exr")) #d
        embedding_paths = None
        if self.load_embeddings:#embedding没用到
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []
        with open(self.pose_path, "r") as f:
            tj = json.load(f)    
        for frame in tj['frames']:
            c2w = np.array(frame['rot_mat']).reshape(4, 4)
            c2w[:3,:3] *= 100 #注意这个处理
            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)
    
