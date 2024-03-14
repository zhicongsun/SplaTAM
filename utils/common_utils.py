import os

import numpy as np
import random
import torch


def seed_everything(seed=42):
    """
    设置随机种子以确保实验的可重复性。
    这个函数的目的是为了确保在使用随机数生成器时，可以获得可重复的结果。这在科学实验和机器学习模型训练中非常重要，因为我们通常希望能够复现实验结果。设置随机种子可以确保每次运行代码时，随机数生成器产生的随机数序列是一样的。这样，即使我们的代码中有随机过程，我们也能得到相同的结果。这对于调试和改进模型非常有帮助。当我们改变模型或参数时，我们可以确信观察到的任何性能变化都是由我们的改变引起的，而不是随机性的结果。这也使得其他人可以复现我们的结果，这对于科学公开和透明非常重要。
    参数:
    - seed: 一个可哈希的种子值
    """
    random.seed(seed)  # 设置Python内置random模块的随机种子
    os.environ["PYTHONHASHSEED"] = str(seed)  # 设置环境变量PYTHONHASHSEED，以影响Python哈希算法的随机性
    np.random.seed(seed)  # 设置numpy的随机种子
    torch.manual_seed(seed)  # 设置torch（PyTorch）的随机种子
    torch.backends.cudnn.deterministic = True  # 确保每次运行程序时，对于相同的输入，cudnn的输出是确定的
    torch.backends.cudnn.benchmark = False  # 禁用cudnn的自动优化功能，这个功能可能会使得计算结果出现微小的差异
    print(f"Seed set to: {seed} (type: {type(seed)})")  # 打印出设置的种子值和种子值的类型



def params2cpu(params):
    res = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            res[k] = v.detach().cpu().contiguous().numpy()
        else:
            res[k] = v
    return res


def save_params(output_params, output_dir):
    # Convert to CPU Numpy Arrays
    to_save = params2cpu(output_params)
    # Save the Parameters containing the Gaussian Trajectories
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params.npz")
    np.savez(save_path, **to_save)


def save_params_ckpt(output_params, output_dir, time_idx):
    # Convert to CPU Numpy Arrays
    to_save = params2cpu(output_params)
    # Save the Parameters containing the Gaussian Trajectories
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params"+str(time_idx)+".npz")
    np.savez(save_path, **to_save)


def save_seq_params(all_params, output_dir):
    params_to_save = {}
    for frame_idx, params in enumerate(all_params):
        params_to_save[f"frame_{frame_idx}"] = params2cpu(params)
    # Save the Parameters containing the Sequence of Gaussians
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params.npz")
    np.savez(save_path, **params_to_save)


def save_seq_params_ckpt(all_params, output_dir,time_idx):
    params_to_save = {}
    for frame_idx, params in enumerate(all_params):
        params_to_save[f"frame_{frame_idx}"] = params2cpu(params)
    # Save the Parameters containing the Sequence of Gaussians
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params"+str(time_idx)+".npz")
    np.savez(save_path, **params_to_save)