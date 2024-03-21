import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

def setup_camera(w, h, k, w2c, near=0.01, far=100):
    # 从相机内参矩阵 k 中提取焦距 (fx, fy) 和光心 (cx, cy)
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]

    # 将世界坐标系到相机坐标系的变换矩阵 w2c 转换为张量，并移动到 GPU 上
    w2c = torch.tensor(w2c).cuda().float()

    # 计算相机中心的坐标，即 w2c 的逆矩阵的最后一列的前三个元素
    cam_center = torch.inverse(w2c)[:3, 3]

    # 增加 w2c 的维度，并转置第二和第三维度
    w2c = w2c.unsqueeze(0).transpose(1, 2)

    # 创建 OpenGL 投影矩阵，并移动到 GPU 上
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)

    # 计算完整的投影矩阵，即 w2c 和 opengl_proj 的批量矩阵乘法
    full_proj = w2c.bmm(opengl_proj)

    # 创建 Camera 对象，并设置相应的参数
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False
    )

    # 返回创建的 Camera 对象
    return cam
