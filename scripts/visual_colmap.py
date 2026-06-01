import pycolmap
import open3d as o3d
import numpy as np
import argparse  # 用于解析命令行参数

# -------------------------- 解析命令行参数（关键） --------------------------
parser = argparse.ArgumentParser(description="Visualize COLMAP sparse reconstruction results")
parser.add_argument(
    "--scene_dir",
    required=True,
    default="/home/sti/Desktop/",
    help="Path to your SCENE_DIR (contains images/ and sparse/)",
)
args = parser.parse_args()

# 拼接正确的sparse路径（scene_dir + /sparse）
SPARSE_DIR = f"{args.scene_dir}/sparse"

# -------------------------- 加载COLMAP模型（适配新版pycolmap） --------------------------
# 新版pycolmap推荐用Reconstruction类加载，兼容新旧版本
try:
    # 方式1：新版pycolmap（>=0.2.0）推荐用法
    model = pycolmap.Reconstruction(SPARSE_DIR)
except Exception as e:
    # 方式2：兼容旧版pycolmap（备用）
    try:
        from pycolmap.io import read_model

        model = read_model(SPARSE_DIR, ext=".bin")
    except:
        raise RuntimeError(f"Failed to load COLMAP model from {SPARSE_DIR}\nError: {e}")

# -------------------------- 处理3D点云 --------------------------
point_coords = []
point_colors = []

# 适配两种加载方式的模型数据结构
if isinstance(model, pycolmap.Reconstruction):
    # 新版Reconstruction类的遍历方式
    for p3d_id in model.points3D:
        p3d = model.points3D[p3d_id]
        point_coords.append(p3d.xyz)
        point_colors.append(p3d.color / 255.0)
else:
    # 旧版read_model的遍历方式
    for p3d in model["points3D"].values():
        point_coords.append(p3d.xyz)
        point_colors.append(p3d.color / 255.0)

# 创建Open3D点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(point_coords))
pcd.colors = o3d.utility.Vector3dVector(np.array(point_colors))

# （可选）点云采样（避免点云过大卡顿）
# pcd = pcd.farthest_point_down_sample(100000)


# -------------------------- 处理相机位姿 --------------------------
def create_camera_marker(pose, scale=0.5):
    """创建相机3D标记（红色金字塔）"""
    center = pose[:3, 3]
    front = center + pose[:3, 2] * scale
    right = center - pose[:3, 0] * scale
    up = center + pose[:3, 1] * scale
    left = center - pose[:3, 1] * scale

    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    points = [center, front, right, up, left]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([1, 0, 0])
    return line_set


camera_markers = []
# 适配两种模型数据结构的相机遍历
if isinstance(model, pycolmap.Reconstruction):
    # 新版Reconstruction类的相机遍历
    for img_id in model.images:
        img = model.images[img_id]
        cam_pose = img.cam_from_world.matrix()  # 4x4位姿矩阵
        camera_markers.append(create_camera_marker(cam_pose))
else:
    # 旧版read_model的相机遍历
    for img in model["images"].values():
        cam_rot = pycolmap.quaternion_to_rotation_matrix(img.qvec)
        cam_pose = np.hstack([cam_rot, img.tvec.reshape(3, 1)])
        cam_pose = np.vstack([cam_pose, [0, 0, 0, 1]])
        camera_markers.append(create_camera_marker(cam_pose))

# -------------------------- 启动可视化 --------------------------
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="COLMAP Sparse Reconstruction")
vis.add_geometry(pcd)
for marker in camera_markers:
    vis.add_geometry(marker)
vis.run()
vis.destroy_window()
