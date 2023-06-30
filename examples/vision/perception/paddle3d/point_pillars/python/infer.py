import fastdeploy as fd
import cv2
import os
import numpy as np
import numba
from fastdeploy import ModelFormat


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Path of smoke paddle model.")
    parser.add_argument(
        '--lidar_file', type=str, help='The lidar path.', required=True)
    parser.add_argument(
        "--num_point_dim",
        type=int,
        default=4,
        help="Dimension of a point in the lidar file.")
    parser.add_argument(
        "--point_cloud_range",
        dest='point_cloud_range',
        nargs='+',
        help="Range of point cloud for voxelize operation.",
        type=float,
        default=None)
    parser.add_argument(
        "--voxel_size",
        dest='voxel_size',
        nargs='+',
        help="Size of voxels for voxelize operation.",
        type=float,
        default=None)
    parser.add_argument(
        "--max_points_in_voxel",
        type=int,
        default=100,
        help="Maximum number of points in a voxel.")
    parser.add_argument(
        "--max_voxel_num",
        type=int,
        default=12000,
        help="Maximum number of voxels.")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Type of inference device, support 'cpu' or 'gpu'.")
    return parser.parse_args()


def read_point(file_path, num_point_dim):
    points = np.fromfile(file_path, np.float32).reshape(-1, num_point_dim)
    points = points[:, :4]
    return points


@numba.jit(nopython=True)
def _points_to_voxel(points, voxel_size, point_cloud_range, grid_size, voxels,
                     coords, num_points_per_voxel, grid_idx_to_voxel_idx,
                     max_points_in_voxel, max_voxel_num):
    num_voxels = 0
    num_points = points.shape[0]
    # x, y, z
    coord = np.zeros(shape=(3, ), dtype=np.int32)

    for point_idx in range(num_points):
        outside = False
        for i in range(3):
            coord[i] = np.floor(
                (points[point_idx, i] - point_cloud_range[i]) / voxel_size[i])
            if coord[i] < 0 or coord[i] >= grid_size[i]:
                outside = True
                break
        if outside:
            continue
        voxel_idx = grid_idx_to_voxel_idx[coord[2], coord[1], coord[0]]
        if voxel_idx == -1:
            voxel_idx = num_voxels
            if num_voxels >= max_voxel_num:
                continue
            num_voxels += 1
            grid_idx_to_voxel_idx[coord[2], coord[1], coord[0]] = voxel_idx
            coords[voxel_idx, 0:3] = coord[::-1]
        curr_num_point = num_points_per_voxel[voxel_idx]
        if curr_num_point < max_points_in_voxel:
            voxels[voxel_idx, curr_num_point] = points[point_idx]
            num_points_per_voxel[voxel_idx] = curr_num_point + 1

    return num_voxels

def build_option(args):
    option = fd.RuntimeOption()
    if args.device.lower() == "gpu":
        option.use_gpu(0)
    if args.device.lower() == "cpu":
        option.use_cpu()
    return option

def hardvoxelize(points, point_cloud_range, voxel_size, max_points_in_voxel,
                 max_voxel_num):
    num_points, num_point_dim = points.shape[0:2]
    point_cloud_range = np.array(point_cloud_range)
    voxel_size = np.array(voxel_size)
    voxels = np.zeros((max_voxel_num, max_points_in_voxel, num_point_dim),
                      dtype=points.dtype)
    coords = np.zeros((max_voxel_num, 3), dtype=np.int32)
    num_points_per_voxel = np.zeros((max_voxel_num, ), dtype=np.int32)
    grid_size = np.round((point_cloud_range[3:6] - point_cloud_range[0:3]) /
                         voxel_size).astype('int32')

    grid_size_x, grid_size_y, grid_size_z = grid_size

    grid_idx_to_voxel_idx = np.full((grid_size_z, grid_size_y, grid_size_x),
                                    -1,
                                    dtype=np.int32)

    num_voxels = _points_to_voxel(points, voxel_size, point_cloud_range,
                                  grid_size, voxels, coords,
                                  num_points_per_voxel, grid_idx_to_voxel_idx,
                                  max_points_in_voxel, max_voxel_num)

    voxels = voxels[:num_voxels]
    coords = coords[:num_voxels]
    num_points_per_voxel = num_points_per_voxel[:num_voxels]

    return voxels, coords, num_points_per_voxel


def preprocess(file_path, num_point_dim, point_cloud_range, voxel_size,
               max_points_in_voxel, max_voxel_num):
    points = read_point(file_path, num_point_dim)
    voxels, coords, num_points_per_voxel = hardvoxelize(
        points, point_cloud_range, voxel_size, max_points_in_voxel,
        max_voxel_num)

    return voxels, coords, num_points_per_voxel

args = parse_arguments()

model_file = os.path.join(args.model, "pointpillars.pdmodel")
params_file = os.path.join(args.model, "pointpillars.pdiparams")
# 配置runtime，加载模型
runtime_option = build_option(args)
model = fd.vision.perception.PointPillars(
    model_file, params_file, runtime_option=runtime_option)

# 预测图片检测结果
voxels, coords, num_points_per_voxel = preprocess(
        args.lidar_file, args.num_point_dim, args.point_cloud_range,
        args.voxel_size, args.max_points_in_voxel, args.max_voxel_num)
result = model.predict(voxels, coords, num_points_per_voxel)
print(result)

# 预测结果可视化
vis_im = fd.vision.vis_perception(im, result, config_file)
cv2.imwrite("visualized_result.jpg", vis_im)
print("Visualized result save in ./visualized_result.jpg")
