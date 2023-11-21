# from my_module import Cube
from typing import List
# import numba as nb
import open3d as o3d
import numpy as np


# @nb.jitclass
# class Point:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y


# @nb.jitclass
class Cube:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.size = np.abs(p1 - p2)

    def is_valid(self, max_size):
        return np.all(self.size <= max_size)

    def __eq__(self, other):
        return np.all(self.p1 == other.p1) and np.all(self.p2 == other.p2)


def find_closest_points(points, pt):
    # 计算每个点与目标点之间的欧几里得距离
    dists = np.sqrt(np.sum((points - pt)**2, axis=1))

    # 找到距离最近的两个点的索引
    closest_idxs = np.argsort(dists)[:2]

    return closest_idxs


# @nb.njit("List[Cube](np.ndarray, float)")
# @nb.jit(nopython=True)
def create_cube(points: np.ndarray, max_size: float) -> List[Cube]:
    cubes = np.empty((len(points), len(points)), dtype=Cube)  # 创建空的NumPy数组
    cube_count = 0  # 立方体计数器

    for i in range(len(points)-1):
        for j in range(i+1, len(points)):
            p1 = points[i]
            p2 = points[j]
            cube = Cube(p1, p2)

            if cube.is_valid(max_size) and cube not in cubes:
                cubes[cube_count] = cube
                cube_count += 1

    return cubes[:cube_count].tolist()  # 返回有效的立方体列表


# @nb.njit
def Sampledata(pcd: o3d.geometry.PointCloud):  # （Open3D点云对象）
    # pcd = o3d.io.read_point_cloud("point_cloud.ply")
    points = np.asarray(pcd.points)
    max_size = 5

    cubes = create_cube(points, max_size)
    # 创建可视化
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 将正方体添加到可视化中
    for cube in cubes:
        box = o3d.geometry.OrientedBoundingBox(cube.p1, cube.p2)
        vis.add_geometry(box)

    # 设置相机视角
    vis.get_view_control().set_front([0, 0, -1])
    vis.run()
    vis.destroy_window()
