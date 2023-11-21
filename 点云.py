# from plyfile import PlyData, PlyElement
import numpy as np

mtx_l = np.array([[990.9992544,   0., 255.77496849],
                  [0., 982.03121621, 257.33982351],
                  [0.,   0.,   1.]])
mtx_r = np.array([[996.86283867,   0., 372.66824011],
                  [0., 994.45882779, 203.59233342],
                  [0.,   0.,   1.]])
dist_l = np.array([[-6.61759052e-01,  1.46735038e+00,  3.06624602e-03,  1.11889872e-03,
                    -3.19970246e+00]])
dist_r = np.array(
    [[-0.42459846,  0.43149149, -0.00584208, -0.00560596, -0.62122897]])
R = np.array([[0.99686024, -0.00255278, -0.07913997],
              [0.0065506,  0.9987128,  0.05029739],
              [0.07890971, -0.05065789,  0.99559381]])
T = np.array([[-56.8560311],
              [0.76523124],
              [1.07410098]])
rect_left = np.array([[0.99501371, -0.01503351, -0.09859876],
                      [0.01750031,  0.99955391,  0.02420154],
                      [0.09819094, -0.02580637,  0.99483294]])
rect_right = np.array([[0.99973109, -0.01345548, -0.01888651],
                       [0.01297841,  0.9995992, -0.02515944],
                       [0.01921748,  0.02490756,  0.99950503]])
proj_left = np.array([[988.245022,   0., 489.65180206,   0.],
                      [0., 988.245022, 122.49853134,   0.],
                      [0.,   0.,   1.,   0.]])
proj_right = np.array([[9.88245022e+02,  0.00000000e+00,  4.89651802e+02, -5.62028033e+04],  [0.00000000e+00,  9.88245022e+02,  1.22498531e+02,  0.00000000e+00],
                       [0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  0.00000000e+00]])
dispartity = np.array([[1.00000000e+00,  0.00000000e+00,  0.00000000e+00, -4.89651802e+02],  [0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -1.22498531e+02],
                       [0.00000000e+00,  0.00000000e+00,
                           0.00000000e+00,  9.88245022e+02],
                       [0.00000000e+00,  0.00000000e+00,  1.75835539e-02, -0.00000000e+00]])
ROI_left = np.array((117, 0, 1163, 590))
ROI_right = np.array((79, 0, 1201, 606))


def disparity_to_point_cloud1(disparity_map, baseline, focal_length):
    # 获取图像尺寸
    height, width = disparity_map.shape[:2]

    # 创建空的点云数组
    point_cloud = np.zeros((height, width, 3), dtype=np.float32)

    # 计算每个像素的三维坐标
    for y in range(height):
        for x in range(width):
            # 获取当前像素的视差值
            disparity = disparity_map[y, x]

            # 计算深度值
            depth = baseline * focal_length / disparity

            # 根据相机模型，计算三维坐标
            point_cloud[y, x, 0] = (x - width / 2) * \
                depth / focal_length  # X 坐标
            point_cloud[y, x, 1] = (y - height / 2) * \
                depth / focal_length  # Y 坐标
            point_cloud[y, x, 2] = depth  # Z 坐标
    save(point_cloud)
    return point_cloud


# 读取视差图
# disparity_map = cv2.imread(
#     "disparity_map.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)

# 设置相机参数
focal_length = mtx_l[0, 0]
baseline = -((T[0, 0] * proj_right[0, 3]) / focal_length)

# 将视差图转换为点云
# point_cloud = disparity_to_point_cloud(disparity_map, baseline, focal_length)


def save(point_cloud):

    # 假设你有点云数据，存储在一个Numpy数组中，每个点包含XYZ坐标
    # point_cloud = np.array([[x1, y1, z1],
    #                     [x2, y2, z2],
    #                     ...
    #                     [xn, yn, zn]])

    # 创建PLY元素
    # plydata = PlyData([PlyElement.describe(point_cloud, 'vertex')])

    # # 将PLY数据写入文件
    # plydata.write('point_cloud.ply')

    # 假设你有点云数据，存储在一个Numpy数组中，每个点包含XYZ坐标
    # point_cloud = np.array([[x1, y1, z1],
    #                     [x2, y2, z2],
    #                     ...
    #                     [xn, yn, zn]])

    # 创建OBJ文件的内容
    lines = []
    for point in point_cloud:
        lines.append(f"v {point[0]} {point[1]} {point[2]}")

    # 将OBJ数据写入文件
    with open('point_cloud.obj', 'w') as file:
        file.write('\n'.join(lines))


def disparity_to_point_cloud(left_disparity_map, right_disparity_map):
    focal_length = mtx_l[0, 0]
    baseline = -((T[0, 0] * proj_right[0, 3]) / focal_length)

    # 获取图像尺寸
    height, width = left_disparity_map.shape[:2]

    # 创建空的点云数组
    point_cloud = np.zeros((height, width, 3), dtype=np.float32)

    # 计算每个像素的三维坐标
    for y in range(height):
        for x in range(width):
            # 获取左右眼图像上的视差值
            left_disparity = left_disparity_map[y, x]
            right_disparity = right_disparity_map[y, x]

            # 根据视差值计算深度值
            depth = baseline * focal_length / \
                (left_disparity - right_disparity)

            # 根据相机模型，计算三维坐标
            point_cloud[y, x, 0] = (x - width / 2) * \
                depth / focal_length  # X 坐标
            point_cloud[y, x, 1] = (y - height / 2) * \
                depth / focal_length  # Y 坐标
            point_cloud[y, x, 2] = depth  # Z 坐标

    # 保存点云数据
    # np.save('point_cloud.obj', point_cloud)
    # save(point_cloud)

    import open3d as o3d
    # 将点云数据重塑为 (480*680, 3) 的形状
    point_cloud_flat = point_cloud.reshape(-1, 3)
    point_cloud1 = o3d.geometry.PointCloud()
    point_cloud1.points = o3d.utility.Vector3dVector(point_cloud_flat)
    # 可视化点云
    o3d.visualization.draw_geometries([point_cloud1])


# 读取点云数据
# point_cloud = o3d.io.read_point_cloud('pointcloud.obj')

# # 可视化点云
    # o3d.visualization.draw_geometries([point_cloud])

    return point_cloud

# 示例用法
# left_disparity_map = ...  # 左眼视差图
# right_disparity_map = ...  # 右眼视差图
# baseline = ...  # 基线长度
# focal_length = ...  # 焦距

# point_cloud = disparity_to_point_cloud(left_disparity_map, right_disparity_map, baseline, focal_length)
