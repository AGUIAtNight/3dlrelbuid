from scipy.interpolate import RectBivariateSpline
# import need.CountdownTimer as CountdownTimer
import cv2
import open3d as o3d
import numpy as np
i = 0
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
# 根据相机内参构建PointCloud对象
fx = mtx_l[0, 0]  # 假设相机内参
fy = mtx_r[0, 0]
# cx = left_image.shape[1] / 2
# cy = left_image.shape[0] / 2
# 设置相机参数
focal_length = mtx_l[0, 0]
baseline = -((T[0, 0] * proj_right[0, 3]) / focal_length)


def voxelization(points, voxel_size=1.0):
    # 转换为NumPy数组
    points_np = np.asarray(points.points)

    # 计算体素网格的大小
    min_coords = np.min(points_np, axis=0)
    max_coords = np.max(points_np, axis=0)
    grid_size = np.ceil((max_coords - min_coords) / voxel_size).astype(int)

    # 创建空的体素网格
    voxel_grid = np.zeros(grid_size, dtype=bool)

    # 遍历每个点，将其投射到对应的体素中
    for point in points_np:
        voxel_idx = ((point - min_coords) / voxel_size).astype(int)
        voxel_grid[tuple(voxel_idx)] = True

    return voxel_grid


def create_reflection_mesh(pcd):
    # 估计法线
    pcd.estimate_normals()

    # 使用法线进行表面重建
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=10)

    # 删除无效的三角形
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    # 为每个顶点分配渐变彩色
    colors = []
    for vertex_id in range(len(mesh.vertices)):
        color = np.array([vertex_id / len(mesh.vertices),
                         1 - vertex_id / len(mesh.vertices), 0])
        colors.append(color)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    # 显示反光面片

    o3d.visualization.draw_geometries([mesh])

    return mesh


def disparity_to_color(disparity_map):
    # 将视差图转换为彩色图像
    # 假设视差图是灰度图像，取值范围为 [0, 255]

    # 类型转换
    disparity_map = cv2.convertScaleAbs(disparity_map)

    # 应用伪彩色映射
    colormap = cv2.COLORMAP_JET
    colored_map = cv2.applyColorMap(disparity_map, colormap)

    return colored_map


def depth_to_color(gray_map):
    # 将深度图转换为彩色图像
    # 假设深度图是灰度图像，取值范围为 [min_depth, max_depth]
    # gray_image = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)

    # 转换为灰度图像
    # gray_map = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    # 归一化深度图像
    normalized_map = cv2.normalize(
        gray_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    # 应用伪彩色映射
    colormap = cv2.COLORMAP_HOT
    colored_map = cv2.applyColorMap(normalized_map, colormap)

    return colored_map


# def create_reflection_mesh1(pcd):
#     # 创建反射面片对象
#     reflection_mesh = o3d.geometry.TriangleMesh()

#     # TODO: 根据点云数据创建反射面片
#     # 这里可以根据需要进行具体操作，例如基于点云数据进行表面重建或其他处理

#     return reflection_mesh


# def create_reflection_mesh_from_depth_image(depth, intrinsic):
#     depth = depth.astype(np.float32) / 255.0
#     # 将深度图像转换为Open3D的图像对象
#     image = o3d.geometry.Image(depth)

#     # 使用create_from_depth_image函数将深度图像转换为点云数据
#     pcd = o3d.geometry.PointCloud.create_from_depth_image(image, intrinsic)

#     # 创建渐变彩色的反射面片
#     reflection_mesh = create_reflection_mesh(pcd)

#     return reflection_mesh


def fill_disparity(disp, x_range=None, y_range=None, ngrid=None, method='linear'):
    """
    给定视差图，使用插值方法将其补全为网格。

    参数：
    disp：输入的视差图。可以是Numpy数组或PIL图像对象。数据类型应为float32。
    x_range：生成的网格在x轴上的范围。默认为None，表示使用整个x轴范围。
    y_range：生成的网格在y轴上的范围。默认为None，表示使用整个y轴范围。
    ngrid：生成的网格数量。默认为None，表示使用图片尺寸。
    method：插值方法。默认为线性插值（'linear'），可选 'nearest', 'cubic' 等。

    返回：
    生成的视差网格数据。

    示例：
    grid_disp = fill_disparity(disp_img, x_range=(
        10, 200), y_range=(20, 400), ngrid=None, method='cubic')
    """

    # 将输入视差图转换为Numpy数组（如果不已经是）
    if isinstance(disp, np.ndarray):
        disp_arr = disp.astype('float32')
    else:
        disp_arr = np.array(disp, dtype='float32')

    # 确定x,y轴范围
    if x_range is None:
        x_range = (0, disp_arr.shape[0])
    if y_range is None:
        y_range = (0, disp_arr.shape[1])

    # 确定网格数量
    if ngrid is None:
        ngrid = (disp_arr.shape[0], disp_arr.shape[1])

    # 生成网格坐标
    grid_x, grid_y = np.mgrid[x_range[0]:x_range[1]
        :ngrid[0]*1j, y_range[0]:y_range[1]:ngrid[1]*1j]

    # 创建二维插值对象
    interp_func = RectBivariateSpline(
        range(disp_arr.shape[1]), range(disp_arr.shape[0]), disp_arr.T)

    # 在网格上进行插值
    grid_disp = interp_func.ev(grid_x.ravel(), grid_y.ravel()).reshape(
        grid_x.shape).astype('float32')

    return grid_disp


def create_mesh_from_point_cloud(point_cloud):
    # 创建KD树来加速最近邻搜索
    tree = o3d.geometry.KDTreeFlann(point_cloud)

    # 遍历每个点，寻找最近的邻居点，并创建面
    triangles = []
    for i in range(len(point_cloud.points)):
        # 寻找最近的3个邻居点
        [k, idx, dist] = tree.search_knn_vector_3d(point_cloud.points[i], 3)
        print('点', i, '的三个邻居点之间的距离为：', dist)
        # 创建面
        triangle = o3d.geometry.TriangleMesh()
        triangle.vertices = o3d.utility.Vector3dVector(
            np.array([point_cloud.points[idx[0]],
                     point_cloud.points[idx[1]], point_cloud.points[idx[2]]])
        )
        triangles.append(triangle)

    # 将所有的面合并成一个对象
    mesh = o3d.geometry.TriangleMesh()
    mesh = mesh + triangles[0]
    for i in range(1, len(triangles)):
        mesh += triangles[i]

    # 获取三角面片数量
    face_count = len(mesh.triangles)

    # 打印三角面片数量
    print("生成的面的数量：", face_count)

    return mesh


def remove_outliers(image, window_size=3):
    # 创建一个与原始图像相同大小的数组用于保存结果
    filtered_image = np.zeros_like(image)

    # 扩展原始图像边界
    expanded_image = cv2.copyMakeBorder(
        image, window_size//2, window_size//2, window_size//2, window_size//2, cv2.BORDER_REFLECT)

    # 迭代遍历每个像素点
    for i in range(window_size//2, image.shape[0] + window_size//2):
        for j in range(window_size//2, image.shape[1] + window_size//2):
            # 获取邻域窗口内的像素
            window = expanded_image[i - window_size//2: i + window_size //
                                    2 + 1, j - window_size//2: j + window_size//2 + 1]

            # 计算除中心点外的平均值
            mean_value = np.mean(window) - \
                window[window_size//2, window_size//2]

            # 判断中心像素是否高于平均值
            if image[i - window_size//2, j - window_size//2] > mean_value:
                # 将中心像素设置为零
                filtered_image[i - window_size//2, j - window_size//2] = 0
            else:
                # 否则将中心像素保留
                filtered_image[i - window_size//2, j - window_size //
                               2] = image[i - window_size//2, j - window_size//2]

    return filtered_image


def remove_discrete_points(depth_map):
    # 创建空的离散点列表
    discrete_points = []

    # 遍历深度图像的像素，获取所有离散点的坐标
    for y in range(depth_map.shape[0]):
        for x in range(depth_map.shape[1]):
            depth_value = depth_map[y, x]
            if depth_value > 0:
                discrete_points.append((x, y))

    # 创建与深度图像大小相同的掩码，初始值为1
    mask = np.ones_like(depth_map)

    # 遍历离散点列表，将对应位置的像素值设为0
    for point in discrete_points:
        x, y = point
        mask[y, x] = 0

    # 将掩码应用于深度图像
    depth_map_filtered = depth_map * mask

    # 计算离散点所占总像素点数的比例
    total_pixels = depth_map.size
    deleted_pixels = len(discrete_points)
    deleted_ratio = deleted_pixels / total_pixels

    print(f"Deleted ratio: {deleted_ratio}")

    return depth_map_filtered


def smooth_outliers(image, threshold):
    # 创建一个3x3的平均滤波器
    kernel = np.ones((3, 3), np.float32) / 9

    # 使用滤波器对图像进行平均滤波
    smoothed_image = cv2.filter2D(image, -1, kernel)

    # 计算中心点与邻域平均值之间的差异
    diff = np.abs(image - smoothed_image)

    # 将差异超过阈值的像素设置为零
    result = np.where(diff <= threshold, image, 0)

    return result


def remove_discrete_points1(depth_map, threshold):
    # 创建空的离散点列表
    discrete_points = []

    # 遍历深度图像的像素，获取所有离散点的坐标
    for y in range(1, depth_map.shape[0]-1):
        for x in range(1, depth_map.shape[1]-1):
            depth_value = depth_map[y, x]
            if depth_value > 0:
                # 获取中心点和周围点的深度值
                center_depth = depth_map[y, x]
                neighbor_depths = depth_map[y-1:y+2, x-1:x+2]

                # 检查中心点与周围点之间的深度差异
                if np.max(neighbor_depths) - np.min(neighbor_depths) > threshold:
                    discrete_points.append((y, x))

    # 创建与深度图像大小相同的掩码，初始值为1
    mask = np.ones_like(depth_map)

    # 遍历离散点列表，将对应位置的像素值设为0
    for point in discrete_points:
        y, x = point
        mask[y, x] = 0

    # 将掩码应用于深度图像
    depth_map_filtered = depth_map * mask

    # 计算离散点所占总像素点数的比例
    total_pixels = depth_map.size
    deleted_pixels = len(discrete_points)
    deleted_ratio = deleted_pixels / total_pixels

    print(f"Deleted ratio: {deleted_ratio}")

    return depth_map_filtered


def remove_discrete_pointsno(depth_map, threshold):
    # 获取深度值
    center_depth = depth_map[1:-1, 1:-1]
    neighbor_depths = depth_map[:-2, :-2]

    # 计算深度差异
    depth_diff = np.abs(center_depth - neighbor_depths)
    max_diff = np.max(depth_diff, axis=(0, 1))

    # 创建掩码
    mask = np.ones_like(depth_map)
    mask[1:-1, 1:-1] = (max_diff <= threshold)

    # 应用掩码
    depth_map_filtered = depth_map * mask

    # 计算删除比例
    total_pixels = depth_map.size
    deleted_pixels = np.count_nonzero(mask == 0)
    deleted_ratio = deleted_pixels / total_pixels

    print(f"Deleted ratio: {deleted_ratio}")

    return depth_map_filtered


def remove_outliers_zscore(image, threshold=3):
    # 计算每个像素与其邻域像素的均值和标准差
    win = 5
    mean_image = cv2.blur(image, (win, win))
    std_image = cv2.blur(image**2, (win, win)) - mean_image**2

    # 计算Z-Score
    zscore_image = (image - mean_image) / std_image.clip(min=1e-6)**0.5

    # 根据阈值去除离群点
    filtered_image = np.where(np.abs(zscore_image) < threshold, image, 0)

    return filtered_image


# @ CountdownTimer.measure_execution_time
def create_point_cloud(left_image, right_image):
    # 读取左右相机图像
    # left_image = cv2.imread(left_image_path)
    # right_image = cv2.imread(right_image_path)
    # 将左右图像转换为灰度图像
    # left_image = cv2.cvtColor(left_image1, cv2.COLOR_BGR2GRAY)
    # right_image = cv2.cvtColor(right_image1, cv2.COLOR_BGR2GRAY)

    # 双目匹配算法（例如SGBM）计算视差图像
    # stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    # disp = stereo.compute(left_image, right_image)
    import need.Disparitycalculation as Disparitycalculation
    disp, trueDisp_right = Disparitycalculation.stereoMatchSGBM(
        left_image, right_image)
    # 进行值滤波
    disp = cv2.medianBlur(disp, ksize=5)
    # import paddlesc

    # disp = fill_disparity(disp)
    # disp = paddlesc.stereo_matching(left_image, right_image)

    # kernel_size = 5  # 中值滤波器的窗口大小，必须为正奇数
    # disp = cv2.medianBlur(
    #     disp, kernel_size)

    # # 使用Z-Score方法去除离群点
    # disp = remove_outliers_zscore(disp, threshold=2)

    # 假设你已加载原始图像到image变量中
    # 获取图像尺寸
    height, width = disp.shape[:2]

    # 创建一个与图像大小相同的数组
    indices = np.indices((height, width))

    # 计算x*y的值
    xy_value = indices[0] + indices[1]

    # 检查是否已经有保存的掩码，如果没有则创建新的
    if 'mask' not in globals():
        global mask
        # 创建一个与图像大小相同的数组
        indices = np.indices((height, width))

        # 计算x*y的值
        xy_value = indices[0] + indices[1]

        # 创建掩码
        print('create mask')
        mask = np.zeros_like(disp, dtype=np.float32)
        mask[xy_value % 2 == 1] = 1.0

    # 将满足条件的像素设为0

    # 应用蒙板到图像
    disp = cv2.multiply(disp, mask)
    disp[disp < 0] = 0

    # # 将xy值为奇数的像素设为0
    # disp[xy_value % 2 == 1] = 0

    disp1 = disparity_to_color(disp)

    # 保存视差图像
    cv2.imwrite('disparity.png', disp1)
    # # 计算深度图像
    # depth = baseline * focal_length / disp

    # 假设disp是视差图像中的像素值
    depth = np.where(disp != 0, baseline * focal_length / disp, 0.0)

    # # 去除离散点
    # depth = remove_discrete_points(depth)
    # depth = smooth_outliers(depth, 10)
    # # 使用Z-Score方法去除离群点
    # depth = remove_outliers_zscore(depth, threshold=2)

    # kernel_size = 3  # 中值滤波器的窗口大小，必须为正奇数
    # depth = cv2.medianBlur(
    #     depth, kernel_size)
    # 删除离散点，默认阈值为20
    # depth = remove_outliers(depth)

    # 过滤小于焦距的点
    # depth = np.where(depth < 0, 0, depth)
    # 过滤小于焦距的点云数据
    # depth[depth < focal_length] = 0

    # 过滤小于焦距的点云数据
    # for i in range(depth.shape[0]):
    #     for j in range(depth.shape[1]):
    #         if depth[i, j] < focal_length:
    #             depth[i, j] = 0

    # 将视差图转换为深度图
    # depth = 1.0 / (disp + 0.001)

    # 将深度图像转换为8位无符号整数类型（范围为0-255）
    depth_uint = (depth * 255).astype(np.uint8)

    depth_uint = depth_to_color(depth_uint)

    # 保存深度图像
    cv2.imwrite('depth_image.png', depth_uint)

    # 根据相机内参构建PointCloud对象
    # fx = mtx_l[0, 0]  # 假设相机内参
    # fy = mtx_r[0, 0]
    # cx = left_image.shape[1] / 2
    # cy = left_image.shape[0] / 2

    fx = mtx_l[0, 0]  # 假设相机内参
    fy = mtx_r[0, 0]
    cx = mtx_l[0, 2]
    cy = mtx_l[1, 2]

    intrinsic = o3d.camera.PinholeCameraIntrinsic(width=left_image.shape[1],
                                                  height=left_image.shape[0],
                                                  fx=fx, fy=fy, cx=cx, cy=cy)

    depth = depth.astype(np.uint16)  # 转换为16位无符号整数

    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image(depth), intrinsic)
    # i += 1

    # # 保存点云为PLY文件
    # o3d.io.write_point_cloud("pcd/{}.ply".format(i), pcd)

    # import need.Cuber as Cuber
    # Cuber.Sampledata(pcd)

    # mesh = create_mesh_from_point_cloud(pcd)

    # # 可视化显示结果
    # # 可视化三角面片
    # o3d.visualization.draw_geometries([mesh])

#     # 删除离散点
#     cl, ind = pcd.remove_statistical_outlier(
#         nb_neighbors=20, std_ratio=2.0)

#    # 通过索引获取未被剔除的点
#     inlier_cloud = pcd.select_by_index(ind)

#     # 可以打印剔除的离散点数量
#     outlier_cloud = pcd.select_by_index(ind, invert=True)
#     print("剔除的离散点数量：", len(outlier_cloud.points))
#     # 可视化剔除离散点后的点云
#     o3d.visualization.draw_geometries([inlier_cloud])

    # # 创建渐变彩色的反光面片# 显示反光面片
    # reflection_mesh = create_reflection_mesh(pcd)

    # 显示点云
    # o3d.visualization.draw_geometries([pcd])
