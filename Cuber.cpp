#include <iostream>
#include <vector>
#include <cmath>

struct Point {
    double x;
    double y;
};

class Cube {
public:
    Point p1;
    Point p2;
    Point size;

    Cube(const Point& p1, const Point& p2) : p1(p1), p2(p2) {
        size.x = std::abs(p1.x - p2.x);
        size.y = std::abs(p1.y - p2.y);
    }

    bool isValid(double max_size) {
        return (size.x <= max_size) && (size.y <= max_size);
    }

    bool operator==(const Cube& other) {
        return (p1.x == other.p1.x) && (p1.y == other.p1.y) && (p2.x == other.p2.x) && (p2.y == other.p2.y);
    }
};

std::vector<Cube> createCube(const std::vector<Point>& points, double max_size) {
    std::vector<Cube> cubes;
    int cube_count = 0;

    for (int i = 0; i < points.size() - 1; i++) {
        for (int j = i + 1; j < points.size(); j++) {
            const Point& p1 = points[i];
            const Point& p2 = points[j];
            Cube cube(p1, p2);

            if (cube.isValid(max_size)) {
                bool exists = false;
                for (const auto& existing_cube : cubes) {
                    if (cube == existing_cube) {
                        exists = true;
                        break;
                    }
                }
                if (!exists) {
                    cubes.push_back(cube);
                    cube_count++;
                }
            }
        }
    }

    return std::vector<Cube>(cubes.begin(), cubes.begin() + cube_count);
}
// #include <iostream>
// #include <Open3D/Visualization/Visualizer.h>
// #include <Open3D/Geometry/OrientedBoundingBox.h>
// void sampleData(const std::vector<Cube>& cubes) {
//     // 创建可视化对象
//     o3d::visualization::Visualizer vis;
//     vis.CreateVisualizerWindow("Open3D Viewer", 1920, 1080);

//     // 将正方体添加到可视化中
//     for (const auto& cube : cubes) {
//         const o3d::geometry::OrientedBoundingBox box = cube.GetBoundingBox();
//         vis.AddGeometry(box);
//     }

//     // 设置相机视角
//     vis.GetViewControl().SetFront({0, 0, -1});

//     // 运行可视化
//     vis.Run();

//     // 销毁窗口
//     vis.DestroyVisualizerWindow();
// }


#include <iostream>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>

typedef pcl::PointXYZ PointT;

void sampleData(const std::vector<Cube>& cubes) {
    // 创建可视化对象
    pcl::visualization::PCLVisualizer visualizer("PCL Viewer");

    // 设置背景颜色和坐标轴
    visualizer.setBackgroundColor(0, 0, 0);
    visualizer.addCoordinateSystem(1.0);

    // 将正方体添加到可视化中
    for (const auto& cube : cubes) {
        const Eigen::Vector3f center = cube.GetCenter();
        const float length = cube.GetLength();

        // 构建正方体的八个顶点
        pcl::PointCloud<PointT>::Ptr cubeCloud(new pcl::PointCloud<PointT>());
        cubeCloud->points.resize(8);
        cubeCloud->points[0].getVector3fMap() = center + Eigen::Vector3f(-length / 2, -length / 2, -length / 2);
        cubeCloud->points[1].getVector3fMap() = center + Eigen::Vector3f(length / 2, -length / 2, -length / 2);
        // ... 添加剩余的点

        // 创建可视化对象并添加到可视化中
        pcl::visualization::PointCloudColorHandlerCustom<PointT> color_handler(cubeCloud, 255, 0, 0);  // 红色
        visualizer.addPointCloud(cubeCloud, color_handler);
    }

    // 设置相机视角
    visualizer.setCameraPosition(0, 0, -5, 0, -1, 0, 0);

    // 运行可视化
    while (!visualizer.wasStopped()) {
        visualizer.spinOnce(100);
    }
}



int main() {
    // 从文件或其他方式获取点数据并存储在std::vector<Point>中
    std::vector<Point> points;

    // 调用sampleData函数进行处理和可视化
    sampleData(points);

    return 0;
}
