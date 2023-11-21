import os
import cv2

# 图片所在目录路径
image_dir = 'F:\\FXXK\\CODE\\3dlrelbuid\\adjustlimg'
path_l = 'F:\\FXXK\\CODE\\3dlrelbuid\\adjustlimg\\left\\'
path_r = 'F:\\FXXK\\CODE\\3dlrelbuid\\adjustlimg\\right\\'
# 遍历目录下的所有文件


def replace_extension(file_path):
    new_file_path = file_path.replace(".jpg", ".png")
    return new_file_path


for filename in os.listdir(image_dir):
    # 构建完整的文件路径
    filename1 = replace_extension(filename)
    file_path_unicode = os.path.join(image_dir, filename)
    # 转换中文目录路径为 Unicode 编码
    # file_path_unicode = file_path.encode('utf-8').decode('unicode_escape')

    # 判断文件是否为图片文件
    if os.path.isfile(file_path_unicode) and any(file_path_unicode.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
        # 使用OpenCV读取图像文件
        image = cv2.imread(file_path_unicode)
        imagel = image[:, 0:1280]
        imager = image[:, 1280:]

        # 进行相应的操作，例如显示图像名称
        print("图像名称:", filename)
        # 创建窗口并显示图像
        # cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        # cv2.imshow('Image', imager)

        # 等待按下任意按键
        # cv2.waitKey(0)

        # 关闭窗口
        # cv2.destroyAllWindows()
        # cv2.imshow("imager", imager)
        cv2.imwrite(path_l + str(filename1),
                    imagel)  # 保存左图
        cv2.imwrite(path_r + str(filename1),
                    imager)  # 保存右图

        # 关闭图像
        # cv2.destroyAllWindows()
