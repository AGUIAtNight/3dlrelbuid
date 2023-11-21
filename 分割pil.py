import os
from PIL import Image

path_l = 'F:\\FXXK\\CODE\\3dlrelbuid\\adjustlimg\\left\\'
path_r = 'F:\\FXXK\\CODE\\3dlrelbuid\\adjustlimg\\right\\'


def split_image(image_path):
    # 打开原始图像
    image = Image.open(image_path)

    # 获取原始图像的宽度和高度
    width, height = image.size

    # 定义每个小图像的宽度
    crop_width = 1280

    # 计算需要分割的小图像的个数
    num_crops = width // crop_width

    # 分割图像并保存小图像
    for i in range(num_crops):
        # 计算当前小图像的左边界和右边界
        left = i * crop_width
        right = left + crop_width

        # 提取当前小图像
        crop = image.crop((left, 0, right, height))

        # 保存左侧图像
        left_crop_path = path_l + "{i}.png"
        crop.save(left_crop_path)
        print(f"Saved {left_crop_path}")

        # 保存右侧图像
        right_crop_path = path_r+"{i}.png"
        crop.transpose(Image.FLIP_LEFT_RIGHT).save(right_crop_path)
        print(f"Saved {right_crop_path}")


# 获取图片所在的目录路径
directory = "F:\\FXXK\\CODE\\3dlrelbuid\\adjustlimg"

# 遍历目录下的所有文件
for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        # 构建图片文件的完整路径
        image_path = os.path.join(directory, filename)

        # 调用函数进行图像分割和保存
        split_image(image_path)
