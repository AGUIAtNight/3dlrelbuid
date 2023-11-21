
import cv2


def mosaic(image, block_size=3):
    # 加载图像
    # image = Image.open(image_path)

    # 缩小图像
    small_image = cv2.resize(
        image, (image.shape[1]//block_size, image.shape[0]//block_size))

    # 放大图像
    mosaic_image = cv2.resize(small_image, (image.shape[1], image.shape[0]))

    # 保存图像
    # mosaic_image.save('path_to_mosaic_image.jpg')
    return mosaic_image
