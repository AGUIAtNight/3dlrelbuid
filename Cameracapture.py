import cv2
# from PIL import Image
import numpy as np

# nuitka  --output-dir=F:\FXXK\CODE\3dlrelbuid\out F:\FXXK\CODE\3dlrelbuid\Cameracapture.py

# nuitka --follow-imports --nofollow-imports numpy,cv2 --output-dir=F:\FXXK\CODE\3dlrelbuid\out F:\FXXK\CODE\3dlrelbuid\Cameracapture.py


def sketch(image):
    # 将彩色图像转换为灰度图像
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 进行边缘检测
    edges = cv2.Canny(image, 30, 100)

    # 进行灰度映射
    ret, threshold = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY_INV)

    return threshold


def MSR(image, scales=[15, 80, 250], sigma=30):
    # 将输入图像转换成浮点数类型
    image = np.float32(image) / 255.0

    # 定义滤波器模板
    width = int(sigma * 3)
    gaussian = cv2.getGaussianKernel(width, sigma)
    gaussian = np.outer(gaussian, gaussian.transpose())

    # 对每个尺度上的图像进行处理
    for scale in scales:
        # 滤波器的大小根据当前尺度计算
        width = int(scale * 3)
        if width % 2 == 0:
            width += 1

        # 高斯模糊
        blurred = cv2.GaussianBlur(image, (width, width), 0)

        # 对数变换
        log_image = np.log10(blurred + 1e-6)

        # 带通滤波
        filtered = cv2.filter2D(log_image, -1, gaussian)

        # 反变换
        restored = np.power(10, filtered) - 1e-6

        # 将当前尺度的图像增强后合并到结果中
        image += restored

    # 将结果限制到[0, 1]范围内，并转换为8位图像
    image = np.clip(image, 0.0, 1.0)
    enhanced = np.uint8(image * 255.0)

    return enhanced


def remove_shadows(image):
    # 将输入图像转换为浮点数类型
    image = np.float32(image) / 255.0

    # 对图像进行多尺度Retinex增强
    enhanced_image = MSR(image)

    # 将增强后的图像转换为LAB颜色空间
    lab_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2LAB)

    # 分离LAB通道
    L, A, B = cv2.split(lab_image)

    # 对亮度通道应用自适应阈值分割，得到阴影掩码
    mask = cv2.adaptiveThreshold(
        L, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10)

    # 对阴影区域进行颜色填充
    result = cv2.inpaint(image, mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)

    # 将结果限制到[0, 255]范围内，并转换为8位图像
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)

    return result


def denoise_image(image):
    # 彩色图像降噪
    denoised_image = cv2.fastNlMeansDenoisingColored(
        image, None, 10, 10, 7, 21)

    return denoised_image


def linear_adjustment(image, alpha, beta):
    # 线性调整函数，alpha为亮度缩放参数，beta为对比度调整参数
    adjusted_image = alpha * image + beta
    # 将像素值限制在 [0, 255] 范围内
    adjusted_image = np.clip(adjusted_image, 0, 255)
    # 转换为 8-bit 无符号整数
    adjusted_image = adjusted_image.astype(np.uint8)
    return adjusted_image


def enhance_contrast1(image):
    # 将图像转换为灰度图像

    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 创建自适应直方图均衡化器
    clahe = cv2.createCLAHE()

    # 应用自适应直方图均衡化增加对比度
    enhanced_image = clahe.apply(image)

    return enhanced_image


def enhance_contrast(image):
    # 将图像转换为灰度图
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 拆分彩色图像通道
    b, g, r = cv2.split(image)

    # 对每个通道进行直方图均衡化
    equalized_b = cv2.equalizeHist(b)
    equalized_g = cv2.equalizeHist(g)
    equalized_r = cv2.equalizeHist(r)

    # 合并通道
    equalized_image = cv2.merge([equalized_b, equalized_g, equalized_r])

    # 降噪
    denoised_disparity_map = denoise_image(equalized_image)

    return denoised_disparity_map


def auto_adjust_brightness_contrast(image, target_brightness, target_contrast):
    # Convert image to float32 for calculations
    image = image.astype(np.float32)

    # Calculate current brightness and contrast
    current_brightness = np.mean(image)
    current_contrast = np.std(image)

    # Calculate adjustment factors
    brightness_adjustment = (target_brightness - current_brightness) / 255.0
    contrast_adjustment = target_contrast / current_contrast

    # Apply brightness and contrast adjustment
    adjusted_image = image + (brightness_adjustment * 255.0)
    adjusted_image = adjusted_image * contrast_adjustment

    # Clip the pixel values to the valid range [0, 255]
    adjusted_image = np.clip(adjusted_image, 0, 255)

    # Convert image back to uint8 format
    adjusted_image = adjusted_image.astype(np.uint8)

    return adjusted_image


try:
    # 摄像头捕获
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    i = 0
    while cap.isOpened():

        # 读取摄像头图像帧
        ret, frame = cap.read()

        # 检测棋盘格并保存
        if ret:
            # cv2.imwrite("{save_path.jpg", frame)
            # detect_chessboard_and_save(frame, i)
            enhanced_disparity_mapl = frame[:, 0:640, :]
            enhanced_disparity_mapr = frame[:, 640:1280, :]
            # 进行多尺度Retinex增强
            enhanced_disparity_mapl = remove_shadows(enhanced_disparity_mapl)
            enhanced_disparity_mapr = remove_shadows(enhanced_disparity_mapr)

            enhanced_disparity_mapl = cv2.cvtColor(
                enhanced_disparity_mapl, cv2.COLOR_BGR2GRAY)
            enhanced_disparity_mapr = cv2.cvtColor(
                enhanced_disparity_mapr, cv2.COLOR_BGR2GRAY)

            # from FastGlobalSmoothFilter算法 import fast_global_smoother_filter

            # imgSmoothl = fast_global_smoother_filter(
            #     left_img, 625.0)
            # imgSmoothr = fast_global_smoother_filter(
            #     right_img, 625.0)
            # 增强对比度
            # enhanced_disparity_mapl = enhance_contrast(left_img)
            # # enhanced_disparity_mapr = enhance_contrast(right_img)
            # enhanced_disparity_mapl = denoise_image(left_img)
            # enhanced_disparity_mapr = denoise_image(right_img)
            import cv2
            # import mask

            # # 对灰度图像进行高斯滤波
            # enhanced_disparity_mapl = cv2.GaussianBlur(left_img, (5, 5), 0)
            # enhanced_disparity_mapr = cv2.GaussianBlur(right_img, (5, 5), 0)
            # enhanced_disparity_mapl = enhance_contrast1(enhanced_disparity_mapl)
            # enhanced_disparity_mapr = enhance_contrast1(enhanced_disparity_mapr)
            # 进行中值滤波
            kernel_size = 3  # 中值滤波器的窗口大小，必须为正奇数
            enhanced_disparity_mapl = cv2.medianBlur(
                enhanced_disparity_mapl, kernel_size)
            enhanced_disparity_mapr = cv2.medianBlur(
                enhanced_disparity_mapr, kernel_size)
            # kernel_size = 5  # 中值滤波器的窗口大小，必须为正奇数

            # # 进行均值滤波
            # enhanced_disparity_mapl = cv2.blur(
            #     enhanced_disparity_mapl, (kernel_size, kernel_size))
            # enhanced_disparity_mapr = cv2.blur(
            #     enhanced_disparity_mapr, (kernel_size, kernel_size))
            # # 进行高斯滤波
            # enhanced_disparity_mapl = cv2.GaussianBlur(
            #     enhanced_disparity_mapl, (kernel_size, kernel_size), 0)
            # enhanced_disparity_mapr = cv2.GaussianBlur(
            #     enhanced_disparity_mapr, (kernel_size, kernel_size), 0)
            # # 进行双边滤波
            # enhanced_disparity_mapl = cv2.bilateralFilter(
            #     enhanced_disparity_mapl, kernel_size, 0, 0)
            # enhanced_disparity_mapr = cv2.bilateralFilter(
            #     enhanced_disparity_mapr, kernel_size, 0, 0)
            # 进行中值滤波
            # 调用sketch函数进行图片素描化处理
            # enhanced_disparity_mapl = sketch(enhanced_disparity_mapl)
            # enhanced_disparity_mapr = sketch(enhanced_disparity_mapr)

            enhanced_disparity_mapl = linear_adjustment(
                enhanced_disparity_mapl, 1.5, 45)
            enhanced_disparity_mapr = linear_adjustment(
                enhanced_disparity_mapr, 1.5, 45)
            # 调整亮度和对比度
            # 进行边缘检测
            # enhanced_disparity_mapl1 = cv2.Canny(
            #     enhanced_disparity_mapl, 60, 100)
            # enhanced_disparity_mapr1 = cv2.Canny(
            #     enhanced_disparity_mapr, 60, 100)
            # # 创建蒙版

            # # _, mask1 = cv2.threshold(
            # #     enhanced_disparity_mapl1, 127, 1, cv2.THRESH_BINARY)
            # enhanced_disparity_mapl1 = cv2.bitwise_not(
            #     enhanced_disparity_mapl1)
            # # # 将0替换为1，1替换为0
            # # mask1 = np.where(mask1 == 0, 1, 0)

            # # 保存图像
            # # cv2.imwrite("mask1.jpg", mask1)

            # # _, mask2 = cv2.threshold(
            # #     enhanced_disparity_mapr1, 127, 1, cv2.THRESH_BINARY)
            # enhanced_disparity_mapr1 = cv2.bitwise_not(
            #     enhanced_disparity_mapr1)
            # # mask2 = np.where(mask2 == 0, 1, 0)

            # # 将蒙版应用于原图像
            # # enhanced_disparity_mapl = mask1*enhanced_disparity_mapl
            # # enhanced_disparity_mapr = mask2*enhanced_disparity_mapr
            # # enhanced_disparity_mapl = cv2.bitwise_and(
            # #     enhanced_disparity_mapl, enhanced_disparity_mapl1)
            # # enhanced_disparity_mapr = cv2.bitwise_and(
            # #     enhanced_disparity_mapr, enhanced_disparity_mapr1)
            # enhanced_disparity_mapl[enhanced_disparity_mapl1 == 0] = 0
            # enhanced_disparity_mapr[enhanced_disparity_mapr1 == 0] = 0

            # # 增强对比度
            # enhanced_disparity_mapl = enhance_contrast1(enhanced_disparity_mapl)
            # enhanced_disparity_mapr = enhance_contrast1(enhanced_disparity_mapr)

            # enhanced_disparity_mapl = auto_adjust_brightness_contrast(
            #     enhanced_disparity_mapl, 50, 1.5)
            # enhanced_disparity_mapr = auto_adjust_brightness_contrast(
            #     enhanced_disparity_mapr, 50, 1.5)

            # import 边缘检测函数
            # enhanced_disparity_mapl = 边缘检测函数.fill_contours(enhanced_disparity_mapl)
            # enhanced_disparity_mapr = 边缘检测函数.fill_contours(enhanced_disparity_mapr)
            # enhanced_disparity_mapl = mask.mosaic(enhanced_disparity_mapl)
            # enhanced_disparity_mapr = mask.mosaic(enhanced_disparity_mapr)

            # 保存图像
            # 水平连接两张图像
            # merged_image = cv2.hconcat(
            #     [enhanced_disparity_mapl, enhanced_disparity_mapr])

            # cv2.imwrite("saved_image.jpg", merged_image)

            # import 图片点云
            # 图片点云.create_point_cloud(enhanced_disparity_mapl,
            #                         enhanced_disparity_mapr)

            import need.Imagecorrection as Imagecorrection
            left_img = Imagecorrection.Correction(enhanced_disparity_mapl,
                                                  enhanced_disparity_mapr)
            # import os
            # # 创建保存目录
            # save_dir = "pcd"
            # os.makedirs(save_dir, exist_ok=True)

        # 显示图像
        # cv2.imshow('frame', frame)

        # 检测键盘输入，如果按下 q 键则退出循环
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # 释放摄像头
    cap.release()

    # 关闭所有图像窗口
    cv2.destroyAllWindows()
except Exception as e:
    print(e)
    import logging
    # 设置日志记录器
    logging.basicConfig(filename='log.txt', level=logging.INFO,
                        format='%(asctime)s - %(message)s')
    logging.error(str(e))
