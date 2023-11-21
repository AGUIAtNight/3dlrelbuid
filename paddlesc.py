import paddle
from paddle.vision.transforms import ToTensor
from paddle.nn import Conv2D


def stereo_matching(left_img, right_img):
    # 转换为PaddlePaddle所需的格式
    left_tensor = ToTensor()(left_img).unsqueeze(0)
    right_tensor = ToTensor()(right_img).unsqueeze(0)
    # 合并左右图像张量
    input_tensor = paddle.concat([left_tensor, right_tensor], axis=1)

    # 构建视差估计模型
    model = Conv2D(3, 64, kernel_size=3, stride=1, padding=1)

    # 模型预测
    with paddle.no_grad():
        disparity_map = model(input_tensor)

    return disparity_map


# 加载左右相机图像
# left_image_path = 'left.jpg'
# right_image_path = 'right.jpg'
# # left_img = Image.open(left_image_path).convert('RGB')
# # right_img = Image.open(right_image_path).convert('RGB')

# # 进行视差图计算
# disparity_map = stereo_matching(left_img, right_img)

# # 可视化结果
# disparity_map = disparity_map.squeeze().numpy()
# plt.imshow(disparity_map, cmap='jet')
# plt.colorbar()
# plt.show()
