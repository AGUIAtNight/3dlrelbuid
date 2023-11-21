import numpy as np
from sklearn.linear_model import RANSACRegressor


def fit_and_remove_outliers(data1, data2):
    # 组合两个点云数据
    combined_data = np.concatenate((data1, data2), axis=0)

    # 拟合模型
    model = RANSACRegressor()
    model.fit(combined_data[:, :3], combined_data[:, 3:])

    # 预测所有数据的拟合结果
    all_predictions = model.predict(combined_data[:, :3])

    # 计算每个数据点与拟合结果之间的残差
    residuals = np.abs(combined_data[:, 3:] - all_predictions)

    # 选择阈值，将残差大于阈值的点标记为异常值
    threshold = 0.1  # 根据实际情况调整阈值
    outliers = np.where(residuals > threshold)[0]

    # 删除异常值
    cleaned_data = np.delete(combined_data, outliers, axis=0)

    # 分离两个点云数据
    cleaned_data_1 = cleaned_data[:len(data1)]
    cleaned_data_2 = cleaned_data[len(data1):]

    return cleaned_data_1, cleaned_data_2
