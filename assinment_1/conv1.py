import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_npy(image):
    for i in range(0,image.shape[0]):
        plt.imshow(image[i,:,:])
        cv2.imwrite(str(i)+".png",image[i,:,:])
        plt.show()

def apply_sobel(image):
    # Sobel算子
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # 对图像应用Sobel算子
    sobel_x_img = cv2.filter2D(image, -1, sobel_x)
    sobel_y_img = cv2.filter2D(image, -1, sobel_y)

    # 将Sobel滤波器的结果转换为浮点类型
    sobel_x_img = sobel_x_img.astype(np.float64)
    sobel_y_img = sobel_y_img.astype(np.float64)

    # 计算总的梯度近似值
    sobel_combined = cv2.magnitude(sobel_x_img, sobel_y_img)

    return sobel_combined

# 读取图像
image = cv2.imread(r"assinment_1\sobel.png")

sobel_result = apply_sobel(image)

# 将结果映射到0-255范围
sobel_result = np.abs(sobel_result)
sobel_result = np.clip(sobel_result, 0, 255).astype(np.uint8)

#***********************************************************************************************************************************************************
# 对图像进行给定卷积核滤波
custom_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

filtered_result = np.zeros_like(image, dtype=np.float32)

for i in range(1, image.shape[0]-1):
    for j in range(1, image.shape[1]-1):
        filtered_result[i, j] = np.sum(image[i-1:i+2, j-1:j+2] * custom_kernel)

filtered_result = np.abs(filtered_result)
filtered_result = np.clip(filtered_result, 0, 255).astype(np.uint8)

# 可视化颜色直方图
color_histogram = np.zeros((256, 3), dtype=int)

for i in range(3):
    color_channel = image[:, :, i]
    hist, _ = np.histogram(color_channel, bins=256, range=[0, 256])
    color_histogram[:, i] = hist

plt.plot(color_histogram)
plt.title('Color Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()

# 保存纹理特征至npy格式
texture_feature = np.array([sobel_result])
texture_feature1=np.array([filtered_result])

load_npy(texture_feature1)
load_npy(texture_feature)


np.save('./assinment_1/texture_feature.npy', texture_feature)


