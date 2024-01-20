import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# 灰度图转换
def grayscale(image):
    return cv.cvtColor(image, cv.COLOR_RGB2GRAY)

# Canny边缘检测
def canny(image, low_threshold, high_threshold):
    return cv.Canny(image, low_threshold, high_threshold)

# 高斯滤波
def gaussian_blur(image, kernel_size):
    return cv.GaussianBlur(image, (kernel_size, kernel_size), 0)

# 生成感兴趣区域即Mask掩模
def region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv.bitwise_and(image, mask)
    return masked_image

# 原图像与车道线图像按照a:b比例融合
def weighted_img(img, initial_img, a=0.8, b=1., c=0.):
    return cv.addWeighted(initial_img, a, img, b, c)

def draw_lines(image, lines, color=[255, 0, 0], thickness=2):
    slope_threshold = 0.5
    right_slope_set = []
    left_slope_set = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]

            if slope > slope_threshold:
                right_slope_set.append(slope)
            elif slope < -slope_threshold:
                left_slope_set.append(slope)

    right_slope = np.median(right_slope_set) if right_slope_set else None
    left_slope = np.median(left_slope_set) if left_slope_set else None

    if right_slope is not None:
        draw_lane_line(image, lines, right_slope, color, thickness)

    if left_slope is not None:
        draw_lane_line(image, lines, left_slope, color, thickness)

def draw_lane_line(image, lines, slope, color, thickness):
    imshape = image.shape
    x_bottom = int((imshape[0] / 2) / slope)
    x_top = int(imshape[1])
    y_bottom = int(imshape[0] / 2)
    y_top = int(slope * x_top + (y_bottom - slope * x_bottom))

    cv.line(image, (x_bottom, y_bottom), (x_top, y_top), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

def process_image(image):
    rho = 1
    theta = np.pi / 180
    hof_threshold = 20
    min_line_len = 30
    max_line_gap = 60
    kernel_size = 5
    canny_low_threshold = 75
    canny_high_threshold = canny_low_threshold * 3
    alpha = 0.8
    beta = 1.
    lambda_ = 0.

    imshape = image.shape
    gray = grayscale(image)
    blur_gray = gaussian_blur(gray, kernel_size)
    edge_image = canny(blur_gray, canny_low_threshold, canny_high_threshold)
    vertices = np.array([[(0, imshape[0]), (9 * imshape[1] / 20, 11 * imshape[0] / 18),
                          (11 * imshape[1] / 20, 11 * imshape[0] / 18), (imshape[1], imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edge_image, vertices)
    lines = hough_lines(masked_edges, rho, theta, hof_threshold, min_line_len, max_line_gap)
    line_image = np.zeros_like(image)
    draw_lines(line_image, lines, thickness=10)
    lines_edges = weighted_img(image, line_image, alpha, beta, lambda_)
    return lines_edges

if __name__ == '__main__':
    image = cv.imread(r'assinment_2\Curve2.jpg')
    line_image = process_image(image)
    cv.imwrite("./assinment_2/1.jpg", line_image)
    cv.imshow('image', line_image)
    cv.waitKey(0)
