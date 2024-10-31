import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载图片
image = cv2.imread('/Volumes/Samsung_T5/result_all/hamer_output_001/Hand_to_table_17_2b/90.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用高斯模糊来减少噪声
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 使用Canny边缘检测
edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

# 使用霍夫线变换检测直线
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# 存储满足条件的水平线坐标
horizontal_lines_in_range = []

# 过滤水平线（基于斜率和y值范围）
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # 计算斜率，避免除以零
        if x2 - x1 != 0:
            slope = abs((y2 - y1) / (x2 - x1))
        else:
            slope = float('inf')  # 垂直线斜率设为无穷大

        # 如果斜率接近0，且y值在330到500之间，认为是符合条件的水平线
        if slope < 0.1 and 330 <= y1 <= 500 and 330 <= y2 <= 500:
            horizontal_lines_in_range.append(((x1, y1), (x2, y2)))

# 找到y值最小的水平线
if horizontal_lines_in_range:
    min_y_line = min(horizontal_lines_in_range, key=lambda line: min(line[0][1], line[1][1]))

    # 绘制选定的最小y值的水平线
    cv2.line(image, min_y_line[0], min_y_line[1], (0, 255, 0), 2)
    print("Selected Line with Minimum Y:", min_y_line)
else:
    print("No horizontal lines found in the specified range.")

# 显示结果图像
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Detected Horizontal Line with Minimum Y in Range")
plt.axis("off")
plt.show()
