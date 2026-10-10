import cv2
import numpy as np
from config import YOLO_IMAGE_TRAIN_DIR, YOLO_LABEL_TRAIN_DIR
import os

# 配置参数
CLASS_ID = 0  # 类别ID（需根据实际情况修改）
# BOX_COLOR = (0, 0, 255)  # 框的颜色（BGR格式，这里是红色）
BOX_COLOR = (90, 90, 220)  # 框的颜色（BGR格式，这里是红色）
COLOR_THRESHOLD = 90  # 颜色检测容差

# 创建labels目录
os.makedirs(YOLO_LABEL_TRAIN_DIR, exist_ok=True)


def extract_boxes(image_path):
    """从图片中提取红色框的坐标"""
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # 颜色掩膜（检测红色框）
    lower_red = np.array([0, 0, 255 - COLOR_THRESHOLD])
    upper_red = np.array([COLOR_THRESHOLD, COLOR_THRESHOLD, 255])
    mask = cv2.inRange(img, lower_red, upper_red)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, box_w, box_h = cv2.boundingRect(cnt)
        boxes.append([x, y, box_w, box_h])
    return boxes, (w, h)


def save_yolo_label(image_path, boxes, img_size):
    """保存为YOLO格式的.txt文件"""
    txt_path = os.path.join(YOLO_LABEL_TRAIN_DIR, os.path.splitext(os.path.basename(image_path))[0] + ".txt")
    with open(txt_path, "w") as f:
        for (x, y, box_w, box_h) in boxes:
            # 转换为YOLO格式（归一化）
            x_center = (x + box_w / 2) / img_size[0]
            y_center = (y + box_h / 2) / img_size[1]
            width = box_w / img_size[0]
            height = box_h / img_size[1]
            f.write(f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


# 处理所有图片
if __name__ == '__main__':
    for img_name in os.listdir(YOLO_IMAGE_TRAIN_DIR):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        img_path = os.path.join(YOLO_IMAGE_TRAIN_DIR, img_name)
        boxes, img_size = extract_boxes(img_path)
        save_yolo_label(img_path, boxes, img_size)
        print(f"Processed: {img_name} -> {len(boxes)} boxes")

