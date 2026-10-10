import cv2
import os

from common.enums import MULTI_CLASSES
from config import YOLO_IMAGE_TRAIN_DIR, YOLO_LABEL_TRAIN_DIR

# 全局变量
annotations = []  # 存储标注结果 [class_id, x_center, y_center, width, height]
current_class_id = 0  # 当前选择的类别ID
drawing = False  # 是否正在绘制
ix, iy = -1, -1  # 起始坐标


# 鼠标回调函数
def draw_bbox(event, x, y, flags, param):
    global annotations, drawing, ix, iy, current_class_id

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # 计算归一化坐标
        height, width = img.shape[:2]
        x_center = (ix + x) / 2 / width
        y_center = (iy + y) / 2 / height
        box_width = abs(x - ix) / width
        box_height = abs(y - iy) / height
        annotations.append([current_class_id, x_center, y_center, box_width, box_height])
        # 绘制矩形
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow("Image", img)


# 主流程
def get_class_id(name):
    for label, clazz in enumerate(MULTI_CLASSES):
        if name.startswith(clazz):
            return label

def resize_to_fit_screen(img, max_width=1200, max_height=800):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


if __name__ == '__main__':

    os.makedirs(YOLO_LABEL_TRAIN_DIR, exist_ok=True)

    for img_name in os.listdir(YOLO_IMAGE_TRAIN_DIR):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        class_id = get_class_id(img_name)
        img_path = os.path.join(YOLO_IMAGE_TRAIN_DIR, img_name)
        label_path = os.path.join(YOLO_LABEL_TRAIN_DIR, os.path.splitext(img_name)[0] + ".txt")

        # 读取图像
        img = cv2.imread(img_path)
        img = resize_to_fit_screen(img)
        img_copy = img.copy()
        annotations = []  # 清空上一张图的标注

        # 创建窗口并绑定鼠标事件
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Image", draw_bbox)

        while True:
            cv2.imshow("Image", img)
            key = cv2.waitKey(1) & 0xFF

            # 按键功能
            if key == ord("s"):  # 保存标注
                with open(label_path, "w") as f:
                    for ann in annotations:
                        f.write(f"{ann[0]} {ann[1]} {ann[2]} {ann[3]} {ann[4]}\n")
                print(f"标注已保存: {label_path}")
                break
            elif key == ord("n"):  # 跳过当前图像
                break
            elif key == ord("r"):  # 重置当前图像的标注
                annotations = []
                img = img_copy.copy()
                cv2.imshow("Image", img)

        cv2.destroyAllWindows()