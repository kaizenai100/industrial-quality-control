import os

from ultralytics import YOLO

from config import CHECKPOINTS_DIR, MULTI_DIRTY_SPOTS_DIR, MULTI_CONVEX_POWDER_DIR

if __name__ == '__main__':

    yolo_model = YOLO(os.path.join(CHECKPOINTS_DIR, 'best.pt'))  # 加载预训练模型

    images = []
    img_paths = []
    defectives = []
    for fname in os.listdir(MULTI_CONVEX_POWDER_DIR):
        if fname.endswith('.jpg') or fname.endswith('.png'):
            img_path = os.path.join(MULTI_CONVEX_POWDER_DIR, fname)
            img_paths.append(img_path)

    for img_path in img_paths:
        results = yolo_model(img_path)  # 推理
        for result in results:
            result.show()