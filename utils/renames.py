import os
from config import MULTI_DIR, MULTI_OTHER_DIR
from common.enums import MULTI_CLASSES

def batch_rename(directory, prefix="file_", digits=4, start_num=1):
    files = sorted(os.listdir(directory))
    total = len(files)
    for idx, filename in enumerate(files, start=start_num):
        old_path = os.path.join(directory, filename)
        if os.path.isfile(old_path):
            ext = os.path.splitext(filename)[1]
            new_name = f"{prefix}{idx:0{digits}d}{ext}"
            os.rename(old_path, os.path.join(directory, new_name))
            print(f"已处理 {idx-start_num+1}/{total} 个文件")


if __name__ == '__main__':
    isMulti = 0
    # 使用示例（4位编号，起始1001）
    # batch_rename(SAMPLE_DEFECTIVE_DIR, prefix="defective", digits=4, start_num=1001)
    batch_rename(MULTI_OTHER_DIR, prefix="other", digits=4, start_num=1001)
    # batch_rename(TEST_DIR, prefix="testa", digits=4, start_num=1001)
    if isMulti == 1 :
        for label, cls in enumerate(MULTI_CLASSES):
            cls_dir = os.path.join(MULTI_DIR, cls)
            if os.path.exists(cls_dir):
                batch_rename(cls_dir, prefix=cls, digits=4, start_num=1001)