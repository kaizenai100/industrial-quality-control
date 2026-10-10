"""
替换字符，用于清晰yolo数据集
"""
import os

from common.enums import MULTI_CLASSES
from config import YOLO_LABEL_TRAIN_DIR


def replace_first_char_based_on_filename(directory):
    """
    遍历指定目录下的所有txt文件，根据文件名替换每行第一个字符
    :param directory: 目标目录路径
    """
    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)

            # 根据文件名生成替换字符（自定义规则）
            replace_char = generate_replacement_char(filename)  # 示例函数，需根据需求实现

            # 读取并处理文件内容
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 逐行替换首字符
            modified_lines = []
            for line in lines:
                if len(line.strip()) > 0:  # 跳过空行
                    new_line = replace_char + line[1:]  # 替换首字符
                    modified_lines.append(new_line)
                else:
                    modified_lines.append(line)  # 保留空行

            # 写回文件
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(modified_lines)


def generate_replacement_char(filename):
    """
    自定义规则：根据文件名生成替换字符
    :param filename: 文件名（如 "data_123.txt"）
    :return: 替换后的首字符（示例规则：取文件名中的第一个数字）
    """
    # 示例规则1：取文件名中第一个数字作为替换字符
    for index, clazz in enumerate(MULTI_CLASSES):
        if filename.startswith(clazz):
            return str(index)
    return '0'  # 默认值（若无匹配）


# 调用示例
if __name__ == '__main__':
    replace_first_char_based_on_filename(YOLO_LABEL_TRAIN_DIR)