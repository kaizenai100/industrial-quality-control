import argparse


class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='model run parameters')
        # 指定配置文件名称，默认使用config.yaml
        self.parser.add_argument('--config', type=str, default='config.yaml', help='number of class: binary, multi')
        self.parser.add_argument('--class_type', type=int, help='number of class: binary, multi')
        self.parser.add_argument('--img_path', type=str, help='picture path')
        self.parser.add_argument('--model_name', type=int, help='model_name: quality, mobilenetv2, efficientnetb0')
        self.parser.add_argument('--num_classes', type=int, help='num_classes, automatic assembly no set up')
        self.parser.add_argument('--cnn', type=str, help='cnn type, automatic assembly no set up')
        self.parser.add_argument('--yolo_name', type=str, help='yolo_name')

    def parse_arguments(self):
        return self.parser.parse_args()
