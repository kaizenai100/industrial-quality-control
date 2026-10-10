import yaml

class ConfigLoader():
    def __init__(self, config_path):
        self.config_path = config_path

    def load_config(self):
        with open(self.config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config