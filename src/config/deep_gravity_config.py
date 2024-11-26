import os
from src.config.model_config import ModelConfig


class DeepGravityConfig(ModelConfig):
    def __init__(self):
        super(DeepGravityConfig, self).__init__()

        self.model_name = "deep_gravity"
        self.model_version = "20230125"
        self.device = "cuda:0"
        self.save_suffix = "_7"

        self.model_status = "train"
        self.reuse_model = False
        self.save_training_model = False
        self.save_tensorboard = False
        # self.train_cities = ['北京市']
        self.train_cities = ["北京市"]

        self.test_cities = ["上海市", "深圳市", "杭州市","武汉市", "天津市", "徐州市", "无锡市",
                            "大连市", "福州市", "南宁市", "兰州市","呼和浩特市", "昆明市", "长春市",
                            "温州市", "石家庄市", "贵阳市", "南昌市", "常州市", "哈尔滨市", "乌鲁木齐市"]

        self.split_frac = 0.1

        self.epoch = 20
        self.train_batch_size = 64
        self.learning_rate = 1e-5
        self.do_augmentation = True

        self.hour = 7

        self.grid_radius = 2
        self.grid_num = (self.grid_radius * 2 - 1) ** 2
        self.layer_channels = [256, 256, 256, 256, 256, 256, 128, 128, 128, 128, 128, 128, 128, 128, 128, 1]

        self.MODEL_SAVE_PATH = os.path.join(self.SAVE_PATH, "{}_{}.pt".format(self.model_name, self.model_version))
        self.BEST_MODEL_SAVE_PATH = os.path.join(self.SAVE_PATH, "{}_{}_best.pt".format(self.model_name, self.model_version))
        self.TENSORBOARD_PATH = os.path.join(self.CUR_PATH, "runs", "{}_{}".format(self.model_name, self.model_version))
        self.CITY_CENTROID_PATH = "/media/lai/5d8d5be8-23c6-40b8-9d77-d15711f9ed67/data/supportive_data/centroid/city_centroid"