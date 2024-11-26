import os

from config import ModelConfig


class GatGegnConfig(ModelConfig):
    def __init__(self):
        super(GatGegnConfig, self).__init__()

        self.model_name = "gat_gegn_combine"
        self.model_version = "0"
        self.device = "cuda:0"

        self.model_status = "train"
        self.reuse_model = False
        self.save_training_model = False
        self.save_tensorboard = True

        self.train_cities = ["合肥市", "南京市", "佛山市", "深圳市"]
        # self.train_cities = ["合肥市", "温州市", "南京市", "徐州市", "无锡市", "东莞市", "厦门市", "常州市", "佛山市", "深圳市"]
        self.infer_cities = ["苏州市"] # ["苏州市"]

        self.split_type = "o_id"
        self.split_frac = 0.1

        self.epoch = 120
        self.eval_gap = 5
        self.train_batch_size = 1024
        self.eval_batch_size = 20480
        self.learning_rate = 1e-3
        self.do_augmentation = True

        self.grid_radius = 2
        self.grid_num = (self.grid_radius * 2 - 1) ** 2
        self.grid_layer_channels = [256, 128]
        self.edge_layer_channels = [16]
        self.external_layer_channels = [16]
        self.out_layer_channels = [256, 256, 128, 128, 64, 64, 1]

        self.MODEL_SAVE_PATH = os.path.join(self.SAVE_PATH, "{}_{}.pt".format(self.model_name, self.model_version))
        self.BEST_MODEL_SAVE_PATH = os.path.join(self.SAVE_PATH, "{}_{}_best.pt".format(self.model_name, self.model_version))
        self.TENSORBOARD_PATH = os.path.join(self.CUR_PATH, "runs", "{}_{}".format(self.model_name, self.model_version))