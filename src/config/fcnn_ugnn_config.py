import os

from config import ModelConfig
from util import get_area_ids


class FcnnGegnConfig(ModelConfig):
    def __init__(self):
        super(FcnnGegnConfig, self).__init__()

        self.model_name = "fcnn_gegn_combine"
        self.model_version = "20221229"

        self.device = "cuda:7"

        self.model_status = "train"
        self.reuse_model = False
        self.save_training_model = False
        self.save_tensorboard = False

        self.train_cities = ["北京市", "广州市",
                             "成都市", "南京市", "郑州市", "宁波市", "佛山市", "东莞市", "合肥市", "长沙市", "青岛市", "苏州市",
                             "厦门市", "济南市", "沈阳市", "大连市", "昆明市", "长春市", "温州市", "石家庄市", "贵阳市", "南昌市", "常州市", "哈尔滨市",
                             "乌鲁木齐市", "徐州市"]
        # self.test_cities = ["上海市", "深圳市", "苏州市",
        #                     "成都市", "南京市",
        #                     "无锡市"]
        # self.test_cities = ["北京市", "上海市", "深圳市",
        #                     "杭州市", "苏州市", "武汉市", "天津市", "长沙市", "青岛市",
        #                     "福州市", "哈尔滨市", "南宁市", "兰州市",
        #                     "徐州市", "呼和浩特市",
        #                     "广州市",
        #                     "成都市", "南京市", "郑州市", "宁波市", "佛山市", "东莞市", "合肥市",
        #                     "无锡市", "厦门市", "济南市", "沈阳市", "大连市", "昆明市", "长春市", "温州市", "石家庄市", "贵阳市", "南昌市", "常州市",
        #                     "乌鲁木齐市"]
        self.test_cities = ["南京市"]
        # self.test_cities = ["厦门市"]
        self.infer_cities = ["上海市"]

        # self.test_ids = get_area_ids(*self.area_scope_dict["中关村"])
        self.test_ids = []

        self.w_analysis_cities = ["北京市"]
        self.w_analysis_ids = get_area_ids(*self.area_scope_dict["北京市"]["CBD"])

        self.grid_type = "poi_aoi"
        self.split_type = "o_id"
        self.split_frac = 0.1
        self.sample_gap = 3
        # self.sample_type = "normal" # use 0.1 sampling for value<1
        self.sample_type = "diff"   # use differential sampling strategy
        self.beta = 0.3

        # self.train_frac = 0.2 # 0.2   1
        self.epoch = 30
        self.eval_gap = 1   # 1 5
        self.train_batch_size = 2048
        self.infer_batch_size = 81920
        self.w_analysis_batch_size = 201
        self.learning_rate = 1e-4
        self.do_augmentation = True

        self.grid_radius = 2
        self.grid_num = (self.grid_radius * 2 - 1) ** 2
        self.grid_layer_channels = [1024, 512, 256] # [128, 64, 32]
        self.edge_layer_channels = [16]
        self.external_layer_channels = [16]
        self.out_layer_channels = [512, 512, 256, 256, 128, 128, 1]   # [128, 128, 64, 64, 1]

        self.MODEL_SAVE_PATH = os.path.join(self.SAVE_PATH, "{}_{}.pt".format(self.model_name, self.model_version))
        self.BEST_MODEL_SAVE_PATH = os.path.join(self.SAVE_PATH, "{}_{}_best.pt".format(self.model_name, self.model_version))
        self.TENSORBOARD_PATH = os.path.join(self.CUR_PATH, "runs", "{}_{}".format(self.model_name, self.model_version))
