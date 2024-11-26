import os

from src.config.model_config import ModelConfig


class FcnnConfig(ModelConfig):
    def __init__(self):
        super(FcnnConfig, self).__init__()

        self.model_name = "fcnn_1000"
        self.model_version = "2024es"
        self.device = "cuda:1"
        self.save_suffix = ""
        self.model_status = "test"
        self.reuse_model = True
        self.grid1000 = True     
        
        if self.grid1000:
            self.ALL_EDGE_PATH = "/media/lai/27dbdec5-6518-47f1-b1af-159ffe3d1c08/2019/edge_2019_by_grid1000"
            print("Using grid1000", self.ALL_EDGE_PATH)
        
        self.MODEL_SAVE_PATH = os.path.join(self.SAVE_PATH, "{}_{}.pt".format(self.model_name, self.model_version))
        print(self.MODEL_SAVE_PATH)
        self.BEST_MODEL_SAVE_PATH = os.path.join(self.SAVE_PATH,
                                                 "{}_{}_best.pt".format(self.model_name, self.model_version))
        if self.reuse_model:
            print("Using model in ", self.BEST_MODEL_SAVE_PATH)
        else:
            print("Making new model ", self.model_name, self.model_version)


        self.save_training_model = False
        self.save_tensorboard = False

        self.train_cities = ["北京市", "广州市",
                             "成都市", "南京市", "郑州市", "宁波市", "佛山市", "东莞市", "合肥市",
                             "无锡市", "厦门市", "济南市", "沈阳市", "大连市", "昆明市", "长春市", "温州市", "石家庄市", "贵阳市", "南昌市", "常州市",
                             "乌鲁木齐市"]
        # self.test_cities = ["北京市", "上海市", "深圳市",
        #                     "杭州市", "苏州市", "武汉市", "天津市", "长沙市", "青岛市",
        #                     "福州市", "哈尔滨市", "南宁市", "兰州市",
        #                     "徐州市", "呼和浩特市",
        #                     "广州市",
        #                     "成都市", "南京市", "郑州市", "宁波市", "佛山市", "东莞市", "合肥市",
        #                     "无锡市", "厦门市", "济南市", "沈阳市", "大连市", "昆明市", "长春市", "温州市", "石家庄市", "贵阳市", "南昌市", "常州市",
        #                     "乌鲁木齐市"]
        # self.train_cities = ['宁波市']
        self.test_cities = ["上海市",
                            "武汉市",
                            "兰州市",
                            "呼和浩特市",
                            "深圳市",
                            "杭州市",
                            "天津市",
                            "无锡市",
                            "福州市",
                            "南宁市"]
        # self.test_cities = ["深圳市",
        #                     "杭州市",
        #                     "天津市",
        #                     "大连市",
        #                     "福州市",
        #                     "南宁市"]
        # self.test_cities = ["无锡市"]

        # self.test_cities = ["杭州市"]
        self.check_paras = False
        self.test_ids = []
        self.grid_type = "poi_aoi"
        self.split_type = "o_id"
        self.split_frac = 0.1
        self.sample_gap = 3
        # self.sample_type = "normal" # use 0.1 sampling for value<1
        self.sample_type = "diff"   # use differential sampling strategy
        self.beta = 0.3
        self.split_type = "o_id"
        self.split_frac = 0.1

        self.sample_gap = 3
        self.epoch = 30
        self.eval_gap = 1
        self.train_batch_size = 2048
        self.eval_batch_size = 200000
        self.infer_batch_size = 200000
        self.learning_rate = 1e-4
        self.do_augmentation = True

        self.grid_radius = 2
        self.grid_num = (self.grid_radius * 2 - 1) ** 2
        self.layer_channels = [1024, 1024, 512, 512, 128, 128, 128, 64, 64, 64, 32, 1]


        self.TENSORBOARD_PATH = os.path.join(self.CUR_PATH, "runs", "{}_{}".format(self.model_name, self.model_version))