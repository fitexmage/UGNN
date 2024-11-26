import os
from src.config.model_config import ModelConfig


class CnnGegnConfig(ModelConfig):
    def __init__(self):
        super(CnnGegnConfig, self).__init__()

        # Get the name of the currently running script's file
        script_filename = os.path.basename(__file__)
        print("Using "+script_filename)
        
        self.grid1000 = False
        self.small_debugging = True
        self.hour = [7]
        self.note = "test"  # Anything you would like to put into the log
        
        
        if self.grid1000:
            self.ALL_EDGE_PATH = "/media/lai/27dbdec5-6518-47f1-b1af-159ffe3d1c08/2019/edge_2019_by_grid1000"
            print("Using grid1000", self.ALL_EDGE_PATH)
        
        # Uncomment below to enable small sample debugging
        if self.small_debugging:
            print("Small sample debugging")
            self.GRID_POI_AOI_PATH = "/home/lai/ugnn/sample_data/GRID_POI_AOI_PATH"
            self.GRID_BASIC_PATH = "/home/lai/ugnn/sample_data/GRID_BASIC_PATH"
            self.CITY_CENTROID_PATH = "/home/lai/ugnn/sample_data/CITY_CENTROID_PATH"
            self.TELECOM_OUTPUT_ID_PATH = "/home/lai/ugnn/sample_data/TELECOM_OUTPUT_ID_PATH"
            self.ALL_EDGE_PATH = "/home/lai/ugnn/sample_data/ALL_EDGE_PATH"
        
        self.model_name = "cnn"
        self.model_version = "20231017_grid1000"
        self.device = "cuda:1"
        
        self.MODEL_SAVE_PATH = f"/home/lai/ugnn/results/gegn_outputs/{self.model_name}_{self.model_version}.pt"
        self.BEST_MODEL_SAVE_PATH = f"/home/lai/ugnn/results/gegn_outputs/{self.model_name}_{self.model_version}_best.pt"
        
        # self.note = "+2020filtered super" # Anything you would like to put into the log
        self.save_suffix = self.model_version

        self.check_paras = False  # Only print model params

        self.model_status = "test"
        self.reuse_model = False
        self.one_hot_hour = False
        
        if self.reuse_model:
            print("Using model in ", self.BEST_MODEL_SAVE_PATH)
        else:
            print("Making new model ", self.model_name, self.model_version)
        
        self.test_custom = False
        self.CUSTOM_EDGE_PATH = "./"
        self.save_training_model = True
        self.save_tensorboard = False
        self.train_cities = ["北京市", "广州市",
                             "成都市", "南京市", "郑州市", "宁波市", "佛山市", "东莞市", "合肥市", "长沙市", "青岛市", "苏州市",
                             "厦门市", "济南市", "沈阳市", "大连市", "昆明市", "长春市", "温州市", "石家庄市", "贵阳市", "南昌市", "常州市", "哈尔滨市",
                             "乌鲁木齐市", "徐州市"]
        # self.train_cities = ["北京市", "成都市", "厦门市", "徐州市","青岛市", "石家庄市", "苏州市", "深圳市", "杭州市"]
        self.train_cities = ["北京市"]
        # self.train_cities = ["乌鲁木齐市"]
        self.infer_cities = ["北京市", "广州市",
                             "成都市", "南京市", "郑州市", "宁波市", "佛山市", "东莞市", "合肥市", "长沙市", "青岛市", "苏州市",
                             "厦门市", "济南市", "沈阳市", "大连市", "昆明市", "长春市", "温州市", "石家庄市", "贵阳市", "南昌市", "常州市", "哈尔滨市",
                             "乌鲁木齐市", "徐州市"]
        # self.test_cities = ["宁波市"]
        # self.test_cities = ["北京市", "广州市",
        #                      "成都市", "南京市", "郑州市", "宁波市", "佛山市", "东莞市", "合肥市", "长沙市"]
        # self.test_cities = ["青岛市", "苏州市",
        # "厦门市", "济南市", "沈阳市", "大连市", "昆明市", "长春市", "温州市", "石家庄市", "贵阳市", "南昌市", "常州市", "哈尔滨市",
        # "乌鲁木齐市", "徐州市"]

        self.test_cities = ["上海市", "深圳市",
                            "杭州市","武汉市", "天津市",
                            "大连市", "福州市", "南宁市", "兰州市",
                            "呼和浩特市"]
        # self.test_cities = ["深圳市"]
        self.test_cities = ["北京市"]
        # self.infer_cities = ["北京市", "上海市", "广州市"]
        # #
        # self.test_cities = ["北京市", "上海市", "广州市"]
        # self.grid_type = "land_use"
        self.grid_type = "poi_aoi"
        self.split_type = "o_id"
        self.split_frac = 0.1
        self.sample_gap = 3
        # self.sample_type = "normal" # use 0.1 sampling for value<1
        self.sample_type = "diff"  # use differential sampling strategy
        self.beta = 0.3

        self.test_ids = []

        self.epoch = 50
        self.eval_gap = 3
        self.train_batch_size = 512
        self.infer_batch_size = 100000
        self.learning_rate = 5e-5
        self.do_augmentation = True

        self.grid_radius = 2
        self.grid_num = (self.grid_radius * 2 - 1) ** 2
        self.o_grid_cnn_layer_channels = [256, 128]
        self.o_grid_fcnn_layer_channels = [128, 128]
        self.o_grid_extra_cnn_layer_channels = [256, 128]
        self.o_grid_extra_fcnn_layer_channels = [128, 128]
        self.d_grid_cnn_layer_channels = [256, 128]
        self.d_grid_fcnn_layer_channels = [128, 128]
        self.d_grid_extra_cnn_layer_channels = [256, 128]
        self.d_grid_extra_fcnn_layer_channels = [128, 128]
        self.edge_layer_channels = [64]
        self.external_layer_channels = [64]
        self.out_layer_channels = [512, 256, 128, 128, 64, 64, 1]


        
        self.TENSORBOARD_PATH = os.path.join(self.CUR_PATH, "runs", "{}_{}".format(self.model_name, self.model_version))
