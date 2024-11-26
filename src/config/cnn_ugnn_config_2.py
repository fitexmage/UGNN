import os
from src.config.model_config import ModelConfig


class CnnGegnConfig(ModelConfig):
    def __init__(self):
        super(CnnGegnConfig, self).__init__()

        # Get the name of the currently running script's file
        script_filename = os.path.basename(__file__)
        print("Using "+script_filename)
        
        self.grid1000 = True
        self.small_debugging = False
        self.one_hot_hour = False

        self.hour = None
        
        
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
        
        self.model_name = "cnn_gegn_combine"
        self.model_version = "20231102"
        self.device = "cuda:1"
        self.note = "fcnn 256+128" # Anything you would like to put into the log
        # self.note = "2019" # Anything you would like to put into the log
        self.save_suffix = self.model_version


        self.model_status = "test"
        self.reuse_model = True
        self.test_custom = False
        self.CUSTOM_EDGE_PATH = "/home/user/disk/custom_edge/five.csv"
        self.save_training_model = False
        self.save_tensorboard = False
        
        self.MODEL_SAVE_PATH = os.path.join(self.SAVE_PATH, "{}_{}.pt".format(self.model_name, self.model_version))

        self.BEST_MODEL_SAVE_PATH = os.path.join(self.SAVE_PATH,
                                                 "{}_{}best.pt".format(self.model_name, self.model_version))
        if self.reuse_model:
            print("Using model in ", self.BEST_MODEL_SAVE_PATH)
        else:
            print("Making new model ", self.model_name, self.model_version)

        # self.train_cities = ["北京市", "成都市", "厦门市", "徐州市","青岛市", "石家庄市", "苏州市", "深圳市", "杭州市"]
        # self.train_cities = ["大连市"]
        self.train_cities = ["北京市", "广州市",
                             "成都市", "南京市", "郑州市", "宁波市", "佛山市", "东莞市", "合肥市", "长沙市", "青岛市", "苏州市",
                             "厦门市", "济南市", "沈阳市", "大连市", "昆明市", "长春市", "温州市", "石家庄市", "贵阳市", "南昌市", "常州市", "哈尔滨市",
                             "乌鲁木齐市", "徐州市"]

        self.test_cities = ["上海市", "深圳市",
                            "杭州市","武汉市", "天津市",
                            "无锡市", "福州市", "南宁市", "兰州市",
                            "呼和浩特市"]
        
        self.test_cities = ["呼和浩特市", "深圳市",
                            "杭州市","武汉市", "天津市",
                            "无锡市", "福州市", "南宁市", "兰州市",
                            "上海市"]
        self.test_cities = ["无锡市"]

        # self.test_cities = ["北京市", "上海市", "广州市"]
        # self.infer_cities = ["北京市", "上海市", "广州市"]
        # #
        # self.test_cities = ["上海市",
        #                     "武汉市",
        #                     "兰州市",
        #                     "呼和浩特市"]
        # self.grid_type = "land_use"
        self.grid_type = "poi_aoi"
        self.split_type = "o_id"
        self.split_frac = 0.1
        self.sample_gap = 3
        # self.sample_type = "normal" # use 0.1 sampling for value<1
        self.sample_type = "diff"   # use differential sampling strategy
        self.beta = 0.3

        self.test_ids=[]

        self.epoch = 50
        self.eval_gap = 1
        self.train_batch_size = 2048
        self.infer_batch_size = 82920
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
        self.o_grid_out_fcnn_layer_channels = [128, 128]
        self.d_grid_out_fcnn_layer_channels = [128, 128]
        
        if self.one_hot_hour:
            self.edge_layer_channels = [128]  # 64+12 -> 128
            self.hour_channels = [64]
            self.others_channels = [12]
            self.hour_dim = 24
            self.others_dim = 3
            self.edge_dim = 76
        else:
            self.edge_layer_channels = [128]
            
        self.external_layer_channels = [64]
        self.out_layer_channels = [512, 256, 128, 128, 64, 64, 1]


        self.TENSORBOARD_PATH = os.path.join(self.CUR_PATH, "runs", "{}_{}".format(self.model_name, self.model_version))
