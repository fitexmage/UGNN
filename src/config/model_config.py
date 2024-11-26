import os

from config import Config


class ModelConfig(Config):
    def __init__(self):
        super(ModelConfig, self).__init__()

        self.DISK_2_PATH = "/media/lai/27dbdec5-6518-47f1-b1af-159ffe3d1c08/gegn_outputs"
        self.INFER_PATH = os.path.join(self.DISK_2_PATH, "infer")
        self.INFER_RESULT_PATH = os.path.join(self.INFER_PATH, "result")
        self.TEST_PATH = os.path.join(self.DISK_2_PATH, "test")
        self.TEST_RESULT_PATH = os.path.join(self.TEST_PATH, "result")
        self.TEST_RESULT_BY_HOUR_PATH = os.path.join(self.TEST_PATH, "result_by_hour")
        self.TEST_CUSTOM_RESULT_PATH = os.path.join(self.TEST_PATH, "result_custom")
        self.TEST_SCATTER_PATH = os.path.join(self.TEST_PATH, "scatter")
        self.TEST_DISTANCE_PATH = os.path.join(self.TEST_PATH, "distance")
        self.TEST_DIFF_PATH = os.path.join(self.TEST_PATH, "diff")
        self.TEST_W_ANALYSIS_PATH = os.path.join(self.TEST_PATH, "weight.csv")

        
