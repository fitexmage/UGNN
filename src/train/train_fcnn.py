import sys
import os
import time

script_dir = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_path)

from src.train.train import ODPredictor
from src.utils.dataset import GegnDataset
from src.config.fcnn_config import FcnnConfig
from src.model.fcnn import Fcnn
import util
import torch


class FcnnPredictor(ODPredictor):
    def __init__(self):
        super(FcnnPredictor, self).__init__()
        self.config = FcnnConfig()
        self.Dataset = GegnDataset
        self.Model = Fcnn

        self.checkpoint = util.load_checkpoint(self.config)
        
        def write_cf2model():
            print("Writng config into model...")
            # A temperary function to load config into model for version control.
            self.checkpoint["config"] = self.config.__dict__
            torch.save(self.checkpoint, self.config.BEST_MODEL_SAVE_PATH)
            print(self.config.__dict__)
            print("Write finished!")
        # write_cf2model()
        
        self.setup_grid_data()
        if self.config.model_status in ["train", "eval"]:
            self.setup_train_edge_data()
        elif self.config.model_status in ["test", "w_analysis"]:
            self.setup_test_edge_data()
        self.prepare_train()
        self.setup_model()


if __name__ == "__main__":
    start_time = time.time()

    fcnn = FcnnPredictor()
    if fcnn.config.model_status == "train":
        fcnn.train()
    elif fcnn.config.model_status == "eval":
        fcnn.eval()
    elif fcnn.config.model_status == "test":
        fcnn.test()
    else:
        fcnn.infer()

    print("Total hour:", (time.time() - start_time) / 3600)