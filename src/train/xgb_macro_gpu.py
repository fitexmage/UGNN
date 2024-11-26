import sys
import os
import shutil
script_dir = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.dirname(os.path.dirname(script_dir))
project_name = "xgb"
sys.path.append(project_path)

from torch.utils.data import DataLoader

import torch
from tqdm import tqdm
import torch
import xgboost as xgb
import numpy as np

from src.config.xgb_config import CnnGegnConfig
from src.model.cnn_ugnn import CnnGegn
from src.utils.dataset import GegnDataset
import util
from src.preprocessor.base_preprocessor import (CoordPreprocessor, IdPreprocessor)
from src.preprocessor.cnn_ugnn_preprocessor import (
    preprocess,
    sample_every_epoch,
    test_preprocess,
    UgnnGridPreprocessor,
    UgnnEdgePreprocessor
)
from src.utils.model_util import get_score_in_scope

import cupy as cp


class Xgb_iterator(xgb.DataIter):
    def __init__(self, torch_dataloader):
        self.og_dataloader = torch_dataloader
        self.torch_dataloader = iter(tqdm(torch_dataloader))
        self._it = 0
        self._len = len(torch_dataloader)
        self.counter = 0
        super().__init__()

    def next(self, input_data):  # input_data
        if self._it == self._len or self.counter == 500:
            # return 0 to let XGBoost know this is the end of iteration
            return 0
        o_grid_basic_x, d_grid_basic_x, o_grid_extra_x, d_grid_extra_x, edge_x, external_x, y = next(self.torch_dataloader)
        X = torch.concat((o_grid_basic_x, d_grid_basic_x, o_grid_extra_x, d_grid_extra_x, edge_x, external_x), dim=1).numpy()
        X = cp.array(X)
        y = cp.array(y.numpy())
        input_data(data=X, label=y)
        self._it += 1
        self.counter += 1
        # print(f"{self._it/self._len}", end='\r')
        
        # Return 1 to let XGBoost know we haven't seen all the files yet.
        return 1
    def reset(self):
        """Reset the iterator to its beginning"""
        self._it = 0
        self.torch_dataloader = iter(tqdm(self.og_dataloader))


def make_preprocessor(cities):
    
    """
    Make preprocessor according to config.
    For training, the cities will involve various cities. However for testing,
    we load one dataset for one city, so the cities will ONLY contain one city.
    """
    
    coord_preprocessor = CoordPreprocessor(cities, config.CITY_CENTROID_PATH)
    id_preprocessor = IdPreprocessor(cities, config.split_frac)
    edge_preprocessor = UgnnEdgePreprocessor(
        cities, config.ALL_EDGE_PATH, checkpoint, config.one_hot_hour
    )
    grid_preprocessor = UgnnGridPreprocessor(
        checkpoint,
        cities,
        config.grid_type,
        config.GRID_BASIC_PATH,
        config.GRID_POI_AOI_PATH,
        config.GRID_LAND_USE_PATH,
        config.RAW_CITY_METRIC_PATH,
        config.CITY_DIVISION_PATH,
        config.grid_radius,
        config.model_name,
        config.one_hot_hour,
    )
    preprocessor = (
        coord_preprocessor,
        id_preprocessor,
        edge_preprocessor,
        grid_preprocessor,
    )
    return preprocessor


def train(train_dataloader_, valid_dataloader, params, valid_edge):
    train_dataloader = iter(tqdm(train_dataloader_))
    
    old_model = None  # No old model for first iteration.
    batch_size = 50  # up most about 785
    for i in range(len(train_dataloader_)//batch_size+1):
        
        # Make Incremental dataset
        Xs = []
        ys = []
        for j in range(batch_size):
            o_grid_basic_x, d_grid_basic_x, o_grid_extra_x, d_grid_extra_x, edge_x, external_x, y = next(train_dataloader)
            X = torch.concat((o_grid_basic_x, d_grid_basic_x, o_grid_extra_x, d_grid_extra_x, edge_x, external_x), dim=1).numpy()
            Xs.append(X)
            ys.append(y)
        Xs = np.concatenate(Xs, axis=0)
        ys = np.concatenate(ys, axis=0)
        Xs = cp.array(Xs)
        ys = cp.array(ys)
        inc_dataset = xgb.DeviceQuantileDMatrix(data=Xs, label=ys)
        
        booster = xgb.train(params, inc_dataset, xgb_model=old_model)
        old_model = 'old_model'
        booster.save_model(old_model)
    
    
    # # Eval
    # y_preds = torch.Tensor(booster.predict(valid_data))
    # gt_trip = torch.Tensor(valid_edge['trip'].values).squeeze(-1)

    
    # log = ''
    # print("\n")
    # score1 = get_score_in_scope(y_preds, gt_trip)
    # log += 'total r_squared, mae, rmse, cpc, pearson: '+str(score1)+'\n'
    # score2 = get_score_in_scope(y_preds, gt_trip, [0, 10])
    # log += '0, 10 r_squared, mae, rmse, cpc, pearson: '+str(score2)+'\n'
    # score3 = get_score_in_scope(y_preds, gt_trip, [10, 100])
    # log += '10, 100 r_squared, mae, rmse, cpc, pearson: '+str(score3)+'\n'
    # score4 = get_score_in_scope(y_preds, gt_trip, [100, None])
    # log += '> 100 r_squared, mae, rmse, cpc, pearson: '+str(score4)+'\n'
    # log += '\n\n'
    
    # Save model
    booster.save_model(config.BEST_MODEL_SAVE_PATH)
    
    
def test(cities):
    log = ''
    for city in cities:
        log += 'City: ' + city + '\n'
        # Initialize data preprocessing for the specified cities
        preprocessor = make_preprocessor([city])
        
        # Perform data preprocessing
        p_data = test_preprocess(*preprocessor)
        test_data = (p_data['grid']['grid_basic_dict'], p_data['grid']['grid_extra_dict'], p_data['test_edge'])
        test_dataset = GegnDataset(*test_data, config.one_hot_hour)
        test_dataloader = DataLoader(test_dataset, batch_size=config.train_batch_size, shuffle=True)
        test_data = xgb.DMatrix(Xgb_iterator(test_dataloader))
        
        booster = xgb.Booster()
        booster.load_model(config.BEST_MODEL_SAVE_PATH)
        # Eval
        y_preds = torch.Tensor(booster.predict(test_data))
        gt_trip = torch.Tensor(p_data['test_edge']['trip'].values).squeeze(-1)

        
        log = ''
        print("\n")
        score1 = get_score_in_scope(y_preds, gt_trip)
        log += 'total r_squared, mae, rmse, cpc, pearson: '+str(score1)+'\n'
        score2 = get_score_in_scope(y_preds, gt_trip, [0, 10])
        log += '0, 10 r_squared, mae, rmse, cpc, pearson: '+str(score2)+'\n'
        score3 = get_score_in_scope(y_preds, gt_trip, [10, 100])
        log += '10, 100 r_squared, mae, rmse, cpc, pearson: '+str(score3)+'\n'
        score4 = get_score_in_scope(y_preds, gt_trip, [100, None])
        log += '> 100 r_squared, mae, rmse, cpc, pearson: '+str(score4)+'\n'
        log += '\n\n'
    


if __name__ == "__main__":
    # Load configuration settings for the xgb model
    config = CnnGegnConfig()

    # Load a checkpoint if available
    checkpoint = util.load_checkpoint(config)

    # Check if the model is in 'train' mode
    if config.model_status == 'train':
        # Get the list of cities for training
        cities = config.train_cities
        print("Training cities:", cities)
        # Initialize data preprocessing for the specified cities
        preprocessor = make_preprocessor(cities)
        edge_preprocessor = preprocessor[2]
        
        # Perform data preprocessing
        p_data = preprocess(*preprocessor)
        
        # Sample data for training, considering augmentation if specified
        train_data = sample_every_epoch(edge_preprocessor, p_data, augmentation=config.do_augmentation)
        
        # Create a DataLoader for the training dataset
        train_dataset = GegnDataset(*train_data, config.one_hot_hour)
        train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
        
        # Prepare validation data
        valid_data = (p_data['grid']['grid_basic_dict'], p_data['grid']['grid_extra_dict'], p_data['valid_edge'])
        valid_dataset = GegnDataset(*valid_data, config.one_hot_hour)
        valid_dataloader = DataLoader(valid_dataset, batch_size=config.train_batch_size, shuffle=True)
        
        # Set parameters
        params = {
            'device': 'cuda',
            'objective': 'reg:squarederror',  # Objective function for regression tasks
            'eval_metric': 'rmse',  # Evaluation metric: Root Mean Squared Error
            'eta': 0.1,  # Learning rate
            'max_depth': 6,  # Maximum depth of a tree
            'subsample': 0.8,  # Subsample ratio of the training instances
            'colsample_bytree': 0.8,  # Subsample ratio of columns when constructing each tree
            # 'n_estimators': 100,  # Number of boosting rounds
            'seed': 42,  # Random seed for reproducibility
            'sampling_method': 'gradient_based',
            'tree_method': 'gpu_hist'}
        # Start the training process
        train(train_dataloader, valid_dataloader, params, p_data['valid_edge'])
    
    elif config.model_status == "test":
        md5 = util.calculate_md5(config.BEST_MODEL_SAVE_PATH)
        save_path = os.path.join(config.SAVE_PATH, 'xgb', md5)
        print(md5)
        # Create the folder path if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # Specify the destination path (including the file name)
        destination_path = os.path.join(save_path, "{}_{}_best.pt".format(config.model_name, config.model_version))

        # Copy the file to the destination path
        shutil.copy(config.BEST_MODEL_SAVE_PATH, destination_path)
        
        test(config.test_cities)








