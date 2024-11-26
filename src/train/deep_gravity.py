# -*- coding: utf-8 -*-
"""
This module is derived from train_deep_gravity_xizhi1, which is derived from train_deep_gravity.
Both train_deep_gravity_xizhi1 and train_deep_gravity are efficient on machine but complex on human
cognition. This module is slightly inefficient but easy to read.

# config

This script get its config from one of the json files below:
1. configs/deep_gravity.json  Basic one
2. configs/deep_gravity_small_sample.json  Very small sample, run very quick
3. configs/deep_gravity_debug.json  Containing only one city, good for debug

# How to use
1. Train from scratch: For json config file, change reuse_model to false and change model_status to train;
2. Continue training: For json config file, change reuse_model to true and model_status to train;
3. Testing: For json config file, change reuse_model to false and model_status to test.
"""

from torch import optim
import sys
import os
import time
import gc
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import matplotlib
import torch
from torch import nn
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np

matplotlib.use("agg")
import sys

project_path = "/home/lai/ugnn/"
project_name = "deep_gravity"
sys.path.append(project_path)
from src.model.deep_gravity import DeepGravity
from config import config

from util import setup_seed, make_save_dir, calculate_md5
from src.utils.metric import DeepGravityCrossEntropyLoss

from src.preprocessor.base_preprocessor import (
    CoordPreprocessor,
    IdPreprocessor,
    GridPreprocessor,
    EdgePreprocessor,
)

from src.utils.model_util import (
    get_score_in_scope,
)

from src.preprocessor.deep_gravity_preprocessor import (
    make_data_for_dataset,
    make_data_for_test_dataset,
)
from src.datasets.deep_gravity_dataset2 import DeepGravityDataset
from config import config


def setup_model(config, checkpoint):
    print("Model name:", config.model_name)
    print("Model version:", config.model_version)
    Model = DeepGravity

    if "grid_dim" in checkpoint.keys() and "edge_dim" in checkpoint.keys():
        model = Model(checkpoint["grid_dim"], checkpoint["edge_dim"], config)

    else:
        print(
            "You didn't save grid_dim and edge_dim in checkpoint, \nwe will\
              use default grid_dim=49 and edge_dim=1, \nsee save_model(checkpoint, path)\
              for more information."
        )
        model = Model(49, 1, config)

    if torch.cuda.is_available():
        print("Device:", config.device)
        model = model.to(config.device)

    criterion = DeepGravityCrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)

    # load all the things in the saved model
    if config.reuse_model:
        epoch = checkpoint["epoch"]
        step = checkpoint["step"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_loss = checkpoint["best_loss"]
    else:
        for name, param in model.named_parameters():
            if ".bn" not in name:
                if ".weight" in name:
                    torch.nn.init.kaiming_normal_(
                        param.data, mode="fan_out", nonlinearity="leaky_relu"
                    )
                else:
                    torch.nn.init.constant_(param.data, 0)
    return model, criterion, optimizer


def train(train_dataset, valid_dataset, config, checkpoint, model_path):
    def save_model(checkpoint, path):
        checkpoint["epoch"] = epoch
        checkpoint["step"] = step
        checkpoint["model"] = model.state_dict()
        checkpoint["optimizer"] = optimizer.state_dict()
        checkpoint["best_loss"] = best_loss
        checkpoint["train_cities"] = config.cities
        checkpoint["hour"] = config.hour
        checkpoint["grid_dim"] = train_dataset.grid_dim
        checkpoint["edge_dim"] = train_dataset.edge_dim

        torch.save(checkpoint, path)

    def eval(best_loss):
        model.eval()
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)
        valid_loader = tqdm(valid_loader)
        eval_loss = 0.0
        for i, valid_data in enumerate(valid_loader):
            torch.cuda.empty_cache()
            for j in range(len(valid_data)):
                valid_data[j] = valid_data[j].squeeze(0).to(config.device)
            y_pred = model(*valid_data[:-1])
            eval_loss += criterion(y_pred, valid_data[-1]).item()
        print("Loss:", eval_loss / len(valid_loader))

        if config.save_tensorboard:
            writer.add_scalar("evaluation loss", eval_loss, global_step=step)
            writer.flush()

        if best_loss is None or best_loss > eval_loss:
            best_loss = eval_loss
            
            save_model(checkpoint, model_path)
            print("Best Model saved!")
        print()
        return best_loss

    print("Train cities:", config.train_cities)
    print("Test cities:", config.test_cities)
    model, criterion, optimizer = setup_model(config, checkpoint)
    # model = nn.DataParallel(model)≤
    epoch = 0
    step = 0
    best_loss = None

    if config.save_tensorboard:
        writer = SummaryWriter(config.TENSORBOARD_PATH)

    for epoch in range(config.epoch):
        model.train()
        torch.cuda.empty_cache()

        train_loss = 0
        train_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=True, num_workers=10
        )
        train_loader = tqdm(train_loader)
        loss = 0.0
        for i, train_data in enumerate(train_loader):
            if torch.cuda.is_available():
                for j in range(len(train_data)):
                    train_data[j] = train_data[j].squeeze(0).to(config.device)
            y_pred = model(*train_data[:-1])
            loss += criterion(y_pred, train_data[-1])

            if (i + 1) % config.train_batch_size == 0:
                train_loss += loss.item()

                train_loader.set_description(
                    "Epoch: {} Step: {} Training Loss: {}".format(
                        epoch, step, str(train_loss / (i + 1))
                    )
                )
                train_loader.refresh()

                optimizer.zero_grad()
                loss.backward()
                loss = 0.0
                optimizer.step()

                step += 1
        train_loss = train_loss / len(train_loader)

        if config.save_tensorboard:
            writer.add_scalar("training loss", train_loss, global_step=step)
            writer.flush()

        epoch += 1

        if config.save_training_model:
            print("Model save path: ", os.path.join(config.result_path, 'deep_gravity'))
            model_name = 'deep_gravity_grid1000'
            save_file_path = os.path.join(config.result_path, 'deep_gravity', 'deep_gravity_grid1000.pt')
            save_model(checkpoint, save_file_path)
        best_loss = eval(best_loss)


def load_checkpoint(config):
    if config.reuse_model:
        if config.model_status == "train":
            checkpoint = torch.load(config.MODEL_SAVE_PATH)
            print(f"Picking up where you last left off:  {config.MODEL_SAVE_PATH}.")
        else:
            checkpoint = torch.load(config.BEST_MODEL_SAVE_PATH)
            print(f"Loading {config.BEST_MODEL_SAVE_PATH} for testing or inferring.")
    else:
        checkpoint = {}
        print(f"Did not load any existed model.")
    return checkpoint


def get_score(result_df, city):
    y_pred = torch.Tensor(result_df["pred_trip"].values)
    y_gt = torch.Tensor(result_df["gt_trip"].values)

    score = [city]
    get_score_in_scope(y_pred, y_gt, None, score)
    get_score_in_scope(y_pred, y_gt, [0, 10], score)
    get_score_in_scope(y_pred, y_gt, [10, 100], score)
    get_score_in_scope(y_pred, y_gt, [100, None], score)
    return score


def make_train_dataset(city: List, checkpoint) -> DeepGravityDataset:
    """
    Make complex dataset.

    Our dataset involves lots of data preprocessing and differ for train and test.
    So this func will make this complex dataset for you.

    :param city:
    For training, it's a list of cities, for we need to train with many cities.
    For testing, it's a list of one city, for we need to give result for each cities separately.
    :return: a deep gravity dataset object
    """
    # Make preprocessor according to config
    coord_preprocessor = CoordPreprocessor(city, config.CITY_CENTROID_PATH)
    id_preprocessor = IdPreprocessor(city, config.split_frac)
    edge_preprocessor = EdgePreprocessor(
        city, config.ALL_EDGE_PATH, config.hour, checkpoint
    )
    grid_preprocessor = GridPreprocessor(
        config.GRID_BASIC_PATH,
        config.GRID_POI_AOI_PATH,
        city,
        checkpoint,
        config.grid_radius,
        config.model_name,
        config.grid_num,
    )

    train_data, eval_data, edge_dim, grid_dim, grid, coord = make_data_for_dataset(
        coord_preprocessor, id_preprocessor, edge_preprocessor, grid_preprocessor
    )
    train_dataset = DeepGravityDataset(train_data, grid, coord, checkpoint)
    valid_dataset = DeepGravityDataset(eval_data, grid, coord, checkpoint)
    train_dataset.edge_dim = edge_dim
    train_dataset.grid_dim = grid_dim
    checkpoint["edge_dim"] = edge_dim
    checkpoint["grid_dim"] = grid_dim

    return train_dataset, valid_dataset


def make_test_dataset(city: List, checkpoint) -> DeepGravityDataset:
    """
    Returns:
        edge: For testing, we need it to present results.
    """

    coord_preprocessor = CoordPreprocessor(city, config.CITY_CENTROID_PATH)
    id_preprocessor = IdPreprocessor(city, config.split_frac)
    edge_preprocessor = EdgePreprocessor(
        city, config.ALL_EDGE_PATH, config.hour, checkpoint
    )
    grid_preprocessor = GridPreprocessor(
        config.GRID_BASIC_PATH,
        config.GRID_POI_AOI_PATH,
        city,
        checkpoint,
        config.grid_radius,
        config.model_name,
        config.grid_num,
    )
    test_data, edge_dim, grid_dim, grid, coord, edge = make_data_for_test_dataset(
        coord_preprocessor, id_preprocessor, edge_preprocessor, grid_preprocessor
    )
    test_dataset = DeepGravityDataset(test_data, grid, coord, checkpoint)
    test_dataset.edge_dim = edge_dim
    test_dataset.grid_dim = grid_dim
    checkpoint["edge_dim"] = edge_dim
    checkpoint["grid_dim"] = grid_dim

    return test_dataset, edge


def test(deep_gravity_dataset, model, scores, save_dir, city, edge, log):
    test_loader = DataLoader(deep_gravity_dataset, batch_size=1, shuffle=False)
    test_loader = tqdm(test_loader)
    all_y = []

    for i, test_data in enumerate(test_loader):
        for j in range(len(test_data)):
            test_data[j] = test_data[j].squeeze(0).to(config.device)
        y_pred = model(*test_data[:-1])
        y_pred = torch.softmax(y_pred, dim=0)
        y_pred = y_pred * torch.sum(test_data[-1])

        o_ids, d_ids = deep_gravity_dataset.get_ids(i)
        y = np.concatenate(
            [
                np.array(o_ids)[:, np.newaxis],
                np.array(d_ids)[:, np.newaxis],
                np.array(y_pred.cpu().detach())[:, np.newaxis],
            ],
            axis=1,
        )
        all_y.append(y)
    all_y = np.concatenate(all_y, axis=0)

    result_df = edge[
        ["city", "hour", "o_id", "d_id", "trip", "surface_distance"]
    ].merge(
        pd.DataFrame(all_y, columns=["o_id", "d_id", "pred_trip"]), on=["o_id", "d_id"]
    )
    result_df.columns = [
        "city",
        "hour",
        "o_id",
        "d_id",
        "gt_trip",
        "surface_distance",
        "pred_trip",
    ]

    score = get_score(result_df, city)
    scores.append(score)
    log += str(city)
    log += '\n'
    score1 = get_score_in_scope(torch.Tensor(result_df['pred_trip'].values).squeeze(-1), torch.Tensor(result_df['gt_trip'].values).squeeze(-1))
    log += 'total r_squared, mae, rmse, ssi, cpc, pearson, ssim: '+str(score1)+'\n'
    score2 = get_score_in_scope(torch.Tensor(result_df['pred_trip'].values).squeeze(-1), torch.Tensor(result_df['gt_trip'].values).squeeze(-1), [0, 10])
    log += '0, 10 r_squared, mae, rmse, ssi, cpc, pearson, ssim: '+str(score2)+'\n'
    score3 = get_score_in_scope(torch.Tensor(result_df['pred_trip'].values).squeeze(-1), torch.Tensor(result_df['gt_trip'].values).squeeze(-1), [10, 100])
    log += '10, 100 r_squared, mae, rmse, ssi, cpc, pearson, ssim: '+str(score3)+'\n'
    score4 = get_score_in_scope(torch.Tensor(result_df['pred_trip'].values).squeeze(-1), torch.Tensor(result_df['gt_trip'].values).squeeze(-1), [100, None])
    log += '> 100 r_squared, mae, rmse, ssi, cpc, pearson, ssim: '+str(score4)+'\n'
    log += '\n\n'
    

    result_df = result_df[["hour", "o_id", "d_id", "pred_trip"]]
    # result_df.columns = ["hour", "o_id", "d_id", "trip"]
    print("output size:", len(result_df))

    # Get the current time as a prefix

    result_df.to_csv(os.path.join(save_dir, f"{city}.csv"), index=False)

    gc.collect()
    return log

def main(checkpoint):
    """
    The main function for managing the execution flow of deep gravity model.

    Args:
        checkpoint: Model checkpoint we will use through our the program.

    Raises:
        Exception: If the `config.model_status` is not one of ["train", "eval", "test"].

    This function performs the following tasks based on the value of `config.model_status`:

    - If `config.model_status` is "train" or "eval":
        - Prepares train and validation datasets based on the specified cities.
        - Initiates 🚋ing of the model using the provided datasets and configuration.

    - If `config.model_status` is "test":
        - Loads a trained model checkpoint.
        - Prepares test datasets for specified cities.
        - Evaluates the model's performance on each city's test data.
        - Saves the evaluation results to a CSV file.
    """
    
    
    if config.model_status in ["train", "eval"]:
        print("🚋ing Models...")

        config.cities = config.train_cities
        train_dataset, valid_dataset = make_train_dataset(
            config.train_cities, checkpoint
        )
        train(train_dataset, valid_dataset, config, checkpoint, model_path)

    elif config.model_status == "test":
        print("Testing...")
        scores = []
        

        checkpoint = load_checkpoint(config)
        cities = config.test_cities
        model, _, __ = setup_model(config, checkpoint)
        model.eval()

        log = ""
        log += str(config.__dict__)
        log += '\n\n\n'
        for city in cities:
            print("City:", city)
            # Yes, here for each city loop, we need to make new dataset.
            # For we want to give separate results for each city
            test_dataset, edge = make_test_dataset([city], checkpoint)
            log = test(test_dataset, model, scores, save_dir, city, edge, log)

        with open(os.path.join(save_dir, 'log_guangzhou'), 'w') as file:
            file.write(log)
    else:
        raise Exception


if __name__ == "__main__":
    print(
        "==============================================================================="
    )
    print("")
    
    config.grid1000 = True
    if config.grid1000:
        config.ALL_EDGE_PATH = "/media/lai/27dbdec5-6518-47f1-b1af-159ffe3d1c08/2019/edge_2019_by_grid1000"
        print("Using grid1000", config.ALL_EDGE_PATH)
    
    config.reuse_model = True
    config.hour = 7
    config.model_status = 'test'
    config.BEST_MODEL_SAVE_PATH = "/home/lai/ugnn/results/gegn_outputs/deep_gravity/deep_gravity_1221_grid1000.pt"
    
    config.train_cities = ["北京市", "广州市",
                   "成都市", "南京市", "郑州市", "宁波市", "佛山市", "东莞市", "合肥市", "长沙市", "青岛市", "苏州市",
                   "厦门市", "济南市", "沈阳市", "大连市", "昆明市", "长春市", "温州市", "石家庄市", "贵阳市", "南昌市", "常州市", "哈尔滨市",
                   "乌鲁木齐市", "徐州市"]
    config.test_cities = ["上海市", "深圳市",
                  "杭州市","武汉市", "天津市", "无锡市",
                 "大连市", "福州市", "南宁市", "兰州市",
                  "呼和浩特市"]
    config.test_cities = ["昆明市", "长春市"]

    print(config.train_cities)
    print(config.test_cities)


    
    config.split_frac = 0.1
    config.model_name = "deep_G"
    config.model_version = None
    config.grid_num = 9
    config.grid_radius = 2
    config.layer_channels = [256, 256, 256, 256, 256, 256, 128, 128, 128, 128, 128, 128, 128, 128, 128, 1]
    config.train_batch_size = 64
    config.device = "cuda:1"
    config.learning_rate = 1e-5
    config.save_tensorboard = None
    config.epoch = 20
    config.save_training_model = True

    
    # <<< Training Part <<<
    # model_name = "deep_gravity_1221_grid1000.pt"
    # model_path = os.path.join(config.result_path, 'deep_gravity', model_name)  # Trained model will be saved in this dir.
    
    # <<< Testing Part <<<
    md5 = calculate_md5(config.BEST_MODEL_SAVE_PATH)
    print(md5)
    save_dir = os.path.join("/home/lai/ugnn/results/gegn_outputs", 'deep_gravity', md5)
    
    setup_seed(95)  # Does it help?
    checkpoint = load_checkpoint(config)

    main(checkpoint)
