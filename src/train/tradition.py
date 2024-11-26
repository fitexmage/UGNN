# -*- coding: utf-8 -*-
import math
import time
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import sys
project_path = "/home/lai/ugnn/"
project_name = "tradition"
sys.path.append(project_path)
from util import (
    setup_seed,
    make_save_dir,
    calculate_md5,
)
from src.utils.model_util import (
    get_score_in_scope,
)
from config import config



"""Load package for temp test."""
import torch
import pandas as pd
import numpy as np
import datetime


# 程序功能：根据给定的出行分布量mob文件和出行发生吸引量pop文件，预测小区间的出行分布量，并输出预测精度SSI值

# 参数设置
alpha = 0.4



def dist(f, t):  # 输入起终点经纬度数组
    # 根据经纬度求距离
    def deg2rad(d):
        return d * math.pi / 180.0
    flon = deg2rad(f[0])
    flat = deg2rad(f[1])
    tlon = deg2rad(t[0])
    tlat = deg2rad(t[1])
    con = math.sin(flat) * math.sin(tlat)
    con += math.cos(flat) * math.cos(tlat) * math.cos(flon - tlon)
    if con > 1.0:
        con = 1.0
    return math.acos(con) * 6378.137


def gravity_func(x, a):
    m_i = x[:, 0]
    m_j = x[:, 1]
    d_ij = x[:, 2]
    return torch.mul(a[0], torch.div(torch.mul(torch.pow(m_i, a[1]), torch.pow(m_j, a[2])), torch.pow(d_ij, a[3])))


def get_data(city, hours, config):
    r"""
    :return:
    DataFrame["o_id", "d_id", "o_resi", "d_resi", "surface_distance", "trip"]
    """
    city_all_grid_df = pd.read_csv(os.path.join(config.ALL_GRID_PATH, "{}.csv".format(city)))
    city_all_edge_df = pd.read_csv(os.path.join(config.ALL_EDGE_PATH, "{}.csv".format(city)))
    city_all_edge_df = city_all_edge_df[city_all_edge_df["hour"].isin(hours)]  # filter

    # Left join edge and grid data.
    city_all_edge_df = city_all_edge_df.merge(city_all_grid_df[["id", "resi"]], how="left", left_on="o_id", right_on="id")
    city_all_edge_df = city_all_edge_df.merge(city_all_grid_df[["id", "resi"]], how="left", left_on="d_id", right_on="id")
    city_all_edge_df.dropna(inplace=True)

    city_all_edge_df = city_all_edge_df[["o_id", "d_id", "resi_x", "resi_y", "surface_distance", "trip"]].reset_index(drop=True)
    city_all_edge_df.columns = ["o_id", "d_id", "o_resi", "d_resi", "surface_distance", "trip"]
    return city_all_edge_df

def train():
    parameters = torch.normal(mean=0.5, std=0.1, size=[4])
    parameters.requires_grad = True

    train_edge_dfs, test_edge_dfs = [], []
    for city in train_cities:
        print("City:", city)
        city_all_edge_df = get_data(city, config.hours, config)
        train_edge_dfs.append(city_all_edge_df)
    train_edge_df = pd.concat(train_edge_dfs, ignore_index=True)
    # ---CAN WE DELETE TRAIN_EDGE_DFS FROM NOW ON?---

    # train
    x = torch.tensor(train_edge_df[["o_resi", "d_resi", "surface_distance"]].values, dtype=torch.float)
    y = torch.tensor(train_edge_df['trip'].values, dtype=torch.float)
    dataset = TensorDataset(x, y)
    data_loader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)

    epochs = 5
    optimizer = torch.optim.Adam([parameters], lr=1e-3)
    criterion = torch.nn.L1Loss()

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        train_loss = 0
        data_loader = tqdm(data_loader)
        for i, data in enumerate(data_loader):
            x_data = data[0]
            y_data = data[1]

            y_pred = gravity_func(x_data, parameters)
            loss = criterion(y_pred, y_data)

            train_loss += loss.item()

            data_loader.set_description("Epoch: {} Training Loss: {}".format(epoch, str(train_loss / (i + 1))))
            data_loader.refresh()
            # https://stackoverflow.com/questions/45862715/how-to-flush-tqdm-progress-bar-explicitly

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # save model
    checkpoint = {}
    checkpoint['model'] = parameters
    
    print("Model save path: ", model_path)
    
    
    torch.save(checkpoint, model_path)

def test(model_path):
    
    stat = os.stat(model_path)
    creation_time = stat.st_mtime
    human_readable_time = datetime.datetime.fromtimestamp(creation_time)
    formatted_time = human_readable_time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"Create time {formatted_time}.")
    
    
    
    model = torch.load(model_path)
    parameters = model['model']
    log = ''
    
    for city in test_cities:
        print("City:", city)
        log += city+'\n'
        
        city_all_edge_df = get_data(city, config.hours, config)
        print("input size:", len(city_all_edge_df))
        x = torch.tensor(city_all_edge_df[["o_resi", "d_resi", "surface_distance"]].values, dtype=torch.float)
        y = torch.tensor(city_all_edge_df['trip'].values, dtype=torch.float)
        y_pred = gravity_func(x, parameters)
        print("y_pred size:", len(y_pred))
        city_all_edge_df['pred_trip'] = pd.DataFrame(y_pred.detach())

        # Amplification
        total_group = []
        gt_sum_total = 0
        amp_sum_total = 0
        for group_df in city_all_edge_df.groupby('o_id'):
            group_df = group_df[1]
            gt_sum = group_df['trip'].sum()
            pred_sum = group_df['pred_trip'].sum()
            gt_sum_total += gt_sum
            if pred_sum == 0:
                group_df['pred_trip'] += 0.1
                pred_sum = group_df['pred_trip'].sum()
            group_df['ratio'] = group_df['pred_trip']/pred_sum
            group_df['amplified_trip'] = gt_sum*group_df['ratio']
            total_group.append(group_df)
            amp_sum_total += group_df['amplified_trip'].sum()
            # assert group_df['amplified_trip'].sum() // 1 == gt_sum // 1
        city_all_edge_df = pd.concat(total_group)
        y_pred = torch.tensor(city_all_edge_df['pred_trip'].values, dtype=torch.float)

        score1 = get_score_in_scope(y_pred, y)
        log += 'total r_squared, mae, rmse, cpc, pearson, ssim: '+str(score1)+'\n'
        score2 = get_score_in_scope(y_pred, y, [0, 10])
        log += '0, 10 r_squared, mae, rmse, cpc, pearson, ssim: '+str(score2)+'\n'
        score3 = get_score_in_scope(y_pred, y, [10, 100])
        log += '10, 100 r_squared, mae, rmse, cpc, pearson, ssim: '+str(score3)+'\n'
        score4 = get_score_in_scope(y_pred, y, [100, None])
        log += '> 100 r_squared, mae, rmse, cpc, pearson, ssim: '+str(score4)+'\n'
        log += '\n\n'
        

        # save
        city_all_edge_df['hour'] = 7
        result_df = city_all_edge_df[["hour", "o_id", "d_id", "amplified_trip"]]
        result_df.columns = ["hour", "o_id", "d_id", "pred_trip"]

        # result_df = result_df[result_df["pred_trip"] >= 5]
        print("output size:", len(result_df))

        
        result_df.to_csv(os.path.join(save_path, city+'.csv'), index=False)
        print("save in ", os.path.join(save_path, city+'.csv'))
    
    with open(os.path.join(config.result_path, 'tradition', md5, 'log_guangzhou'), 'w') as file:
        file.write(log)



if __name__ == "__main__":
    start_time = time.time()
    setup_seed(95)
    train_cities = ["北京市", "广州市",
                         "成都市", "南京市", "郑州市", "宁波市", "佛山市", "东莞市", "合肥市", "长沙市", "青岛市", "苏州市",
                         "厦门市", "济南市", "沈阳市", "大连市", "昆明市", "长春市", "温州市", "石家庄市", "贵阳市", "南昌市", "常州市", "哈尔滨市",
                         "乌鲁木齐市", "徐州市"]
    test_cities = ["上海市", "深圳市", "杭州市", "武汉市", "天津市", "徐州市", "无锡市",
                         "福州市", "南宁市", "兰州市","呼和浩特市"]
    test_cities = ["昆明市", "长春市"]
    # train_cities = ["北京市"]
    # test_cities = ["北京市"]
    config.hours = [7]
    print("Train cities:", train_cities)
    print("Test cities:", test_cities)
    
    
    config.ALL_EDGE_PATH = "/media/lai/27dbdec5-6518-47f1-b1af-159ffe3d1c08/2019/edge_2019_by_grid1000"
    

    
    # <<< Training Part <<<
    # model_name = "tradition_grid1000.pt"
    # model_path = os.path.join(config.result_path, 'tradition', model_name)  # Trained model will be saved in this dir.
    # train()
    
    # <<< Test Part <<<
    model_path = "/home/lai/ugnn/results/gegn_outputs/tradition_grid1000.pt"
    md5 = calculate_md5(model_path)
    print(f"md5: {md5}")
    save_path = os.path.join(config.result_path, 'tradition', md5)
    test(model_path)
