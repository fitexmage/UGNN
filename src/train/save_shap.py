# %%
import sys
import os
import shutil
script_dir = os.path.dirname("/home/lai/ugnn/src/train/cnn_ugnn_shap.ipynb")
project_path = os.path.dirname(os.path.dirname(script_dir))
project_name = "cnn_ugnn"
sys.path.append(project_path)
import shap


from torch.utils.data import DataLoader
from torch import optim 
from torch import nn
import pandas as pd
import torch
from tqdm import tqdm


from src.config.cnn_ugnn_config import CnnGegnConfig
from src.model.cnn_ugnn import CnnGegn
from src.utils.dataset import GegnDataset
import util
from src.preprocessor.base_preprocessor import (CoordPreprocessor, IdPreprocessor)
from src.preprocessor.cnn_ugnn_preprocessor import (
    preprocess,
    sample_every_epoch,
    test_preprocess,
    test_preprocess_,
    UgnnGridPreprocessor,
    UgnnEdgePreprocessor
)
from src.utils.model_util import get_score_in_scope


def setup_model(model):
    print("Model name:", config.model_name)
    print("Model version:", config.model_version)

    if torch.cuda.is_available():
        print("Device:", config.device)
        model = model.to(config.device)

    criterion = nn.L1Loss()
    # criterion = nn.MSELoss()
    # optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)
    
    # load all the things in the saved model
    if config.reuse_model:
        epoch = checkpoint["epoch"]
        step = checkpoint["step"]
        print(step)
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

# %%
def test(cities, model):
    
    if config.model_status == "test" and len(config.test_ids) != 0:
        test_df = test_df[test_df["o_id"].isin(config.test_ids)].reset_index(drop=True)
    elif config.model_status == "w_analysis":
        test_df = test_df[(test_df["o_id"].isin(config.w_analysis_ids) |
        (test_df["d_id"].isin(config.w_analysis_ids)))].reset_index(drop=True)

    md5 = util.calculate_md5(config.BEST_MODEL_SAVE_PATH)
    save_path = os.path.join(config.SAVE_PATH, 'cnn', md5)
    print(md5)
    # Create the folder path if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Specify the destination path (including the file name)
    destination_path = os.path.join(save_path, "{}_{}_best.pt".format(config.model_name, config.model_version))

    # Copy the file to the destination path
    shutil.copy(config.BEST_MODEL_SAVE_PATH, destination_path)
    
    scores = []
    log = ""
    log += str(config.__dict__)
    log += '\n\n\n'
    
    
    for city in cities:
        log += 'City: '+city+'\n'
        preprocessor = make_preprocessor([city])
        p_data = test_preprocess(*preprocessor)

        parameters = {
            "model_version": config.model_version,
            "lr": config.learning_rate,
            "sample_type": config.sample_type,
            "beta": config.beta,
        }
        print(parameters)
        
        model.eval()
        
        test_dataset = GegnDataset(p_data['grid']['grid_basic_dict'], p_data['grid']['grid_extra_dict'], p_data['test_edge'], config.one_hot_hour)
        if config.model_status == "test":
            test_loader = DataLoader(test_dataset, batch_size=config.infer_batch_size,
                                        shuffle=False)
        elif config.model_status == "w_analysis":
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        bg_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)
        bg_loader_iter = iter(bg_loader)
        shap_loader = DataLoader(test_dataset, batch_size=400, shuffle=False)
        
        shap_loader = tqdm(shap_loader)
        
        y_preds = []
        for i_batch, train_data in enumerate(shap_loader):
            # o_grid_x, d_grid_x, edge_x, external_x, y = train_data
            # because there will be different data inside train_data, we will not assign the name for each of them
            if torch.cuda.is_available():
                
                for j in range(len(train_data)):
                    train_data[j] = train_data[j].to(config.device)
                
                y_pred = model(*train_data[:-1])
                y_preds.append(y_pred.cpu().detach())
                
                
                # build an explainer using a token masker
                shap_batch = next(bg_loader_iter)
                for j in range(len(shap_batch)):
                    shap_batch[j] = shap_batch[j].to(config.device)
                explainer = shap.DeepExplainer(model, shap_batch[:-1])

                # explain the model's predictions on IMDB reviews
                # train_data = [i.unsqueeze(-1) for i in train_data[:-1]]
                shap_values = explainer.shap_values(train_data[:-1])
                break
    return shap_values
    
    
def make_preprocessor(cities):
    
    """
    Make preprocessor according to config.
    For training, the cities will involve various cities. However for testing,
    we load one dataset for one city, so the cities will ONLY contain one city.
    """
    
    coord_preprocessor = CoordPreprocessor(cities, config.CITY_CENTROID_PATH)
    id_preprocessor = IdPreprocessor(cities, config.split_frac)
    edge_preprocessor = UgnnEdgePreprocessor(
        cities, config.ALL_EDGE_PATH, checkpoint, config.one_hot_hour, config.hour
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

# %%
util.setup_seed(95)
util.get_gpu_memory_usage()


config = CnnGegnConfig()

print("Using "+config.device+"\n\n")
checkpoint = util.load_checkpoint(config)

print("Testing Model...")
print(f"Test cities: {config.test_cities}")


cities = config.test_cities        
preprocessor = make_preprocessor([cities[0]])
p_data = test_preprocess_(*preprocessor)

model = CnnGegn(
p_data["grid"]["grid_basic_dim"],
p_data["grid"]["grid_extra_dim"],
p_data["edge_dim"],
p_data['grid']["external_dim"],
config,
)

model, criterion, optimizer= setup_model(model)

# shap_values = test(cities, model)

# %%
bg_preprocessor = make_preprocessor(["北京市"])
bg_data = test_preprocess(*bg_preprocessor)

model.eval()

test_dataset = GegnDataset(p_data['grid']['grid_basic_dict'], p_data['grid']['grid_extra_dict'], p_data['test_edge'], config.one_hot_hour)
bg_dataset = GegnDataset(bg_data['grid']['grid_basic_dict'], bg_data['grid']['grid_extra_dict'], bg_data['test_edge'], config.one_hot_hour)

# %%
bg_loader = DataLoader(bg_dataset, batch_size=100, shuffle=True)
bg_loader_iter = iter(bg_loader)
shap_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)

shap_loader = tqdm(shap_loader)

y_preds = []
for i_batch, train_data in enumerate(shap_loader):
    # o_grid_x, d_grid_x, edge_x, external_x, y = train_data
    # because there will be different data inside train_data, we will not assign the name for each of them
    if torch.cuda.is_available():
        
        for j in range(len(train_data)):
            train_data[j] = train_data[j].to(config.device)
        
        y_pred = model(*train_data[:-1])
        y_preds.append(y_pred.cpu().detach())
        
        
        # build an explainer using a token masker
        shap_batch = next(bg_loader_iter)
        for j in range(len(shap_batch)):
            shap_batch[j] = shap_batch[j].to(config.device)
        explainer = shap.DeepExplainer(model, shap_batch[:-1])

        # explain the model's predictions on IMDB reviews
        # train_data = [i.unsqueeze(-1) for i in train_data[:-1]]
        shap_values = explainer.shap_values(train_data[:-1])
        break
import numpy as np
# Calculate means along the second axis (axis=1) for each array
means_list = [np.mean(arr, axis=0) for arr in shap_values]
# means_list[:4] = [np.mean(arr, axis=1) for arr in means_list[:4]]
# means_list[:4] = [np.mean(arr, axis=1) for arr in means_list[:4]]

# Display the means
for mean in means_list:
    print(mean.shape)

# %%
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


import matplotlib.pyplot as plt
# List of labels for each shape
shape_labels = [
    ['resi', 'work', 'road_length', 'nighttime_lights', 'total_building_area', 'subway_distance', 'bus_distance'],
    ['resi', 'work', 'road_length', 'nighttime_lights', 'total_building_area', 'subway_distance', 'bus_distance'],
    ['宗教设施用地_poi', '娱乐康体设施用地_poi', '医疗卫生用地_poi', '一类工业用地_poi',
       '文化设施用地_poi', '通信用地_poi', '体育用地_poi', '社会停车场用地_poi', '商业用地_poi',
       '商务用地_poi', '其他交通设施用地_poi', '其他服务设施用地_poi', '金融保险用地_poi', '教育科研用地_poi',
       '交通枢纽用地_poi', '环卫设施用地_poi', '行政办公用地_poi', '轨道交通线路用地_poi', '广场用地_poi',
       '公园绿地_poi', '公用设施营业网点用地_poi', '二类居住用地_poi', '物流仓储用地_poi', '宗教设施用地_aoi',
       '娱乐康体设施用地_aoi', '医疗卫生用地_aoi', '一类工业用地_aoi', '文化设施用地_aoi', '体育用地_aoi',
       '社会停车场用地_aoi', '商业用地_aoi', '商务用地_aoi', '其他交通设施用地_aoi', '其他服务设施用地_aoi',
       '教育科研用地_aoi', '交通枢纽用地_aoi', '行政办公用地_aoi', '广场用地_aoi', '公园绿地_aoi',
       '公用设施营业网点用地_aoi', '二类居住用地_aoi', '物流仓储用地_aoi'],
    ['宗教设施用地_poi', '娱乐康体设施用地_poi', '医疗卫生用地_poi', '一类工业用地_poi',
       '文化设施用地_poi', '通信用地_poi', '体育用地_poi', '社会停车场用地_poi', '商业用地_poi',
       '商务用地_poi', '其他交通设施用地_poi', '其他服务设施用地_poi', '金融保险用地_poi', '教育科研用地_poi',
       '交通枢纽用地_poi', '环卫设施用地_poi', '行政办公用地_poi', '轨道交通线路用地_poi', '广场用地_poi',
       '公园绿地_poi', '公用设施营业网点用地_poi', '二类居住用地_poi', '物流仓储用地_poi', '宗教设施用地_aoi',
       '娱乐康体设施用地_aoi', '医疗卫生用地_aoi', '一类工业用地_aoi', '文化设施用地_aoi', '体育用地_aoi',
       '社会停车场用地_aoi', '商业用地_aoi', '商务用地_aoi', '其他交通设施用地_aoi', '其他服务设施用地_aoi',
       '教育科研用地_aoi', '交通枢纽用地_aoi', '行政办公用地_aoi', '广场用地_aoi', '公园绿地_aoi',
       '公用设施营业网点用地_aoi', '二类居住用地_aoi', '物流仓储用地_aoi'],
    ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00', "surface_distance", "grid_to_grid_distance", "grid_to_grid_time"],
    ["城市人口", "总GDP", "行政区面积", "建城区面积", "lng", "lat"]
]
shape_labels = [
    ['O residence', 'O work', 'O road length', 'O nighttime lights', 'O total building area', 'o_subway_distance', 'o_bus_distance'],
    ['D residence', 'D work', 'D road length', 'D nighttime lights', 'D total building area', 'd_subway_distance', 'd_bus_distance'],
    ['o_宗教设施用地_poi',
 'O entertainment POI',
 'O 医疗卫生用地_poi',
 'o_一类工业用地_poi',
 'o_文化设施用地_poi',
 'o_通信用地_poi',
 'O sports POI',
 'O parking POI',
 'o_商业用地_poi',
 'O business POI',
 'o_其他交通设施用地_poi',
 'o_其他服务设施用地_poi',
 'O financial POI',
 'O educational POI',
 'o_交通枢纽用地_poi',
 'o_环卫设施用地_poi',
 'o_行政办公用地_poi',
 'O rail transit POI',
 'o_广场用地_poi',
 'o_公园绿地_poi',
 'o_公用设施营业网点用地_poi',
 'O residential POI',
 'o_物流仓储用地_poi',
 'o_宗教设施用地_aoi',
 'O entertainment AOI',
 'o_医疗卫生用地_aoi',
 'o_一类工业用地_aoi',
 'o_文化设施用地_aoi',
 'O sports AOI',
 'O parking AOI',
 'o_商业用地_aoi',
 'o_商务用地_aoi',
 'o_其他交通设施用地_aoi',
 'o_其他服务设施用地_aoi',
 'O educational AOI',
 'o_交通枢纽用地_aoi',
 'o_行政办公用地_aoi',
 'o_广场用地_aoi',
 'o_公园绿地_aoi',
 'o_公用设施营业网点用地_aoi',
 'O residential AOI',
 'o_物流仓储用地_aoi'],
['d_宗教设施用地_poi',
 'D entertainment POI',
 'd_医疗卫生用地_poi',
 'd_一类工业用地_poi',
 'd_文化设施用地_poi',
 'd_通信用地_poi',
 'D sports POI',
 'D parking POI',
 'd_商业用地_poi',
 'D business POI',
 'd_其他交通设施用地_poi',
 'D service POI',
 'D financial POI',
 'D educational POI',
 'd_交通枢纽用地_poi',
 'd_环卫设施用地_poi',
 'd_行政办公用地_poi',
 'D rail transit POI',
 'd_广场用地_poi',
 'd_公园绿地_poi',
 'd_公用设施营业网点用地_poi',
 'D residential POI',
 'd_物流仓储用地_poi',
 'd_宗教设施用地_aoi',
 'D entertainment AOI',
 'd_医疗卫生用地_aoi',
 'd_一类工业用地_aoi',
 'd_文化设施用地_aoi',
 'D sports AOI',
 'd_社会停车场用地_aoi',
 'd_商业用地_aoi',
 'd_商务用地_aoi',
 'd_其他交通设施用地_aoi',
 'd_其他服务设施用地_aoi',
 'D educational AOI',
 'd_交通枢纽用地_aoi',
 'd_行政办公用地_aoi',
 'd_广场用地_aoi',
 'd_公园绿地_aoi',
 'd_公用设施营业网点用地_aoi',
 'D residential AOI',
 'd_物流仓储用地_aoi'],
    ["surface distance", "route distance", "route time"],
    ["城市人口", "总GDP", "行政区面积", "建城区面积", "lng", "lat", "hour"]
]

# Plotting
# for i, (mean, labels) in enumerate(zip(means_list, shape_labels)):
#    fig, ax = plt.subplots(figsize=(10, 6))
#    x_values = np.arange(len(labels))
#    if i in [0,1,2,3]:
#       # ax.bar(x_values, mean[:,1,1], color='skyblue')
#       ax.bar(x_values, np.mean(mean, axis=(1, 2)), color='skyblue')
#    else:
#       ax.bar(x_values, mean, color='skyblue')
#    ax.set_xticks(x_values)
#    ax.set_xticklabels(labels, rotation=90, ha='right')
#    ax.set_xlabel('Categories')
#    ax.set_ylabel('Mean Values')
#    ax.set_title(f'Mean Values - Shape {i+1}')
#    plt.tight_layout()
#    plt.show()

# %%
def save_shap_v(shap_values, train_data, f_name):
    train_data = [i.cpu().numpy() for i in train_data]
    import pickle
    import os
    sp = (shap_values, train_data)
    with open(os.path.join("/home/lai/ugnn/imgs", f_name+".pickle"), "wb") as f:
        pickle.dump(sp, f)
# save_shap_v(shap_values, train_data, "真武庙shap")

# %%
# All means data
all_means = []
pos = []
all_labels = []
for labels in shape_labels:
   all_labels += labels
# Plotting
for i, (mean, labels) in enumerate(zip(means_list, shape_labels)):

   if i in [0,1,2,3]:
      all_means += mean[:,1,1].tolist()
   else:
      all_means += mean.tolist()
pos = np.arange(len(all_means))

# %%
# import numpy as np

# # Get indices that would sort the absolute values of all_means
# sorted_indices = np.argsort(np.abs(all_means))[::-1]

# # Take only the top k values
# top_k = 10
# top_k_indices = sorted_indices[:top_k]
# top_k_means = [all_means[i] for i in top_k_indices]
# top_k_labels = [all_labels[i] for i in top_k_indices]

# fig, ax = plt.subplots(figsize=(10, 8))

# # Set different colors for positive and negative values
# colors = ['red' if x > 0 else 'blue' for x in top_k_means]

# ax.barh(range(top_k), top_k_means, align='center', color=colors, edgecolor='none')

# ax.set_yticks(range(top_k), labels=top_k_labels)
# ax.invert_yaxis()  # labels read top-to-bottom
# ax.set_xlabel('Shap Values')
# ax.set_title(f'Top {top_k} Shap values')

# plt.show()

# %%
o_grid_basic = shap_values[0]
o_grid_basic = o_grid_basic[:, :, 1, 1]
d_grid_basic = shap_values[1]
d_grid_basic = d_grid_basic[:, :, 1, 1]
o_grid_extra = shap_values[2]
o_grid_extra = o_grid_extra[:, :, 1, 1]
d_grid_extra = shap_values[3]
d_grid_extra = d_grid_extra[:, :, 1, 1]
edge = shap_values[4]
city_metric = shap_values[5]

shap_features = np.concatenate((o_grid_basic, d_grid_basic, o_grid_extra, d_grid_extra, edge, city_metric), axis=1)
shap_features.shape

# %%
o_grid_basic = train_data[0]
o_grid_basic = o_grid_basic[:, :, 1, 1]
d_grid_basic = train_data[1]
d_grid_basic = d_grid_basic[:, :, 1, 1]
o_grid_extra = train_data[2]
o_grid_extra = o_grid_extra[:, :, 1, 1]
d_grid_extra = train_data[3]
d_grid_extra = d_grid_extra[:, :, 1, 1]
edge = train_data[4]
city_metric = train_data[5]
train_data_plot = torch.cat([o_grid_basic, d_grid_basic, o_grid_extra, d_grid_extra, edge, city_metric], dim=1)
train_data_plot.shape

# %%
def plot_fillter(shap_features, train_data_plot):
    # Create a boolean mask based on the condition
    mask = shap_features[:, 99] <= 2

    # Apply the mask to filter the arrays
    filtered_shap_features = shap_features[mask]
    filtered_train_data_plot = train_data_plot[mask]
    return filtered_shap_features, filtered_train_data_plot

filtered_shap_features, filtered_train_data_plot = plot_fillter(shap_features, train_data_plot.cpu().numpy())

# %%
feature_names = []
for i in shape_labels:
    feature_names += i
shap.summary_plot(filtered_shap_features, filtered_train_data_plot, feature_names=feature_names, show=False)
plt.savefig("shap_guomao.pdf")

# %%
# import pandas as pd
# df = pd.read_csv("/media/lai/27dbdec5-6518-47f1-b1af-159ffe3d1c08/2019/edge_2019_by_grid1000/北京市.csv")
# # # df = df.drop("Unnamed: 0", axis=1)

# guomao = df[df["o_id"]==116440039900040]
# zhenwumiao = df[df["o_id"]==116350039900040]
# huilongguan = df[df["o_id"]==116320040060040]
# # guomao = guomao[(guomao['trip'] >= 5) & (guomao['trip'] <= 15)]

# # # work = df[df['o_id'].isin(work)]
# # # guomao = df[df['o_id'].isin(guomao)]
# # # live = df[df['o_id'].isin(live)]
# # # guomao.to_csv("/media/lai/5d8d5be8-23c6-40b8-9d77-d15711f9ed67/2019/edge_filtered_2019/北京市国贸.csv", index=False)
# huilongguan.to_csv("/media/lai/27dbdec5-6518-47f1-b1af-159ffe3d1c08/2019/edge_2019_by_grid1000/北京市回龙观.csv", index=False)
# # len(live)

# %%
# guomao = pd.read_csv("/media/lai/5d8d5be8-23c6-40b8-9d77-d15711f9ed67/2019/edge_filtered_2019/北京市国贸.csv")
# # print(len(guomao))
# guomao = guomao[guomao["o_id"]==116440039900040]
# guomao = guomao[(guomao['trip'] >= 5) & (guomao['trip'] <= 15)]
# print(len(guomao))
# guomao.to_csv("/media/lai/27dbdec5-6518-47f1-b1af-159ffe3d1c08/2019/edge_2019_by_grid1000/北京市国贸116440039900040.csv", index=False)

# %%



