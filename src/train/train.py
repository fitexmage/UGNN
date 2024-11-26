import math
# from torchsummary import summary
import geopandas
import numpy as np
from torch import optim
import os
import time
import gc
from torch.utils.data import DataLoader
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm
# from tensorboardX import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt
# from util import setup_seed, add_to_id, analyze_distance, get_id, get_area_ids
from src.utils.model_util import *
from numpy import mean
import json
import util


matplotlib.use('agg')


class ODPredictor:
    def __init__(self):
        super(ODPredictor, self).__init__()

        self.normalized_all_grid_df = None
        self.normalized_indi_grid_df = None
        self.dynamic_normalized_grid_df = None
        self.grid_dict = None
        self.cities = None
        self.edge_dim = None
        self.config = None
        self.Dataset = None
        self.Model = None

        util.setup_seed(95)

    def load_checkpoint(self):
        if self.config.reuse_model:
            if self.config.model_status == "train":
                self.checkpoint = torch.load(self.config.MODEL_SAVE_PATH, map_location=self.config.device)
            else:
                self.checkpoint = torch.load(self.config.BEST_MODEL_SAVE_PATH, map_location=self.config.device)
        else:
            self.checkpoint = {}


    # grid
    def setup_grid_data(self):
        if self.config.model_status in ["train", "eval"]:
            self.cities = self.config.train_cities
        elif self.config.model_status == "test":
            self.cities = self.config.test_cities
        elif self.config.model_status == "w_analysis":
            self.cities = self.config.w_analysis_cities
        elif self.config.model_status == "infer":
            self.cities = self.config.infer_cities
        else:
            raise Exception

        self.normalized_grid_basic_df = []
        self.normalized_grid_extra_df = []

        for city in self.cities:
            """
        This function do a left join to join basic and poi,aoi data, get grid dimention,
        and we can also do feature selection(not yet)
        Finally, do above for all cities and concat.
        """
            city_grid_basic_df = pd.read_csv(os.path.join(self.config.GRID_BASIC_PATH, "{}.csv".format(city)))
            if self.config.grid_type == "land_use":
                city_grid_extra_df = pd.read_csv(os.path.join(self.config.GRID_LAND_USE_PATH, "{}.csv".format(city)))
            else:  # poi_aoi
                city_grid_extra_df = pd.read_csv(
                    os.path.join(self.config.GRID_POI_AOI_PATH, "{}.csv".format(city)))

            self.grid_basic_dim = city_grid_basic_df.shape[1] - 1  # remove id
            self.grid_extra_dim = city_grid_extra_df.shape[1] - 1  # remove id

            self.normalized_grid_basic_df.append(city_grid_basic_df)
            self.normalized_grid_extra_df.append(city_grid_extra_df)

        self.raw_metric_df = self.load_city_metric()

        self.normalized_grid_basic_df = pd.concat(self.normalized_grid_basic_df, ignore_index=True)
        self.normalized_grid_extra_df = pd.concat(self.normalized_grid_extra_df, ignore_index=True)
        self.normalized_grid_basic_df = self.normalize_df(self.normalized_grid_basic_df, ['id'], "grid_basic_scaler")
        self.normalized_grid_extra_df = self.normalize_df(self.normalized_grid_extra_df, ['id'], "grid_extra_scaler")

        self.normalized_metric_df = self.normalize_df(self.raw_metric_df, ["city"], "metric_scaler")
        self.preprocess_grid_data()

        print("Grid data loaded!")

    def load_city_metric(self):
        general_df = pd.read_csv(self.config.RAW_CITY_METRIC_PATH, encoding='GB18030')
        # print(general_df)
        city_metric_df = general_df[["城市", "2022城市人口", "2022年GDP", "2022行政区面积", "2022建城区面积"]]
        city_metric_df.columns = ["city", "城市人口", "总GDP", "行政区面积", "建城区面积"]

        city_division_gdf = geopandas.read_file(os.path.join(self.config.CITY_DIVISION_PATH))

        self.external_dim = city_metric_df.shape[1] + 2  # remove city; add lng, lat, and hour

        metric_dfs = []
        for row in np.array(city_metric_df):
            city = row[0]
            center = city_division_gdf[city_division_gdf["NAME"] == city].representative_point()
            metric_array = np.concatenate([row, center.x.values, center.y.values])
            metric_array = metric_array[np.newaxis, :]
            metric_df = pd.DataFrame(metric_array, columns=list(city_metric_df.columns) + ["lng", "lat"])
            metric_dfs.append(metric_df)
        return pd.concat(metric_dfs, ignore_index=True)

    def normalize_df(self, df, frozen_columns, scaler_name):
        """
        
        This function will frozen some columns according to frozen_columns, 
        normalize the df. If it's the 1st time, it will create a minmaxscaler and
        save it in checkpoint, next time it will call that from checkpoint directly.
        """
        frozen_df = df[frozen_columns]
        df = df.drop(columns=frozen_columns)

        if scaler_name in self.checkpoint:
            scaler = self.checkpoint[scaler_name]
            normalized_array = scaler.transform(df)
        else:
            scaler = preprocessing.MinMaxScaler()
            normalized_array = scaler.fit_transform(df)
            self.checkpoint[scaler_name] = scaler

        df = pd.DataFrame(normalized_array, columns=df.columns)
        df = pd.concat([frozen_df, df], axis=1)
        return df



    def preprocess_grid_data(self):
        """
                Enlarge grid by adding surrounding directions. e.g.
                              +-----+-----+-------+
                              | -1,0|..   | ..    |
                +---+         +-----+-----+-------+
                |0, 0| ------>| -1,0| 0,0 |1,0    |
                +---+         +-----+-----+-------+
                              | ..  |..   | ..    |
                              +-----+-----+-------+
                                9 directions(for self.config.grid_radius=2)
                """
        center_id_df = self.normalized_grid_basic_df[["id"]]
        grid_basic_dfs = []
        grid_extra_dfs = []
        for row in range(-self.config.grid_radius + 1, self.config.grid_radius):
            for col in range(-self.config.grid_radius + 1, self.config.grid_radius):
                id_df = util.add_to_id(center_id_df, [row, col])
                grid_basic_df = id_df.merge(self.normalized_grid_basic_df, how="left", on="id")
                grid_basic_df = grid_basic_df.drop(columns="id")
                grid_basic_dfs.append(grid_basic_df)
                grid_extra_df = id_df.merge(self.normalized_grid_extra_df, how="left", on="id")
                grid_extra_df = grid_extra_df.drop(columns="id")
                grid_extra_dfs.append(grid_extra_df)
        self.normalized_grid_basic_df = pd.concat([center_id_df] + grid_basic_dfs, axis=1)
        self.normalized_grid_basic_df = self.normalized_grid_basic_df.fillna(0)
        self.normalized_grid_extra_df = pd.concat([center_id_df] + grid_extra_dfs, axis=1)
        self.normalized_grid_extra_df = self.normalized_grid_extra_df.fillna(0)
        self.grid_to_dict()

    def dynamic_preprocess_grid_data(self):
        center_id_df = self.dynamic_normalized_grid_basic_df[["id"]]
        grid_basic_dfs = []
        grid_extra_dfs = []
        for row in range(-self.config.grid_radius + 1, self.config.grid_radius):
            for col in range(-self.config.grid_radius + 1, self.config.grid_radius):
                id_df = util.add_to_id(center_id_df, [row, col])
                new_basic_dfs = []
                new_extra_dfs = []
                dynamic_id_df = id_df.loc[id_df['id'].isin(np.array(center_id_df).flatten().tolist())]
                if dynamic_id_df.shape[0] > 0:
                    new_basic_dfs.append(
                        dynamic_id_df.merge(self.dynamic_normalized_grid_basic_df, how="left", on="id"))
                    new_extra_dfs.append(
                        dynamic_id_df.merge(self.dynamic_normalized_grid_extra_df, how="left", on="id"))

                origin_id_df = id_df.loc[~id_df['id'].isin(np.array(center_id_df).flatten().tolist())]
                if origin_id_df.shape[0] > 0:
                    new_basic_dfs.append(origin_id_df.merge(self.dynamic_normalized_grid_basic_df, how="left", on="id"))
                    new_extra_dfs.append(origin_id_df.merge(self.dynamic_normalized_grid_extra_df, how="left", on="id"))

                grid_basic_df = pd.concat(new_basic_dfs, ignore_index=True)
                grid_basic_df = grid_basic_df.fillna(0)
                grid_basic_df = grid_basic_df.drop(columns="id")
                grid_basic_dfs.append(grid_basic_df)

                grid_extra_df = pd.concat(new_extra_dfs, ignore_index=True)
                grid_extra_df = grid_extra_df.fillna(0)
                grid_extra_df = grid_extra_df.drop(columns="id")
                grid_extra_dfs.append(grid_extra_df)

        self.dynamic_normalized_grid_basic_df = pd.concat([center_id_df] + grid_basic_dfs, axis=1)
        self.dynamic_normalized_grid_extra_df = pd.concat([center_id_df] + grid_extra_dfs, axis=1)

        self.dynamic_grid_to_dict()

    def dynamic_grid_to_dict(self):
        self.dynamic_grid_basic_dict = {}
        self.dynamic_grid_extra_dict = {}

        for row in np.array(self.dynamic_normalized_grid_basic_df):
            gid = int(row[0])
            grid_data = row[1:]
            if self.config.model_name.startswith("cnn"):
                grid_data = grid_data.reshape([3, 3, self.grid_basic_dim]).T
            elif self.config.model_name.startswith("gat"):
                grid_data = grid_data.reshape(self.config.grid_basic_num, self.grid_basic_dim)
            self.dynamic_grid_basic_dict[gid] = torch.Tensor(grid_data)

        for row in np.array(self.dynamic_normalized_grid_extra_df):
            gid = int(row[0])
            grid_data = row[1:]
            if self.config.model_name.startswith("cnn"):
                grid_data = grid_data.reshape([3, 3, self.grid_extra_dim]).T
            elif self.config.model_name.startswith("gat"):
                grid_data = grid_data.reshape(self.config.grid_num, self.grid_extra_dim)
            self.dynamic_grid_extra_dict[gid] = torch.Tensor(grid_data)

    def grid_to_dict(self):
        self.grid_basic_dict = {}
        self.grid_extra_dict = {}

        for row in np.array(self.normalized_grid_basic_df):
            gid = int(row[0])
            grid_data = row[1:]
            if self.config.model_name.startswith("cnn"):
                grid_data = grid_data.reshape([3, 3, self.grid_basic_dim]).T
            elif self.config.model_name.startswith("gat"):
                grid_data = grid_data.reshape(self.config.grid_basic_num, self.grid_basic_dim)
            self.grid_basic_dict[gid] = torch.Tensor(grid_data)

        for row in np.array(self.normalized_grid_extra_df):
            gid = int(row[0])
            grid_data = row[1:]
            if self.config.model_name.startswith("cnn"):
                grid_data = grid_data.reshape([3, 3, self.grid_extra_dim]).T
            elif self.config.model_name.startswith("gat"):
                grid_data = grid_data.reshape(self.config.grid_num, self.grid_extra_dim)
            self.grid_extra_dict[gid] = torch.Tensor(grid_data)

    # edge
    def setup_train_edge_data(self):
        self.train_id_dict, self.valid_id_dict = {}, {}
        for city in self.cities:
            city_id_df = util.get_id(city)
            train_id_df, valid_id_df = train_test_split(city_id_df, test_size=self.config.split_frac, shuffle=True,
                                                        random_state=95)
            self.train_id_dict[city] = list(
                train_id_df[train_id_df["id"].isin(self.grid_basic_dict)].values.squeeze(-1))
            self.valid_id_dict[city] = list(
                valid_id_df[valid_id_df["id"].isin(self.grid_basic_dict)].values.squeeze(-1))
        self.valid_edge_df = self.load_valid_edge()
        # not put this inside the function, because valid_edge_df will be used in the evaluation part
        self.normalized_valid_edge_df = self.normalize_df(self.valid_edge_df, ["o_id", "d_id", "trip", "city"],
                                                          "edge_scaler")
        self.edge_dim = self.normalized_valid_edge_df.shape[
                            1] - 5  # remove hour, o_id, d_id, trip, city (hour is in metric)
        print("Edge data loaded!")

        self.load_valid_data()

    def setup_test_edge_data(self):
        self.edge_dim = 3

    def setup_infer_edge_data(self):
        self.edge_dim = 3

    def load_valid_edge(self):
        valid_edge_dfs = []
        for city in self.cities:
            iterator = pd.read_csv(os.path.join(self.config.ALL_EDGE_PATH, "{}.csv".format(city)), chunksize=10000000)
            city_valid_edge_dfs = []
            for edge_df in iterator:
                edge_df = edge_df[edge_df["o_id"].isin(self.valid_id_dict[city])]
                city_valid_edge_dfs.append(edge_df)
            city_valid_edge_df = pd.concat(city_valid_edge_dfs, ignore_index=True)
            city_valid_edge_df.loc[:, "city"] = city
            valid_edge_dfs.append(city_valid_edge_df)
        valid_edge_df = pd.concat(valid_edge_dfs, ignore_index=True)
        return valid_edge_df

    # dataset
    def load_dataset(self, edge_df):
        return self.Dataset(self.grid_basic_dict, self.grid_extra_dict, edge_df, False)  # False means no one_hot_hour

    def load_dynamic_dataset(self, edge_df):
        # dynamic_basic_dict = dict(self.dict)
        # dynamic_dict.update(self.dynamic_grid_dict)
        return self.Dataset(self.dynamic_grid_basic_dict, self.dynamic_grid_extra_dict, edge_df)

    def load_train_data(self, epoch):
        if epoch % self.config.sample_gap == 0:
            sample_start_time = time.time()
            if self.config.sample_type == 'normal':
                sampled_edge_df = self.load_sampled_edge()
            elif self.config.sample_type == 'diff':
                sampled_edge_df = self.load_differential_sampled_edge(beta=self.config.beta)
            else:
                raise Exception
            # sampled_edge_df = pd.concat(sampled_edge_df.values())
            sampled_edge_df = sampled_edge_df.reset_index(drop=True)
            self.normalized_train_edge_df = self.normalize_df(sampled_edge_df, ["o_id", "d_id", "trip", "city"],
                                                              "edge_scaler")
            print("Sampled training data loaded in {} min!".format((time.time() - sample_start_time) / 60))
        # must do merging here or it will cost more memory
        # moreover, merging can reset index
        self.train_df = self.normalized_train_edge_df.merge(self.normalized_metric_df, on="city")

        if self.config.do_augmentation:
            # data augmentation for each epoch
            augmented_columns = ["城市人口", "总GDP", "行政区面积", "建城区面积", "lng", "lat"]
            bias_df = pd.DataFrame(np.random.normal(0, 0.01, [self.train_df.shape[0], len(augmented_columns)]),
                                   columns=augmented_columns)
            augmented_df = self.train_df[augmented_columns] + bias_df
            augmented_df[augmented_df < 0] = 0
            augmented_df[augmented_df > 1] = 1
            self.train_df[augmented_columns] = augmented_df
        self.train_dataset = self.load_dataset(self.train_df)
        # pd.set_option("display.max_columns", None)
        # print(self.train_df.head())
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.train_batch_size, shuffle=True)

    def load_differential_sampled_edge(self, threshold=100, step=1, alpha=1.0, beta=0.3, hour=[7]):
        edge = []
        for city in self.cities:
            # Read it all at once?
            sub_edge = []
            iterator = pd.read_csv(os.path.join(self.config.ALL_EDGE_PATH, "{}.csv".format(city)), chunksize=10000000)
            for df in iterator:
                df = df[df["o_id"].isin(self.train_id_dict[city])]
                pure_df = df.dropna()
                assert (len(df) == len(pure_df))
                max_trip = min(math.ceil(df.max()['trip']), threshold)
                min_trip = max(math.floor(df.min()['trip']), 0)
                max_weight = 0
                total_num = len(df)
                d_edges = []
                sub_weights = []
                class_sizes = []
                non_zero = 0.01

                for i in range(min_trip, max_trip):
                    edges = df[(df['trip'] >= i) & (df['trip'] < i + 1)]
                    trip_frequency = len(edges)
                    d_edges.append(edges)
                    class_sizes.append(len(edges))
                    # POW?
                    # IF
                    if trip_frequency != 0:
                        sub_weight = alpha / (pow(len(edges), beta))
                    else:
                        sub_weight = 0  # Zero will not affect calculation below
                    max_weight = max(max_weight, sub_weight)
                    sub_weights.append(sub_weight)

                total_weight = sum(sub_weights)
                avg_weight = mean(sub_weights)

                for i in range(len(sub_weights)):
                    # divide by avg?
                    sub_weights[i] = min(1.0, sub_weights[i] / avg_weight)
                
                # Rest
                class_sizes.append(len(df[df['trip'] >= threshold]))
                sub_weights.append(1.0)
                weight_df = pd.DataFrame(sub_weights, columns=["weight"])
                weight_df['size'] = class_sizes
                # weight_df.to_csv("./city_weights/weights_{}.csv".format(city))
                for i in range(len(d_edges)):
                    d_edges[i] = d_edges[i].sample(int(sub_weights[i] * len(d_edges[i])), ignore_index=True)
                city_sampled = pd.concat(d_edges, ignore_index=True)
                city_sampled = pd.concat([city_sampled, df[df['trip'] >= threshold]], ignore_index=True)
                city_sampled.loc[:, "city"] = city
                sub_edge.append(city_sampled)
            sub_edge = pd.concat(sub_edge)
            edge.append(sub_edge)
        edge = pd.concat(edge)

        return edge

    
    def load_sampled_edge(self, threshold=1, sample_rate=0.1):
        sampled_edge_dfs = []
        for city in self.cities:
            iterator = pd.read_csv(os.path.join(self.config.ALL_EDGE_PATH, "{}.csv".format(city)), chunksize=10000000)
            city_sampled_edge_dfs = []
            for edge_df in iterator:
                # Hour?
                edge_df = edge_df[edge_df["o_id"].isin(self.train_id_dict[city])]
                
                # Duplication?
                small_edge_df = edge_df[edge_df["trip"] <= threshold]
                small_edge_df = small_edge_df.sample(round(small_edge_df.shape[0] * sample_rate))
                
                big_edge_df = edge_df[edge_df["trip"] > threshold]
                
                sampled_edge_df = pd.concat([small_edge_df, big_edge_df], ignore_index=True)
                city_sampled_edge_dfs.append(sampled_edge_df)
            
            city_sampled_edge_df = pd.concat(city_sampled_edge_dfs, ignore_index=True)
            city_sampled_edge_df.loc[:, "city"] = city
            sampled_edge_dfs.append(city_sampled_edge_df)
        sampled_edge_df = pd.concat(sampled_edge_dfs, ignore_index=True)
        return sampled_edge_df

    def load_valid_data(self):
        self.valid_df = self.normalized_valid_edge_df.merge(self.normalized_metric_df, on="city")
        self.valid_dataset = self.load_dataset(self.valid_df)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.config.infer_batch_size, shuffle=False)

    # train and test
    def prepare_train(self):
        self.epoch = 0
        self.step = 0
        self.best_loss = None

        if self.config.model_status == "train" and self.config.save_tensorboard:
            self.writer = SummaryWriter(self.config.TENSORBOARD_PATH)

    def setup_model(self):
        print("Model name:", self.config.model_name)
        print("Model version:", self.config.model_version)
        self.model = self.Model(self.grid_basic_dim,
                                self.grid_extra_dim,
                                self.edge_dim,
                                self.external_dim,
                                self.config)

        if torch.cuda.is_available():
            print("Device:", self.config.device)
            self.model = self.model.to(self.config.device)

        self.criterion = nn.L1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        # load all the things in the saved model
        if self.config.reuse_model:
            self.epoch = self.checkpoint["epoch"]
            self.step = self.checkpoint["step"]
            self.model.load_state_dict(self.checkpoint["model"])
            self.optimizer.load_state_dict(self.checkpoint["optimizer"])
            self.best_loss = self.checkpoint["best_loss"]
        else:
            for name, param in self.model.named_parameters():
                if ".bn" not in name:
                    if ".weight" in name:
                        torch.nn.init.kaiming_normal_(param.data, mode="fan_out", nonlinearity="leaky_relu")
                    else:
                        torch.nn.init.constant_(param.data, 0)

    def prepare_infer(self):
        self.city_edges_dict = self.load_differential_sampled_edge()
        self.city_gid_to_co_dict = {}
        for city in self.cities:
            city_dict = {}
            gid_to_co_df = pd.read_csv(os.path.join(self.config.CITY_ID_TO_CO_PATH, "{}.csv".format(city)))
            for row in gid_to_co_df.itertuples():
                city_dict[getattr(row, "id")] = (getattr(row, 'lng'), getattr(row, 'lat'))
            self.city_gid_to_co_dict[city] = city_dict

    def train(self):
        parameters = {"model_version": self.config.model_version, "lr": self.config.learning_rate,
                      "sample_type": self.config.sample_type, "beta": self.config.beta,
                      # "grid_type": self.config.grid_type, "o_grid_cnn_layer_channels": self.config.o_grid_cnn_layer_channels,
                      # "o_grid_fcnn_layer_channels": self.config.o_grid_fcnn_layer_channels,
                      # "o_grid_extra_cnn_layer_channels": self.config.o_grid_extra_cnn_layer_channels,
                      # "o_grid_extra_fcnn_layer_channels": self.config.o_grid_extra_fcnn_layer_channels,
                      # "d_grid_cnn_layer_channels": self.config.d_grid_cnn_layer_channels,
                      # "d_grid_fcnn_layer_channels": self.config.d_grid_fcnn_layer_channels,
                      # "d_grid_extra_cnn_layer_channels": self.config.d_grid_extra_cnn_layer_channels,
                      # "d_grid_extra_fcnn_layer_channels": self.config.d_grid_extra_fcnn_layer_channels,
                      # "edge_layer_channels": self.config.edge_layer_channels,
                      # "external_layer_channels": self.config.external_layer_channels,
                      # "out_layer_channels": self.config.out_layer_channels,
                      # "note":self.config.note
                      }
        # For some models, we don't have that much info, so we comment.
        print(parameters)
        para_json = json.dumps(parameters)

        f1 = open(os.path.join(self.config.SAVE_PATH,
                               "{}_{}_paras.json".format(self.config.model_name, self.config.model_version)), 'w')
        f1.write(para_json)
        f1.close()
        for epoch in range(self.config.epoch):
            torch.cuda.empty_cache()
            self.model.train()

            train_loss = 0

            self.load_train_data(epoch)  # repick train data every epoch
            train_loader = tqdm(self.train_loader)
            for i, train_data in enumerate(train_loader):
                # o_grid_x, d_grid_x, edge_x, external_x, y = train_data
                # because there will be different data inside train_data, we will not assign the name for each of them
                if torch.cuda.is_available():
                    for j in range(len(train_data)):
                        train_data[j] = train_data[j].to(self.config.device)

                y_pred = self.model(*train_data[:-1])

                loss = self.criterion(y_pred, train_data[-1])
                train_loss += loss.item()

                if (i + 1) % 100 == 0:
                    train_loader.set_description("Epoch: {} Step: {} Training Loss: {}".format(
                        self.epoch,
                        self.step,
                        str(train_loss / (i + 1))))
                    train_loader.refresh()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.step += 1

            train_loss = train_loss / len(self.train_loader)

            if self.config.save_tensorboard:
                self.writer.add_scalar("training loss", train_loss, global_step=self.step)
                self.writer.flush()

            self.epoch += 1
            if (epoch + 1) % self.config.eval_gap == 0:
                if self.config.save_training_model:
                    print(f"epoch{epoch}, saving model in {self.config.MODEL_SAVE_PATH}.")
                    self.save_model(self.config.MODEL_SAVE_PATH)
                self.eval()
            

    def eval(self):
        self.model.eval()

        valid_loader = tqdm(self.valid_loader)
        y_preds = []

        for i, valid_data in enumerate(valid_loader):
            torch.cuda.empty_cache()
            for j in range(len(valid_data)):
                valid_data[j] = valid_data[j].to(self.config.device)

            y_pred = self.model(*valid_data[:-1])
            y_preds.append(y_pred.cpu().detach())

        y_pred = torch.cat(y_preds, dim=0)
        y_gt = torch.Tensor(self.valid_edge_df[["trip"]].values).squeeze(-1)

        # result_df = pd.concat([self.valid_edge_df[["city", "hour", "o_id", "d_id", "trip", "surface_distance"]], pd.DataFrame(np.array(y_pred))], axis=1)
        # result_df.columns = ["city", "hour", "o_id", "d_id", "gt_trip", "surface_distance", "pred_trip"]
        #
        # y_pred = torch.Tensor(result_df["pred_trip"].values)
        # y_gt = torch.Tensor(result_df["gt_trip"].values)

        r_squared, mae, rmse, ssi, cpc, pearson, ssim = get_score_in_scope(y_pred, y_gt)

        eval_loss = self.criterion(y_pred, y_gt)
        if self.config.model_status == "train":
            if self.config.save_tensorboard:
                self.writer.add_scalar("evaluation loss", eval_loss, global_step=self.step)
                self.writer.flush()

            if self.best_loss is None or self.best_loss > eval_loss:
                self.best_loss = eval_loss

                self.save_model(self.config.BEST_MODEL_SAVE_PATH)
                print("Best Model saved!")
                f1 = open(os.path.join(self.config.SAVE_PATH,
                                       "{}_{}_eval.json".format(self.config.model_name, self.config.model_version)),
                          'a')
                f1.write("Best model:\nr_squared:{}, mae:{}, rmse:{}, ssi:{}, cpc:{}, pearson, ssim:{}\n".format(r_squared, mae, rmse, ssi, cpc, pearson, ssim))
                f1.close()
            print()

    def test_custom(self, city="北京市"):
        scores = []
        self.model.eval()
        print("City:", city)

        test_custom_ids_df = pd.read_csv(self.config.CUSTOM_EDGE_PATH)
        self.city_test_edge_df = pd.read_csv(os.path.join(self.config.ALL_EDGE_PATH, "{}.csv".format(city)))
        self.city_test_edge_df.loc[:, "city"] = city
        print("self.city_test_edge_df.:\n", self.city_test_edge_df)
        test_edge_df = self.city_test_edge_df[
            self.city_test_edge_df['o_id'].isin(list(test_custom_ids_df['id'])) | self.city_test_edge_df['d_id'].isin(
                list(test_custom_ids_df['id']))]
        test_edge_df = test_edge_df.reset_index(drop=True)
        print("test_edge_df:\n", test_edge_df)
        test_edge_df = test_edge_df.drop_duplicates()
        print("test_edge_df cols:\n", test_edge_df.columns)
        self.normalized_test_edge_df = self.normalize_df(test_edge_df, ["o_id", "d_id", "trip", "city"],
                                                         "edge_scaler")

        test_df = self.normalized_test_edge_df.merge(self.normalized_metric_df, on="city")
        self.test_dataset = self.load_dataset(test_df)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.config.infer_batch_size, shuffle=False)
        test_loader = tqdm(self.test_loader)
        y_preds = []
        for i, test_data in enumerate(test_loader):
            for j in range(len(test_data)):
                test_data[j] = test_data[j].to(self.config.device)
            y_pred = self.model(*test_data[:-1])
            y_preds.append(y_pred.cpu().detach())
        y_pred = torch.cat(y_preds, dim=0)
        y_pred[y_pred < 0] = 0.0
        print(pd.DataFrame(y_pred))
        assert (len(test_edge_df) == len(y_pred))
        result_df = pd.concat([test_edge_df[["city", "hour", "o_id", "d_id", "trip", "surface_distance"]],
                               pd.DataFrame(y_pred)], axis=1)
        result_df.columns = ["city", "hour", "o_id", "d_id", "gt_trip", "surface_distance", "pred_trip"]
        # result_df.to_csv(os.path.join(self.config.TEST_CUSTOM_RESULT_PATH, "{}_tongzhou_raw.csv".format(city)), index=False)

        # score
        score = self.get_score(result_df, city)
        scores.append(score)

        # test result
        result_df = result_df[["hour", "o_id", "d_id", "pred_trip"]]
        result_df.columns = ["hour", "o_id", "d_id", "trip"]
        result_df.to_csv(os.path.join(self.config.TEST_CUSTOM_RESULT_PATH, "{}_tongzhou.csv".format(city)), index=False)

        del self.city_test_edge_df, self.normalized_test_edge_df, self.test_dataset, self.test_loader
        del y_pred, y_preds, result_df
        gc.collect()

        score_df = pd.DataFrame(scores, columns=["city"] + ["{}_{}".format(metric, scope)
                                                            for scope in ["total", "0", "10", "100"]
                                                            for metric in ["mae", "rmse", "cpc", "pearson"]])
        score_df.to_csv(os.path.join(self.config.TEST_PATH, "custom_score.csv"), index=False)

    def test(self):
        
        log = ""
        md5 = util.calculate_md5(self.config.BEST_MODEL_SAVE_PATH)
        save_path = os.path.join(self.config.SAVE_PATH, 'fcnn', md5)
        print(md5)
        
        scores = []
        self.model.eval()

        for city in self.cities:
            log += city
            log += "\n"
            print("City:", city)

            self.city_test_edge_df = pd.read_csv(os.path.join(self.config.ALL_EDGE_PATH, "{}.csv".format(city)))
            
            # <<< Fillter <<<
            print("Using hour 7")
            self.city_test_edge_df = self.city_test_edge_df[self.city_test_edge_df["hour"].isin([7])]  # filter
            
            print("input size:", len(self.city_test_edge_df))
            self.city_test_edge_df.loc[:, "city"] = city
            self.normalized_test_edge_df = self.normalize_df(self.city_test_edge_df.reset_index(drop=True),
                                                             ["o_id", "d_id", "trip", "city"],
                                                             "edge_scaler")

            test_df = self.normalized_test_edge_df.merge(self.normalized_metric_df, on="city")
            # pd.set_option("display.max_columns", None)
            # print(test_df[:1000])
            if self.config.model_status == "test" and len(self.config.test_ids) != 0:
                test_df = test_df[test_df["o_id"].isin(self.config.test_ids)].reset_index(drop=True)
            elif self.config.model_status == "w_analysis":
                test_df = test_df[(test_df["o_id"].isin(self.config.w_analysis_ids) |
                                   (test_df["d_id"].isin(self.config.w_analysis_ids)))].reset_index(drop=True)
            self.test_dataset = self.load_dataset(test_df)
            if self.config.model_status == "test":
                self.test_loader = DataLoader(self.test_dataset, batch_size=self.config.infer_batch_size,
                                              shuffle=False)
            elif self.config.model_status == "w_analysis":
                self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)

            test_loader = tqdm(self.test_loader)
            y_preds = []
            for i, test_data in enumerate(test_loader):
                for j in range(len(test_data)):
                    test_data[j] = test_data[j].to(self.config.device)
                y_pred = self.model(*test_data[:-1])
                y_preds.append(y_pred.cpu().detach())
            y_preds = torch.cat(y_preds, dim=0)
            y_preds[y_preds < 0] = 0.0
            result_df = pd.concat(
                [self.city_test_edge_df[["city", "hour", "o_id", "d_id", "trip", "surface_distance"]].reset_index(drop=True),
                        pd.DataFrame(y_preds)], axis=1
            ).reset_index(drop=True)
            result_df.columns = ["city", "hour", "o_id", "d_id", "gt_trip", "surface_distance", "pred_trip"]
            score = util.get_score(result_df, city)
            scores.append(score)
            
            score1 = get_score_in_scope(y_preds, torch.Tensor(result_df['gt_trip'].values).squeeze(-1))
            log += 'total r_squared, mae, rmse, ssi, cpc, pearson: '+str(score1)+'\n'
            score2 = get_score_in_scope(y_preds, torch.Tensor(result_df['gt_trip'].values).squeeze(-1), [0, 10])
            log += '0, 10 r_squared, mae, rmse, ssi, cpc, pearson: '+str(score2)+'\n'
            score3 = get_score_in_scope(y_preds, torch.Tensor(result_df['gt_trip'].values).squeeze(-1), [10, 100])
            log += '10, 100 r_squared, mae, rmse, ssi, cpc, pearson: '+str(score3)+'\n'
            score4 = get_score_in_scope(y_preds, torch.Tensor(result_df['gt_trip'].values).squeeze(-1), [100, None])
            log += '> 100 r_squared, mae, rmse, ssi, cpc, pearson: '+str(score4)+'\n'
            log += '\n\n'

            result_df.to_csv(
                os.path.join(save_path, city+'.csv'),
                index=False)

        with open(os.path.join(self.config.SAVE_PATH, 'fcnn', md5, 'log_res'), 'w') as file:
            file.write(log)

    # def custom_infer(self, city, update_list):
    #     extra_col_dict = {"living_area": "二类居住用地", "work_area": "行政办公用地", "industrial_area": "一类工业用地", "bussiness_area": "商业用地"}
    #     self.model.eval()
    #     print("Preprocessing data...")
    #     grid_basic_df = self.raw_grid_basic_df_dict[city]
    #     grid_extra_df = self.raw_grid_extra_df_dict[city]

    #     grid_ids = []
    #     for update in update_list:
    #         update_gid = update['grid_id']
    #         grid_ids.append(update_gid)
    #         for k, v in update.items():
    #             if k == "grid_id":
    #                 continue
    #             if k in extra_col_dict:
    #                 extra_k = extra_col_dict[k]
    #                 grid_extra_df.loc[grid_extra_df["id"]==update_gid, extra_k] *= float(v)
    #             else:
    #                 grid_basic_df.loc[grid_extra_df["id"]==update_gid, k] *= float(v)

    #     # self.normalized_all_grid_df = self.normalize_df(self.raw_all_grid_df, ["id"], "grid_scaler")
    #     self.normalized_grid_basic_df = self.normalize_df(grid_basic_df, ['id'], "grid_basic_scaler")
    #     self.normalized_grid_extra_df = self.normalize_df(grid_extra_df, ['id'], "grid_extra_scaler")
    #     print("normalized_grid_extra_df shape:", self.normalized_grid_extra_df.shape)
    #     self.normalized_metric_df = self.normalize_df(self.raw_metric_df, ["city"], "metric_scaler")
    #     self.preprocess_grid_data()
    #     print("Preprocess finished")

    #     city_edges = self.city_edges_dict[city]
    #     infer_edge_df = city_edges[city_edges['o_id'].isin(grid_ids).isin(grid_ids) | city_edges['d_id'].isin(grid_ids)]
    #     infer_edge_df.reset_index(inplace=True, drop=True)
    #     infer_dataset = self.load_dataset(infer_edge_df)
    #     infer_loader = DataLoader(infer_dataset, batch_size=self.config.infer_batch_size, shuffle=False)
    #     infer_loader = tqdm(infer_loader)
    #     y_preds = []
    #     for i, test_data in enumerate(infer_loader):
    #         for j in range(len(test_data)):
    #             test_data[j] = test_data[j].to(self.config.device)
    #         y_pred = self.model(*test_data[:-1])
    #         y_preds.append(y_pred.cpu().detach())
    #     y_pred = torch.cat(y_preds, dim=0)
    #     y_pred[y_pred < 0] = 0.0
    #     infer_edge_df['pred_trip'] = y_pred
    #     infer_edge_df.fillna(0, inplace=True)
    #     result_df = infer_edge_df[["city", "hour", "o_id", "d_id", "trip", "pred_trip"]]
    #     return result_df

    def infer(self):
        # pd.set_option('display.max_columns', None)
        self.model.eval()
        for city in self.cities:
            print("City:", city)
            city_infer_edge_df = pd.read_csv(os.path.join(self.config.ALL_EDGE_PATH, "{}.csv".format(city)))
            city_infer_edge_df.loc[:, "city"] = city

            for area in self.config.area_scope_dict[city].keys():
                print("Area:", area)
                area_ids = util.get_area_ids(*self.config.area_scope_dict[city][area])
                infer_edge_df = city_infer_edge_df[
                    city_infer_edge_df['o_id'].isin(area_ids) | city_infer_edge_df['d_id'].isin(area_ids)]
                infer_edge_df.reset_index(inplace=True, drop=True)
                # print("infer_edge_df:\n", infer_edge_df)
                change_title = self.config.area_change_dict[city][area]
                # print("change title:", change_title)
                dynamic_grid_basic_df = self.raw_grid_basic_df.copy()
                dynamic_grid_basic_df[change_title] = dynamic_grid_basic_df[dynamic_grid_basic_df['id'].isin(area_ids)][
                                                          change_title] * 0.5

                self.dynamic_normalized_grid_basic_df = self.normalize_df(dynamic_grid_basic_df, ["id"],
                                                                          "grid_basic_scaler")
                self.dynamic_normalized_grid_extra_df = self.normalized_origin_grid_extra_df.copy()
                self.dynamic_preprocess_grid_data()
                # print("dynamic preprocess finished")
                infer_df = self.normalize_df(infer_edge_df, ["o_id", "d_id", "trip", "city"], "edge_scaler")
                # print("nomalized:\n", infer_df)
                infer_df = infer_df.merge(self.normalized_metric_df, on="city")
                infer_dataset = self.load_dynamic_dataset(infer_df)
                infer_loader = DataLoader(infer_dataset, batch_size=self.config.infer_batch_size, shuffle=False)
                infer_loader = tqdm(infer_loader)
                y_preds = []
                for i, test_data in enumerate(infer_loader):
                    for j in range(len(test_data)):
                        test_data[j] = test_data[j].to(self.config.device)
                    y_pred = self.model(*test_data[:-1])
                    y_preds.append(y_pred.cpu().detach())
                y_pred = torch.cat(y_preds, dim=0)
                y_pred[y_pred < 0] = 0.0
                infer_edge_df['pred_trip'] = y_pred
                infer_edge_df.fillna(0, inplace=True)
                # result_df = pd.concat([infer_edge_df[["city", "hour", "o_id", "d_id", "trip", "surface_distance"]],
                #                        pd.DataFrame(y_pred)], axis=1)
                # print(result_df)
                result_df = infer_edge_df[["city", "hour", "o_id", "d_id", "trip", "pred_trip"]]

                # result_df.columns = ["hour", "o_id", "d_id", "trip"]
                result_df.to_csv(
                    os.path.join(self.config.INFER_RESULT_PATH,
                                 "{}_{}_{}{}.csv".format(city, area, self.config.model_name, self.config.save_suffix)),
                    index=False)

    def save_model(self, path):
        self.checkpoint["epoch"] = self.epoch
        self.checkpoint["step"] = self.step
        self.checkpoint["model"] = self.model.state_dict()
        self.checkpoint["optimizer"] = self.optimizer.state_dict()
        self.checkpoint["best_loss"] = self.best_loss

        torch.save(self.checkpoint, path)

    def get_score(self, result_df, city=None, hour=None):
        if hour:
            result_df = result_df[result_df["hour"] == hour]

        y_pred = torch.Tensor(result_df["pred_trip"].values)
        y_gt = torch.Tensor(result_df["gt_trip"].values)

        if city:
            score = [city]
        else:
            score = [hour]
        get_score_in_scope(y_pred, y_gt, None, score)
        get_score_in_scope(y_pred, y_gt, [0, 10], score)
        get_score_in_scope(y_pred, y_gt, [10, 100], score)
        get_score_in_scope(y_pred, y_gt, [100, None], score)
        return score

    def get_score_by_city_hour(self, result_df, city):
        scores = []
        for hour in range(0, 24):
            score = self.get_score(result_df, hour=hour)
            scores.append(score)
        score_df = pd.DataFrame(scores, columns=["hour"] + ["{}_{}".format(metric, scope)
                                                            for scope in ["all", "0", "10", "100"]
                                                            for metric in ["mae", "rmse", "cpc", "pearson"]])
        score_df.to_csv(
            os.path.join(self.config.TEST_RESULT_BY_HOUR_PATH, "{}_{}.csv".format(city, self.config.model_name)),
            index=False)

    def plot_scatter_graph(self, city, result_df):
        y_pred = result_df["pred_trip"].values
        y_pred_list = np.round(y_pred).tolist()
        y_gt = result_df["gt_trip"].values
        y_gt_list = np.round(y_gt).tolist()

        # 绘制流量对比图
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.scatter(y_gt_list, y_pred_list, c='darkgray', s=5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_aspect(1)
        plt.xlim(1, max(max(y_gt_list), max(y_pred_list)))
        plt.ylim(1, max(max(y_gt_list), max(y_pred_list)))
        fig.savefig(os.path.join(self.config.TEST_SCATTER_PATH, "{}.jpg".format(city)))
        plt.close()

    def get_distance_analysis(self, city, result_df):
        gt_df = result_df[["trip", "surface_distance"]].copy(deep=True)
        # gt_df.columns = ["trip", "surface_distance"]
        pred_df = result_df[["pred_trip", "surface_distance"]].copy(deep=True)
        pred_df.columns = ["trip", "surface_distance"]

        gt_distance_df, gt_average_distance = util.analyze_distance(gt_df)
        pred_distance_df, pred_average_distance = util.analyze_distance(pred_df)

        plt.plot(gt_distance_df["surface_distance"], gt_distance_df["trip"], "b", label="gt")
        plt.plot(pred_distance_df["surface_distance"], pred_distance_df["trip"], "r", label="test")
        plt.legend(loc=1)
        plt.savefig(os.path.join(self.config.TEST_DISTANCE_PATH, "{}.jpg".format(city)))
        plt.close()

        plt.plot(gt_distance_df["surface_distance"], gt_distance_df["trip_prop"], "b", label="gt")
        plt.plot(pred_distance_df["surface_distance"], pred_distance_df["trip_prop"], "r", label="test")
        plt.legend(loc=1)
        plt.savefig(os.path.join(self.config.TEST_DISTANCE_PATH, "{}_prop.jpg".format(city)))
        plt.close()

        trip_prop_df = pd.concat([gt_distance_df[["trip_prop"]], pred_distance_df[["trip_prop"]]], axis=1)
        coincidence = trip_prop_df.min(axis=1).sum() / trip_prop_df.max(axis=1).sum()
        compare_data_df = pd.DataFrame([[gt_average_distance, coincidence], [pred_average_distance, coincidence]],
                                       columns=["average_distance", "coincidence"])
        compare_data_df.to_csv(os.path.join(self.config.TEST_DISTANCE_PATH, "{}.csv".format(city)), index=False)

    def get_diff(self, city, result_df):
        result_df["diff"] = ((result_df["gt_trip"] - result_df["pred_trip"]) / result_df["gt_trip"]).abs()

        # test result
        diff_df = result_df[["hour", "o_id", "d_id", "diff"]]
        diff_df.columns = ["hour", "o_id", "d_id", "trip"]
        diff_df.to_csv(os.path.join(self.config.TEST_DIFF_PATH, "{}.csv".format(city)), index=False)


def get_distance_analysis(city, result_df):
    distance_path = "/home/user/PycharmProjects/distribution_prediction/gegn/gegn_combine/test/distance"
    gt_df = result_df[["trip", "surface_distance"]].copy(deep=True)
    # gt_df.columns = ["trip", "surface_distance"]
    pred_df = result_df[["pred_trip", "surface_distance"]].copy(deep=True)
    pred_df.columns = ["trip", "surface_distance"]

    gt_distance_df, gt_average_distance = util.analyze_distance(gt_df)
    pred_distance_df, pred_average_distance = util.analyze_distance(pred_df)

    plt.plot(gt_distance_df["surface_distance"], gt_distance_df["trip"], "b", label="gt")
    plt.plot(pred_distance_df["surface_distance"], pred_distance_df["trip"], "r", label="test")
    plt.legend(loc=1)
    plt.savefig(os.path.join(distance_path, "{}.jpg".format(city)))
    plt.close()

    plt.plot(gt_distance_df["surface_distance"], gt_distance_df["trip_prop"], "b", label="gt")
    plt.plot(pred_distance_df["surface_distance"], pred_distance_df["trip_prop"], "r", label="test")
    plt.legend(loc=1)
    plt.savefig(os.path.join(distance_path, "{}_prop.jpg".format(city)))
    plt.close()

    trip_prop_df = pd.concat([gt_distance_df[["trip_prop"]], pred_distance_df[["trip_prop"]]], axis=1)
    coincidence = trip_prop_df.min(axis=1).sum() / trip_prop_df.max(axis=1).sum()
    compare_data_df = pd.DataFrame([[gt_average_distance, coincidence], [pred_average_distance, coincidence]],
                                   columns=["average_distance", "coincidence"])
    compare_data_df.to_csv(os.path.join(distance_path, "{}.csv".format(city)), index=False)


def get_batch_distance_analysis(city, models, result_dfs):
    distance_path = "/home/user/PycharmProjects/distribution_prediction/gegn/gegn_combine/test/distance"
    gt_df = result_dfs[0][["trip", "surface_distance"]]
    # gt_df.columns = ["trip", "surface_distance"]
    print("len result dfs:", len(result_dfs))
    pred_dfs = []
    for df in result_dfs:
        pred_df = df[["pred_trip", "surface_distance"]]
        pred_df.columns = ["trip", "surface_distance"]
        pred_dfs.append(pred_df)

    gt_distance_df, gt_average_distance = util.analyze_distance(gt_df)

    pred_distance_dfs = []
    pred_average_distances = []

    for pred_df in pred_dfs:
        pred_distance_df, pred_average_distance = util.analyze_distance(pred_df)
        pred_distance_dfs.append(pred_distance_df)
        pred_average_distances.append(pred_average_distance)

    # print("len pred_distance_dfs:", len(pred_distance_dfs))
    # print("len pred_average_distances:", len(pred_average_distances))

    plt.plot(gt_distance_df["surface_distance"], gt_distance_df["trip"], "b", label="gt")
    colors = ['r', 'g', 'c']
    for i in range(len(models)):
        plt.plot(pred_distance_dfs[i]["surface_distance"], pred_distance_dfs[i]["trip"], colors[i],
                 label="test_{}".format(models[i]))
    plt.legend(loc=1)
    plt.savefig(os.path.join(distance_path, "{}_{}.jpg".format(city, "_".join(models))))
    plt.close()

    plt.plot(gt_distance_df["surface_distance"], gt_distance_df["trip_prop"], "b", label="gt")
    for i in range(len(models)):
        plt.plot(pred_distance_dfs[i]["surface_distance"], pred_distance_dfs[i]["trip_prop"], colors[i],
                 label="test_{}".format(models[i]))
    plt.legend(loc=1)
    plt.savefig(os.path.join(distance_path, "{}_{}_prop.jpg".format(city, "_".join(models))))
    plt.close()

    for i in range(len(models)):
        trip_prop_df = pd.concat([gt_distance_df[["trip_prop"]], pred_distance_dfs[i][["trip_prop"]]], axis=1)
        coincidence = trip_prop_df.min(axis=1).sum() / trip_prop_df.max(axis=1).sum()
        compare_data_df = pd.DataFrame([[gt_average_distance, coincidence], [pred_average_distances[i], coincidence]],
                                       columns=["average_distance", "coincidence"])
        compare_data_df.to_csv(os.path.join(distance_path, "{}_{}.csv".format(city, models[i])), index=False)


if __name__ == "__main__":
    cities = ["上海市", "成都市", "南京市", "石家庄市", "杭州市"]
    for city in cities:
        gt_ods = pd.read_csv("/home/user/disk/2019/edge_filtered_2019/{}.csv".format(city))
        # gt_ods.rename(columns={"trip": "gt_trip"})
        models = ['gravity', 'deep_gravity', 'fcnn_combine']
        pred_odss = [pd.read_csv("/home/user/PycharmProjects/distribution_prediction/gegn/gegn_combine/test/result/{}_{}.csv".format(city, model)) for model in models]
        result_dfs = [pd.concat([gt_ods[['trip', 'surface_distance']], pred_ods], axis=1) for pred_ods in pred_odss]
        # result_dfs = [gt_ods] + pred_odss
        get_batch_distance_analysis(city, models, result_dfs)
