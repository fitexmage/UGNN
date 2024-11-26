import numpy as np
import torch
from torch.utils.data import Dataset
import argparse
import pandas as pd

import os
from util import setup_seed, get_id_coord, add_to_id, get_id
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict
from torch import Tensor
import random
import torch.nn.functional as F
from util import get_surface_distance, normalize_df


class DeepGravityDataset(Dataset):

    """
    A Deep Gravity Dataset with a focus on Data Preprocessing.

    The class involves two parts, data preprocessing in the init, and XXX in the get item.
    1. Init
        1.1 Preprocess
        We get edge data and grid data of telecom id and left join with coord from website in a particular
        city within a certain hour.
        1.2 Resample
        For some edge with frequency less than 512, we resample it with other random destinations.
    2. Get Item

    """

    def __init__(self, reuse_model, MODEL_SAVE_PATH, BEST_MODEL_SAVE_PATH, model_status, coord_preprocessor,
                 id_preprocessor, edge_preprocessor, grid_preprocessor, checkpoint):

        """
        TODO
        self.checkpoint?
        edge_dim

        """
        print(self.__class__.__name__)
        self.grid_dim = None
        self.reuse_model = reuse_model
        self.MODEL_SAVE_PATH = MODEL_SAVE_PATH
        self.BEST_MODEL_SAVE_PATH = BEST_MODEL_SAVE_PATH
        self.model_status = model_status
        self.train_or_valid = None

        self.checkpoint = checkpoint
        
        if self.model_status == 'train' or self.model_status == 'valid':
            self.train_preprocess(
                        coord_preprocessor,
                        id_preprocessor,
                        edge_preprocessor,
                        grid_preprocessor)
        if self.model_status == 'test':
            ...
            # You will need to manually call test_preprocess iteratively.


    def __len__(self):
        if self.train_or_valid == 'train':
            return len(self.train_preprocessed_resampled[3])
        elif self.train_or_valid == 'valid':
            return len(self.valid_preprocessed_resampled[3])
        elif self.model_status == 'test':
            return len(self.test_preprocessed_resampled[3])
        else:
            "Train, valid or test? Set Dataset Attribute."
            raise ValueError

    def __getitem__(self, index):
        """
        :return: item: Dict[str, pd.DataFrame]
        """
        if self.train_or_valid == 'train':
            [o_id_lt, d_id_lt, cities, trips] = self.train_preprocessed_resampled
        elif self.train_or_valid == 'valid':
            [o_id_lt, d_id_lt, cities, trips] = self.valid_preprocessed_resampled
        elif self.model_status == 'test':
            [o_id_lt, d_id_lt, cities, trips] = self.test_preprocessed_resampled
        else:
            "Set DeepGravityDataset.train_or_valid!"
            raise ValueError

        o_ids = o_id_lt[index]
        d_ids = d_id_lt[index]

        o_grid_x = torch.concat([self.preprocessed_data['grid'][o_id].unsqueeze(0) for o_id in o_ids])
        d_grid_x = torch.concat([self.preprocessed_data['grid'][d_id].unsqueeze(0) for d_id in d_ids])
        y = trips[index]

        id_df = pd.concat([pd.DataFrame(o_ids, columns=["o_id"], dtype="int64"),
                           pd.DataFrame(d_ids, columns=["d_id"], dtype="int64")], axis=1)
        id_coord_df = self.preprocessed_data['id_coord_dict'][cities[index]]
        id_df = id_df.merge(id_coord_df, left_on="o_id", right_on="id").merge(id_coord_df, left_on="d_id",
                                                                              right_on="id")
        id_df = id_df[["lng_x", "lat_x", "lng_y", "lat_y"]]
        id_df.columns = ["o_lng", "o_lat", "d_lng", "d_lat"]
        o_coord_array = np.array(id_df[["o_lng", "o_lat"]]).T
        d_coord_array = np.array(id_df[["d_lng", "d_lat"]]).T
        city_distance_df = pd.DataFrame(get_surface_distance(o_coord_array, d_coord_array),
                                        columns=["surface_distance"])
        city_distance_array = self.checkpoint["edge_scaler"].transform(city_distance_df)
        edge_x = torch.Tensor(city_distance_array)
        return o_grid_x, d_grid_x, edge_x, y

    def get_preprocessed_data(self, coord_preprocessor, id_preprocessor, edge_preprocessor, grid_preprocessor) \
            -> Dict[str, pd.DataFrame]:
        """
        :param coord_preprocessor: It provides a map between Telecom's id and WGS-84 coord.

        :param id_preprocessor:
        It provides all the id that is in WGS-84 coord within a city. Id is the Telecom's id,
        which is using their own coord, however, WGS-84 is more universal and is the coord of which most of our data
        rely on.

        :param edge_preprocessor:
        It provides all the edge that is in WGS-84 within a city. Edge is original and destination.

        :param grid_preprocessor: It provides all the grid that is in WGS-84 within a city.
        :return:
        For more information about our original data, please refer to the corresponding class definition.

        GOAL:
        Preprocessors solve a conflict between id, edge, grid data from Telecom and our Poi, etc data from Internet,
        for they are using different coord, by doing multiple right join and transformation.
        """
        preprocessed_data = {}
        preprocessed_data['id_coord_dict'] = coord_preprocessor()
        preprocessed_data['train_id'], preprocessed_data['valid_id'] = id_preprocessor(
            preprocessed_data['id_coord_dict'])
        preprocessed_data['train_edge'], preprocessed_data['valid_edge'] = edge_preprocessor(
            preprocessed_data['train_id'], preprocessed_data['valid_id'])
        preprocessed_data['grid'], self.grid_dim = grid_preprocessor()
        return preprocessed_data

    def load_checkpoint(self):
        if self.reuse_model:
            if self.model_status == "train":
                checkpoint = torch.load(self.MODEL_SAVE_PATH)
            else:
                checkpoint = torch.load(self.BEST_MODEL_SAVE_PATH)
        else:
            checkpoint = {}
        return checkpoint

    def resample(self, edge, id):
        """
        Actually, it should be called regenerate.
        Our data is very imbalanced, here we resample the edge data.???
        If num_of_edges_of_an_o_id is less than 512, we resample the edges from all the other ds,
        and pad the trip by 0.
        :param edge: preprocessed edge data
        :param id:Dict[str, List] preprocessed all ids of different cities. We resample the edges according to these
        id
        :return:
        o_id_lt: A list of list of grouped ids.
        [[115450039775040, 115450039775040, 115450039775040, 115450039775040, 115450039775040, 115450039775040,
        115450039775040, 115450039775040, 115450039775040, 115450039775040, 115450039775040, 115450039775040,
        115450039775040,
        d_id_lt: A list of list of grouped ids.
        cities: Duplicated cities of num of groups.
        trip_lt: List of tensors, each tensor records the trip distance from corresponding o_id to d_it in an item of
        o_id_lt.
        trip_lt
        [tensor([0.0983, 0.0491, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,

        """
        o_id_lt, d_id_lt, cities, trip_lt = [], [], [], []
        groups = edge.groupby("o_id")
        for group in groups:
            o_edge_df = group[1]
            o_ids = o_edge_df[["o_id"]].values.squeeze(-1).tolist()
            d_ids = o_edge_df[["d_id"]].values.squeeze(-1).tolist()
            city = o_edge_df["city"].values[0]
            trips = torch.Tensor(o_edge_df[["trip"]].values.squeeze(-1))

            max_length = 512
            num_of_edges_of_an_o_id = o_edge_df.shape[0]
            if num_of_edges_of_an_o_id < max_length:
                o_ids = [o_ids[0] for _ in range(max_length)]
                d_id_set = set(o_edge_df[["d_id"]].values.squeeze(-1).tolist())
                if self.model_status == 'train' or self.model_status == 'valid':
                    neg_sample_d_ids = random.sample(list(set(id[city]) - d_id_set),
                                                 k=max_length - o_edge_df.shape[0])
                if self.model_status == 'test':
                    # In func test there is a large loop to iterate cities, so here we
                    # remove them.
                    neg_sample_d_ids = random.sample(list(set(id) - d_id_set),
                                                 k=max_length - o_edge_df.shape[0])
                d_ids.extend(neg_sample_d_ids)
                trips = F.pad(trips, [0, max_length - o_edge_df.shape[0]])

            o_id_lt.append(o_ids)
            d_id_lt.append(d_ids)
            cities.append(city)
            trip_lt.append(trips)
        return o_id_lt, d_id_lt, cities, trip_lt

    def train_preprocess(self,
                        coord_preprocessor,
                        id_preprocessor,
                        edge_preprocessor,
                        grid_preprocessor):
        self.preprocessed_data = self.get_preprocessed_data(coord_preprocessor,
                                                            id_preprocessor,
                                                            edge_preprocessor,
                                                            grid_preprocessor)

        self.train_preprocessed_resampled = self.resample(self.preprocessed_data['train_edge'], self.preprocessed_data['train_id'])
        self.valid_preprocessed_resampled = self.resample(self.preprocessed_data['valid_edge'], self.preprocessed_data['valid_id'])
        self.edge_dim = self.preprocessed_data['train_edge'].shape[1] - 7

    def test_preprocess(self, test_item):
        self.test_preprocessed_resampled = self.resample(*test_item)

class CoordPreprocessor(object):
    """
    This class will find the IDs for all the following preprocessor's reference.
    """

    def __init__(self, cities, CITY_CENTROID_PATH):
        self.cities = cities
        self.CITY_CENTROID_PATH = CITY_CENTROID_PATH

    def __call__(self):
        print(self.__class__.__name__)
        return self.get_coord(self.cities, self.CITY_CENTROID_PATH)

    @staticmethod
    def get_coord(cities: List[str], CITY_CENTROID_PATH: str) -> Dict[str, pd.DataFrame]:
        """
        Get ID with longitude and latitude for all cities, need data from .shp file
        id_coord_dict:
        {'北京市':                  id         lng        lat
        0      1.154250e+14  115.420976  39.771141
        1      1.154250e+14  115.420952  39.951131
        """
        id_coord_dict = {}
        for city in cities:
            id_coord_dict[city] = get_id_coord(city, CITY_CENTROID_PATH)
        return id_coord_dict


class IdPreprocessor(object):

    def __init__(self, cities, split_frac):

        self.cities = cities
        self.split_frac = split_frac

    def __call__(self, id_coord_dict):
        print(self.__class__.__name__)
        train_id, valid_id = self.right_join_get_id(id_coord_dict, self.cities, self.split_frac)
        return train_id, valid_id

    @staticmethod
    def right_join_get_id(id_coord_dict, cities, split_frac):
        """
        id is data from Telecom, which is defined by them, not our data grid(coord). So for some ids that are not
        in our data grid, we simply remove them by a right join.
        train_id_dict/valid_id_dict:
        {'北京市': [115550039920040, 115760040110040, 116660039870040, 115985040240040, 116125040280040,
        """
        train_id_dict, valid_id_dict = {}, {}
        for city in cities:
            city_id_df = get_id(city)
            train_id_df, valid_id_df = train_test_split(city_id_df, test_size=split_frac, shuffle=True,
                                                        random_state=55)
            train_id_dict[city] = list(
                train_id_df[train_id_df["id"].isin(id_coord_dict[city]["id"])].values.squeeze(-1))
            valid_id_dict[city] = list(
                valid_id_df[valid_id_df["id"].isin(id_coord_dict[city]["id"])].values.squeeze(-1))
        return train_id_dict, valid_id_dict


class GridPreprocessor(object):
    """
    This preprocessor involves data in GRID_BASIC_PATH, GRID_POI_AOI_PATH and CITY_CENTROID_PATH
    """

    def __init__(self, GRID_BASIC_PATH, GRID_POI_AOI_PATH, cities, checkpoint, grid_radius, model_name, grid_num):

        self.GRID_BASIC_PATH = GRID_BASIC_PATH
        self.GRID_POI_AOI_PATH = GRID_POI_AOI_PATH
        self.cities = cities
        self.checkpoint = checkpoint
        self.grid_radius = grid_radius
        self.model_name = model_name
        self.grid_num = grid_num

    def __call__(self):
        """
        Do pipeline, add final data to item
        reshaped_normalized_all_grid_dict: Dict[str, Tensor] Tensor.shape:441, dict length:69445

        reshaped_normalized_all_grid_dict[117460040670040]
        tensor([0.0000e+00, 0.0000e+00, 1.8238e-02, 8.5981e-04, 0.0000e+00, 8.6033e-01,
        8.7145e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        """
        print(self.__class__.__name__)
        raw_all_grid_df, grid_dim = self.joinNselect_features(self.GRID_BASIC_PATH, self.GRID_POI_AOI_PATH, self.cities)
        normalized_all_grid_df = normalize_df(raw_all_grid_df, ["id"], "grid_scaler", self.checkpoint)
        normalized_all_grid_df = self.enlarge_grid(normalized_all_grid_df, self.grid_radius)
        reshaped_normalized_all_grid_dict = self.grid_reshape(normalized_all_grid_df, self.model_name, grid_dim,
                                                            self.grid_num)

        return reshaped_normalized_all_grid_dict, grid_dim

    @staticmethod
    def joinNselect_features(GRID_BASIC_PATH, GRID_POI_AOI_PATH, cities):
        """
        This function do a left join to join basic and poi,aoi data, get grid dimention,
        and we can also do feature selection(not yet)
        Finally, do above for all cities and concat.

        city_grid_basic_df:
                            id   resi  ...  subway_distance  bus_distance
        0      115440039960040   26.0  ...     59917.815164   2692.868440

        raw_all_grid_df
                    id   resi   work  ...  公用设施营业网点用地_aoi  二类居住用地_aoi  物流仓储用地_aoi
        0      115440039960040   26.0   52.0  ...             0.0         0.0         0.0
        1      115450039775040    7.0    7.0  ...             0.0         0.0         0.0
        2      115450039960040   46.0  144.0  ...             0.0         0.0         0.0
        3      115465039775040  542.0  359.0  ...             0.0         0.0         0.0
        4      115465039960040  444.0  791.0  ...             0.0         0.0         0.0
        [69445 rows x 50 columns]

        """
        all_grid_dfs = []
        for city in cities:
            city_grid_basic_df = pd.read_csv(os.path.join(GRID_BASIC_PATH, "{}.csv".format(city)))
            city_grid_extra_df = pd.read_csv(os.path.join(GRID_POI_AOI_PATH, "{}.csv".format(city)))
            city_all_grid_df = city_grid_basic_df.merge(city_grid_extra_df, how="left", on="id")
            city_all_grid_df = city_all_grid_df.fillna(0)
            all_grid_dfs.append(city_all_grid_df)
        raw_all_grid_df = pd.concat(all_grid_dfs, ignore_index=True)
        grid_dim = raw_all_grid_df.shape[1] - 1  # remove id
        print("Grid data loaded!")
        return raw_all_grid_df, grid_dim

    @staticmethod
    def enlarge_grid(normalized_all_grid_df, grid_radius):
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
        center_id_df = normalized_all_grid_df[["id"]]

        directions = []
        for row in range(-grid_radius + 1, grid_radius):
            for col in range(-grid_radius + 1, grid_radius):
                directions.append([row, col])

        grid_dfs = []
        for direction in directions:
            """
            Because id contains coord info, and each grid is 0.05 km wide, here we just add 0.05 to id's coord,
            and we can get a new id, that is the surrounded neighbor of the original one.
            """
            id_df = add_to_id(center_id_df, direction)
            grid_df = id_df.merge(normalized_all_grid_df, how="left", on="id")
            grid_df = grid_df.fillna(0)
            grid_df = grid_df.drop(columns="id")
            # Here grid_df is the 49 columns of one of the neighbors.
            grid_dfs.append(grid_df)
        normalized_all_grid_df = pd.concat([center_id_df] + grid_dfs, axis=1)
        """
        normalized_all_grid_df
                    id  resi  work  ...  公用设施营业网点用地_aoi  二类居住用地_aoi  物流仓储用地_aoi
        0      115440039960040   0.0   0.0  ...             0.0         0.0         0.0
        [69445 rows x 442 columns]
        
        442 = 1 + 9*49. At the beginning, each grid has 50 cols, but after we remove the id, only 49.
        """
        return normalized_all_grid_df

    @staticmethod
    def grid_reshape(normalized_all_grid_df, model_name, grid_dim, grid_num) -> Dict[str, Tensor]:
        """
        Reshape data according to model name.
        For cnn model,
        """
        reshaped_normalized_all_grid_dict = {}
        print("normalized_all_grid_df:\n", normalized_all_grid_df)
        for row in np.array(normalized_all_grid_df):
            id = int(row[0])
            grid_data = row[1:]
            if model_name.startswith("cnn"):
                grid_data = grid_data.reshape([3, 3, grid_dim]).T
            elif model_name.startswith("gat"):
                grid_data = grid_data.reshape(grid_num, grid_dim)
            reshaped_normalized_all_grid_dict[id] = torch.Tensor(grid_data)
        return reshaped_normalized_all_grid_dict

class EdgePreprocessor(object):
    def __init__(self, cities, ALL_EDGE_PATH, hour, checkpoint):

        self.cities = cities
        self.ALL_EDGE_PATH = ALL_EDGE_PATH
        self.checkpoint = checkpoint
        self.hour = hour

    def __call__(self, train_id_dict, valid_id_dict):
        """
        normalized_train_edge_df
        PyDev console: starting.
            hour             o_id  ...  grid_to_grid_time  surface_distance
        0      7  115450039775040  ...          19.830328          0.012185
        1      7  115450039775040  ...          24.901353          0.011608
        """
        print(self.__class__.__name__)
        normalized_train_edge_df, normalized_valid_edge_df = (
            self.load_edge_iteratively(self.cities, self.ALL_EDGE_PATH, self.hour, train_id_dict, valid_id_dict,
                                       self.checkpoint))
        return normalized_train_edge_df, normalized_valid_edge_df

    @staticmethod
    def load_edge_iteratively(cities, ALL_EDGE_PATH, hour, train_id_dict, valid_id_dict, checkpoint):
        """
        ALL_EDGE_PATH is a very huge dataset, each city about 20GB, so we can't read it as a whole and then
        filter one particular hour, we can only read in an iterative way.
        """
        train_edge_dfs: List[pd.DataFrame] = []
        valid_edge_dfs: List[pd.DataFrame] = []
        for city in cities:
            iterator = pd.read_csv(os.path.join(ALL_EDGE_PATH, "{}.csv".format(city)), chunksize=10000000)
            city_edge_dfs = []
            for edge_df in iterator:
                edge_df = edge_df[edge_df["hour"] == hour].reset_index(drop=True)
                city_edge_dfs.append(edge_df)
            city_edge_df = pd.concat(city_edge_dfs, ignore_index=True)
            city_edge_df.loc[:, "city"] = city
            train_edge_dfs.append(city_edge_df[city_edge_df["o_id"].isin(train_id_dict[city])])
            valid_edge_dfs.append(city_edge_df[city_edge_df["o_id"].isin(valid_id_dict[city])])
        normalized_train_edge_df = normalize_df(pd.concat(train_edge_dfs, ignore_index=True),
                                                ["hour", "o_id", "d_id", "trip", "city", "grid_to_grid_distance",
                                                 "grid_to_grid_time"], "edge_scaler", checkpoint)
        normalized_valid_edge_df = normalize_df(pd.concat(valid_edge_dfs, ignore_index=True),
                                                ["hour", "o_id", "d_id", "trip", "city", "grid_to_grid_distance",
                                                 "grid_to_grid_time"], "edge_scaler", checkpoint)
        return normalized_train_edge_df, normalized_valid_edge_df
