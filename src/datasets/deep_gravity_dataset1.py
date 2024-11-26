import numpy as np
import torch
from torch.utils.data import Dataset
import argparse
import pandas as pd

import os
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict
from torch import Tensor
import random
import torch.nn.functional as F

import util
from datasets.____base_dataset import BaseDataset

class DeepGravityDataset(BaseDataset):

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
        super().__init__(coord_preprocessor, id_preprocessor)
        
        print(self.__class__.__name__)
        
        self.reuse_model = reuse_model
        self.MODEL_SAVE_PATH = MODEL_SAVE_PATH
        self.BEST_MODEL_SAVE_PATH = BEST_MODEL_SAVE_PATH
        self.model_status = model_status
        self.checkpoint = checkpoint
        self.preprocessor = (edge_preprocessor, grid_preprocessor)
        
        self.grid_dim = None
        self.edge_dim = None
        self.train_or_valid = None
        self.data = None
        
        self.preprocessed_data['id_coord_dict'] = coord_preprocessor()
        
        self.preprocessed_data['train_id'], self.preprocessed_data['valid_id'] = \
        id_preprocessor(self.preprocessed_data['id_coord_dict'])
            

    def __len__(self):
        return len(self.data[3])

    def __getitem__(self, index):
        """
        :return: item: Dict[str, pd.DataFrame]
        """
        [o_id_lt, d_id_lt, cities, trips] = self.data
        

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
        city_distance_df = pd.DataFrame(util.get_surface_distance(o_coord_array, d_coord_array),
                                        columns=["surface_distance"])
        city_distance_array = self.checkpoint["edge_scaler"].transform(city_distance_df)
        edge_x = torch.Tensor(city_distance_array)
        return o_grid_x, d_grid_x, edge_x, y


    def train(self):
        
        """
        Set dataset to train.
        Load train data as self.data for __getitem__.
        """
        
        if hasattr(self, "resampled_train_data"):
            self.data = self.resampled_train_data
        else:  # first time called
            self.preprocessed_data = self.get_train_preprocessed_data(*self.preprocessor)
            self.resampled_train_data = self.resample(self.preprocessed_data['train_edge'], self.preprocessed_data['train_id'])
            self.data = self.resampled_train_data


    def eval(self):
                
        """
        Set dataset to eval.
        Load eval data as self.data for __getitem__.
        """
        if hasattr(self, "resampled_eval_data"):
            self.data = self.resampled_eval_data
        else:  # first time called
            self.resampled_eval_data = self.resample(self.preprocessed_data['valid_edge'], self.preprocessed_data['valid_id'])
            self.data = self.resampled_eval_data

    
    def test(self):
                
        """
        Set dataset to test.
        Load test data.
        """
        
        self.preprocessed_data = self.get_test_preprocessed_data(*self.preprocessor)
        self.data = self.resample(self.preprocessed_data['test_edge'], self.preprocessed_data['test_id'])

    def get_train_preprocessed_data(self, edge_preprocessor, grid_preprocessor) \
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
        Preprocessor solve a conflict between id, edge, grid data from Telecom and our Poi, etc data from Internet,
        for they are using different coord, by doing multiple right join and transformation.
        """
        
        
        self.preprocessed_data['train_edge'], self.preprocessed_data['valid_edge'], edge_dim\
        = edge_preprocessor(self.preprocessed_data['train_id'], self.preprocessed_data['valid_id'])
        
        self.preprocessed_data['grid'], self.grid_dim = grid_preprocessor()
        
        self.edge_dim = edge_dim
        
        return self.preprocessed_data

    def get_test_preprocessed_data(self, edge_preprocessor, grid_preprocessor):

        self.preprocessed_data['test_id'] = {**self.preprocessed_data['train_id'], **self.preprocessed_data['valid_id']}
        
        self.preprocessed_data['train_edge'], self.preprocessed_data['valid_edge'], edge_dim = edge_preprocessor(
            self.preprocessed_data['train_id'], self.preprocessed_data['valid_id'])
        self.preprocessed_data['test_edge'] = pd.concat((self.preprocessed_data['train_edge'], self.preprocessed_data['valid_edge']))
        self.preprocessed_data['grid'], self.grid_dim = grid_preprocessor()
        return self.preprocessed_data


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

            sub_edge = group[1]
            sub_o_ids = sub_edge['o_id'].tolist()
            sub_d_ids = sub_edge['d_id'].tolist()
            sub_city = sub_edge["city"].values[0]
            sub_trips = torch.Tensor(sub_edge["trip"].values)

            max_length = 512
            num_of_edges_of_an_o_id = sub_edge.shape[0]
            
            if num_of_edges_of_an_o_id < max_length:
                
                sub_o_ids = [sub_o_ids[0] for _ in range(max_length)]
                d_id_set = set(sub_edge[["d_id"]].values.squeeze(-1).tolist())
                neg_sample_d_ids = random.sample(list(set(id[sub_city]) - d_id_set),
                                                 k=max_length - sub_edge.shape[0])
                sub_d_ids.extend(neg_sample_d_ids)
                sub_trips = F.pad(sub_trips, [0, max_length - sub_edge.shape[0]])

            o_id_lt.append(sub_o_ids)
            d_id_lt.append(sub_d_ids)
            cities.append(sub_city)
            trip_lt.append(sub_trips)
        return o_id_lt, d_id_lt, cities, trip_lt

    def get_ids(self, index):
        o_id_lt = self.data[0]
        d_id_lt = self.data[1]
        return o_id_lt[index], d_id_lt[index]


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
        normalized_all_grid_df = util.normalize_df(raw_all_grid_df, ["id"], "grid_scaler", self.checkpoint)
        normalized_all_grid_df = util.enlarge_grid(normalized_all_grid_df, self.grid_radius)
        reshaped_normalized_all_grid_dict = util.grid_reshape(normalized_all_grid_df, self.model_name, grid_dim,
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


class EdgePreprocessor(object):
    def __init__(self, cities, ALL_EDGE_PATH, hour, checkpoint):

        self.cities = cities
        self.ALL_EDGE_PATH = ALL_EDGE_PATH
        self.checkpoint = checkpoint
        self.hour = hour

    def __call__(self, train_id_dict, valid_id_dict):
        """
        normalized_train_edge_df
        
            hour             o_id  ...  grid_to_grid_time  surface_distance
        0      7  115450039775040  ...          19.830328          0.012185
        1      7  115450039775040  ...          24.901353          0.011608
        """
        print(self.__class__.__name__)
        
        normalized_train_edge_df, normalized_valid_edge_df = (
            self.load_edge_iteratively(self.cities, self.ALL_EDGE_PATH, self.hour, train_id_dict, valid_id_dict,
                                       self.checkpoint))
        # remove o_id, d_id, trip, city, hour, grid_to_grid_distance, grid_to_grid_time
        edge_dim = normalized_train_edge_df.shape[1] - 7  
        
        
        return normalized_train_edge_df, normalized_valid_edge_df, edge_dim

    @staticmethod
    def load_edge_iteratively(cities, ALL_EDGE_PATH, hour, train_id_dict, valid_id_dict, checkpoint):
        
        """
        ALL_EDGE_PATH is a very huge dataset, each city about 20GB, so we can't read it as a whole and then
        filter one particular hour, we can only read in an iterative way.
        """

        train_edges: List[pd.DataFrame] = []
        valid_edges: List[pd.DataFrame] = []
        
        cities: List[str]
        for city in cities:
            edges = util.get_edge(ALL_EDGE_PATH, city, hour)

            # filter
            train_edges.append(edges[edges["o_id"].isin(train_id_dict[city])])
            valid_edges.append(edges[edges["o_id"].isin(valid_id_dict[city])])
        
        
        frozen_columns = ["hour", "o_id", "d_id", "trip", "city", "grid_to_grid_distance", "grid_to_grid_time"]
        scaler_name = "edge_scaler"

        train_edges = util.normalize_df(pd.concat(train_edges, ignore_index=True),
                                                frozen_columns, scaler_name, checkpoint)
        valid_edges = util.normalize_df(pd.concat(valid_edges, ignore_index=True),
                                                frozen_columns, scaler_name, checkpoint)
        return train_edges, valid_edges
