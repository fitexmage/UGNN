import numpy as np
import torch
from torch.utils.data import Dataset
import argparse
import pandas as pd

import os
# from util import setup_seed, get_id_coord, add_to_id, get_id
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict
from torch import Tensor
import random
import torch.nn.functional as F
# from util import get_surface_distance, normalize_df
import util


class BaseDataset(Dataset):
    def __init__(self, coord_preprocessor, id_preprocessor):
        self.preprocessed_data = {}

        self.preprocessed_data['id_coord_dict'] = coord_preprocessor()

        self.preprocessed_data['train_id'], self.preprocessed_data['valid_id']\
        = id_preprocessor(self.preprocessed_data['id_coord_dict'])



class CoordPreprocessor(object):

    """
    Load id, lng, lat as dict of df from geo file.
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
            id_coord_dict[city] = util.get_id_coord(city, CITY_CENTROID_PATH)
        return id_coord_dict

class IdPreprocessor(object):

    def __init__(self, cities, split_frac):

        self.cities = cities
        self.split_frac = split_frac

    def __call__(self, id_coord_dict):
        print(self.__class__.__name__)
        train_id, valid_id = self.get_filtered_id(id_coord_dict, self.cities, self.split_frac)
        return train_id, valid_id

    @staticmethod
    def get_filtered_id(id_coord_dict, cities, split_frac):
        
        """
        id is data from Telecom, which is defined by them, not our data grid(coord). So for some ids that are not
        in our data grid, we simply remove them by a right join.
        train_id_dict/valid_id_dict:
        {'北京市': [115550039920040, 115760040110040, 116660039870040, 115985040240040, 116125040280040,
        """
        train_id_dict, valid_id_dict = {}, {}
        for city in cities:
            city_id_df = util.get_id(city)
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
        reshaped_normalized_all_grid_dict: Dict[str, Tensor] Tensor.shape: (N, 441)

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
        normalized_train_edge_df.shape: (N, 8)
        normalized_train_edge_df.columns
        Index(['hour', 'o_id', 'd_id', 'trip', 'city', 'grid_to_grid_distance',
            'grid_to_grid_time', 'surface_distance'],
            dtype='object')
        
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
