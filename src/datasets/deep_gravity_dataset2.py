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
from src.datasets.____base_dataset import BaseDataset

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

    def __init__(self, data, grid, coord, checkpoint):
        print(self.__class__.__name__)
        self.edge = data
        self.grid = grid
        self.coord = coord
        self.checkpoint = checkpoint

    def __len__(self):
        return len(self.edge[3])

    def __getitem__(self, index):
        """
        :return: item: Dict[str, pd.DataFrame]
        """
        [o_id_lt, d_id_lt, cities, trips] = self.edge
        

        o_ids = o_id_lt[index]
        d_ids = d_id_lt[index]

        # Pad grid data to o and d
        o_grid_x = torch.concat([self.grid[o_id].unsqueeze(0) for o_id in o_ids])
        d_grid_x = torch.concat([self.grid[d_id].unsqueeze(0) for d_id in d_ids])
        y = trips[index]

        # Calculate edge distance between o_id and d_id
        id_df = pd.concat([pd.DataFrame(o_ids, columns=["o_id"], dtype="int64"),
                           pd.DataFrame(d_ids, columns=["d_id"], dtype="int64")], axis=1)
        id_coord_df = self.coord[cities[index]]
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

    def get_ids(self, index):
        o_id_lt = self.edge[0]
        d_id_lt = self.edge[1]
        return o_id_lt[index], d_id_lt[index]







