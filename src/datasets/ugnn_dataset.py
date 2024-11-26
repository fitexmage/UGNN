from typing import Any
import pandas as pd
import os
import geopandas
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets.____base_dataset import BaseDataset
import util


class UgnnDataset(BaseDataset):

    """
    An Universal geo Neural Network(Ugnn) Dataset with a focus on Data Preprocessing
    and Resampling.
    Why is this dataset so complex?
    Well, this is because for one thing, it needs to concat data from various sources.
    Secondly, it resamples data for every training epoch.
    """
    def __init__(self, xxx):
        super().__init__(coord_preprocessor, id_preprocessor)
        print(self.__class__.__name__)

    def __len__(self):
        ...
    
    def __getitem__(self, index):
        ...
    
    def train_preprocess(self):
        ...
    
    def test_preprocess(self):
        ...
    
    def resample(self):
        ...


class GridPreprocessor(object):
    def __init__(self, checkpoint, cities, grid_type, GRID_BASIC_PATH, GRID_POI_AOI_PATH,
                  GRID_LAND_USE_PATH, RAW_CITY_METRIC_PATH, CITY_DIVISION_PATH, grid_radius,
                    model_name, grid_dim):
        self.checkpoint = checkpoint
        self.cities = cities
        self.grid_type = grid_type
        self.GRID_BASIC_PATH = GRID_BASIC_PATH
        self.GRID_POI_AOI_PATH = GRID_POI_AOI_PATH
        self.GRID_LAND_USE_PATH = GRID_LAND_USE_PATH
        self.RAW_CITY_METRIC_PATH = RAW_CITY_METRIC_PATH
        self.CITY_DIVISION_PATH = CITY_DIVISION_PATH
        self.grid_radius = grid_radius
        self.model_name = model_name
        self.grid_dim = grid_dim
        

    def __call__(self, item) -> Any:
        # Print class name
        print(self.__class__.__name__)

        # Join and select features
        grid_basic, grid_extra, grid_basic_dim, grid_extra_dim = self.joinNselect_features(
            self.grid_type,
            self.GRID_BASIC_PATH,
            self.GRID_POI_AOI_PATH,
            self.GRID_LAND_USE_PATH,
            self.cities
        )

        # Update item dictionary
        item['grid_basic_dim'] = grid_basic_dim
        item['grid_extra_dim'] = grid_extra_dim

        # Normalize data
        grid_basic = util.normalize_df(grid_basic, ['id'], "grid_basic_scaler", self.checkpoint)
        grid_extra = util.normalize_df(grid_extra, ['id'], "grid_extra_scaler", self.checkpoint)

        # Enlarge grids
        grid_basic = util.enlarge_grid(grid_basic, self.grid_radius)
        grid_extra = util.enlarge_grid(grid_extra, self.grid_radius)

        # Update item dictionary
        item['grid_basic_df'] = grid_basic
        item['grid_extra_df'] = grid_extra

        # Reshape grids
        grid_basic = util.grid_reshape(grid_basic, self.model_name, grid_basic_dim, "_")
        grid_extra = util.grid_reshape(grid_extra, self.model_name, grid_extra_dim, "_")

        # Update item dictionary
        item['grid_basic_dict'] = grid_basic
        item['grid_extra_dict'] = grid_extra

        # Load city metric data
        city_metric, external_dim = self.load_city_metric(
            self.RAW_CITY_METRIC_PATH, self.CITY_DIVISION_PATH)

        # Normalize city metric data
        city_metric = util.normalize_df(city_metric, ["city"], "metric_scaler", self.checkpoint)

        # Update item dictionary
        item['city_metric'] = city_metric

        return item

    @staticmethod
    def joinNselect_features(grid_type, GRID_BASIC_PATH, GRID_POI_AOI_PATH, GRID_LAND_USE_PATH, cities):
        """
        This function got its name from deep_gravity_dataset.py, however, it actually doesn't
        join the features but just return them.
        Land Use is temporarily useless because of the low quality, in the future we might need
        it. In theory, it's of vital importance.
        """
        grid_basic = []
        grid_extra = []

        for city in cities:
            city_grid_basic_df = pd.read_csv(os.path.join(
                GRID_BASIC_PATH, "{}.csv".format(city)))
            
            if grid_type == "land_use":
                city_grid_extra_df = pd.read_csv(os.path.join(
                    GRID_LAND_USE_PATH, "{}.csv".format(city)))
            else:  # poi_aoi
                city_grid_extra_df = pd.read_csv(
                    os.path.join(GRID_POI_AOI_PATH, "{}.csv".format(city)))

            grid_basic_dim = city_grid_basic_df.shape[1] - 1  # remove id
            grid_extra_dim = city_grid_extra_df.shape[1] - 1  # remove id

            grid_basic.append(city_grid_basic_df)
            grid_extra.append(city_grid_extra_df)

        grid_basic = pd.concat(grid_basic, ignore_index=True)
        grid_extra = pd.concat(grid_extra, ignore_index=True)

        return grid_basic, grid_extra, grid_basic_dim, grid_extra_dim

    @staticmethod
    def load_city_metric(RAW_CITY_METRIC_PATH, CITY_DIVISION_PATH):
        """
        This function load city metric in RAW_CITY_METRIC_PATH, and add city center coord to it.
        """
        general_df = pd.read_csv(RAW_CITY_METRIC_PATH, encoding='GB18030')

        city_metric_df = general_df[[
            "城市", "2022城市人口", "2022年GDP", "2022行政区面积", "2022建城区面积"]]
        city_metric_df.columns = ["city", "城市人口", "总GDP", "行政区面积", "建城区面积"]

        city_division_gdf = geopandas.read_file(
            os.path.join(CITY_DIVISION_PATH))

        # remove city; add lng, lat, and hour
        external_dim = city_metric_df.shape[1] + 2

        metric_dfs = []
        for row in np.array(city_metric_df):
            city = row[0]
            center = city_division_gdf[city_division_gdf["NAME"]
                                       == city].representative_point()
            metric_array = np.concatenate(
                [row, center.x.values, center.y.values])
            metric_array = metric_array[np.newaxis, :]
            metric_df = pd.DataFrame(metric_array, columns=list(
                city_metric_df.columns) + ["lng", "lat"])
            metric_dfs.append(metric_df)
        return pd.concat(metric_dfs, ignore_index=True), external_dim


class EdgePreprocessor(object):
    def __init__(self) -> None:
        pass


