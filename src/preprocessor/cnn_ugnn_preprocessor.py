import random
import torch.nn.functional as F
import torch
import pandas as pd
import math
from numpy import mean
import util
import os
import geopandas
import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm
    
class UgnnGridPreprocessor(object):
    
    """
    
    Modified from deep gravity.
    
    
    """
    def __init__(self, checkpoint, cities, grid_type, GRID_BASIC_PATH, GRID_POI_AOI_PATH,
                  GRID_LAND_USE_PATH, RAW_CITY_METRIC_PATH, CITY_DIVISION_PATH, grid_radius,
                    model_name, one_hot_hour):
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
        self.one_hot_hour = one_hot_hour
        

    def __call__(self):
        # Print class name
        print(self.__class__.__name__)
        grids = {}

        # Join and select features
        grid_basic, grid_extra, grid_basic_dim, grid_extra_dim = self.joinNselect_features(
            self.grid_type,
            self.GRID_BASIC_PATH,
            self.GRID_POI_AOI_PATH,
            self.GRID_LAND_USE_PATH,
            self.cities
        )

        # Update item dictionary
        grids['grid_basic_dim'] = grid_basic_dim
        grids['grid_extra_dim'] = grid_extra_dim
        

        # Normalize data
        grid_basic = util.normalize_df(grid_basic, ['id'], "grid_basic_scaler", self.checkpoint)
        grid_extra = util.normalize_df(grid_extra, ['id'], "grid_extra_scaler", self.checkpoint)

        # Enlarge grids
        grid_basic = util.enlarge_grid(grid_basic[['id']], grid_basic, self.grid_radius)
        grid_extra = util.enlarge_grid(grid_basic[['id']], grid_extra, self.grid_radius)

        # Update item dictionary
        grids['grid_basic_df'] = grid_basic
        grids['grid_extra_df'] = grid_extra

        # Reshape grids
        grid_basic = util.grid_reshape(grid_basic, self.model_name, grid_basic_dim, "_", self.grid_radius)
        grid_extra = util.grid_reshape(grid_extra, self.model_name, grid_extra_dim, "_", self.grid_radius)

        # Update item dictionary
        grids['grid_basic_dict'] = grid_basic
        grids['grid_extra_dict'] = grid_extra

        # Load city metric data
        city_metric, external_dim = self.load_city_metric(
            self.RAW_CITY_METRIC_PATH, self.CITY_DIVISION_PATH)
        if self.one_hot_hour:
            external_dim -= 1  # remove hour

        # Normalize city metric data
        city_metric = util.normalize_df(city_metric, ["city"], "metric_scaler", self.checkpoint)

        # Update item dictionary
        grids['city_metric'] = city_metric
        grids['external_dim'] = external_dim
        

        return grids

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

            city_grid_extra_df['二类居住用地_poi'] = city_grid_extra_df['二类居住用地_poi'] + 10
            print('\n\n\nadding 1 to resi poi')
            
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

        city_metric_df = general_df[["城市", "2022城市人口", "2022年GDP", "2022行政区面积", "2022建城区面积"]]
        city_metric_df.columns = ["city", "城市人口", "总GDP", "行政区面积", "建城区面积"]

        city_division_gdf = geopandas.read_file(
            os.path.join(CITY_DIVISION_PATH))

        
        external_dim = city_metric_df.shape[1] + 2  # remove city; add lng, lat, and hour

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

class UgnnEdgePreprocessor(object):
    def __init__(self, cities, ALL_EDGE_PATH, checkpoint, one_hot_hour, hour):

        self.cities = cities
        self.ALL_EDGE_PATH = ALL_EDGE_PATH
        self.checkpoint = checkpoint
        self.one_hot_hour = one_hot_hour
        self.hour = hour
        if self.hour:
            print("Using: ", self.hour)

    def __call__(self, train_id_dict):
        """
        Load train edge iteratively and sample it.
        train_edge.shape: N, 7
        """
        print(self.__class__.__name__)
        
        train_edges: List[pd.DataFrame] = []
        
        for city in self.cities:
            
            city_edge = []
            iterator = pd.read_csv(os.path.join(self.ALL_EDGE_PATH, "{}.csv".format(city)), chunksize=10000000, engine="c")
            
            iterator = tqdm(iterator, leave=False)
            for sub_edge in iterator:
                sub_edge = sub_edge[sub_edge["o_id"].isin(train_id_dict[city])]
                # TEMP
                sub_edge = sub_edge[sub_edge["d_id"].isin(train_id_dict[city])]
                if self.hour:
                    sub_edge = sub_edge[sub_edge["hour"].isin(self.hour)]  # filter
                # sub_edge = self.differential_sample(sub_edge)
                
                sub_edge = self.simple_sample(sub_edge)
                iterator.set_description("Simple_sample")
                iterator.set_description(f"Loading 🏙️ {city}")
                city_edge.append(sub_edge)
            iterator.close()
            city_edge = pd.concat(city_edge, ignore_index=True)
            city_edge['city'] = city
            train_edges.append(city_edge)

        train_edges = pd.concat(train_edges, ignore_index=True)
        train_edges = train_edges.reset_index(drop=True)
        
        if self.one_hot_hour:
            # Set hour to category
            hours = list(range(0, 24))
            hour_type = pd.CategoricalDtype(categories=hours)
            train_edges['hour'] = pd.Series(train_edges['hour'], dtype=hour_type)
        
        frozen_columns = ["hour", "o_id", "d_id", "trip", "city"]
        scaler_name = "edge_scaler"
        train_edges = util.normalize_df(train_edges,
                                                frozen_columns, scaler_name, self.checkpoint)
        
        return train_edges
    
    def load_test(self, train_id_dict):
        for city in self.cities:
            print("City:", city)

            edge = pd.read_csv(os.path.join(self.ALL_EDGE_PATH, "{}.csv".format(city)))
            print(edge.shape)
            # TEMP
            edge = edge[edge["o_id"].isin(train_id_dict[city])]
            edge = edge[edge["d_id"].isin(train_id_dict[city])]
            print(edge.shape)
            if self.hour:
                edge = edge[edge["hour"].isin(self.hour)]  # filter
            edge.loc[:, "city"] = city
            edge = util.normalize_df(edge,
                                    ["o_id", "d_id", "trip", "city", "hour"],
                                    "edge_scaler", self.checkpoint)
            edge_dim = edge.shape[1] - 5
        if self.one_hot_hour:
            # Set hour to category
            hours = list(range(0, 24))
            hour_type = pd.CategoricalDtype(categories=hours)
            edge['hour'] = pd.Series(edge['hour'], dtype=hour_type)
            edge_dim = edge.shape[1] - 5
        return edge, edge_dim
    
    def load_test_(self):
        """
        Made tempraryly for shap values.
        """
        for city in self.cities:
            print("City:", city)

            # edge = pd.read_csv(os.path.join(self.ALL_EDGE_PATH, "{}.csv".format(city)))
                        # TEMP
            edge = pd.read_csv("/media/lai/27dbdec5-6518-47f1-b1af-159ffe3d1c08/2019/edge_2019_by_grid1000/北京市真武庙.csv")
            print("TTTTTTHHHHHHIIIIIISSSSS   IIIIIIIISSSSSS TTTTTTEEEEEEMMMMMPPPPP!!!!!!")
            print(len(edge))
            
            
            if self.hour:
                edge = edge[edge["hour"].isin(self.hour)]  # filter
            edge.loc[:, "city"] = city
            edge = util.normalize_df(edge,
                                    ["o_id", "d_id", "trip", "city", "hour"],
                                    "edge_scaler", self.checkpoint)
            edge_dim = edge.shape[1] - 5
        if self.one_hot_hour:
            # Set hour to category
            hours = list(range(0, 24))
            hour_type = pd.CategoricalDtype(categories=hours)
            edge['hour'] = pd.Series(edge['hour'], dtype=hour_type)
            edge_dim = edge.shape[1] - 5
        return edge, edge_dim

    def load_valid_iteratively(self, valid_id_dict):
        """
        Load valid iteratively, saving memory
        In cnn_ugnn model, the valid edge will be read iteratively in order to keep
        low memory usage.
        For train edge, sampler will handle.
        """
        valid_edges: List[pd.DataFrame] = []
        self.cities: List[str]
        
        print("In loading valid data...")
        
        for city in self.cities:
            
            city_edge = []
            iterator = pd.read_csv(os.path.join(self.ALL_EDGE_PATH, "{}.csv".format(city)), chunksize=10000000, engine="c")
            iterator = tqdm(iterable=iterator, leave=False)
            for sub_edge in iterator:
                sub_edge = sub_edge[sub_edge["o_id"].isin(valid_id_dict[city])]
                # TEMP
                sub_edge = sub_edge[sub_edge["d_id"].isin(valid_id_dict[city])]
                city_edge.append(sub_edge)
                iterator.set_description(f"Loading 🏙️ {city}")
            iterator.close()
            city_edge = pd.concat(city_edge, ignore_index=True)
            
            # Add city column for later merging city metric.
            city_edge['city'] = city
            valid_edges.append(city_edge)

        valid_edges = pd.concat(valid_edges, ignore_index=True)
        if self.one_hot_hour:
            # Set hour to category
            hours = list(range(0, 24))
            hour_type = pd.CategoricalDtype(categories=hours)
            valid_edges['hour'] = pd.Series(valid_edges['hour'], dtype=hour_type)
        
        
        frozen_columns = ["hour", "o_id", "d_id", "trip", "city"]
        scaler_name = "edge_scaler"
        valid_edges = util.normalize_df(valid_edges,
                                                frozen_columns, scaler_name, self.checkpoint)
        edge_dim = valid_edges.shape[1] - 5
        return valid_edges, edge_dim
    
    @staticmethod
    def differential_sample(edge, threshold=100, step=1, alpha=1.0, beta=0.3):

        """
        Calculate the sampled weights according to pow.
        Sample the edge. Sample rate is about 3%.
        """
            
        # We won't process data beyond threshold.
        max_trip = min(math.ceil(edge.max()['trip']), threshold)
        min_trip = max(math.floor(edge.min()['trip']), 0)
        max_weight = 0
        total_num = len(edge)
        d_edges = []
        weights = []
        class_sizes = []
        trip_frequencies = []

        for i in range(min_trip, max_trip):
            edges = edge[(edge['trip'] >= i) & (edge['trip'] < i + 1)]
            trip_frequencies.append(len(edges))
            d_edges.append(edges)

            if len(edges) == 0:
                # For trip that has no edges, weight equals zero, so that it will
                # not affect avg.
                weight = 0
            else:
                weight = alpha / pow(len(edges), beta)  # More edges, smaller weight
            max_weight = max(max_weight, weight)
            weights.append(weight)

        avg_weight = mean(weights)

        for i in range(len(weights)):
            # If smaller than avg, it will sample.
            # Bigger than avg, weight change to 1, which mean do not sample.
            weights[i] = min(1.0, weights[i] / avg_weight)

        for i in range(len(d_edges)):
            d_edges[i] = d_edges[i].sample(int(weights[i] * trip_frequencies[i]), ignore_index=True)

        edge_sampled = pd.concat(d_edges, ignore_index=True)
        edge_sampled = pd.concat([edge_sampled, edge[edge['trip'] >= threshold]], ignore_index=True)

        return edge_sampled
    
    @staticmethod
    def simple_sample(edge, threshold=1, sample_rate=0.1):
        """If less than 1, only take 10%. Bigger than i, don't touch.
        """
        small_edge = edge[edge["trip"] <= threshold]
        small_edge = small_edge.sample(round(small_edge.shape[0] * sample_rate))
        
        big_edge = edge[edge["trip"] > threshold]
                
        sampled_edge = pd.concat([small_edge, big_edge], ignore_index=True)
        return sampled_edge
        

def preprocess(coord_preprocessor, id_preprocessor, edge_preprocessor, ugnn_grid_preprocessor):
    
    """
    Preprocess grid, edge data.
    """
    

    p_data = {}  # preprocessed data
    
    p_data['id_coord_dict'] = coord_preprocessor()
        
    p_data['train_id'], p_data['valid_id'] = id_preprocessor(p_data['id_coord_dict'])
        
    p_data['valid_edge'], p_data['edge_dim'] = edge_preprocessor.load_valid_iteratively(p_data['valid_id'])
        
    p_data['grid'] = ugnn_grid_preprocessor()
    p_data["valid_edge"] = p_data["valid_edge"].merge(p_data["grid"]["city_metric"], on="city")
    
    return p_data

def test_preprocess(coord_preprocessor, id_preprocessor, edge_preprocessor, ugnn_grid_preprocessor):
    
    """
    Preprocess grid, test edge data.
    """
    

    p_data = {}  # preprocessed data
    
    p_data['id_coord_dict'] = coord_preprocessor()
        
    p_data['train_id'], p_data['valid_id'] = id_preprocessor(p_data['id_coord_dict'])
        
    p_data['test_edge'], p_data['edge_dim'] = edge_preprocessor.load_test(p_data['train_id'])  # only difference
        
    p_data['grid'] = ugnn_grid_preprocessor()
    p_data["test_edge"] = p_data["test_edge"].merge(p_data["grid"]["city_metric"], on="city")
    
    return p_data





def test_preprocess_(coord_preprocessor, id_preprocessor, edge_preprocessor, ugnn_grid_preprocessor):
    
    """
    Preprocess grid, test edge data.
    """
    

    p_data = {}  # preprocessed data
    
    p_data['id_coord_dict'] = coord_preprocessor()
        
    p_data['train_id'], p_data['valid_id'] = id_preprocessor(p_data['id_coord_dict'])
        
    p_data['test_edge'], p_data['edge_dim'] = edge_preprocessor.load_test_()  # only difference
        
    p_data['grid'] = ugnn_grid_preprocessor()
    p_data["test_edge"] = p_data["test_edge"].merge(p_data["grid"]["city_metric"], on="city")
    
    return p_data


def sample_every_epoch(edge_preprocessor, p_data, augmentation):
    
    """
    Sample the edge data for different epoch, so that model can learn better
    in terms of short trip edges.

    """
    
    sampled_edge = edge_preprocessor(p_data["train_id"])
    
    sampled_edge = sampled_edge.merge(p_data['grid']['city_metric'], on='city')
    
    if augmentation:
        # Data augmentation for each epoch
        augmented_columns = ["城市人口", "总GDP", "行政区面积", "建城区面积", "lng", "lat"]
        bias_df = pd.DataFrame(np.random.normal(0, 0.01, [sampled_edge.shape[0], len(augmented_columns)]),
                                columns=augmented_columns)
        augmented_df = sampled_edge[augmented_columns] + bias_df
        augmented_df[augmented_df < 0] = 0
        augmented_df[augmented_df > 1] = 1
        sampled_edge[augmented_columns] = augmented_df
        
    return p_data['grid']['grid_basic_dict'], p_data['grid']['grid_extra_dict'], sampled_edge


