from decimal import Decimal
import numpy as np
import torch
import random
import geopandas
# import matplotlib.pyplot as plt
import os
from config import *
import argparse
import json
from sklearn import preprocessing
from typing import List, Tuple, Dict
from torch import Tensor
from src.utils.model_util import get_score_in_scope
import datetime

# from data_processing.trans_coord import GisTransform


project_path = "/home/lai/ugnn/"

def setup_seed(seed):
    torch.manual_seed(seed)  # set parameters to be the same for cpu
    torch.cuda.manual_seed(seed)  # set parameters to be the same for gpu
    torch.cuda.manual_seed_all(seed) # set parameters to be the same for multi gpus
    np.random.seed(seed)



# path util

def calculate_md5(file_path):
    import hashlib
    # Create an MD5 hash object
    md5_hash = hashlib.md5()

    # Open the file in binary mode for reading
    with open(file_path, "rb") as file:
        # Read the entire file into memory
        file_contents = file.read()
        md5_hash.update(file_contents)

    # Get the hexadecimal representation of the MD5 hash
    md5_hex = md5_hash.hexdigest()

    return md5_hex


def get_all_dir_paths(dir_path):
    # get all the file paths in a given directory
    all_dir_paths = []
    for root, dirs, files in os.walk(dir_path):
        for dir in dirs:
            all_dir_paths.append(os.path.join(root, dir))
    return all_dir_paths


def get_all_file_paths(dir_path):
    # get all the file paths in a given directory
    all_file_paths = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            all_file_paths.append(os.path.join(root, file))
    return all_file_paths


def get_all_shp_file_paths(dir_path):
    # get all shape file path in a given directory.
    # because the shape file has volume limitation, it is used when the shape file is separated to parts.
    file_set = set()
    for file in os.listdir(dir_path):
        file_name = file.split(".")[0]  # not use os.path.split because file with ".shp.xml" may exist
        file_set.add(file_name)
    all_shp_files = [os.path.join(dir_path, "{}.shp".format(file_name)) for file_name in file_set]
    return all_shp_files


# ID and coordinate util

def get_id_coord(city: str, CITY_CENTROID_PATH) -> pd.DataFrame:
    """
    Load id, lng, lat as dict of df from geo file.
    :return:
    id_coord_df: 
                id         lng        lat
    0      1.154250e+14  115.420976  39.771141
    1      1.154250e+14  115.420952  39.951131

    Lng and lat are from WGS-84, id are from Telecom
    """
    city_centroid_gdf = geopandas.read_file(os.path.join(CITY_CENTROID_PATH, city, "{}.shp".format(city)))


    dicts = []
    for row in np.array(city_centroid_gdf):
        dicts.append({"id": row[0], "lng": row[1].x, "lat": row[1].y})
    id_coord_df = pd.DataFrame(dicts)
    return id_coord_df


def get_coord_from_id(id, transform=False):
    # get the coordinates from the given cell ID. should be attention to float computing bias.
    id = str(int(id))  # id can be int or float
    if len(id) == 11:
        lng, lat = float(Decimal(str(id[: 6])) / 1000), float(Decimal(str(id[6: 11])) / 1000)
    elif len(id) == 14:
        lng, lat = float(Decimal(str(id[: 6])) / 10000), float(Decimal(str(id[6: 12])) / 10000)
    elif len(id) == 15:
        lng, lat = float(Decimal(str(id[: 7])) / 10000), float(Decimal(str(id[7: 13])) / 10000)
    else:
        raise Exception
    if transform:
        lng, lat = transform_coord(lng, lat)
    return lng, lat


def get_id_from_coord(lng, lat):
    # get the ID from the given coordinates. should be attention to float computing bias.
    id = str(int(Decimal(str(lng)) * 10000)) + str(int(Decimal(str(lat)) * 10000)) + "40"
    if len(id) not in [14, 15] or "." in id or id[-3] != "0" or id[-9] != "0":
        print("Format of ID is wrong!")
        print(lng, lat, id)
    return id


def transform_coord(lng, lat, transform=None, transform_dict=None):
    # transform the coordinate to another standard
    if transform is None:
        transform = GisTransform("gcj02", "wgs84")  # 经纬度: wgs84 墨卡托: webMercator 国测局: gcj02
    if transform_dict is None:
        trans_lng, trans_lat = transform.transform_func(lng, lat)
    else:
        if lng in transform_dict and lat in transform_dict[lng]:
            trans_lng, trans_lat = transform_dict[lng][lat]
        else:
            trans_lng, trans_lat = transform.transform_func(lng, lat)
            transform_dict[lng][lat] = (trans_lng, trans_lat)
    return trans_lng, trans_lat


def add_to_coordinate(coordinate, num):
    # add a number to the coordinate. should be attention to float computing bias.
    return float((Decimal(str(coordinate)) + Decimal(str(num))))


def add_to_id(id, units):
    """
    :param id: DataFrame: n by 1 ids
    :param units: one of the directions the grid needs to add
    :return: DataFrame: n by 1 ids
    """
    # add longitude and latitude to the ID or data frame of ID. ID must be integer.
    if isinstance(id, int):
        first_id = id
    elif isinstance(id, pd.DataFrame):
        first_id = id["id"][0]
    else:
        raise Exception

    if len(str(int(first_id))) == 11:
        new_id = id + 500000 * units[0] + 5 * units[1]
    elif len(str(int(first_id))) in [14, 15]:
        new_id = id + 5000000000 * units[0] + 5000 * units[1]
    else:
        print("id:", str(first_id))
        raise Exception
    return new_id

def add_to_id1000(id, units):
    """
    :param id: DataFrame: n by 1 ids
    :param units: one of the directions the grid needs to add
    :return: DataFrame: n by 1 ids
    """
    # add longitude and latitude to the ID or data frame of ID. ID must be integer.
    if isinstance(id, int):
        first_id = id
    elif isinstance(id, pd.DataFrame):
        first_id = id["id"][0]
    else:
        raise Exception

    if len(str(int(first_id))) == 11:
        new_id = id + 1000000 * units[0] + 10 * units[1]
    elif len(str(int(first_id))) in [14, 15]:
        new_id = id + 10000000000 * units[0] + 10000 * units[1]
    else:
        print("id:", str(first_id))
        raise Exception
    return new_id


def get_area_ids(ld_id, ru_id=None):
    if ru_id is None:
        return [ld_id]
    ld_lng, ld_lat = get_coord_from_id(ld_id)
    ru_lng, ru_lat = get_coord_from_id(ru_id)

    area_ids = []
    for i in range(round((ru_lng - ld_lng) / 0.005) + 1):
        for j in range(round((ru_lat - ld_lat) / 0.005) + 1):
            id = add_to_id(ld_id, [i, j])
            area_ids.append(id)
    return area_ids


def combine_id(id_df, combine_rate):
    # id must be int or float, default is 15 digits
    end_df = id_df % 5000000000 - 40
    front_df = id_df - end_df - 40
    front_df = front_df - front_df % (combine_rate * 5000000000)
    end_df = end_df - end_df % (combine_rate * 5000)
    combined_id_df = front_df + end_df + 40
    return combined_id_df


def combine_od(od_df, combine_rate):
    od_df["o_id"] = combine_id(od_df["o_id"], combine_rate)
    od_df["d_id"] = combine_id(od_df["d_id"], combine_rate)
    od_df = od_df.groupby(["hour", "o_id", "d_id"])["trip"].sum().reset_index()

    return od_df


# PA and OD util


def get_pa(city):
    if config.od_type == "gt":
        pa_df = pd.read_csv(os.path.join(config.TELECOM_PA_PATH, "{}.csv".format(city)))
    elif config.od_type == "unicom":
        pa_df = pd.read_csv(os.path.join(config.UNICOM_PA_PATH, "{}.csv".format(city)))
    elif config.od_type == "telecom":
        pa_df = pd.read_csv(os.path.join(config.TEMP_TELECOM_OUTPUT_PA_PATH, "{}.csv".format(city)))
    else:
        raise Exception
    return pa_df


def get_od(city):
    if config.od_type == "gt":
        city_od_df = pd.read_csv(os.path.join(config.TELECOM_OUTPUT_OD_PATH, "{}.csv".format(city)))
    elif config.od_type == "filtered":
        city_od_df = pd.read_csv(os.path.join(config.TELECOM_FILTERED_OD_PATH, "{}.csv".format(city)))
    elif config.od_type == "telecom":
        city_od_df = pd.read_csv(os.path.join(config.TEMP_TELECOM_OUTPUT_OD_PATH, "{}.csv".format(city)))
    elif config.od_type == "unicom":
        city_od_df = pd.read_csv(os.path.join(config.UNICOM_OUTPUT_OD_PATH, "{}.csv".format(city)))
    else:
        raise Exception
    return city_od_df


def get_id(city: str) -> pd.DataFrame:
    """
    Return a df of ids.

    The ids are o and d data stored by Telecom or Unicom.
    Specifically, at the beginning, the data from Telecom looks like below:

    | hour | o_id(original) | d_id(destination) | trip |
    | ---- | -------------- | ----------------- | ---- |
    |      |                |                   |      |
    |      |                |                   |      |
    We combine hours to days and then split the big data into cities according to CITY_CENTROID data.
    Note that CITY_CENTROID data will update. So we need to do `isin` as filtering after calling this
    function.

    """
    if config.od_type == "gt":
        if config.id_version == 2020:
            city_id_df = pd.read_csv(os.path.join(config.TELECOM_OUTPUT_ID_2020_PATH, "{}.csv".format(city)))
        else:
            city_id_df = pd.read_csv(os.path.join(config.TELECOM_OUTPUT_ID_PATH, "{}.csv".format(city)))
    elif config.od_type == "filtered":
        if config.id_version == 2020:
            city_id_df = pd.read_csv(os.path.join(config.TELECOM_OUTPUT_ID_2020_PATH, "{}.csv".format(city)))
        else:
            city_id_df = pd.read_csv(os.path.join(config.TELECOM_OUTPUT_ID_PATH, "{}.csv".format(city)))
    elif config.od_type == "telecom":
        city_id_df = pd.read_csv(os.path.join(config.TEMP_TELECOM_OUTPUT_ID_PATH, "{}.csv".format(city)))
    elif config.od_type == "unicom":
        city_id_df = pd.read_csv(os.path.join(config.UNICOM_OUTPUT_ID_PATH, "{}.csv".format(city)))
    else:
        raise Exception
    return city_id_df


def get_edge(ALL_EDGE_PATH, city, hour):
    """Return edge DataFrame from a city within an hour.

    Returns:
        pd.DataFrame: 
        edge.columns
        ['hour', 'o_id', 'd_id', 'trip', 'surface_distance', 'grid_to_grid_distance', 'grid_to_grid_time', 'city]
    """
    edges: List[pd.DataFrame] = []
    
    iterator = pd.read_csv(os.path.join(ALL_EDGE_PATH, "{}.csv".format(city)), chunksize=10000000)
    for edge in iterator:
        edge = edge[edge["hour"] == hour].reset_index(drop=True)
        edges.append(edge)
    edges = pd.concat(edges, ignore_index=True)
    
    edges.loc[:, "city"] = city  # add one column, city
    
    return edges

def get_edge_no_hour(ALL_EDGE_PATH, city):
    """Return edge DataFrame from a city within an hour.

    Returns:
        pd.DataFrame: 
        edge.columns
        ['hour', 'o_id', 'd_id', 'trip', 'surface_distance', 'grid_to_grid_distance', 'grid_to_grid_time', 'city]
    """
    edges: List[pd.DataFrame] = []
    
    iterator = pd.read_csv(os.path.join(ALL_EDGE_PATH, "{}.csv".format(city)), chunksize=10000000)
    for edge in iterator:
        edges.append(edge)
    edges = pd.concat(edges, ignore_index=True)
    
    edges.loc[:, "city"] = city  # add one column, city
    
    return edges

def enlarge_grid(center_id_df, grid, grid_radius):
    
    """
    Enlarge grid by adding surrounding directions. e.g.
                              +-----+-----+-------+
                              | -1,1|..   | ..    |
                +---+         +-----+-----+-------+
                |0, 0| ------>| -1,0| 0,0 |1,0    |
                +---+         +-----+-----+-------+
                              | ..  |..   | ..    |
                              +-----+-----+-------+
    For self.config.grid_radius=2, there are 9 directions.

    Args:
        grid (pd.DataFrame): The original grid data.
        grid_radius (int): The radius for enlargement.

    Returns:
        pd.DataFrame: Enlarged grid data. shape: (N, 442)
    """

    directions = []
    for row in range(-grid_radius + 1, grid_radius):
        for col in range(-grid_radius + 1, grid_radius):
            directions.append([row, col])

    enlarged_grid = []
    for direction in directions:
        """
        Add 0.05 to id's coordinates to get the surrounding neighbor's id.
        Because id contains coord info, and each grid is 0.05 km wide, here we just add 0.05 to id's coord,
        and we can get a new id, that is the surrounded neighbor of the original one.
        """
        id_df = add_to_id(center_id_df, direction)
        grid_df = id_df.merge(grid, how="left", on="id")
        grid_df = grid_df.fillna(0)
        grid_df = grid_df.drop(columns="id")
        # Here grid_df is the 49 columns of one of the neighbors.
        enlarged_grid.append(grid_df)
    enlarged_grid = pd.concat([center_id_df] + enlarged_grid, axis=1)

    return enlarged_grid


def grid_reshape(grid, model_name, grid_dim, grid_num, grid_radius) -> Dict[str, Tensor]:
    """
    Reshape data according to model name.
    For cnn model,
    """
    reshaped_grid = {}
    for row in np.array(grid):
        id = int(row[0])
        grid_data = row[1:]
        if model_name.startswith("cnn"):
            grid_data = grid_data.reshape([grid_radius*2-1, grid_radius*2-1, grid_dim]).T
        elif model_name.startswith("gat"):
            grid_data = grid_data.reshape(grid_num, grid_dim)
        reshaped_grid[id] = torch.Tensor(grid_data)
    return reshaped_grid



def wavg(group, value_name, weight_name):
    # weighted average algorithm used in pandas
    v = group[value_name]
    w = group[weight_name]
    try:
        return (v * w).sum() / w.sum()
    except ZeroDivisionError:
        return v.mean()


def weighted(group, value_name, weight_name):
    # weighted algorithm used in pandas
    v = group[value_name]
    w = group[weight_name]
    return (v * w).sum()


# analysis util


def get_pa_in_area(df, city, area):
    if isinstance(area, int):
        df = df[df["id"] == area]
    elif area in config.area_scope_dict[city]:
        area_ids = get_area_ids(*config.area_scope_dict[city][area])
        df = df[df["id"].isin(area_ids)]
    elif area.endswith("区"):
        city_centroid_gdf = geopandas.read_file(os.path.join(config.CITY_CENTROID_PATH, city, "{}.shp".format(city)))
        county_gdf = geopandas.read_file(config.COUNTY_DIVISION_PATH)
        county_gdf = county_gdf[county_gdf["NAME"] == area]
        within_gdf = geopandas.sjoin(city_centroid_gdf, county_gdf, op="within")
        df = df[df["id"].isin(within_gdf["id"])]
    else:
        print("Wrong area name!")
    return df


def get_od_in_area(df, city, area, direction=None):
    if area in config.area_scope_dict[city]:
        area_ids = get_area_ids(*config.area_scope_dict[city][area])
    elif area.endswith("区"):
        city_centroid_gdf = geopandas.read_file(os.path.join(config.CITY_CENTROID_PATH, city, "{}.shp".format(city)))
        county_gdf = geopandas.read_file(config.COUNTY_DIVISION_PATH)
        county_gdf = county_gdf[county_gdf["NAME"] == area]
        within_gdf = geopandas.sjoin(city_centroid_gdf, county_gdf, op="within")
        area_ids = within_gdf["id"]
    else:
        raise Exception
    if direction == "o":
        df = df[df["o_id"].isin(area_ids)]
    elif direction == "d":
        df = df[df["d_id"].isin(area_ids)]
    else:
        df = df[(df["o_id"].isin(area_ids)) | (df["d_id"].isin(area_ids))]
    return df


def get_pa_in_hour(df, hour):
    if isinstance(hour, int):
        df = df[df["hour"] == hour]
        del df["hour"]
    else:
        hour_span = hour.split("-")
        df = df[df["hour"].isin(list(range(int(hour_span[0]), int(hour_span[1]) + 1)))]
        df = df.groupby(["id", "lng", "lat"])["trip"].sum().reset_index()
    return df


def get_od_in_hour(df, hour):
    if isinstance(hour, int):
        df = df[df["hour"] == hour]
        del df["hour"]
    else:
        hour_span = hour.split("-")
        df = df[df["hour"].isin(list(range(int(hour_span[0]), int(hour_span[1]) + 1)))]
        df = df.groupby(["o_id", "d_id", "o_lng", "o_lat", "d_lng", "d_lat"])["trip"].sum().reset_index()
    return df


def get_visualized_pa(pa_df, city, type, hour, area):
    if area:
        pa_df = get_pa_in_area(pa_df, city, area)

    id_coord_df = get_id_coord(city)
    pa_df = pa_df.merge(id_coord_df, on="id")

    city_visualize_dir_path = os.path.join(config.VISUALIZE_PATH, type, city)
    if not os.path.exists(city_visualize_dir_path):
        os.makedirs(city_visualize_dir_path)

    pa_by_hour_df = get_pa_in_hour(pa_df, hour)
    pa_by_hour_df.to_csv(os.path.join(city_visualize_dir_path, "{}_{}_{}_gt_tongzhou.csv".format(city, type, hour)), index=False)


def get_visualized_od(od_df, city, type, hour, area, max_percent=0.03):
    if area:
        od_df = get_od_in_area(od_df, city, area)

    id_coord_df = get_id_coord(city)
    od_df = od_df.merge(id_coord_df, left_on="o_id", right_on="id").merge(id_coord_df, left_on="d_id", right_on="id")
    # od_df['hour'] = [8]*len(od_df)
    od_df = od_df[["hour", "o_id", "d_id", "trip", "lng_x", "lat_x", "lng_y", "lat_y"]]
    # od_df = od_df[["hour", "o_id", "d_id", "pred_trip", "lng_x", "lat_x", "lng_y", "lat_y"]]
    od_df.columns = ["hour", "o_id", "d_id", "trip", "o_lng", "o_lat", "d_lng", "d_lat"]

    city_visualize_dir_path = os.path.join(config.VISUALIZE_PATH, type, city)
    if not os.path.exists(city_visualize_dir_path):
        os.makedirs(city_visualize_dir_path)
    od_by_hour_df = get_od_in_hour(od_df, hour)
    print("hour size:", len(od_by_hour_df))
    random.seed(95)
    od_by_hour_df = od_by_hour_df.nlargest(round(od_by_hour_df.shape[0] * max_percent), "trip")
    # od_by_hour_df = od_by_hour_df.sample(frac=0.03)
    od_by_hour_df.to_csv(os.path.join(city_visualize_dir_path, "{}_{}_{}.csv".format(city, type, hour)), index=False)


def analyze_distance(od_df):
    od_df = od_df.reset_index(drop=True)
    distance_range_df = pd.cut(x=od_df["surface_distance"],
                               bins=[i * 1000 for i in range(0, 51)] + [float('inf')],
                               labels=[i for i in range(0, 51)]).reset_index()
    distance_range_df = pd.concat([od_df[["trip"]], distance_range_df[["surface_distance"]]], axis=1)
    distance_range_sum_df = distance_range_df.groupby("surface_distance")["trip"].sum().reset_index()
    distance_range_sum_df["trip_prop"] = distance_range_sum_df["trip"] / distance_range_sum_df["trip"].sum()

    od_df["weighted_trip_distance"] = od_df["surface_distance"] * od_df["trip"]
    average_distance = od_df["weighted_trip_distance"].sum() / od_df["trip"].sum()
    return distance_range_sum_df, average_distance


def get_surface_distance(coord_array_1, coord_array_2):
    # get the surface distance between two numpy array
    rad_array_1 = np.divide(np.multiply(coord_array_1, config.pi), 180.0)
    rad_array_2 = np.divide(np.multiply(coord_array_2, config.pi), 180.0)
    a = rad_array_1[1] - rad_array_2[1]
    b = rad_array_1[0] - rad_array_2[0]
    s = np.multiply(2, np.arcsin(np.sqrt(
        np.add(np.power(np.sin(np.divide(a, 2)), 2),
               np.multiply(np.multiply(np.cos(rad_array_1[1]), np.cos(rad_array_2[1])),
                           np.power(np.sin(np.divide(b, 2)), 2))))))

    s = np.multiply(np.multiply(s, config.earth_radius), 1000)
    return s


def get_od_distance(city_od_df, city, hour=None, area=None):
    if hour:
        city_od_df = get_od_in_hour(city_od_df, hour)
    if area:
        city_od_df = get_od_in_area(city_od_df, city, area)

    id_coord_df = get_id_coord(city)
    city_od_df = city_od_df.merge(id_coord_df, left_on="o_id", right_on="id").merge(id_coord_df, left_on="d_id",
                                                                                    right_on="id").reset_index(
        drop=True)
    city_od_df = city_od_df[["o_id", "d_id", "trip", "lng_x", "lat_x", "lng_y", "lat_y"]]
    city_od_df.columns = ["o_id", "d_id", "trip", "o_lng", "o_lat", "d_lng", "d_lat"]
    o_coord_array = np.array(city_od_df[["o_lng", "o_lat"]]).T
    d_coord_array = np.array(city_od_df[["d_lng", "d_lat"]]).T
    city_distance_array = get_surface_distance(o_coord_array, d_coord_array).T
    city_surface_distance_df = pd.DataFrame(city_distance_array, columns=["surface_distance"])
    city_od_df = pd.concat([city_od_df[["o_id", "d_id", "trip"]], city_surface_distance_df], axis=1)
    distance_df, average_distance = analyze_distance(city_od_df)

    plt.plot(distance_df["surface_distance"], distance_df["trip"], "b", label="telecom")
    plt.legend(loc=1)
    plt.savefig(os.path.join(config.DATA_ANALYSIS_PATH, "distance", "{}.jpg".format(city)))
    plt.close()
    distance_df.to_csv(os.path.join(config.DATA_ANALYSIS_PATH, "distance", "{}.csv".format(city)), index=False)

    average_distance_df = pd.DataFrame([average_distance], columns=["average_distance"])
    average_distance_df.to_csv(os.path.join(config.DATA_ANALYSIS_PATH, "distance", "{}_average.csv".format(city)),
                               index=False)


def get_od_distance_diff(city, hour=None, area=None):
    city_telecom_od_df = pd.read_csv(os.path.join(config.TELECOM_OUTPUT_OD_PATH, "{}.csv".format(city)))
    city_unicom_od_df = pd.read_csv(os.path.join(config.UNICOM_OUTPUT_OD_PATH, "{}.csv".format(city)))

    if hour:
        city_telecom_od_df = get_od_in_hour(city_telecom_od_df, hour)
        city_unicom_od_df = get_od_in_hour(city_unicom_od_df, hour)

    if area:
        city_telecom_od_df = get_od_in_area(city_telecom_od_df, city, area)
        city_unicom_od_df = get_od_in_area(city_unicom_od_df, city, area)

    city_telecom_surface_distance_df = pd.read_csv(
        os.path.join("../data/processed_data/surface_distance/{}.csv".format(city)))
    city_telecom_od_df = city_telecom_od_df.merge(city_telecom_surface_distance_df, on=["o_id", "d_id"])
    telecom_distance_df, telecom_average_distance = analyze_distance(city_telecom_od_df)

    city_unicom_surface_distance_df = pd.read_csv(
        os.path.join("../unicom_data/processed_data/surface_distance/{}.csv".format(city)))
    city_unicom_od_df = city_unicom_od_df.merge(city_unicom_surface_distance_df, on=["o_id", "d_id"])
    unicom_distance_df, unicom_average_distance = analyze_distance(city_unicom_od_df)

    plt.plot(telecom_distance_df["surface_distance"], telecom_distance_df["trip"], "b", label="telecom")
    plt.plot(unicom_distance_df["surface_distance"], unicom_distance_df["trip"], "r", label="unicom")
    plt.legend(loc=1)
    plt.savefig(os.path.join(config.DATA_ANALYSIS_PATH, "distance", "{}.jpg".format(city)))
    plt.close()

    trip_df = pd.concat([telecom_distance_df[["trip"]], unicom_distance_df[["trip"]]], axis=1)
    coincidence = trip_df.min(axis=1).sum() / trip_df.max(axis=1).sum()
    compare_data_df = pd.DataFrame([[telecom_average_distance, coincidence], [unicom_average_distance, coincidence]],
                                   columns=["average_distance", "coincidence"])
    compare_data_df.to_csv(os.path.join(config.DATA_ANALYSIS_PATH, "distance", "{}.csv".format(city)), index=False)

def load_checkpoint(config):
    
    def right_checkpoint(checkpoint, config):
        
        def compare_dicts(long_dict, short_dict):
            # Check if all keys and values in short_dict match long_dict
            for key, value in short_dict.items():
                if key not in long_dict or long_dict[key] != value:
                    raise ValueError(f"Key '{key}' or its value does not match in the dictionaries.")

            # Print the remaining keys in long_dict
            remaining_keys = set(long_dict.keys()) - set(short_dict.keys())
            if len(remaining_keys) > 0:
                print(f"Remaining keys in long_dict: {remaining_keys}")
        
        if "config" in checkpoint:
            # Compare if checkpoint["config"] == config
            try:
                if len(checkpoint["config"] ) > len(config.__dict__):
                    compare_dicts(checkpoint["config"], config.__dict__)
                else:
                    compare_dicts(config.__dict__, checkpoint["config"])
                print("Model config is correct.")
            except ValueError as e:
                print(str(e))
                import sys
                user_input = input("Type 'y' to continue: ")
                if user_input.lower() == 'y':
                    print("Continuing with the program...")
                    # Your code logic goes here
                else:
                    print("Terminating the program.")
                    sys.exit()
        else:
            print(">>>This is a old model, can't check model config.")
    print()
    print("Loading checkpoint")
    if config.reuse_model:
        if config.model_status == "train":
            checkpoint = torch.load(config.MODEL_SAVE_PATH)
            
            print(f"Training, Reading model from {config.MODEL_SAVE_PATH}.")
            stat = os.stat(config.MODEL_SAVE_PATH)
            creation_time = stat.st_mtime
            human_readable_time = datetime.datetime.fromtimestamp(creation_time)
            formatted_time = human_readable_time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"Create time {formatted_time}.")
            
        else:
            try:
                checkpoint = torch.load(config.BEST_MODEL_SAVE_PATH)
            except:
                print("You may be using some heathen model like xgb, try escaping torch.load.")
                checkpoint = {}
                
            print(f"Testing, reading model from {config.BEST_MODEL_SAVE_PATH}")
            stat = os.stat(config.BEST_MODEL_SAVE_PATH)
            creation_time = stat.st_mtime
            human_readable_time = datetime.datetime.fromtimestamp(creation_time)
            formatted_time = human_readable_time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"Create time {formatted_time}.")
        
        # Check if checkpint in of right config.
        right_checkpoint(checkpoint, config)    
    
    else:
        checkpoint = {}
        print("Creating empty checkpoint.")
    return checkpoint

def normalize_df(df, frozen_columns, scaler_name, checkpoint):
    """Normalize grid data for selected cities in full version(include basic info
    and sparce matrices, i.e. city_grid_extra_df)"""
    keep_df = df[frozen_columns].reset_index()
    df = df.drop(columns=frozen_columns)

    if scaler_name in checkpoint:
        scaler = checkpoint[scaler_name]
        normalized_array = scaler.transform(df)
    else:
        scaler = preprocessing.MinMaxScaler()
        normalized_array = scaler.fit_transform(df)
        checkpoint[scaler_name] = scaler

    df = pd.DataFrame(normalized_array, columns=df.columns)
    df = pd.concat([keep_df, df], axis=1).drop(columns='index')
    return df

def make_save_dir(project_name):
    """
    
    save_dir = os.path.join(config.SAVE_PATH, project_name)
    
    make dir if not exist
    
    """
    # Combine save_path and project_name
    save_dir = os.path.join(config.SAVE_PATH, project_name)
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def get_gpu_memory_usage():
    """
    Prints the memory usage of all available GPUs and returns the name of the GPU with the largest available space.
    
    Returns:
        str: The name of the GPU with the largest available space.
    """
    if torch.cuda.is_available():
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        
        if num_gpus > 0:
            print(f"Number of available GPUs: {num_gpus}")
            
            # Initialize variables to keep track of the largest memory and its corresponding GPU
            av_memories = []
            devices = []
            
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.mem_get_info(i)[0]  # Available mem
                gpu_memory_gb = gpu_memory / (1024 ** 3)  # Convert to MB
                
                print(f"GPU {i} ({gpu_name}): {gpu_memory_gb:.2f} GB")
                
                av_memories.append(gpu_memory_gb)
                devices.append('cuda:'+str(i))
            max_mem = max(av_memories)
            max_device = devices[av_memories.index(max_mem)]
            print(f"GPU with the largest available memory: {max_device}")
            return max_device
        else:
            print("No GPUs available.")
    else:
        print("CUDA is not available. Make sure you have an NVIDIA GPU and PyTorch with CUDA support installed.")
        return None

def get_score(result_df, city=None, hour=None):
    """Return predicted score.
    result_df has two columns: pred_trip, gt_trip

    if city:
        score = [city]
    else:
        score = [hour]
    This part might bare imperfection.
    
    """
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
    score = None  # score has been abandoned.
    return score



if __name__ == "__main__":
    # ids = get_area_ids(*[116265040035040, 116305040050040])
    # print(ids)
    from config import *
    gr_df = pd.read_csv("武汉市_gravity.csv")
    print(gr_df['pred_trip'].sum())
    print(len(gr_df))
    gt_path = "/home/user/disk/2019/edge_filtered_2019/武汉市.csv"
    gt_df = pd.read_csv(gt_path)
    print(gt_df[gt_df['hour'] == 8]['trip'].sum())
    print(len(gt_df[gt_df['hour'] == 8]))

