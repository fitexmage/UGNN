import random
import torch.nn.functional as F
import torch
import pandas as pd


def preprocess(coord_preprocessor, id_preprocessor, edge_preprocessor, grid_preprocessor):
    preprocessed_data = {}
    preprocessed_data['id_coord_dict'] = coord_preprocessor()
        
    preprocessed_data['train_id'], preprocessed_data['valid_id']\
        = id_preprocessor(preprocessed_data['id_coord_dict'])
        
    preprocessed_data['train_edge'], preprocessed_data['valid_edge'], edge_dim\
        = edge_preprocessor(preprocessed_data['train_id'], preprocessed_data['valid_id'])
        
    preprocessed_data['grid'], grid_dim = grid_preprocessor()
        
    return preprocessed_data, edge_dim, grid_dim

def resample(edge, id):
    """
    Actually, it should be called regenerate.
    Our data is very imbalanced(Too many short trip, too few long trip),
    here we resample the edge data.
    If num_of_edges_of_an_o_id is less than 512, we resample the edges from all the other ids,
    and pad the trip by 0.
    
    Here, resampling is because of the need of paper. Author need to use grouped y sum to do augmentation. We need to replicate that
    method. So we write it on.
    
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
            
            rest_d_id = list(set(id[sub_city]) - d_id_set)
            rest_to_fill_num = max_length - sub_edge.shape[0]
            neg_sample_d_ids = random.sample(rest_d_id, k=rest_to_fill_num)
            
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


def make_data_for_dataset(*preprocessor):
    
    # Preprocess the data
    # p_data: preprocessed_data
    p_data, edge_dim, grid_dim = preprocess(*preprocessor)

    # Resample and prepare training data
    train_data = resample(p_data["train_edge"], p_data["train_id"])

    # Resample and prepare evaluation data
    eval_data = resample(p_data["valid_edge"], p_data["valid_id"])

    # Return the prepared data along with relevant information
    return train_data, eval_data, edge_dim, grid_dim, p_data['grid'], p_data['id_coord_dict']


def make_data_for_test_dataset(*preprocessor):
    
    # Preprocess the data
    # p_data: preprocessed_data
    p_data, edge_dim, grid_dim = preprocess(*preprocessor)

    # Concatenate ID and edges for the test data
    p_data['test_id'] = {**p_data['train_id'], **p_data['valid_id']}
    p_data['test_edge'] = pd.concat((p_data['train_edge'], p_data['valid_edge']))

    # Resample and prepare test data
    test_data = resample(p_data['test_edge'], p_data['test_id'])

    # Return the prepared test data along with relevant information
    return test_data, edge_dim, grid_dim, p_data['grid'], p_data['id_coord_dict'], p_data['test_edge']
