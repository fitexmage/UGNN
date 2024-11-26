from torch.utils.data import Dataset
# from torch_geometric.data import Dataset as GeoDataset, Data
import torch.nn.functional as F
from util import *


class GegnDataset(Dataset):
    def __init__(self, grid_basic_dict, grid_extra_dict, edge_df, one_hot_hour):
        super(GegnDataset, self).__init__()

        # self.grid_dict = grid_dict
        self.grid_basic_dict = grid_basic_dict
        self.grid_extra_dict = grid_extra_dict
        self.o_ids = edge_df[["o_id"]].values
        self.d_ids = edge_df[["d_id"]].values
        

        self.trips = edge_df[["trip"]].values
        if one_hot_hour:
            self.externals = edge_df[["城市人口", "总GDP", "行政区面积", "建城区面积", "lng", "lat"]].values
            hour = F.one_hot(torch.tensor(edge_df["hour"]))
            others = torch.Tensor(edge_df[["surface_distance", "grid_to_grid_distance", "grid_to_grid_time"]].values)
            self.edges = torch.cat((hour, others), dim=1)
        else:
            self.edges = edge_df[["surface_distance", "grid_to_grid_distance", "grid_to_grid_time"]].values
            # self.edges[:, 0] = self.edges[:, 0] + 10
            # print('\n\n\nadding 1 to resi')
            self.externals = edge_df[["城市人口", "总GDP", "行政区面积", "建城区面积", "lng", "lat", "hour"]].values
            # self.externals[:, 3] = self.externals[:, 3] + 10
            # print('\n\n\nadding 1 to resi')
            self.edges = torch.Tensor(self.edges)
            print("No one hot hour.")
        self.externals = torch.Tensor(self.externals)
        self.trips = torch.Tensor(self.trips).squeeze(-1)

    def __len__(self):
        return self.trips.shape[0]

    def __getitem__(self, item):

        o_grid_basic_x = self.grid_basic_dict[self.o_ids[item][0]]
        d_grid_basic_x = self.grid_basic_dict[self.d_ids[item][0]]
        o_grid_extra_x = self.grid_extra_dict[self.o_ids[item][0]]
        d_grid_extra_x = self.grid_extra_dict[self.d_ids[item][0]]

        edge_x = self.edges[item]
        external_x = self.externals[item]
        y = self.trips[item]
        # print("o_grid_extra_x:\n", o_grid_extra_x.shape)
        # print("d_grid_extra_x:\n", d_grid_extra_x.shape)
        return o_grid_basic_x, d_grid_basic_x, o_grid_extra_x, d_grid_extra_x, edge_x, external_x, y


class DeepGravityDataset(Dataset):
    def __init__(self, grid_dict, edge_df, id_dict, id_coord_df, checkpoint):
        super(DeepGravityDataset, self).__init__()

        self.grid_dict = grid_dict
        self.id_dict = id_dict
        self.id_coord_df = id_coord_df
        self.checkpoint = checkpoint

        self.o_ids, self.d_ids, self.cities, self.trips = [], [], [], []
        groups = edge_df.groupby("o_id")
        for group in groups:
            o_edge_df = group[1]
            o_ids = o_edge_df[["o_id"]].values.squeeze(-1).tolist()
            d_ids = o_edge_df[["d_id"]].values.squeeze(-1).tolist()
            city = o_edge_df["city"].values[0]
            trips = torch.Tensor(o_edge_df[["trip"]].values).squeeze(-1)

            max_length = 512
            # print("o_edge_df.shape[0]:", o_edge_df.shape[0])
            if o_edge_df.shape[0] < max_length:
                o_ids = [o_ids[0] for _ in range(max_length)]
                d_id_set = set(o_edge_df[["d_id"]].values.squeeze(-1).tolist())
                neg_sample_d_ids = random.sample(list(set(self.id_dict[city]) - d_id_set), k=max_length-o_edge_df.shape[0])
                d_ids.extend(neg_sample_d_ids)
                trips = F.pad(trips, [0, max_length - o_edge_df.shape[0]])
            # print("o_edge_df.shape[0]:", o_edge_df.shape[0])
            self.o_ids.append(o_ids)
            self.d_ids.append(d_ids)
            self.cities.append(city)
            self.trips.append(trips)

    def __len__(self):
        return len(self.trips)

    def __getitem__(self, index):
        """
        :param index: index for dataloader
        :return:
        1. o_grid_x: grouped original locations
        2. d_grid_x: corresponding destination locations
        3. edge_x: surface distance
        4. expected flow
        """
        o_ids = self.o_ids[index]
        d_ids = self.d_ids[index]

        o_grid_x = torch.concat([self.grid_dict[o_id].unsqueeze(0) for o_id in o_ids])
        d_grid_x = torch.concat([self.grid_dict[d_id].unsqueeze(0) for d_id in d_ids])
        y = self.trips[index]

        id_df = pd.concat([pd.DataFrame(o_ids, columns=["o_id"], dtype="int64"),
                           pd.DataFrame(d_ids, columns=["d_id"], dtype="int64")], axis=1)
        id_coord_df = self.id_coord_df[self.cities[index]]
        id_df = id_df.merge(id_coord_df, left_on="o_id", right_on="id").merge(id_coord_df, left_on="d_id", right_on="id")
        id_df = id_df[["lng_x", "lat_x", "lng_y", "lat_y"]]
        id_df.columns = ["o_lng", "o_lat", "d_lng", "d_lat"]
        o_coord_array = np.array(id_df[["o_lng", "o_lat"]]).T
        d_coord_array = np.array(id_df[["d_lng", "d_lat"]]).T
        city_distance_df = pd.DataFrame(get_surface_distance(o_coord_array, d_coord_array), columns=["surface_distance"])
        city_distance_array = self.checkpoint["edge_scaler"].transform(city_distance_df)
        edge_x = torch.Tensor(city_distance_array)
        return o_grid_x, d_grid_x, edge_x, y

    def get_ids(self, item):
        return self.o_ids[item], self.d_ids[item]

#
# class GraphGegnDataset(GeoDataset):
#     def __init__(self, grid_dict, edge_index, edge_df):
#         super(GraphGegnDataset, self).__init__()
#
#         self.grid_dict = grid_dict
#         self.o_ids = edge_df[["o_id"]].values
#         self.d_ids = edge_df[["d_id"]].values
#         self.edges = edge_df[["surface_distance", "grid_to_grid_distance", "grid_to_grid_time"]].values
#         self.externals = edge_df[["城市人口", "总GDP", "行政区面积", "lng", "lat", "hour"]].values
#         self.trips = edge_df[["trip"]].values
#
#         self.edges = torch.Tensor(self.edges)
#         self.externals = torch.Tensor(self.externals)
#         self.trips = torch.Tensor(self.trips).squeeze(-1)
#
#         self.edge_index = torch.tensor(edge_index, dtype=torch.long).T
#
#     def __len__(self):
#         return self.trips.shape[0]
#
#     def __getitem__(self, item):
#         o_grid_x = self.grid_dict[self.o_ids[item][0]]
#         d_grid_x = self.grid_dict[self.d_ids[item][0]]
#         edge_index = self.edge_index
#         edge_x = self.edges[item]
#         external_x = self.externals[item]
#         y = self.trips[item]
#         return o_grid_x, d_grid_x, edge_index, edge_x, external_x, y



class SAGEDataset(Dataset):
    def __init__(self, o_ids, d_ids, id_dict, edge_df, one_hot_hour, edge_pair):
        super(SAGEDataset, self).__init__()

        # self.grid_dict = grid_dict

        self.o_ids = o_ids
        self.d_ids = d_ids
        self.id_dict = id_dict
        self.edge_pair = edge_pair
        

        self.trips = edge_df[["trip"]].values
        if one_hot_hour:
            self.externals = edge_df[["城市人口", "总GDP", "行政区面积", "建城区面积", "lng", "lat"]].values
            hour = F.one_hot(torch.tensor(edge_df["hour"]))
            others = torch.Tensor(edge_df[["surface_distance", "grid_to_grid_distance", "grid_to_grid_time"]].values)
            self.edges = torch.cat((hour, others), dim=1)
        else:
            self.edges = edge_df[["surface_distance", "grid_to_grid_distance", "grid_to_grid_time"]].values
            # self.edges[:, 0] = self.edges[:, 0] + 10
            # print('\n\n\nadding 1 to resi')
            self.externals = edge_df[["城市人口", "总GDP", "行政区面积", "建城区面积", "lng", "lat", "hour"]].values
            # self.externals[:, 3] = self.externals[:, 3] + 10
            # print('\n\n\nadding 1 to resi')
            self.edges = torch.Tensor(self.edges)
            print("No one hot hour.")
        self.externals = torch.Tensor(self.externals)
        self.trips = torch.Tensor(self.trips).squeeze(-1)
        self.o_edge = edge_df['o_id']
        self.d_edge = edge_df['d_id']

    def __len__(self):
        return len(self.o_edge)

    def __getitem__(self, item):
        o_ids, d_ids = self.o_edge[item], self.d_edge[item]
        # o_ids = self.o_ids[item]
        # d_ids = self.d_ids[item]
        # o_idx = self.id_dict[o_ids]
        # d_idx = self.id_dict[d_ids]

        edge_x = self.edges[item]
        external_x = self.externals[item]
        y = self.trips[item]
        # print("o_grid_extra_x:\n", o_grid_extra_x.shape)
        # print("d_grid_extra_x:\n", d_grid_extra_x.shape)
        return o_ids, d_ids, edge_x, external_x, y
