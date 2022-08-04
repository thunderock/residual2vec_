# @Filename:    geometric_datasets.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        8/2/22 12:47 PM
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data, download_url
import pandas as pd
from utils.config import *


class GeometricDataset(InMemoryDataset):
    def __init__(self, root='/tmp'):
        super(GeometricDataset, self).__init__(root=root, transform=None, pre_transform=None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self): return ['data.pt']

    def download(self): raise NotImplementedError

    def __repr__(self): return str(self.name.capitalize())

    @property
    def name(self): raise NotImplementedError("need to implement this!")

    @staticmethod
    def create_node_masks(d):
        print("Creating classification masks")
        cnt = len(d.x)
        # actually the index to the nodes
        nums = np.arange(cnt)
        np.random.shuffle(nums)

        train_size = int(cnt * 0.75)
        test_size = int(cnt * 0.25)

        train_set = nums[0:train_size]
        test_set = nums[train_size:train_size + test_size]

        assert abs(len(train_set) + len(test_set) - cnt) <= 1, "The split should be coherent. {} + {} != {}".format(len(train_set), len(test_set), cnt)

        train_mask = torch.zeros(cnt, dtype=torch.long, device=DEVICE)
        for i in train_set:
            train_mask[i] = 1.

        test_mask = torch.zeros(cnt, dtype=torch.long, device=DEVICE)
        for i in test_set:
            test_mask[i] = 1.
        d.train_mask = train_mask
        d.test_mask = test_mask

# https://github.com/Orbifold/pyg-link-prediction
class Pokec(GeometricDataset):

    @property
    def name(self): return "pokec"

    def __init__(self, root='/tmp'):
        super().__init__(root=root, )
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["soc-pokec-profiles.txt.gz", "soc-pokec-relationships.txt.gz"]

    def download(self):
        download_url("https://snap.stanford.edu/data/soc-pokec-profiles.txt.gz", self.raw_dir)
        download_url("https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz", self.raw_dir)

    def load_frames(self):
        """
        Returns the Pandas node and edge frames.
        """
        print("Loading data frames")
        hdf_path = os.path.join(self.processed_dir, "frames.h5")
        if os.path.exists(hdf_path):
            node_frame = pd.read_hdf(hdf_path, "/dfn")
            edge_frame = pd.read_hdf(hdf_path, "/dfe")
            return node_frame, edge_frame
        node_fields = [
            "public",
            "completion_percentage",
            "gender",
            "region",
            "last_login",
            "registration",
            "AGE",
            "body",
            "I_am_working_in_field",
            "spoken_languages",
            "hobbies",
            "I_most_enjoy_good_food",
            "pets",
            "body_type",
            "my_eyesight",
            "eye_color",
            "hair_color",
            "hair_type",
            "completed_level_of_education",
            "favourite_color",
            "relation_to_smoking",
            "relation_to_alcohol",
            "sign_in_zodiac",
            "on_pokec_i_am_looking_for",
            "love_is_for_me",
            "relation_to_casual_sex",
            "my_partner_should_be",
            "marital_status",
            "children",
            "relation_to_children",
            "I_like_movies",
            "I_like_watching_movie",
            "I_like_music",
            "I_mostly_like_listening_to_music",
            "the_idea_of_good_evening",
            "I_like_specialties_from_kitchen",
            "fun",
            "I_am_going_to_concerts",
            "my_active_sports",
            "my_passive_sports",
            "profession",
            "I_like_books",
            "life_style",
            "music",
            "cars",
            "politics",
            "relationships",
            "art_culture",
            "hobbies_interests",
            "science_technologies",
            "computers_internet",
            "education",
            "sport",
            "movies",
            "travelling",
            "health",
            "companies_brands",
            "more",
            ""
        ]
        dfn = pd.read_csv(self.raw_paths[0], sep="\t", names=node_fields, nrows=None)[["gender", "AGE"]]
        dfe = pd.read_csv(self.raw_paths[1], sep="\t", names=["source", "target"], nrows=None)
        dfn["age"] = dfn["AGE"].drop(columns=["AGE"])
        dfn = dfn.astype({'gender': 'float', 'age': 'float'})

        # transform edges
        dfe = dfe.astype({'source': 'str', 'target': 'str'})
        node_frame = dfn
        edge_frame = dfe

        # save as hdf
        store = pd.HDFStore(hdf_path)
        store["dfn"] = dfn
        store["dfe"] = dfe
        store.close()
        print("Save data frames to 'frames.h5'.")
        return node_frame, edge_frame

    def __transform_nodes(self, node_frame):
        print("Transforming nodes")
        # sorting the index does not make sense here
        node_index_map = {str(index): i for i, index in enumerate(node_frame.index.unique())}
        # filling nans with 0
        gender_series = node_frame["gender"].fillna(0.).values.astype(int)
        gender_tensor = torch.from_numpy(gender_series).unsqueeze(-1)
        age_tensor = torch.from_numpy(node_frame['age'].fillna(17.0).values.astype(np.float32)).unsqueeze(-1)
        x = torch.cat((gender_tensor, age_tensor), dim=1)
        return x, node_index_map

    def __transform_edges(self, edge_frame, node_index_map):
        print("Transforming edges")
        src = [node_index_map[src_id] if src_id in node_index_map else -1 for src_id in edge_frame.source]
        dst = [node_index_map[tgt_id] if tgt_id in node_index_map else -1 for tgt_id in edge_frame.target]
        return torch.tensor([src, dst])

    def process(self):
        node_frame, edge_frame = self.load_frames()
        nodes_x, nodes_mapping = self.__transform_nodes(node_frame)
        edges_index = self.__transform_edges(edge_frame, nodes_mapping)
        d = Data(x=nodes_x, edge_index=edges_index, edge_attr=None, y=None)
        Pokec.create_node_masks(d)
        print("Saving data to Pyg file")
        torch.save(self.collate([d]), self.processed_paths[0])
        self.data, self.slices = self.collate([d])