# @Filename:    dataset.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        8/2/22 12:47 PM

from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data, download_url
import pandas as pd
from utils.config import *

# https://github.com/Orbifold/pyg-link-prediction
class Pokec(InMemoryDataset):
    """
    Pokec is the most popular on-line social network in Slovakia. The popularity of network has not changed even after the coming of Facebook.
    Pokec has been provided for more than 10 years and connects more than 1.6 million people. Datasets contains anonymized data of the whole network.
    Profile data contains gender, age, hobbies, interest, education etc. P
    Profile data are in Slovak language. Friendships in Pokec are oriented.
    This is a Pyg dataset wrapping the Pokec.
    See https://snap.stanford.edu/data/soc-pokec.html.
    """

    def __init__(self, root=None):
        self.name = "pokec"
        if root is None:
            root = '/tmp/'

        self.node_data_map = None
        """ Maps the original label to a vector."""
        self.node_index_map = None
        """ Maps the original key to a node index."""
        self.edge_data_map = None
        """Maps the original label to a vector."""
        self.node_frame = None
        """The original nodes frame."""
        self.edge_frame = None
        """The original edges frame."""
        super().__init__(root, transform=None, pre_transform=None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["soc-pokec-profiles.txt.gz", "soc-pokec-relationships.txt.gz"]

    @property
    def processed_file_names(self):
        return ["pokec.pt"]

    def __repr__(self):
        return f'{self.name.capitalize()}()'

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
            self.node_frame = pd.read_hdf(hdf_path, "/dfn")
            self.edge_frame = pd.read_hdf(hdf_path, "/dfe")
            return self.node_frame, self.edge_frame
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
        dfn = pd.read_csv(self.raw_paths[0], sep = "\t", names = node_fields, nrows = None)
        dfe = pd.read_csv(self.raw_paths[1], sep = "\t", names = ["source", "target"], nrows = None)

        # transform nodes, only keep gender and age
        dfn = dfn[["gender", "AGE"]]
        dfn["age"] = dfn["AGE"]

        dfn = dfn.drop(columns=["AGE"])
        dfn = dfn.astype({'gender': 'float', 'age': 'float'})

        # transform edges
        dfe = dfe.astype({'source': 'str', 'target': 'str'})
        self.node_frame = dfn
        self.edge_frame = dfe

        # save as hdf
        store = pd.HDFStore(hdf_path)
        store["dfn"] = dfn
        store["dfe"] = dfe
        store.close()
        print("Save data frames to 'frames.h5'.")
        return dfn, dfe

    def __transform_nodes(self):
        print("Transforming nodes")
        if self.node_frame is None:
            self.node_frame, self.edge_frame = self.load_frames()
        # sorting the index does not make sense here
        self.node_index_map = {str(index): i for i, index in enumerate(self.node_frame.index.unique())}
        gender_series = self.node_frame["gender"]
        gender_tensor = torch.zeros(len(gender_series), 2, dtype = torch.float)
        for i, v in enumerate(gender_series.values):
            gender_tensor[i, 0 if np.isnan(v) else int(v)] = 1.0
        age_tensor = torch.tensor(self.node_frame['age'].values, dtype = torch.float).reshape(len(gender_series), -1)
        x = torch.cat((gender_tensor, age_tensor), dim=-1)  # 1x3 tensor
        return x, self.node_index_map

    def __transform_edges(self):
        print("Transforming edges")

        src = [self.node_index_map[src_id] if src_id in self.node_index_map else -1 for src_id in self.edge_frame.source]
        dst = [self.node_index_map[tgt_id] if tgt_id in self.node_index_map else -1 for tgt_id in self.edge_frame.target]
        edge_index = torch.tensor([src, dst])

        return edge_index, None

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

        print("REACHED HERE!")
        d.train_mask = train_mask
        d.test_mask = test_mask

    def process(self):
        self.load_frames()
        nodes_x, nodes_mapping = self.__transform_nodes()
        edges_index, edges_label = self.__transform_edges()

        d = Data(x=nodes_x, edge_index=edges_index, edge_attr=edges_label, y=None)

        if self.pre_filter is not None:
            d = self.pre_filter(d)

        if self.pre_transform is not None:
            d = self.pre_transform(d)
        Pokec.create_node_masks(d)
        print("Saving data to Pyg file")
        torch.save(self.collate([d]), self.processed_paths[0])
        self.data, self.slices = self.collate([d])

    def describe(self):
        print("Pokec Pyg Dataset")
        print("Nodes:", len(self.data.x), "Edges:", len(self.data.edge_index[0]))

