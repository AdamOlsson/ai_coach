# torch
from torch_geometric.data import Dataset

# libs
import networkx as nx

# native python
from os import listdir
from os.path import isfile, join, splitext
import json

# Create graph-dataset
# https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html

# create_graph as an example
# https://github.com/dawidejdeholm/dj_graph/blob/master/Utils/SECParser.py


"""
In the cases where nodes are missing from the human graph, investigate if it
is possible to temporally connect the node to frame t+2, i.e skipping connecting 
to frame t+1.

t=2   ^ O--O--O
      | |  |  |
      | |  |  |
t=1   | |  O--O <- node missing from this timestep
      | |  |  |
      | |  |  | <- temporal edge
t=0   | O--O--O <- nodes with spatial edges
      |
    time
"""

class GraphDataset(Dataset):
    '''
    A graph dataset built from human pose predictions. The unprocessed graphs are adapted to easy use of 
    the PosePrediction package. Therefore, these graphs are initially processed here by contatenating the
    temporal domain over frames, creating one single graph spanning the entire video.

    Args:
        raw_dir (string): Root directory to the unprocessed graphs from the PosePrediction package.
        save_dir (string): Root directory where the processed graphs are saved.
        pre_transform (callable, optional):
        transform (callable, optional): 
    '''
    def __init__(self, save_dir, raw_dir, transform=None, pre_transform=None):
        self.raw_f_name = [f for f in listdir(raw_dir) if isfile(join(raw_dir, f))]
        self.processed_f_names = ["{}.pth".format(splitext(f)[0]) for f in self.raw_f_name]
        self.sd = save_dir
        self.rd = raw_dir

        # Seems to override above variables?
        super(GraphDataset, self).__init__(save_dir, transform, pre_transform)


    def len(self):
        pass

    def get(self, idx):
        pass

    def process(self):
        def build_bodies():
            pass
        
        def connect_temporally():
            pass

        for name in [join(self.rd, f) for f in self.raw_file_names]:
            print("Processing {}...".format(name))
            with open(name) as f:
                data = json.load(f)

            metadata, frames = data["metadata"], data["frames"]
            scheme = metadata["body_construction"] # How to build the body
            frame_dt = 1.0/metadata["info"]["video_fps"] # time between frames (1 sec / fps)

            bodies = frames[0]["bodies"]
            body = bodies[0]
            body_parts = body["body_parts"]

            graph = nx.Graph() # undirected graph for now

    @property
    def raw_file_names(self):
        return self.raw_f_name

    @property
    def processed_file_names(self):
        return self.processed_f_names
