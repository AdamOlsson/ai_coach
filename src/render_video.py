# installed
import torchvision
import numpy as np
from torch import from_numpy

# native
import sys, getopt, json
from os.path import join

# misc
from paf.util import load_humans
from paf.common import draw_humans

def main(graph_path, video_path):
    graph_data = load_humans(graph_path)
    metadata, humans = graph_data["metadata"], graph_data["frames"]

    vframes, _, _ = torchvision.io.read_video(video_path, pts_unit="sec") # Tensor[T, H, W, C]) – the T video frames
    vframes = np.flip(vframes.numpy(), axis=3)

    # graph data is truncated due to dividing clip into subparts
    vframes = vframes[:len(humans)]

    for frame_idx in range(len(humans)):
        vframes[frame_idx] = draw_humans(np.float32(vframes[frame_idx]), humans[frame_idx])

    vframes = np.flip(vframes, axis=3).copy()

    save_path = join("results", "skeleton_" + metadata["filename"])
    torchvision.io.write_video(save_path, from_numpy(vframes), int(metadata["video_properties"]["video_fps"]))

    print("Results written to {}".format(save_path))


def parse_args(argv):
    try:
        opts, _ = getopt.getopt(argv, 'v:g:', ['video=', 'graph='])
    except getopt.GetoptError:
       sys.exit(2)
    video_path = ""
    graph_path = ""
    for opt, arg in opts:
        if opt == '-h':
            sys.exit()
        elif opt in ("-v", "--video"):
            video_path = arg
        elif opt in ("-g", "--graph"):
            graph_path = arg
    return video_path, graph_path 

if __name__ == "__main__":
    video_path, graph_path = parse_args(sys.argv[1:])
    main(graph_path, video_path)