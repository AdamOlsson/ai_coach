from paf.paf_to_pose import paf_to_pose_cpp
import torch
import numpy as np

def remove_bg_humans(frames):
    for frame in frames:
        # only 1 body found, no need to find largest body
        if len(frame) == 1:
            continue
        frame = _remove_bg_humans_from_frame(frame)

    return frames

def _remove_bg_humans_from_frame(bodies):
    """ Remove background humans from a single frame. The function computes the difference 
    between the highest and lowest point and assumes the largest difference is the main subject.

    Input:
        bodies (iteratable) - an iteratable object with all the humans in the frame.

    Output:
        List containing the main subject of the frame.
    """    

    largest_body_idx = None
    best_delta = 0

    for i, body in enumerate(bodies):
        top_y = 1000000
        bottom_y = 0
        
        for _, part in body.body_parts.items():
            y = part.y
            
            if y < top_y:
                top_y = y
            elif y > bottom_y:
                bottom_y = y
            
        delta = bottom_y - top_y

        if delta > best_delta:
            best_delta = delta
            largest_body_idx = i

    if largest_body_idx == None:
        return []
    else:
        return [bodies[largest_body_idx]]


def poseEstimation(model, config, video, device, remove_bg_objects=True):
    """ Perfors body pose estimation.
    Params:
        model (torch.nn)     - Model to use during pose estimations
        video (torch.Tensor) - Video to do pose estimations on
        config (dict)        - Model configuration
    Returns:
        A list containing the body pose estimations for each frame.
    """

    batch = 15
    idx = zip(np.arange(0,len(video),batch), np.arange(batch,len(video)+batch,batch))

    pafs = []
    heatmaps = []
    for start, stop in idx:
        tmp_vid = video[start:stop].to(device)
        with torch.no_grad():
            (branch1, branch2), _ = model(tmp_vid)
        del tmp_vid
        paf = branch1.data.cpu().numpy().transpose(0, 2, 3, 1)
        heatmap = branch2.data.cpu().numpy().transpose(0, 2, 3, 1)

        pafs.append(paf)
        heatmaps.append(heatmap)

    pafs = np.concatenate(pafs, axis=0)
    heatmaps = np.concatenate(heatmaps, axis=0)

    # Construct humans on every frame
    frames = []
    for frame in range(len(video)):
        humans = paf_to_pose_cpp(heatmaps[frame], pafs[frame], config)
        frames.append(humans)

    if remove_bg_objects:
        frames = remove_bg_humans(frames)

    return frames
