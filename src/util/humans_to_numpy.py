import numpy as np

def humansToNumpy(frames, C=3, T=300, V=18, M=1):
    """Takes a list of humans and converts it to a numpy array.
    """

    data_numpy = np.zeros((C, T, V, M))
    for t in range(T):
        body = frames[t%len(frames)] # replay video if shorter than 300 frames

        if len(body) == 0: # no persons on this frame
            continue
        
        for _, part in body[0].body_parts.items():
            data_numpy[0, t, int(part.part_idx), 0] = part.x
            data_numpy[1, t, int(part.part_idx), 0] = part.y
            data_numpy[2, t, int(part.part_idx), 0] = part.score
    
    return data_numpy