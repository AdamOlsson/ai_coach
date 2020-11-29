
def remove_bg_humans(frames):
    for frame in frames:
        # only 1 body found, no need to find largest body
        if len(frame["bodies"]) == 1:
            continue
        frame["bodies"] = _remove_bg_humans_from_frame(frame["bodies"])

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
        body_parts = body["body_parts"]
        top_y = 1000000
        bottom_y = 0
        
        for _, part in body_parts.items():
            y = part["y"]
            
            if y < top_y:
                top_y = y
            elif y > bottom_y:
                bottom_y = y
            
        delta = bottom_y - top_y
        #print("{}. largest delta found for body: {}".format(i, delta))
        if delta > best_delta:
            best_delta = delta
            largest_body_idx = i

    #print("largest body index found {}".format(largest_body_idx))
    if largest_body_idx == None:
        return []
    else:
        return [bodies[largest_body_idx]]