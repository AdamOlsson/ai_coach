from paf.common import Human, BodyPart
import json

def save_humans(path, frames, metadata):
    frames_list = []
    for i, frame in enumerate(frames):
        bodies = []
        for human in frame:
            body = {}
            body['pairs'] = human.pairs
            body["uidx_list"] = list(human.uidx_list)
            body["score"] = human.score

            body_parts = {}
            for _, bp in human.body_parts.items():
                body_parts[bp.part_idx] = {
                    "x":bp.x,
                    "y":bp.y,
                    "score":bp.score,
                    "uidx":bp.uidx,
                }
            body["body_parts"] = body_parts

            bodies.append(body)
        frames_list.append({"bodies": bodies, "frame_id":i})
    
    with open(path, 'w') as f:
        f.write(json.dumps({ "metadata":metadata, "frames":frames_list }, indent=4, sort_keys=True))


def load_humans(path):

    ret = {}
    with open(path) as jsfile:
        data = json.load(jsfile)

    frames = []
    for frame in data["frames"]:
        bodies = []
        for body in frame["bodies"]:
            human = Human([])
            human.pairs = body["pairs"]
            human.uidx_list = body["uidx_list"]
            human.score = body["score"]
            for key, body_part in body["body_parts"].items():
                human.body_parts[int(key)] = BodyPart(body_part["uidx"], int(key), body_part["x"], body_part["y"], body_part["score"])
            
            bodies.append(human)
        frames.append(bodies)

    ret["frames"] = frames
    ret["metadata"] = data["metadata"]

    return ret