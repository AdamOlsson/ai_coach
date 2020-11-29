import json

def load_config(path):
    with open(path) as jsfile:
        config = json.load(jsfile)
    return config