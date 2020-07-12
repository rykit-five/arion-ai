import json


def load_json_as_dict(json_file):
    with open(json_file, "r") as f:
        data_dict = json.load(f)
    return data_dict


def flatten_dict(data_dict):
    flattened_dict = {}
    for k, v in data_dict.items():
        if isinstance(v, dict):
            for k_, v_ in v.items():
                flattened_dict[k_] = v_
        else:
            flattened_dict[k] = v
    return flattened_dict
