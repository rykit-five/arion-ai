import json
from datetime import datetime, timedelta


def str_to_digit(data):
    if not isinstance(data, str) or data == '':
        raise
    if data.isdigit():
        return int(data)
    else:
        return float(data)


def str_to_sec(data):
    time = datetime.strptime("00.00.0", "%M.%S.%f")
    data = datetime.strptime(data, "%M.%S.%f")
    data = timedelta.total_seconds(data - time)
    return data


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
