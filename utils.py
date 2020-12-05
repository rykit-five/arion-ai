import re
import json
from datetime import datetime, timedelta


class Utils():

    def __init__(self):
        pass

    @staticmethod
    def str_to_int_or_float(data):
        if isinstance(data, str):
            if re.search("\d+\.\d+", data):
                return float(data)
            elif re.search("\d+", data):
                return int(data)
            else:
                return data
        elif isinstance(data, int) or isinstance(data, float):
            return data

    @staticmethod
    def str_to_sec(data):
        if isinstance(data, str):
            time = datetime.strptime("00.00.0", "%M.%S.%f")
            data = datetime.strptime(data, "%M.%S.%f")
            data = timedelta.total_seconds(data - time)
        return data

    @staticmethod
    def load_json_as_dict(json_file):
        with open(json_file, "r") as f:
            data_dict = json.load(f)
        return data_dict

    @staticmethod
    def flatten_dict(data_dict):
        flattened_dict = {}
        for k, v in data_dict.items():
            if isinstance(v, dict):
                for _k, _v in v.items():
                    flattened_dict[_k] = _v
            else:
                flattened_dict[k] = v
        return flattened_dict
