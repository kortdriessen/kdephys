import pandas as pd


class data_dict(dict):
    def __init__(self, dict):
        self._dict = dict

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._dict, attr)

    def __getitem__(self, item):
        return self._dict[item]

    def __setitem__(self, item, data):
        self._dict[item] = data

    def __repr__(self):
        return repr(self._dict)

    def __len__(self):
        return len(self._dict)

    def mush(self):
        df_list = []
        for k in self._dict.keys():
            df_list.append(self._dict[k])
        return pd.concat(df_list)
