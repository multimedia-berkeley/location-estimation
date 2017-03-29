from utils import *

class DataFile(object):
    def __init__(self, name, prefix, entry_delimiter, field_delimiter, tag_delimiter, field_names):
        self.name = name
        self.prefix = prefix
        self.entry_delimiter = entry_delimiter
        self.field_delimiter = field_delimiter
        self.tag_delimiter = tag_delimiter
        self.field_names = field_names

def get():
    files = []

    field_names = ['title', 'keywords', 'description', 'userID', 'idk what this is', 'userlocation', 'latitude', 'longitude', 'region', 'locality', 'country', 'watchlink']
    files.append(DataFile('placing2011_train', '/Users/daniel/Developer/ICME/', '\n', ' @#|#@ ', ', ', field_names))
    
    data = []

    for data_file in files:
        with open(data_file.prefix + data_file.name, 'r') as f:
            raw_data = f.read().split(data_file.entry_delimiter)


        for i in range(len(raw_data)):
            entry = raw_data[i].split(data_file.field_delimiter)
            if len(entry) < len(data_file.field_names):
                continue
            entry_dict = {}
            for j in range(len(data_file.field_names)):
                entry_dict[data_file.field_names[j]] = entry[j]
            data.append(entry_dict)
    return data

def split(lst, proportion=0.5):
    def safe_div(a, b):
        if b == 0:
            return a
        else:
            return a / b

    test = []
    train = []
    train_users = set()
    test_users = set()
    
    # Ensures all images from any given user are all in only train or all in only test
    for entry in lst:
        user = entry['userID']
        if user in train_users:
            train.append(entry)
        elif user in test_users:
            test.append(entry)
        else:
            if safe_div(len(train), len(test) + len(train)) < proportion:
                train.append(entry)
                train_users.add(user)
            else:
                test.append(entry)
                test_users.add(user)

    return train, test

def filter(data, place):
    LAT_LON_BOUNDS = {
        'western_europe': [Location(35.606383, -11.094409), Location(58.715069, 15.972913)],
        'us': [Location(30, -126), Location(49, -67)],
        'ca': [Location(32.18, -124.49), Location(41.94, -114.77)],
    }
    result = []
    bounds = LAT_LON_BOUNDS[place]
    for entry in data:
        lat = float(entry['latitude'])
        lon = float(entry['longitude'])
        if lat > bounds[0].lat and lat < bounds[1].lat \
                and lon > bounds[0].lon and lon < bounds[1].lon:
            result.append(entry)
    return result

