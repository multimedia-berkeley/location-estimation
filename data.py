from utils import *

def get():
    DATA_ENTRY_SEPARATOR = ' @#|#@ '
    PATH_PREFIX = '/Users/daniel/Developer/ICME/'
    FILE_NAME = 'placing2011_train'

    with open(PATH_PREFIX + FILE_NAME, 'r') as myfile:
        raw_data = myfile.read().split('\n')

    entry_keys = ['title', 'keywords', 'description', 'userID', 'idk what this is', 'userlocation', 'latitude', 'longitude', 'region', 'locality', 'country', 'watchlink']
    data = []

    for i in range(len(raw_data)):
        entry = raw_data[i].split(DATA_ENTRY_SEPARATOR)
        if len(entry) < len(entry_keys):
            continue
        entry_dict = {}
        for j in range(len(entry_keys)):
            entry_dict[entry_keys[j]] = entry[j]
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
        'western_europe': [Location(36, -27), Location(70, 28)],
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

