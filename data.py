#-*- coding: utf-8 -*-
import sys
import codecs
#reload(sys)
#sys.setdefaultencoding("utf-8")
from utils import *

class DataFile(object):
    def __init__(self, name, prefix, entry_delimiter, field_delimiter, tag_delimiter, field_names, tag_field_name):
        self.name = name
        self.prefix = prefix
        self.entry_delimiter = entry_delimiter
        self.field_delimiter = field_delimiter
        self.tag_delimiter = tag_delimiter
        self.field_names = field_names
        self.tag_field_name = tag_field_name

    def is_tag_field_name(self, field_name):
        return field_name == self.tag_field_name

def get_train_test():
    train_files = []
    field_names = ['userID', 'watchlink', 'geoData', 'tags', 'idk', 'idk']
    train_files.append(DataFile('train2012', './', '\n', ' : ', ' ', field_names, 'tags'))
    train_data = get(train_files)

    test_files = []
    field_names = ['userID', 'watchlink', 'geoData', 'tags', 'idk', 'idk']
    test_files.append(DataFile('test2012', './', '\n', ' : ', ' ', field_names, 'tags'))
    test_data = get(test_files)
    return train_data, test_data


def get_small():
    files = []
    field_names = ['IGNORE', 'tags', 'IGNORE', 'userID', 'IGNORE', 'IGNORE', 'latitude', 'longitude', 'IGNORE', 'IGNORE', 'IGNORE', 'watchlink']
    files.append(DataFile('placing2011_train', './', '\n', ' @#|#@ ', ', ', field_names, 'tags'))
    return get(files)


def get_medium():
    files = []
    field_names = ['userID', 'watchlink', 'geoData', 'tags', 'IGNORE', 'IGNORE']
    files.append(DataFile('train2012_subset', './', '\n', ' : ', ' ', field_names, 'tags'))
    return get(files)


def get_large():
    files = []
    field_names = ['userID', 'watchlink', 'geoData', 'tags', 'IGNORE', 'IGNORE']
    files.append(DataFile('train2012', './', '\n', ' : ', ' ', field_names, 'tags'))
    return get(files)


def get(files):
    data = []

    for data_file in files:
        raw_data = []
        with open(data_file.prefix + data_file.name, 'r', encoding='ISO-8859-1') as f:
            raw_data = f.read().split('\n')

        for i in range(len(raw_data)):
            entry = raw_data[i].split(data_file.field_delimiter)
            if len(entry) < len(data_file.field_names):
                continue
            entry_dict = {}
            for j in range(len(data_file.field_names)):
                if data_file.field_names[j] == 'IGNORE':
                    continue

                if data_file.is_tag_field_name(data_file.field_names[j]):
                    entry_dict[data_file.field_names[j]] = entry[j].split(data_file.tag_delimiter)
                else:
                    entry_dict[data_file.field_names[j]] = entry[j]

                if data_file.field_names[j] == 'geoData':
                    geoData = entry_dict['geoData'].replace('GeoData[longitude=', '').replace('latitude=', '').replace('accuracy=', '').strip('[]').split()
                    entry_dict['latitude'] = geoData[1]
                    entry_dict['longitude'] = geoData[0]
                    entry_dict['accuracy'] = geoData[2]
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

