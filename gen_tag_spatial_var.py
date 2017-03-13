import numpy as np
import data
from utils import *

all_data = data.get()

def get_tag_locations(data):
    tag_locations = {}

    for img in data:
        tags = img['keywords'].split(', ')
        lat = float(img['latitude'])
        lon = float(img['longitude'])
        
        for tag in tags:
            if tag not in tag_locations:
                tag_locations[tag] = []
            tag_locations[tag].append(Location(lat, lon))
    return tag_locations

tag_locations = get_tag_locations(all_data)

def get_tag_spatial_var(tag_locations):
    def get_avg(lst):
        if len(lst) == 0:
            return 0
        else:
            return sum(lst)/len(lst)

    def get_var(l, avg):
        total = 0.0
        for n in l:
            n = float(n)
            total += (n-avg)*(n-avg)
        return total / len(l)

    tag_mean_loc = {}
    for tag, locations in tag_locations.items():
        lst_lat = []
        lst_lon = []
        for loc in locations:
            lst_lat.append(loc.lat)
            lst_lon.append(loc.lon)
            avg_lat = get_avg(lst_lat)
            avg_lon = get_avg(lst_lon)

            num_point = len(lst_lat)
            list_distance = []
            for i in range(num_point):
                distance = Location.dist(Location(avg_lat, avg_lon), Location(lst_lat[i], lst_lon[i]))
                list_distance.append(distance)
                avg_dist = get_avg(list_distance)
                var = get_var(list_distance, avg_dist)
                tag_mean_loc[tag] = Location(avg_lat, avg_lon, var)
    return tag_mean_loc

tag_spatial_var = get_tag_spatial_var(tag_locations)

def to_str_array(tag_spatial_var):
    result = []
    for tag, loc in tag_spatial_var.items():
        result.append('{0},{1},{2},{3}'.format(tag, loc.lat, loc.lon, loc.var))
    return result

tag_spatial_var_formatted = to_str_array(tag_spatial_var)

def write_to_file(lst, name):
    f = open(name, 'w')
    for item in lst:
        f.write(item + '\n')
    f.close()

write_to_file(tag_spatial_var_formatted, 'tag_spatial_var.csv')


