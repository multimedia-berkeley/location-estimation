import csv
import numpy as np
import data
from utils import *

all_data = data.get()
all_data = data.filter(all_data, 'us')
print(len(all_data))
#all_data = all_data[:0]  # TODO: delete later
'''
# Basic four node test
all_data = []
all_data.append({'watchlink': 0, 'latitude': 80, 'longitude': 80, 'keywords': 'berkeley'})
all_data.append({'watchlink': 1, 'latitude': 80, 'longitude': 80, 'keywords': 'berkeley, campanile'})
all_data.append({'watchlink': 2, 'latitude': 80, 'longitude': 80, 'keywords': 'haas, campanile'})
all_data.append({'watchlink': 3, 'latitude': 90, 'longitude': 80, 'keywords': 'haas'})
'''
train_data, test_data = data.split(all_data, 0.8)

NUM_ITERATIONS = 1
DROP_EDGE_THRESHOLD = len(all_data) * 0.1
img_data_mappings = {}
G = UndirectedGraph()

print('Num iterations: {0}'.format(NUM_ITERATIONS))


def get_tag_approx_loc():
    tag_approx_loc = {}
    reader = csv.reader(open('tag_spatial_var.csv'), delimiter=",")
    for tag, lat, lon, var in reader:
        tag_approx_loc[tag] = Location(float(lat), float(lon), float(var))
    return tag_approx_loc

tag_approx_loc = get_tag_approx_loc()

def process_training_data(train_data, img_data_mappings, tag_approx_loc):
    tag_locations = {}
    tag_mean_locations = {}
    train_tag_counts = Counter()

    for train_img in train_data:
        img_id = train_img['watchlink']
        img_tags = train_img['keywords'].split(', ')

        # Initialize lat, lon, var values
        lat, lon = float(train_img['latitude']), float(train_img['longitude'])
        var = min([tag_approx_loc[tag].var for tag in img_tags])
        img_data_mappings[img_id] = Location(lat, lon, var)
    
        # Count tags
        train_tag_counts.add_counts(img_tags)

        # Create lists of locations for each tags
        for tag in img_tags:
            if tag not in tag_locations:
                tag_locations[tag] = []
            tag_locations[tag].append(Location(lat, lon, var))

    return train_tag_counts, tag_locations

train_tag_counts, tag_locations = process_training_data(train_data, img_data_mappings, tag_approx_loc)

def get_tag_approx_loc_from_train(tag_locations):
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

tag_approx_loc = get_tag_approx_loc_from_train(tag_locations)
print(tag_approx_loc['newyork'])
print(tag_approx_loc['nyc'])
print(tag_approx_loc['timessquare'])
print(tag_approx_loc['manhattan'])
def process_test_data(test_data, train_tag_counts, img_data_mappings, tag_approx_loc):
    total_tag_counts = train_tag_counts.copy()
    
    # Process test data
    for test_img in test_data:
        img_id = test_img['watchlink']
        img_tags = test_img['keywords'].split(', ')

        # Count tags
        total_tag_counts.add_counts(img_tags)
   
        # Initialize lat, lon, and var
        min_var_loc = Location(37, 122, 15098163) # Approx lat/lon of SF, and approx var of tag 'iphone'
        for tag in img_tags:
            if tag in tag_approx_loc and tag_approx_loc[tag].var < min_var_loc.var:
                min_var_loc = tag_approx_loc[tag]
        img_data_mappings[img_id] = min_var_loc
    return total_tag_counts

total_tag_counts = process_test_data(test_data, train_tag_counts, img_data_mappings, tag_approx_loc)

# Add vertices to graph
for img_id in img_data_mappings:
    G.add_vertex(img_id)


# Add edges to graph
edge_count = 0
for img1_i in range(len(all_data)):
    img1_id = all_data[img1_i]['watchlink']
    img1_tags = set(all_data[img1_i]['keywords'].split(', '))

    for img2_i in range(img1_i + 1, len(all_data)): # start at i+1 to ensure no duplicate pairs or self pairs
        img2_id = all_data[img2_i]['watchlink']
        img2_tags = all_data[img2_i]['keywords'].split(', ')
        for tag in img2_tags:
            if total_tag_counts.get_count(tag) < DROP_EDGE_THRESHOLD and tag in img1_tags:
                edge_count += 1
                G.add_edge(img1_id, img2_id)
                break

print('Num edges: ' + str(edge_count))

###################
'''
G = UndirectedGraph()
G.add_vertex('berkeley')
G.add_vertex('sathergate')
G.add_vertex('iphone')
G.add_vertex('test')
G.add_edge('iphone', 'test')
G.add_edge('berkeley', 'test')
G.add_edge('sathergate', 'test')
img_data_mappings = {}
img_data_mappings['iphone'] = Location(36.02264117460317,-36.72232646031748,15098163.310764369)
img_data_mappings['berkeley'] = Location(80, 80, .00001)
img_data_mappings['sathergate'] = Location(90, 90, .0000001)
img_data_mappings['test'] = Location(90, 90, .01)
test_data = []
test_data.append({'watchlink': 'test', 'latitude': '90', 'longitude': '90'})
'''
###################

def calc_update(img, loc, G):
    def safe_div(num1, num2):
        if num2 == 0:
            return num1 / .0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
        else:
            return num1 / num2

    def calc_mean():
        neighbors = G.neighbors(img)
        lat_lon = np.array([loc.lat, loc.lon])
        
        summation = np.zeros(2)
        for neighbor in neighbors:
            neighbor_loc = img_data_mappings[neighbor]
            neighbor_lat_lon = np.array([neighbor_loc.lat, neighbor_loc.lon])
            summation = summation + safe_div(neighbor_lat_lon, neighbor_loc.var)
        numerator = safe_div(lat_lon, loc.var) + summation

        summation = 0
        for neighbor in neighbors:
            neighbor_loc = img_data_mappings[neighbor]
            summation += safe_div(1, neighbor_loc.var)
        denominator = safe_div(1, loc.var) + summation

        return safe_div(numerator, denominator).tolist()

    def calc_var():
        neighbors = G.neighbors(img)
        numerator = 1

        summation = 0
        for neighbor in neighbors:
            neighbor_loc = img_data_mappings[neighbor]
            summation += safe_div(1, neighbor_loc.var)
        denominator = safe_div(1, loc.var) + summation

        return safe_div(numerator, denominator)

    mean = calc_mean()
    var = calc_var()
    return mean[0], mean[1], var

# Perform location estimate update algorithm
for _ in range(NUM_ITERATIONS):
    new_img_data_mappings = img_data_mappings.copy()
    for test_img in test_data:
        img_id = test_img['watchlink']
        loc = img_data_mappings[img_id]
        lat, lon, var = calc_update(img_id, loc, G)
        new_img_data_mappings[img_id] = Location(lat, lon, var)
    img_data_mappings = new_img_data_mappings

# Calculate error
median_error_finder = MedianFinder()
one_km_count = 0
five_km_count = 0
ten_km_count = 0
hundred_km_count = 0
thousand_km_count = 0
other_count = 0
for test_img in test_data:
    img_id = test_img['watchlink']
    img_result_loc = img_data_mappings[img_id]
    img_actual_loc = Location(float(test_img['latitude']), float(test_img['longitude']))
    error = Location.dist(img_result_loc, img_actual_loc)
    median_error_finder.add(error)
    if error < 1:
        one_km_count += 1
        print(img_result_loc)
        print()
    elif error < 5:
        five_km_count += 1
    elif error < 10:
        ten_km_count += 1
    elif error < 100:
        hundred_km_count += 1
    elif error < 1000:
        thousand_km_count += 1
    else:
        other_count += 1
print('Less than 1 km: {0}'.format(one_km_count))
print('Less than 5 km: {0}'.format(five_km_count))
print('Less than 10 km: {0}'.format(ten_km_count))
print('Less than 100 km: {0}'.format(hundred_km_count))
print('Less than 1000 km: {0}'.format(thousand_km_count))
print('Greater than 100 km: {0}'.format(other_count))
print('Med error: {0}'.format(median_error_finder.med()))

