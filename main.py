import csv
import numpy as np
import data
from utils import *

all_data = data.get()
all_data = None
with open('/Users/daniel/Developer/ICME/data', 'r') as f:
    all_data = eval(f.read())

all_data = data.filter(all_data, 'ca')
print(len(all_data))

'''
# Basic four node test
all_data = []
all_data.append({'watchlink': 0, 'latitude': 80, 'longitude': 80, 'keywords': 'berkeley'})
all_data.append({'watchlink': 1, 'latitude': 80, 'longitude': 80, 'keywords': 'berkeley, campanile'})
all_data.append({'watchlink': 2, 'latitude': 80, 'longitude': 80, 'keywords': 'haas, campanile'})
all_data.append({'watchlink': 3, 'latitude': 90, 'longitude': 80, 'keywords': 'haas'})
'''
train_data, test_data = data.split(all_data, 0.8)

MAX_ITERATIONS = 0
CONVERGENCE_THRESHOLD = 0.00006288 # About the mean sqaured difference of 1km
DROP_EDGE_THRESHOLD = len(all_data) * 0.1
LOCALITY_THRESHOLD = 1 # Locality of 'newyorkcity' is 0.057
img_data_mappings = {}
G = UndirectedGraph()


def get_tag_locality():
    locality_str = ''
    with open('/Users/daniel/Developer/ICME/large_tag_weights.tsv', 'r') as f:
        locality_str = f.read()
    return eval(locality_str)

locality = get_tag_locality()

def process_train_tags(train_data, locality):
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
    
    tag_locations = {}
    train_tag_counts = Counter()
    for train_img in train_data:
        lat, lon = float(train_img['latitude']), float(train_img['longitude'])
        img_tags = train_img['tags']

        remove_low_locality_tags(img_tags)

        # Gather locations for each tag
        for tag in img_tags:
            if tag not in tag_locations:
                tag_locations[tag] = []
            tag_locations[tag].append(Location(lat, lon))

        # Count tags
        train_tag_counts.add_counts(img_tags)
    
    # Get average loc and spatial var for each tag
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
    return tag_mean_loc, train_tag_counts

def remove_low_locality_tags(tags_list):
    '''Drop low locality tags. Mutates original list'''
    tags_to_remove = []
    for tag in tags_list:
        if tag not in locality:
            tags_to_remove.append(tag)
        else:
            locality_score = locality[tag]
            if type(locality_score) is tuple:
                locality_score = locality_score[0]
            if locality_score < LOCALITY_THRESHOLD: 
                tags_to_remove.append(tag)
    for tag in tags_to_remove:
        tags_list.remove(tag)

   

tag_approx_loc, train_tag_counts = process_train_tags(train_data, locality)


def process_training_data(train_data, img_data_mappings, tag_approx_loc, train_tag_counts):
    tag_to_imgs = {}
    for train_img in train_data:
        img_id = train_img['watchlink']
        img_tags = train_img['tags']

        # Initialize lat, lon, var values
        lat, lon = float(train_img['latitude']), float(train_img['longitude'])
        min_var = 10 ** 5
        for tag in img_tags:
            if tag_approx_loc[tag].var < min_var and train_tag_counts.get_count(tag) > .01 * len(train_data):
                min_var = tag_approx_loc[tag].var
            
            # Add img to tag_to_imgs
            if tag not in tag_to_imgs:
                tag_to_imgs[tag] = []
            tag_to_imgs[tag].append(train_img['watchlink'])

        img_data_mappings[img_id] = Location(lat, lon, min_var)
    return tag_to_imgs
    

tag_to_imgs = process_training_data(train_data, img_data_mappings, tag_approx_loc, train_tag_counts)

def process_test_data(test_data, train_tag_counts, img_data_mappings, tag_approx_loc, tag_to_imgs):
    total_tag_counts = train_tag_counts.copy()
    
    # Process test data
    for test_img in test_data:
        img_id = test_img['watchlink']
        img_tags = test_img['tags']

        remove_low_locality_tags(img_tags)

        # Count tags
        total_tag_counts.add_counts(img_tags)
   
        # Initialize lat, lon, and var
        min_var_loc = Location(37.7749, -122.4194, 15098163) # Approx lat/lon of SF, and approx var of tag 'iphone'
        #min_var_loc = Location(52.52, 13.405, 15098163) # Approx lat/lon of Berlin, and approx var of tag 'iphone'

        for tag in img_tags:
            if tag in tag_approx_loc and tag_approx_loc[tag].var < min_var_loc.var and train_tag_counts.get_count(tag) > .01 * len(train_data):
                min_var_loc = tag_approx_loc[tag]

            # Add img to tag_to_imgs
            if tag not in tag_to_imgs:
                tag_to_imgs[tag] = []
            tag_to_imgs[tag].append(test_img['watchlink'])

        img_data_mappings[img_id] = min_var_loc
    return total_tag_counts

total_tag_counts = process_test_data(test_data, train_tag_counts, img_data_mappings, tag_approx_loc, tag_to_imgs)

# Add vertices to graph
for img_id in img_data_mappings:
    G.add_vertex(img_id)


# Add edges to graph
edge_count = 0

for tag in tag_to_imgs:
    neighbors = tag_to_imgs[tag]
    for i in range(len(neighbors) - 1):
        for j in range(i + 1, len(neighbors)):
            edge_count += 1
            G.add_edge(neighbors[i], neighbors[j])
                

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
    '''
    if img == 'http://flickr.com/photos/32409718@N00/4189020951':
        for neighbor in G.neighbors(img):
            print(neighbor + ': ' + repr(img_data_mappings[neighbor]))
    print()
    print()
    '''
    def safe_div(num1, num2):
        if num2 == 0:
            return num1 / (10 ** -40)
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
    delta_squared = ((loc.lat - mean[0])**2 + (loc.lon - mean[1])**2)/2
    return mean[0], mean[1], var, delta_squared

# Perform location estimate update algorithm
mean_squared_change = 100 # Arbitrary number above CONVERGENCE_THRESHOLD
num_iter = 0
has_converged = True
while mean_squared_change > CONVERGENCE_THRESHOLD:
    if num_iter >= MAX_ITERATIONS:
        has_converged = False
        break
    num_iter += 1

    new_img_data_mappings = img_data_mappings.copy()
    mean_squared_change = 0
    for iteration, test_img in enumerate(all_data):
        img_id = test_img['watchlink']
        loc = img_data_mappings[img_id]
        lat, lon, var, delta_squared = calc_update(img_id, loc, G)
        new_img_data_mappings[img_id] = Location(lat, lon, var)
        mean_squared_change = mean_squared_change / (iteration+1) * iteration + (delta_squared / (iteration + 1))
    img_data_mappings = new_img_data_mappings
img = 'http://flickr.com/photos/51035718466@N01/2536809519'
'''
for neighbor in G.neighbors(img):
    print(neighbor + ': ' + repr(img_data_mappings[neighbor]))
    for test_img in all_data:
        if neighbor == test_img['watchlink']:
            print(test_img['keywords'].split(', '))
            break

print(img + ': ' + repr(img_data_mappings[img]))
for test_img in test_data + train_data:
    if img == test_img['watchlink']:
        print(test_img['keywords'].split(', '))
        break


print()
print()
'''
print('Num iterations: {0}'.format(num_iter))
print('Converged?: {0}'.format(has_converged))
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
        print(img_id + ': ' + repr(img_result_loc))
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
print('Greater than 1000 km: {0}'.format(other_count))
print('Med error: {0}'.format(median_error_finder.med()))

