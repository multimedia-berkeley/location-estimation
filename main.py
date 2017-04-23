import argparse
import csv
import multiprocessing as mp
import os.path
import pickle
import time

import numpy as np

import data
from utils import *


def main(MAX_ITERATIONS=20):
    main_timer = Timer()
    output_log = []

    data_fetch_timer = Timer()
    train_data, test_data = get_data()
    output_log.append('Took {0}s to parse the data'.format(data_fetch_timer.time()))

    img_data_mappings = {}

    preprocess_timer = Timer()
    locality = get_tag_locality()
    tag_approx_loc, train_tag_counts = process_train_tags(train_data, locality)
    tag_to_imgs = process_training_data(train_data, img_data_mappings, tag_approx_loc, train_tag_counts)
    total_tag_counts = process_test_data(test_data, train_data, train_tag_counts, img_data_mappings, tag_approx_loc, tag_to_imgs, locality)
    output_log.append('Took {0}s to preprocess the data'.format(preprocess_timer.time()))

    create_graph_timer = Timer()
    G = create_graph(train_data, test_data, tag_to_imgs)
    output_log.append('Took {0}s to create the graph'.format(create_graph_timer.time()))
               
    update_timer = Timer()
    img_data_mappings, num_iter, has_converged = run_update(G, test_data, img_data_mappings, MAX_ITERATIONS)
    output_log.append('Took {0}s to perform the update'.format(update_timer.time()))
    output_log.append('Total runtime: {0}'.format(main_timer.time()))

    errors = calc_errors(test_data, img_data_mappings)

    print('Num train points: {0}'.format(len(train_data)))
    print('Num test points: {0}'.format(len(test_data)))
    print('Num edges: {0}'.format(G.num_edges))
    print('Num iterations: {0}'.format(num_iter))
    print('Converged?: {0}'.format(has_converged))
    print('Less than 1 km: {0}'.format(errors[0]))
    print('Less than 5 km: {0}'.format(errors[1]))
    print('Less than 10 km: {0}'.format(errors[2]))
    print('Less than 100 km: {0}'.format(errors[3]))
    print('Less than 1000 km: {0}'.format(errors[4]))
    print('Greater than 1000 km: {0}'.format(errors[5]))
    print('')
    print('\n'.join(output_log))


def get_data():
    '''
    if os.path.isfile('/int_train') and os.path.isfile('int_test'):
        with open('int_train.pickle', 'rb') as f:
            train_data = pickle.load(f)
        with open('int_test.pickle', 'rb') as f:
            test_data = pickle.load(f)
    else:
        train_data, test_data = data.get_train_test()
        with open('int_train.pickle', 'wb') as f:
            pickle.dump(train_data, f)
        with open('int_test.pickle', 'wb') as f:
            pickle.dump(test_data, f)
    '''
    all_data = data.get()
    train_data, test_data = data.split(all_data, 0.8)
    return train_data, test_data


def get_tag_locality():
    locality_str = ''
    with open('/Users/daniel/Developer/ICME/tagweights.tsv', 'r') as f:
        locality_str = f.read()
    return eval(locality_str)


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

        remove_low_locality_tags(locality, img_tags)

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


def remove_low_locality_tags(locality, tags_list):
    '''Drop low locality tags. Mutates original list'''
    LOCALITY_THRESHOLD = 1 # Locality of 'newyorkcity' is 0.057
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


def process_training_data(train_data, img_data_mappings, tag_approx_loc, train_tag_counts):
    tag_to_imgs = {}
    for train_img in train_data:
        img_id = train_img['watchlink']
        img_tags = train_img['tags']

        # Initialize lat, lon, var values
        lat, lon = float(train_img['latitude']), float(train_img['longitude'])
        min_var = 10 ** 5
        for tag in img_tags:
            if tag_approx_loc[tag].var < min_var:
                min_var = tag_approx_loc[tag].var
            
            # Add img to tag_to_imgs
            if tag not in tag_to_imgs:
                tag_to_imgs[tag] = []
            tag_to_imgs[tag].append(train_img['watchlink'])

        img_data_mappings[img_id] = Location(lat, lon, min_var)
    return tag_to_imgs
 

def process_test_data(test_data, train_data, train_tag_counts, img_data_mappings, tag_approx_loc, tag_to_imgs, locality):
    total_tag_counts = train_tag_counts.copy()
    
    # Process test data
    for test_img in test_data:
        img_id = test_img['watchlink']
        img_tags = test_img['tags']

        remove_low_locality_tags(locality, img_tags)

        # Count tags
        total_tag_counts.add_counts(img_tags)
   
        # Initialize lat, lon, and var
        min_var_loc = Location(37.7749, -122.4194, 15098163) # Approx lat/lon of SF, and approx var of tag 'iphone'
        #min_var_loc = Location(52.52, 13.405, 15098163) # Approx lat/lon of Berlin, and approx var of tag 'iphone'

        for tag in img_tags:
            if tag in tag_approx_loc and tag_approx_loc[tag].var < min_var_loc.var:
                min_var_loc = tag_approx_loc[tag]

            # Add img to tag_to_imgs
            if tag not in tag_to_imgs:
                tag_to_imgs[tag] = []
            tag_to_imgs[tag].append(test_img['watchlink'])

        img_data_mappings[img_id] = min_var_loc
    return total_tag_counts


def create_graph(train_data, test_data, tag_to_imgs):
    def add_vertices():
        for img in train_data:
            G.add_vertex(img['watchlink'])

        for img in test_data:
            G.add_vertex(img['watchlink'])
    def add_edges():
        for tag in tag_to_imgs:
            neighbors = tag_to_imgs[tag]
            for i in range(len(neighbors) - 1):
                for j in range(i + 1, len(neighbors)):
                    G.add_edge(neighbors[i], neighbors[j])

    G = UndirectedGraph()
    add_vertices()
    add_edges()

    return G

def run_update(G, test_data, img_data_mappings, MAX_ITERATIONS):
    CONVERGENCE_THRESHOLD = 0.00006288 # About the mean sqaured difference of 1km
    mean_squared_change = 100 # Arbitrary number above CONVERGENCE_THRESHOLD
    num_iter = 0
    has_converged = True
    while mean_squared_change > CONVERGENCE_THRESHOLD:
        if num_iter >= MAX_ITERATIONS:
            has_converged = False
            break
        num_iter += 1

        global update
        def update(test_img):
                img_id = test_img['watchlink']
                loc = img_data_mappings[img_id]
                lat, lon, var, delta_squared = calc_update(img_id, loc, G, img_data_mappings)
                return Location(lat, lon, var), delta_squared
 
        new_img_data_mappings = img_data_mappings.copy()
        with mp.Pool(mp.cpu_count()) as p:
            updates = p.map(update, test_data)
            mean_squared_change = 0
            for i, test_img in enumerate(test_data):
                img_id = test_img['watchlink']
                new_loc = updates[i][0]
                delta_squared = updates[i][1]
                new_img_data_mappings[img_id] = new_loc
                mean_squared_change = mean_squared_change / (i+1) * i + (delta_squared / (i + 1))
        
        img_data_mappings = new_img_data_mappings
    return img_data_mappings, num_iter, has_converged



def calc_update(img, loc, G, img_data_mappings):

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


def calc_errors(test_data, img_data_mappings):
    # Calculate error
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
        if error < 1:
            one_km_count += 1
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
    return [one_km_count, five_km_count, ten_km_count, hundred_km_count, thousand_km_count, other_count]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--maxiter', nargs=1, type=int,
            help='Max number of iterations to run.')
    arguments = parser.parse_args() # pylint: disable=invalid-name
    if arguments.maxiter is None:
        main()
    else:
        main(arguments.maxiter[0])

