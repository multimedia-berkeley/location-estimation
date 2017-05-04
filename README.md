# location-estimation

This project attempts to estimate the location of images/videos based off of user-defined tags.

## Overview

Based on this [paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.799.9639&rep=rep1&type=pdf). This project creates a Markov random field in which each node represents an image/video and its estimated location. The existence of an edge between nodes indicates that the nodes share a common tag. On each iteration of the algorithm, the Markov random field updates each node's estimated location based on the estimated locations of that node's neighbors.

## Lower Level Description of the Code

1. Delete all low locality tags (see more under the "Locality" heading below)
2. Find the average location in the training data for each tag
    * Also calculate the variance of locations for each tag
3. Create an undirected graph
    * Each vertex represents an image and contains the data for the image's location and variance (where lower variance represents higher confidence in the location)
    * An edge exists between a pair of vertices if the images have at least one common tag
4. Initialize each vertex's location and variance
    * For each image, initialize the vertex to be have the mean location and variance of the image's lowest variance tag
5. Run the update algorithm on each test image vertex until convergence
    * The update algorithm essentially takes a weighted average of all of a vertex's neighbors
        * The average is of the neighbor's location and the neighbor's variance determines its weight
6. Calculate the errors by finding the distance from each test image's estimated location to the ground truth location

## Locality

The locality of a tag describes how useful it is for determining location. For instance, a tag with a low locality score might be a tag such as ```"unitedstates"```. A tag with a high locality score might be a tag such as ```"timessquare"```.

The metrics that the locality script ```tagWeighting.py``` uses to calculate locality scores is described in this paper http://ceur-ws.org/Vol-1436/Paper58.pdf.

## Things to Consider

### General Improvements

* When evaluating the entire dataset, the error does rise with each iteration of the update algorithm. However, when the test/training data points are limited to a region, such as California, and the tag locality scores are evaluated based on all non-test data points (training + other excluded data points worldwide), the error does fall on each iteration of the update algorithm.

### Accuracy Improvements

* The code simply deletes all tags that are considered to be of "low locality".  This could potentially result in isolated nodes that did not have to be isolated. Utilizing these "low locality" tags could be useful for a node that would otherwise be isolated.

### Memory Improvements

* The graph is implemented using an adjacency matrix, which can get quite large and cause memory issues. Consider keeping track of each node's neighbors using a dict (e.g. ```{node: [neighbor1, neighbor2, ...]}```).

* The code stores lat, lon, and variance in a Location object. There may be minor memory improvements by not using this Location object and instead simply using a 3-element numpy array.

* The code uses each image/video's url as a unique identifier. There could be memory and runtime improvements by simply using 0, 1, 2, ... as IDs instead. The code would no longer have to store the urls in memory, and many dicts could be turned into lists.

### Runtime Improvements

* The vast majority of the runtime is spent in the update algorithm. There are likely optimizations to be made, such as smarter multiprocessing or batching calculations into linear operations.

* The graph is re-generated on every run of the script. Intelligently saving the graph or other intermediate calculations could save a lot of time.
