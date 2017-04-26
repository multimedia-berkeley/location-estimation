# location-estimation

Steps

1. Delete all low locality tags (TODO: describe locality more)
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
