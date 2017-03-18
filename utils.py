import math
from heapq import heappush, heappop

class Location(object):
    def __init__(self, lat=0.0, lon=0.0, var=9999999999999.0):
        self.lat = lat
        self.lon = lon
        self.var = var

    def __repr__(self):
        return '(lat: {0}, lon: {1}, var: {2})'.format(self.lat, self.lon, self.var)

    def __eq__(self, other):
        return self.lat == other.lat and self.lon == other.lon and self.var == other.var

    def copy(self):
        return Location(self.lat, self.lon, self.var)

    def avg(locations):
        total_lat = 0
        total_lon = 0
        for loc in locations:
            total_lat += loc.lat
            total_lon += loc.lon

        avg_lat = total_lat / len(locations)
        avg_lon = total_lon / len(locations)
        return Location(avg_lat, avg_lon)

    def dist(loc1, loc2, method=1):
        loc1 = loc1.copy()
        loc2 = loc2.copy()

        if abs(loc1.lat)>90 or abs(loc2.lat)>90 or abs(loc1.lon)>360 or abs(loc2.lon)>360:
            dist = -99999
            print('Degree(s) illegal! distance = -99999')
            return dist

        if loc1.lon < 0:
            loc1.lon += 360
        
        if loc2.lon < 0:
            loc2.lon += 360

        # Default method is 1.
        if method == 1:
            km_per_deg_la = 111.3237
            km_per_deg_lo = 111.1350
            km_la = km_per_deg_la * (loc1.lat-loc2.lat)
            # Always calculate the shorter arc.
            if abs(loc1.lon-loc2.lon) > 180:
                dif_lo = abs(loc1.lon-loc2.lon)-180
            else:
                dif_lo = abs(loc1.lon-loc2.lon)

            km_lo = km_per_deg_lo * dif_lo * math.cos((loc1.lat+loc2.lat) * math.pi / 360)
            dist = math.sqrt(km_la**2 + km_lo**2)
        else:
            R_aver = 6374
            deg2rad = math.pi/180
            loc1.lat = loc1.lat * deg2rad
            loc1.lon = loc1.lon * deg2rad
            loc2.lat = loc2.lat * deg2rad
            loc2.lon = loc2.lon * deg2rad
            dist = R_aver * math.acos(cos(loc1.lat)* cos(loc2.lat) * cos(loc1.lon-loc2.lon) + math.sin(loc1.lat) * math.sin(loc2.lat))

        return dist

class Counter(object):
    def __init__(self):
        self.counts = {}

    def __repr__(self):
        return repr(self.counts)

    def copy(self):
        counter_copy = Counter()
        counter_copy.counts = self.counts.copy()
        return counter_copy

    def add_counts(self, lst):
        for item in lst:
            if item not in self.counts:
                self.counts[item] = 0
            self.counts[item] += 1

    def get_count(self, item):
        if item in self.counts:
            return self.counts[item]
        else:
            return 0

class UndirectedGraph(object):
    def __init__(self):
        self.vertex_mappings = {}
        self.backwards_mapping = {}
        self.next_vert_id = 0
        self.adj_mtx = [[0]] #Adjacency matrix

    def add_vertex(self, label):
        self.vertex_mappings[label] = self.next_vert_id
        self.backwards_mapping[self.next_vert_id] = label

        if self.next_vert_id >= len(self.adj_mtx[0]):
            # Double adj matrix size
            delta_size = len(self.adj_mtx[0])
            for row in self.adj_mtx:
                row.extend([0] * delta_size)

            for _ in range(delta_size):
                self.adj_mtx.append([0] * len(self.adj_mtx[0]))

        self.next_vert_id += 1

    def add_edge(self, label1, label2):
        if label1 in self.vertex_mappings and label2 in self.vertex_mappings:
            vert1_id = self.vertex_mappings[label1]
            vert2_id = self.vertex_mappings[label2]

            self.adj_mtx[vert1_id][vert2_id] = 1
            self.adj_mtx[vert2_id][vert1_id] = 1  # Since its an undirected graph
            return 1
        else:
            return 0
    
    def vertices(self):
        return list(self.vertex_mappings.keys())

    def neighbors(self, label):
        neighbors = []
        vert_id = self.vertex_mappings[label]
        for i in range(len(self.adj_mtx[0])):
           if self.adj_mtx[vert_id][i] == 1:
               neighbors.append(self.backwards_mapping[i])
        return neighbors


class MedianFinder(object):
    def __init__(self):
        self.lower = []
        self.upper = []

    def add(self, el):
        if len(self.lower) == 0 and len(self.upper) == 0:
            heappush(self.upper, el)
        elif el < self.upper[0]:
            heappush(self.lower, -el)
        else:
            heappush(self.upper, el)

        if len(self.lower) - len(self.upper) > 1:
            heappush(self.upper, -heappop(self.lower))
        elif len(self.upper) - len(self.lower) > 1:
            heappush(self.lower, -heappop(self.upper))
    
    def med(self):
        if len(self.lower) == 0 and len(self.upper) == 0:
            return None
        elif len(self.upper) > len(self.lower):
            return self.upper[0]
        else:
            return -self.lower[0]

