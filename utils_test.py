import unittest
import random
from utils import *

class GraphTest(unittest.TestCase):

    def test_basic(self):
        G = UndirectedGraph()
        self.assertEqual(G.vertices(), [])

        vertices = [chr(i) for i in range(ord('a'), ord('z') + 1)]
        for ch in vertices:
            G.add_vertex(ch)
        
        for vert in G.vertices():
            self.assertTrue(vert in vertices)

        G.add_edge('a', 'b')
        G.add_edge('c', 'a')
        G.add_edge('b', 'c')
        G.add_edge('b', 'd')

        a_neighbors = ['b', 'c']
        count = 0
        for vert in G.neighbors('a'):
            count += 1
            self.assertTrue(vert in a_neighbors)
        self.assertEqual(count, len(a_neighbors))

    def test_big_graph(self):
        G = UndirectedGraph()
        for i in range(ord('a'), ord('z') + 1):
            G.add_vertex(chr(i))

        for _ in range(5000):
            G.add_vertex(chr(random.randint(ord('a'), ord('z'))))

        for _ in range(5000):
            rand1 = random.randint(ord('a'), ord('z'))
            rand2 = random.randint(ord('a'), ord('z'))
            G.add_edge(rand1, rand2)


class MedianTest(unittest.TestCase):
    
    def test_basic(self):
        M = MedianFinder()
        self.assertEqual(M.med(), None)
        M.add(1)
        self.assertEqual(M.med(), 1)

        nums = [1]
        for _ in range(100):
            nums.append(random.randint(-100, 100))
            M.add(nums[-1])
            nums.sort()
            self.assertEqual(M.med(), nums[(len(nums)-1) // 2])

        nums.append(random.randint(-100, 100))
        M.add(nums[-1])
        nums.sort()
        self.assertEqual(M.med(), nums[(len(nums)-1) // 2])

class LocationTest(unittest.TestCase):

    def test_dist(self):
        place1 = Location(37.789216, -122.401476) # Montgomery Bart
        place2 = Location(37.784020, -122.408071) # Powell Bart
        self.assertTrue(Location.dist(place1, place2) < 1)

        place1 = Location(37.871692, -122.259381) # Campanile
        place2 = Location(37.778971, -122.419160) # SF City Hall
        self.assertTrue(Location.dist(place1, place2) > 17)
        self.assertTrue(Location.dist(place1, place2) < 18)

        place1 = Location(37.778971, -122.419160) # SF City Hall
        place2 = Location(38.897675, -77.036592) # White House
        self.assertTrue(Location.dist(place1, place2) > 3900)
        self.assertTrue(Location.dist(place1, place2) < 4000)

    def test_avg(self):
        locs = [Location(0, 0), Location(1, 2), Location(2, 4)]
        self.assertEqual(Location(1, 2), Location.avg(locs))

class CounterTest(unittest.TestCase):

    def test_basic(self):
        counter = Counter()
        items = []
        for _ in range(10):
            items.append('a')
        for _ in range(20):
            items.append('b')
        
        counter.add_counts(items)
        self.assertEqual(10, counter.get_count('a'))
        self.assertEqual(20, counter.get_count('b'))

        counter.add_counts(items)
        self.assertEqual(20, counter.get_count('a'))
        self.assertEqual(40, counter.get_count('b'))



if __name__ == '__main__':
    unittest.main()

