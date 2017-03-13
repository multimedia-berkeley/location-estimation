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

        a_neighbors = ['b', 'c']
        count = 0
        for vert in G.neighbors('a'):
            count += 1
            self.assertTrue(vert in a_neighbors)
        self.assertEqual(count, len(a_neighbors))

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

if __name__ == '__main__':
    unittest.main()

