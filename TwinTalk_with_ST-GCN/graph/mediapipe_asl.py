# graph/mediapipe_asl.py

import numpy as np

class Graph:
    def __init__(self, layout='mediapipe_asl', strategy='spatial'):
        self.layout = layout
        self.strategy = strategy

        self.num_node = 21
        self.edge = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]

        self.A = self.get_adjacency_matrix()

    def get_adjacency_matrix(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[i][j] = 1
            A[j][i] = 1
        return A
