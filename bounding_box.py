import numpy as np

class BoundingBox:
    def __init__(self, box_shape=np.zeros((1, 2)), centroid=np.zeros((1, 2))) -> None:
        self.box_shape = box_shape
        self.centroid = centroid