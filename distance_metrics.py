# distance_metrics.py
# Created by abdularis on 12/07/18


import numpy as np


class DistanceMetric(object):

    def __init__(self, threshold):
        self.threshold = threshold

    def filter(self, query_features, gallery_images):
        result = [(img, self.calc_distance(query_features, img.features)) for img in gallery_images]
        result = sorted(result, key=lambda d: d[1])

        return [img for img in filter(lambda d: d[1] < self.threshold, result)]

    def calc_distance(self, query_features, img_features):
        pass


class CosineDistance(DistanceMetric):

    def __init__(self, threshold=0.5):
        super().__init__(threshold)

    def calc_distance(self, query_features, img_features):
        return CosineDistance._cd(query_features, img_features)

    @staticmethod
    def _cd(a, b):
        return 1 - (np.dot(a, b) / (np.sqrt((a ** 2).sum()) * np.sqrt((b ** 2).sum())))


class EuclideanDistance(DistanceMetric):

    def __init__(self, threshold=50.):
        super().__init__(threshold)

    def calc_distance(self, query_features, img_features):
        return np.linalg.norm(query_features - img_features)
