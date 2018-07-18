# image_search.py
# Created by abdularis on 16/07/18


import numpy as np
import model_client
import distance_metrics
import data_config as cfg
from skimage import transform


class Image(object):
    def __init__(self, path, pred_labels, features):
        self.path = path
        self.pred_labels = pred_labels
        self.features = features


class ImageTest(Image):
    def __init__(self, path, truth, pred_labels, features):
        super().__init__(path, pred_labels, features)
        self.truth = truth


def query_images_in_test_db(db, query_labels):
    db_images = db.execute(
        "SELECT path, truth, pred_labels, features FROM images_repo "
        "WHERE pred_labels LIKE '%{}%' OR pred_labels LIKE '%{}%'".format(query_labels[0], query_labels[1]))
    return [ImageTest(row[0], row[1], row[2], row[3]) for row in db_images]


def query_images(db, query_labels):
    db_images = db.execute(
        "SELECT path, pred_labels, features FROM images_repo "
        "WHERE pred_labels LIKE '%{}%' OR pred_labels LIKE '%{}%'".format(query_labels[0], query_labels[1]))
    return [Image(row[0], row[1], row[2]) for row in db_images]


def search_image(image, db):
    image = transform.resize(image, (128, 128))
    probs, features = model_client.inference(np.array([image], dtype=np.float32))
    probs_labels = cfg.get_predictions_labels([probs], 2)[0]
    images_result = query_images(db, probs_labels)
    images_result = distance_metrics.CosineDistance(threshold=0.4).filter(features, images_result)
    images_result = [img[0].path for img in images_result]

    return probs_labels, images_result
