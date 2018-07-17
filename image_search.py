# image_search.py
# Created by abdularis on 16/07/18


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
