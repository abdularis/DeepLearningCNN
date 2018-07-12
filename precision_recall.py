# precision_recall.py
# Created by abdularis on 11/07/18


import argparse
import importlib
import json
import numpy as np
import sqlite3
import pickle
import tensorflow as tf
import data_config as cfg
import tqdm
from data_reader import DirDataSet
import distance_metrics
import database_ext


class Image(object):
    def __init__(self, path, truth, pred_labels, features):
        self.path = path
        self.truth = truth
        self.pred_labels = pred_labels
        self.features = features


def _query_images(db, query_labels):
    db_images = db.execute(
        "SELECT path, truth, pred_labels, features FROM images_repo "
        "WHERE pred_labels LIKE '%{}%' OR pred_labels LIKE '%{}%'".format(query_labels[0], query_labels[1]))
    return [Image(row[0], row[1], row[2], row[3]) for row in db_images]


def _query_items_count(db, truth):
    return db.execute("SELECT COUNT(*) FROM images_repo WHERE truth = '{}'".format(truth)).fetchone()[0]


def _count_relevant_items(truth, images):
    count = 0
    for img, _ in images:
        if img.truth == truth:
            count = count + 1
    return count


def _read_precision_recall_configs(config_path):
    with open(config_path, 'r') as f:
        j = json.load(f)
        configs = []
        for cfg in j:
            distance_metric = None
            if cfg['alg'] == 'euc':
                distance_metric = distance_metrics.EuclideanDistance(threshold=cfg['threshold'])
            elif cfg['alg'] == 'cos':
                distance_metric = distance_metrics.CosineDistance(threshold=cfg['threshold'])

            configs.append(PrecisionRecallCalcConfig(distance_metric=distance_metric, prefix=cfg['name']))
        return configs


def _save_obj(obj, name):
    with open('obj_' + name + ".pkl", 'wb') as f:
        pickle.dump(obj, f)


class PrecisionRecallCalcConfig(object):

    def __init__(self, distance_metric, prefix):
        self.per_class_precisions = {key: [] for key in cfg.one_hot_labels}
        self.per_class_recalls = {key: [] for key in cfg.one_hot_labels}
        self.distance_metric = distance_metric
        self.prefix = prefix

    def calculate(self, db, query_truth, query_features, gallery_images):
        filtered_results = self.distance_metric.filter(query_features, gallery_images)
        relevant_items_count = _count_relevant_items(query_truth, filtered_results)

        precision = relevant_items_count / len(filtered_results)
        recall = relevant_items_count / _query_items_count(db, query_truth)

        self.per_class_precisions[query_truth].append(precision)
        self.per_class_recalls[query_truth].append(recall)

    def save(self):
        _save_obj(self.per_class_precisions, '%s_per_class_precisions' % self.prefix)
        _save_obj(self.per_class_recalls, '%s_per_class_recalls' % self.prefix)


def calculate_precision_recall(test_dir_split, model_arch_module, model_path, db_path, config_path):

    db = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    model = model_arch_module.build_model_arch()
    extractor = model.stored_ops['features']
    data_test = DirDataSet(64, test_dir_split, cfg.one_hot)

    pr_configs = _read_precision_recall_configs(config_path)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path)

        for _ in tqdm.tqdm(range(data_test.batch_count)):
            images, one_hot = data_test.next_batch()
            truth_indexes = np.argmax(one_hot, 1)

            truth_labels = [cfg.one_hot_labels[i] for i in truth_indexes]

            preds_probs, features = model.predict(sess, images, extra_fetches=[extractor])
            preds_labels = cfg.get_predictions_labels(preds_probs, 2)

            for i in range(len(truth_labels)):
                curr_query_features = features[i]
                curr_query_truth = truth_labels[i]
                retrieved_gallery_images = _query_images(db, preds_labels[i])

                for config in pr_configs:
                    config.calculate(db, curr_query_truth, curr_query_features, retrieved_gallery_images)

            for config in pr_configs:
                config.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate precision & recall')
    parser.add_argument('--model-module', type=str, help='Python module string untuk model cnn', required=True)
    parser.add_argument('--test-dataset-dir', type=str, help='Direktori dataset', required=True)
    parser.add_argument('--model-path', type=str, help='Path model CNN', required=True)
    parser.add_argument('--config-path', type=str, help='Precision recall calculation configuration', required=True)
    parser.add_argument('--db-path', type=str, help='Image database path', required=True)

    args = parser.parse_args()

    calculate_precision_recall(
        args.test_dataset_dir,
        importlib.import_module(args.model_module),
        args.model_path,
        args.db_path,
        args.config_path
    )
