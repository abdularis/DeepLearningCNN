# precision_recall.py
# Created by abdularis on 11/07/18


import os
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
import image_search
import database_ext


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
    with open(name + ".pkl", 'wb') as f:
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

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        _save_obj(self.per_class_precisions, os.path.join(out_dir, '%s_per_class_precisions' % self.prefix))
        _save_obj(self.per_class_recalls, os.path.join(out_dir, '%s_per_class_recalls' % self.prefix))


def _f_measure(precision, recall):
    return 2 * precision * recall / (precision + recall)


def display_per_class_precision_recall(precision_dict, recalls_dict):
    if len(precision_dict) != len(recalls_dict):
        print('Precision and Recall dictionary should be in the same size!')
        return

    all_precisions = []
    all_recalls = []
    keys = precision_dict.keys()
    print('Average per class precision & recall: %d classes' % len(precision_dict))
    print('\t{:<15s} {:<12s} {:<12s} {:<12s}'.format('CLASS', 'PRECISION', 'RECALL', 'F-MEASURE'))
    print('------------------------------------------------------------')
    for key in keys:
        all_precisions.extend(precision_dict[key])
        all_recalls.extend(recalls_dict[key])

        avg_precision = np.mean(precision_dict[key])
        avg_recall = np.mean(recalls_dict[key])
        f_measure = _f_measure(avg_precision, avg_recall)

        print('\t{:<15s} {:<12f} {:<12f} {:<12f}'.format(key, avg_precision, avg_recall, f_measure))

    p = np.mean(all_precisions)
    r = np.mean(all_recalls)
    print('------------------------------------------------------------')
    print('\t{:<15s} {:<12f} {:<12f} {:<12f}'.format('ALL CLASSES', p, r, _f_measure(p, r)))


def calculate_precision_recall(test_dir_split, model_arch_module, model_path, db_path, config_path, out_dir):

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
                retrieved_gallery_images = image_search.query_images_in_test_db(db, preds_labels[i])

                for config in pr_configs:
                    config.calculate(db, curr_query_truth, curr_query_features, retrieved_gallery_images)

        for config in pr_configs:
            config.save(out_dir)
            print('\t --- configuration name: %s ---' % config.prefix)
            display_per_class_precision_recall(config.per_class_precisions, config.per_class_recalls)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate precision & recall')
    parser.add_argument('--model-module', type=str, help='Python module string untuk model cnn', required=True)
    parser.add_argument('--test-dataset-dir', type=str, help='Direktori dataset', required=True)
    parser.add_argument('--model-path', type=str, help='Path model CNN', required=True)
    parser.add_argument('--config-path', type=str, help='Precision recall calculation configuration', required=True)
    parser.add_argument('--db-path', type=str, help='Image database path', required=True)
    parser.add_argument('--out-dir', type=str, help='Direktori output data', required=True)

    args = parser.parse_args()

    calculate_precision_recall(
        args.test_dataset_dir,
        importlib.import_module(args.model_module),
        args.model_path,
        args.db_path,
        args.config_path,
        args.out_dir
    )
