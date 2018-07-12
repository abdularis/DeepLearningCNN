# feature_extractor.py.py
# Created by abdularis on 02/07/18


import argparse
import importlib
import os
import numpy as np
import sqlite3
import tqdm
import data_config as cfg
from data_reader import DirDataSet
import database_ext


def _create_database(path):
    if os.path.exists(path):
        os.remove(path)

    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.execute("CREATE TABLE images_repo ("
                 "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                 "path TEXT,"
                 "truth TEXT,"
                 "pred_labels TEXT,"
                 "features ARRAY)")
    conn.commit()
    return conn


def _insert_feature(db, record):
    db.executemany("INSERT INTO images_repo ('path', 'truth', 'pred_labels', 'features') "
                   "VALUES (?, ?, ?, ?)", record)
    db.commit()


def extract_features(test_dir_split, model_arch_module, model_path, out_db_path):
    import tensorflow as tf

    db = _create_database(out_db_path)
    model = model_arch_module.build_model_arch()
    extractor = model.stored_ops['features']
    data_test = DirDataSet(64, test_dir_split, cfg.one_hot)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path)

        for _ in tqdm.tqdm(range(data_test.batch_count)):
            images, one_hot = data_test.next_batch()
            truth_indexes = np.argmax(one_hot, 1)

            truth_labels = [cfg.one_hot_labels[i] for i in truth_indexes]
            file_paths = data_test.current_batch_file_paths

            preds_probs, features = model.predict(sess, images, extra_fetches=[extractor])
            preds_labels = [','.join(labels) for labels in cfg.get_predictions_labels(preds_probs, 2)]

            data_records = [(file_paths[i], truth_labels[i], preds_labels[i], features[i]) for i in range(len(file_paths))]

            _insert_feature(db, data_records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature extractor')
    parser.add_argument('--model-module', type=str, help='Python module string untuk model cnn', required=True)
    parser.add_argument('--test-dataset-dir', type=str, help='Direktori dataset', required=True)
    parser.add_argument('--model-path', type=str, help='Path model CNN', required=True)
    parser.add_argument('--out-db-path', type=str, help='Output database path', required=True)

    args = parser.parse_args()

    extract_features(
        args.test_dataset_dir,
        importlib.import_module(args.model_module),
        args.model_path,
        args.out_db_path
    )
