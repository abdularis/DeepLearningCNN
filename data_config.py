# data_config.py.py
# Created by abdularis on 08/06/18


import numpy as np

one_hot = {
    'backpack':     [1., 0., 0., 0., 0., 0.],
    'microwave':    [0., 1., 0., 0., 0., 0.],
    'mug':          [0., 0., 1., 0., 0., 0.],
    'shoe':         [0., 0., 0., 1., 0., 0.],
    'teapot':       [0., 0., 0., 0., 1., 0.],
    'wristwatch':   [0., 0., 0., 0., 0., 1.]
}

one_hot_labels = np.array(['backpack', 'microwave', 'mug', 'shoe', 'teapot', 'wristwatch'])


def get_predictions_labels(preds_probs, top_k=1):
    return [one_hot_labels[indices[:top_k]] for indices in np.flip(np.argsort(preds_probs), axis=1)]
