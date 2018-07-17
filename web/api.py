# api.py
# Created by abdularis on 16/07/18

from flask import Flask, request, jsonify, g, url_for, render_template
from skimage import transform
import numpy as np
import scipy.misc
import sqlite3
import model_client
import data_config as cfg
import image_search
import distance_metrics

app = Flask(__name__)


DATABASE = 'gallery.db'


def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        import database_ext
        db = g._database = sqlite3.connect(DATABASE, detect_types=sqlite3.PARSE_DECLTYPES)
    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


@app.route('/api/search', methods=['POST'])
def search():
    if request.method == 'POST':
        print(request)
        f = request.files['image']
        f.save('/home/abdularis/image_query.jpg')

        img = scipy.misc.imread('/home/abdularis/image_query.jpg')
        img = transform.resize(img, (128, 128))

        probs, features = model_client.inference(np.array([img], dtype=np.float32))
        probs_labels = cfg.get_predictions_labels([probs], 2)[0]
        images_result = image_search.query_images(get_db(), probs_labels)
        images_result = distance_metrics.CosineDistance().filter(features, images_result)
        images_result = [img[0].path for img in images_result]

        return jsonify(pred_labels=probs_labels[0],
                       result=images_result)


@app.route('/')
def index():
    return render_template('index.html')


app.run(
    debug=True
)
