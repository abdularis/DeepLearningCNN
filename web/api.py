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

        probs_labels, images_result = image_search.search_image(img, get_db())

        return jsonify(pred_labels=probs_labels[0],
                       result=images_result)


@app.route('/')
def index():
    return render_template('index.html', title='Image Search')


@app.route('/browse')
def browse():
    return render_template('browse.html', title='Browse Gallery')


app.run(
    host='0.0.0.0',
    debug=True
)
