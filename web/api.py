# api.py
# Created by abdularis on 16/07/18

from flask import Flask, request, jsonify, g, render_template
import scipy.misc
import os
import sqlite3
import image_search

app = Flask(__name__)


TEMP_QUERY_IMAGE_PATH = '/tmp/image_query.jpg'
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


def _get_paths(images_result):
    return [img[0].path for img in images_result]
    # final_result = []
    # for img, distance in images_result:
    #     del img.features
    #     final_result.append({
    #         'path': img.path,
    #         'pred_labels': img.pred_labels
    #     })
    # return final_result


@app.route('/api/search', methods=['POST'])
def search():
    if request.method == 'POST':

        image_path = TEMP_QUERY_IMAGE_PATH
        if request.args.get('query_gallery', ''):
            image_path = request.args.get('query_gallery', image_path)
            image_path = os.path.join('web', image_path)
        else:
            f = request.files['image']
            f.save(image_path)

        img = scipy.misc.imread(image_path)

        threshold = float(request.form['threshold'])
        probs_labels, images_result = image_search.search_image(img, get_db(), request.form['metric'], threshold)

        return jsonify(pred_labels=probs_labels[0],
                       result=_get_paths(images_result))


@app.route('/')
def index():
    if request.args.get('search_by_gallery', ''):
        gallery_path = request.args.get('search_by_gallery')
        img_path = os.path.join('web/static/gallery/', gallery_path)
        img = scipy.misc.imread(img_path)
        probs_labels, images_result = image_search.search_image(img, get_db())
        return render_template('index.html', title='Image Search - search by gallery',
                               gallery_path=os.path.join('static/gallery/', gallery_path),
                               pred_labels=probs_labels[0], img_path_list=_get_paths(images_result))
    return render_template('index.html', title='Image Search')


@app.route('/browse')
def browse():
    if request.args.get('class', ''):
        cat = request.args.get('class')
        img_list = os.listdir('web/static/gallery/%s' % cat)
        img_list = [os.path.join(cat, i_name) for i_name in img_list]
        return render_template('browse.html', title='Gallery - %s' % cat, img_path_list=img_list, category=cat)
    return render_template('browse.html', title='Gallery')


app.run(
    host='0.0.0.0',
    debug=True
)
