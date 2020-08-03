import os

from flask import Flask, render_template, request, jsonify
from flask import send_from_directory
from bov_cbir import get_bovw,find_scene
from skimage import io
import binascii
import cv2
import numpy as np
import glob
import h5py
import pickle
INDEX = os.path.join(os.path.dirname(__file__), 'index.csv')

# create flask instance
app = Flask(__name__)


<<<<<<< HEAD
embeddings = h5py.File('/ssd_scratch/cvit/ashishmenon/embeddings/normalized_hist_tf_idf_db.h5', 'r')
db_embeddings = np.array(embeddings["embed"])
cluster_centres = h5py.File('/ssd_scratch/cvit/ashishmenon/embeddings/clusters_hist.h5',"r")
db_cluster_centres = np.array(cluster_centres["embed"])
idf = h5py.File('/ssd_scratch/cvit/ashishmenon/embeddings/idf_hist.h5',"r")
db_idf = np.array(idf["embed"])
with open ('/ssd_scratch/cvit/ashishmenon/embeddings/file_paths_hist_db', 'rb') as fp:
    db_fp = pickle.load(fp)
=======
# embeddings = h5py.File('/ssd_scratch/cvit/ashishmenon/v_google_frame/embeddings/normalized_tf_idf_db.h5', 'r')
# db_embeddings = np.array(embeddings["embed"])
# cluster_centres = h5py.File('/ssd_scratch/cvit/ashishmenon/v_google_frame/embeddings/clusters.h5',"r")
# db_cluster_centres = np.array(cluster_centres["embed"])
# idf = h5py.File('/ssd_scratch/cvit/ashishmenon/v_google_frame/embeddings/idf.h5',"r")
# db_idf = np.array(idf["embed"])
# with open ('/ssd_scratch/cvit/ashishmenon/v_google_frame/embeddings/file_paths_db', 'rb') as fp:
#     db_fp = pickle.load(fp)
>>>>>>> 734805c6f32a9299ff763d97d7e879ca6e11db35

def encode(x):
    return binascii.hexlify(x.encode('utf-8')).decode()

def decode(x):
    return binascii.unhexlify(x.encode('utf-8')).decode()


# main route
@app.route('/')
def index():
    #image_names = os.listdir('/home/ashish95/Desktop/Desktop_files/ui_try/flask_testing_jup/static/images/')
    #image_address = ['images/' + name for name in image_names]
    image_address_raw = glob.glob('./static/images/*.tif')
    # image_address_raw = random.sample(image_address_raw,6)
    image_address = [encode(i) for i in image_address_raw]
    return render_template('index.html',images=image_address)


@app.route('/cdn/<path:filepath>')
def download_file(filepath):
    directory,filename = os.path.split(decode(filepath))
    return send_from_directory(directory, filename, as_attachment=False)



@app.route('/search', methods=['POST'])
def search():

    if request.method == "POST":

        RESULTS_ARRAY = []

        image_url_encoded = request.form.get('img')
        image_url = decode(image_url_encoded.split('/')[-1])
        try:
            # ret_results,score = find_scene(image_url.split(),db_embeddings,db_fp,db_cluster_centres,db_idf)
            # query = io.imread(image_url)
            # for i in range(5):
            #     RESULTS_ARRAY.append(
            #             {"image": 'cdn/'+encode(ret_results[i]), "score": str(score[i])})
            image_address = ['cdn/'+encode(i) for i in glob.glob('./static/images/*.png')]
            RESULTS_ARRAY = {"image":image_address, "score":["0.8","0.7","0.9","0.6","0.5","0.8"]}
            return jsonify(results=(RESULTS_ARRAY))

        except:
            print('ERROR')
            # return error
            return (jsonify({"sorry": "Sorry, no results! Please try again."}), 500)




if __name__ == '__main__':
    app.run(debug=True,port=int(os.environ.get('PORT', 6793)))
