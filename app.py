import argparse
import os
import os.path as osp
import cv2
import sys
import glob
import re
import numpy as np
from paddleseg.utils import get_sys_env, logger
from deploy.infer import Predictor
import shutil
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

def background_replace(args):
    
    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.img_path is not None:
        img = cv2.imread(args.img_path)
        bg = get_bg_img(args.bg_img_path, img.shape)
        args.input_shape = img.shape
        predictor = Predictor(args)
        comb = predictor.run(img, bg)
        save_name = osp.basename(args.img_path)
        save_path = osp.join(args.save_dir, save_name)
        cv2.imwrite(save_path, comb)
        return save_path

def get_bg_img(bg_img_path, img_shape):
    if bg_img_path is None:
        bg = 255 * np.ones(img_shape)
    else:
        bg = cv2.imread(bg_img_path)
    return bg

class input_arguments():
        def __init__(self):
            self.img_path = 'img.jpg'   
            self.cfg = 'export_model/ppseg_lite_portrait_398x224_with_softmax/deploy.yaml'
            self.bg_img_path = 'bg.jpg'
            self.bg_video_path = None
            self.input_shape = (192,192)
            self.save_dir = 'output'
            self.use_gpu = False
            self.use_optic_flow = False
            self.test_speed = True
            self.soft_predict = True

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        args.save_dir = os.path.join(basepath, 'output')
        shutil.rmtree(args.save_dir)
        os.mkdir(args.save_dir)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        f1 = request.files['file2']
        file_path1 = os.path.join(
            basepath, 'uploads', secure_filename(f1filename))
        f1.save(file_path1)

        args = input_arguments()
        args.img_path = file_path
        args.bg_img_path = file_path1
        
        pth = background_replace(args)
        os.remove(args.img_path)
        os.remove(args.bg_img_path)
        return render_template('index.html',filename = pth)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)