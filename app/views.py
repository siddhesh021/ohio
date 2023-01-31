from flask import render_template,request
from flask import redirect, url_for
import os
from PIL import Image
from app import utils
from app.utils import pipeline_model

def get_width(path):
    img=Image.open(path)
    size=img.size
    aspect_ratio=size[0]/size[1]
    w=300*aspect_ratio
    return int(w)

UPLOAD_FOLDER='static/uploads'

def base():
    return render_template('base.html')

def index():
    return render_template('index.html')

def faceApp():
    return render_template('faceApp.html')

def classify():
    if request.method == 'POST':
        f=request.files['images']
        filename=f.filename
        path=os.path.join(UPLOAD_FOLDER,filename)
        f.save(path)
        w=get_width(path)
        pipeline_model(path,filename=filename,color='bgr')
        return render_template('classify.html',upload=True,img_name=filename,w=w)
    return render_template('classify.html',upload=False,img_name="xyz",w=300)