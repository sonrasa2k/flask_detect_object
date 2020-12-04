import cv2 as cv
import numpy as np
import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import  base64
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
OUTPUT_FOLDER = 'static/output'
def chuyen_base64_sang_anh(anh_base64):

    anh_base64 = np.fromstring(base64.b64decode(anh_base64), dtype=np.uint8)
    anh_base64 = cv.imdecode(anh_base64, cv.IMREAD_ANYCOLOR)
    return anh_base64

whT = 320
confThreshold = 0.5
nmsThreshold = 0.2

#### LOAD MODEL
## Coco Names
classesFile = "coco.names"
classNames = []
classids = []
name = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
## Model Files
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3_608.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                   (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return classIds

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_image():
    global classids
    global  name
    name = []
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = cv.imread('static/uploads/'+filename)
        blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)
        classids= findObjects(outputs,img)
        filename = 'out_'+filename
        cv.imwrite('static/output/'+filename,img)
        # print('upload_image filename: ' + filename)
        flash('Ảnh Đã Được Tải lên và nhận diện')
        for i in classids:
            name.append(classNames[i])
        print(name)
        return render_template('index.html', filename=filename,foobar=name)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename = 'output/'+filename) ,code=301)


if __name__ == "__main__":
    app.run()