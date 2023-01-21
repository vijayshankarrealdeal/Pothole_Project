from fastapi import FastAPI,UploadFile,File
import sys
import cv2
import numpy as np
import os
import aiofiles
import base64
from colormap import rgb2hex

sys.path.append("../backend/mrcnn_demo")
from mrcnn_demo.m_rcnn import load_inference_model
from visualize import random_colors, get_mask_contours, draw_mask
app = FastAPI()
load_model = True
test_model = inference_config = None
@app.get('/api')
def api_call():
    global load_model,test_model,inference_config
    d = {}
    s = os.listdir("../backend/uploadedimages")
    file_ = "../backend/uploadedimages/" + s[0]
    img = cv2.imread(file_)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if load_model:
        test_model, inference_config = load_inference_model(1, "../backend/mask_rcnn_object_0005.h5")
    r = test_model.detect([image])[0]
    object_count = len(r["class_ids"])
    colors = random_colors(100)
    data = []
    for i in range(object_count):
        mask = r["masks"][:, :, i]
        h, w = mask.shape
        k = {}
        k['color'] = list(colors[i])
        contours = get_mask_contours(mask)
        RATIO_PIXEL_TO_SQUARE_CM = h*w
        for cnt in contours:
            cv2.polylines(img, [cnt], True, colors[i], 2)
            img = draw_mask(img, [cnt], colors[i],alpha=0.4)
            area_px = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt,True)
            perimeter = round(perimeter/ w, 3)
            area_cm = round(area_px/ RATIO_PIXEL_TO_SQUARE_CM ,5)
            center,radius = cv2.minEnclosingCircle(cnt)
            k['perimeter'] = perimeter
            k['area'] = area_cm
            k['width'] = radius/ w
            data.append(k)
    cv2.imwrite("../backend/detect.jpg", img)
    with open("../backend/detect.jpg", "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode('ascii')
    d["itemNumber"] = object_count            
    d["image"] = b64_string
    d['dimension'] = data
    os.remove(file_)
    load_model = False
    os.remove("../backend/detect.jpg")
    return d

@app.post('/upload')
async def upload(file:UploadFile = File(...)):
    filename = file.filename
    async with aiofiles.open('../backend/uploadedimages/'+filename, 'wb') as out_file:
        content = await file.read()  
        await out_file.write(content)  

    return {"Result": "OK"}
    
        