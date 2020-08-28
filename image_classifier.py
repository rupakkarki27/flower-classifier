import tensorflow as tf 
import numpy as np 
from PIL import Image, ImageOps	
import cv2

def process_image(img, img_size=(224, 224)):
    image = ImageOps.fit(img, img_size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = (cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_CUBIC))/255.

    img_reshape = img_resize[np.newaxis,...]
    
    return img_reshape


def prediction_result(model, image_data):
    classes = {0: "Daisy", 
               1: "Dandelion",
               2: "Rose",
               3: "Sunflower",
               4: "Tulip"}
    
    pred = model.predict(image_data)
    pred = pred.round(2)
    result = np.argmax(pred)
    
    prediction = {"class": classes[result],
                  "accuracy": np.round(np.max(pred) * 100, 2)}
    
    return prediction

