import tensorflow as tf 
import numpy as np 
from PIL import Image, ImageOps	
import cv2


def process_image(img, img_size=(224, 224)):
  """
  This function is used to pre-process any chosen picture by the user
  to the appropriate format that the model accepts.
  Parameters:
    img: The input Image, opened using the PIL library
    img_size: Defaults to (224, 224) because it is the size that the 
              model accepts

  Returns:
    An Image array that is ready to be fed into the model.
  """
  # reshapes the image
  image = ImageOps.fit(img, img_size, Image.ANTIALIAS)
  # converts the image into numpy array
  image = np.asarray(image)
  # converts image from BGR color space to RGB
  img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  img_resize = (cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_CUBIC))/255.

  img_reshape = img_resize[np.newaxis,...]
  
  return img_reshape


def prediction_result(model, image_data):
  """
  The function that returns the prediction result from the model
  Parameters:
    model: The model to be used to classify
    image_data: Image array that is returned by the process_image function
  
  Returns --> Dictionary with class and accuracy values

  """
  # Mapping prediction results to the Flower type
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

