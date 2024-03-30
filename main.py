import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('x_classifier.h5')

def load_and_process_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (512, 512))                                          
    img = img / 255.0                 
    img = np.expand_dims(img, axis=0)      
    return img

def classify_image(image_path):
    img = load_and_process_image(image_path)
    prediction = model.predict(img)
    if prediction < 0.5:
        return "1"
        print(prediction)
    else:
        return "2"

image_path = "test.jpg"

result = classify_image(image_path)
print(result)
