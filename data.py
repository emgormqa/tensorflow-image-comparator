import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# load and resize images function
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (512, 512))
            images.append(img)
    return images

# load images from 2 folders
1_images = load_images_from_folder("1")
2_images = load_images_from_folder("2")

# tagging
labels = [1] * len(1_images) + [0] * len(2_images)

# creating array from all images
images = np.array(1_images + 2_images)
labels = np.array(labels)

# separation of data into training and test data
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# save all data
np.save('train_images.npy', train_images)
np.save('train_labels.npy', train_labels)
np.save('test_images.npy', test_images)
np.save('test_labels.npy', test_labels)
