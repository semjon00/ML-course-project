import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
from PIL import Image

from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

# Paths
aug_wsi_images = '/kaggle/input/augmented/augmented/wsi'
aug_tma_images = '/kaggle/input/augmented/augmented/tma'
model_file = '/kaggle/working/model.h5'

classes = ['CC', 'EC', 'HGSC', 'LGSC', 'MC']

# If the best prediction probability is below that threshold, it's labelled as 'Other'
threshold = 0.3

# To convert labels to one-hot vectors and vice-versa
def values_to_one_hot(values, classes):
    vector = []
    for value in values:
        one_hot = np.zeros(5)
        one_hot[classes.index(value)] = 1
        vector.append(one_hot)
    return np.array(vector)

def one_hot_to_values(vector, classes):
    values = []
    for one_hot in vector:
        if np.max(one_hot) < threshold:
            value = 'Other'
        else:
            value = classes[np.argmax(one_hot)]
        values.append(value)
    return np.array(values)

# To load images and labels from 'root' folder
def load_data(root):
    files = os.listdir(root)
    n = len(files)
    images = []
    labels = []
    confidences = []    
    # Loading augmented images and labels
    for i, file in enumerate(files):
        if file.endswith(".png"):
            image_id = os.path.splitext(file)[0]
            label_path = os.path.join(root, f"{image_id}.txt")
            if os.path.exists(label_path):
                with open(label_path, "r") as label_file:
                    label, confidence = label_file.read().strip().split()
                image_path = os.path.join(root, file)
                image = Image.open(image_path)
                image = image.resize((224,224)) # Resizing to 244x244 for Resnet model
                images.append(np.array(image))
                labels.append(label)
                confidences.append(confidence)
        print(f'Loading images ({root}): {i+1} / {n}',end='\r')
    print()
    # Converting labels to one-hot vectors 
    labels_one_hot = values_to_one_hot(labels, classes)
    # Reshaping image array for model training
    images = np.array(images).reshape(-1, 224, 224, 3)
    return images, labels_one_hot, confidences

# Loading all training images 
images_wsi, labels_wsi, confidences_wsi = load_data(aug_wsi_images)
images_tma, labels_tma, confidences_tma = load_data(aug_tma_images)
images = np.concatenate((images_wsi, images_tma))
labels_one_hot = np.concatenate((labels_wsi, labels_tma))

# Loading pre-trained ResNet50 model
# adding a final layer to change the number of output classes
base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(len(classes), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
# Not changing pre-trained layers
for layer in base_model.layers:
  layer.trainable = False
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Training model
model.fit(images, labels_one_hot, epochs=20, batch_size=64)
