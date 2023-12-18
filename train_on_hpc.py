import random
import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import pandas
import time
import datetime

from augmentor import yolo_confidence_score

from tensorflow.keras.applications.resnet import ResNet101
from keras.models import Model
from keras.models import load_model, save_model
from keras.layers import Dense, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

# Paths
data_path = Path('./data/augmented/')
main_csv = './data/train.csv'
model_file = './models/model_last'

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


def loaddddd(root):
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
                image = image.resize((224, 224)) # Resizing to 244x244 for Resnet model
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


def load_data():
    # Eyeballed at 15Mb in total, acceptable

    img_filenames = list(Path(data_path).rglob("*.[pP][nN][gG]"))
    csv_data = pandas.read_csv(main_csv)

    input_to_csv_id = {}
    for index, row in csv_data.iterrows():
        input_to_csv_id[row['image_id']] = index

    ans = []

    for fn in img_filenames:
        fnp = fn.parts[len(data_path.parts):]
        input_name = fnp[-1].lstrip('crop_').rstrip('.png').split('_')
        txt = open(str(fn).strip('.png') + '.txt').read().split(' ')
        label = txt[0]
        conf = float(txt[1])
        is_masked = abs(conf - yolo_confidence_score) < 0.0001

        ans.append([int(input_name[0]), int(input_name[1]), label, conf, is_masked, fn, fnp[0], fnp[1]])
    return pandas.DataFrame(ans, columns=['image_id', 'crop_nr', 'label', 'conf', 'is_masked',
                                          'filename', 'type', 'split'])


if __name__ == '__main__':
    print(f'{datetime.utcfromtimestamp(time.time())} The eternaly dormant beast has awaken to channel light and feel heat.')

    master_data = load_data()
    train_data = master_data.loc[master_data['split'] == 'train']
    val_data = master_data.loc[master_data['split'] == 'val']

    random.seed(1337)
    for obj in [train_data, val_data]:
        random.shuffle(obj)

    # Loading pre-trained ResNet101 model
    # adding a final layer to change the number of output classes
    model = ResNet101(include_top=True, classes=5)
    # x = base_model.output
    # x = GlobalAveragePooling2D()(x)
    # predictions = Dense(len(classes), activation='softmax')(x)
    # model = Model(inputs=base_model.input, outputs=predictions)

    #save_model(model, model_file)
    #model = load_model(model_file)
    #print(1)

    # Not changing pre-trained layers
    # for layer in base_model.layers:
    #     layer.trainable = False
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # # Training model
    # model.fit(images, labels_one_hot, epochs=1, batch_size=64, validation_data=(images_val, labels_one_hot_val))
    #
    # # Predicting validation set labels
    # predicted = one_hot_to_values(model.predict(images_val), classes)
