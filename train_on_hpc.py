import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image
from pathlib import Path
import pandas
import time
from datetime import datetime

from augmentor import yolo_confidence_score, aug_side_px

from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model, save_model

# Paths
data_path = Path('./data/augmented/')
main_csv = './data/train.csv'
model_file_start = './models/model_'

classes = ('CC', 'EC', 'HGSC', 'LGSC', 'MC')

lr = 0.001

batch_n = 64
mega_batch_n = 4 * batch_n
save_on = 256 * batch_n

# To convert labels to one-hot vectors and vice-versa
def set_target(to, fro, cl=classes):
    labels = list(fro['label'])
    confs = list(fro['conf'])
    for i, label in enumerate(labels):
        to[i][cl.index(label)] = confs[i]


def load_master_data():
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


def load_images(out, filenames):
    # Last batch might feed some images the second time, whatever
    if isinstance(filenames, pandas.DataFrame):
        filenames = filenames['filename']
    for i, fn in enumerate(filenames):
        img = Image.open(fn)
        out[i] = np.array(img)
        img.close()


if __name__ == '__main__':
    print(f'{datetime.fromtimestamp(time.time())} '
          f'The eternally dormant beast has awaken to channel light and cause heat.')

    master_data = load_master_data()
    train_data = master_data.loc[master_data['split'] == 'train']
    val_data = master_data.loc[master_data['split'] == 'val']
    val_tma = val_data.loc[val_data['type'] == 'tma']
    val_wsi = val_data.loc[val_data['type'] == 'wsi']

    random.seed(1337)
    for obj in [train_data, val_tma, val_wsi]:
        order = list(obj.index)
        random.shuffle(order)
        obj.reindex(order)

    starting_from_scratch = True
    if starting_from_scratch:
        model = ResNet101(weights="imagenet", include_top=False, classes=5, input_shape=(512, 512, 3))
        for layer in model.layers:
            layer.trainable = True
        x = model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        #x = layers.Dropout(0.05)(x)
        predictions = layers.Dense(len(classes), activation='softmax')(x)
        model = Model(inputs=model.input, outputs=predictions)
        # TODO: integration hazard: check that it works with the notebook
    else:
        raise Exception('starting_from_scratch=False is not implemented, '
                        'please implement yourself, commit-push-pull, and restart the job.')
    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy', 'mse'])

    # Training model
    hot_images = np.zeros((mega_batch_n, aug_side_px, aug_side_px, 3))
    target = np.zeros((mega_batch_n, len(classes)))
    for cur_start_i in range(0, len(train_data), 256):
        print(f'{datetime.fromtimestamp(time.time())} Done: {cur_start_i}/{len(train_data)}')
        if cur_start_i % save_on == 0:
            save_model(model, model_file_start + str(cur_start_i))

        cur = train_data.loc[cur_start_i:cur_start_i+mega_batch_n]
        load_images(hot_images, cur)
        set_target(target, cur)
        model.fit(hot_images, target, epochs=1, batch_size=batch_n, validation_data=())
        print(f'{datetime.fromtimestamp(time.time())} Eval not implemented, please implement!')
        # TODO: eval. Keep in mind - we can not eval on the entire eval set, its too large!
        # TODO: Make sure to load reasonable ammount of things, for both TSI and non-TSI images
        # TODO: We also want eval for training set

    save_model(model, model_file_start + 'last')
    print(f'{datetime.fromtimestamp(time.time())} '
          f'The process as suffered a glorious death. Its job is complete.')
