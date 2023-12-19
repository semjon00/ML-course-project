import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image
from pathlib import Path
import pandas
import time
from datetime import datetime

import tensorflow.keras.backend as K
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model, save_model

# Paths
data_path = Path('./data/augmented/')
main_csv = './data/train.csv'
model_file_start = './models/model_'

yolo_confidence_score = 0.2
aug_side_px = 512

classes = ('CC', 'EC', 'HGSC', 'LGSC', 'MC')

lr = 0.002

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
    for obj in [train_data, val_data, val_tma, val_wsi]:
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
    def custom_accuracy(true, pred):
        return K.mean(K.equal(K.argmax(true, axis=-1), K.argmax(pred, axis=-1)))

    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['mse', custom_accuracy])

    # Eval things
    # TODO: We want eval for training set, TSI and non-TSI
    val_images = np.zeros((mega_batch_n, aug_side_px, aug_side_px, 3))
    val_target = np.zeros((mega_batch_n, len(classes)))
    cur_eval = val_data.iloc[0:mega_batch_n]
    load_images(val_images, cur_eval)
    set_target(val_target, cur_eval)

    metrics = model.evaluate(val_images, val_target, batch_size=batch_n)

    # Training
    hot_images = np.zeros((mega_batch_n, aug_side_px, aug_side_px, 3))
    target = np.zeros((mega_batch_n, len(classes)))
    for cur_start_i in range(0, len(train_data), mega_batch_n):
        print(f'{datetime.fromtimestamp(time.time())} Done: {cur_start_i}/{len(train_data)}')
        if cur_start_i % save_on == 0:
            save_model(model, model_file_start + str(cur_start_i))
            print(f'{datetime.fromtimestamp(time.time())} Saved the snapshot of the model.')
            metrics = model.evaluate(val_images, val_target, batch_size=batch_n)
            print(f'{datetime.fromtimestamp(time.time())} Eval metrics: {metrics}')

        cur = train_data.iloc[cur_start_i:cur_start_i+mega_batch_n]
        load_images(hot_images, cur)
        set_target(target, cur)
        model.fit(hot_images, target, epochs=1, batch_size=batch_n, validation_data=(val_images, val_target))
        print(f'{datetime.fromtimestamp(time.time())} Eval not implemented, please implement!')

    save_model(model, model_file_start + 'last')
    print(f'{datetime.fromtimestamp(time.time())} '
          f'The process as suffered a glorious death. Its job is complete.')
