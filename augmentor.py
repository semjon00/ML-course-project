import os
import random
import math
from pathlib import Path
import pandas
import PIL.Image
import PIL
import numpy
import time
from datetime import datetime
import traceback
PIL.Image.MAX_IMAGE_PIXELS = None  # Otherwise it thinks that the images are oversized and may be compression bombs

yolo_confidence_score = 0.2

tma_crops = 1200  # Per image, there are 25 TMA images in the training set
non_tma_crops = 300  # Per image, there are 513 non-TMA (wsi) images

boring_cutoff = 0.45  # pictures with bigger proportion of boring pixels will be discarded
borders_expansion = 0.1  # Expand picture by how much on the sides? Useful for better representing borders
aug_side_proportion = 0.075  # Crop length as a fraction of longest side of input, double for TMA
aug_side_proportion_var = 0.2  # Crop length variation
aug_side_px = 512  # Crop will be downscaled to size (aug_side_px, aug_side_px)
do_random_flip = True  # Whether to allow augmenter to flip cuts.

# From-path
source_path = Path('data') / 'source'
mask_path = Path('data') / 'masks'
to_path = Path('data') / 'augmented'
main_csv = Path('data') / 'train.csv'

def augmenting_generator(picture: PIL.Image, n_crops: int, mask=None, it='unsure'):
    max_side = max(picture.size)

    def p_to_px(p):
        return int(p * max_side)

    def pre_crop_size(angle, size):
        pc_size = angle % (math.tau / 4)
        pc_size = min(pc_size, math.tau / 4 - pc_size)
        pc_size = size * (math.sin(pc_size) + math.cos(pc_size))
        return pc_size

    def make_crop(picture, chosen, pc_size):
        pre_crop = picture.crop((p_to_px(chosen[0]), p_to_px(chosen[1]),
                                 p_to_px(chosen[0]) + p_to_px(pc_size), p_to_px(chosen[1]) + p_to_px(pc_size)))
        pre_crop = pre_crop.rotate(angle / math.tau * 360.0)
        rot_off = (pre_crop.size[0] - p_to_px(size)) // 2
        return pre_crop.crop((rot_off, rot_off, pre_crop.size[0] - rot_off, pre_crop.size[1] - rot_off))

    succesful = 0
    total = 0
    while succesful < n_crops:
        # Failsafe: one black image shall never just indefinitely hang the entire code
        if total > 2 * n_crops:
            break

        angle = random.uniform(0, math.tau)
        size = aug_side_proportion * (1.0 + random.uniform(-aug_side_proportion_var, aug_side_proportion_var))
        if it == 'tma' or it == 'unsure' and random.random() > 0.5:
            size *= 2.0
        pc_size = pre_crop_size(angle, size)

        chosen = []
        for side_prop in [x / max_side for x in picture.size]:
            chosen.append(random.uniform(-borders_expansion, side_prop + borders_expansion - pc_size))

        crop = make_crop(picture, chosen, pc_size).resize((aug_side_px, aug_side_px), PIL.Image.Resampling.LANCZOS)
        if do_random_flip and random.randint(0, 1) == 1:
            crop = crop.transpose(PIL.Image.FLIP_TOP_BOTTOM)

        # Ouch, I don't like this code. Looks not performant at all.
        total += 1
        boring = numpy.sum(numpy.sum(numpy.array(crop), axis=2) == 0) / aug_side_px / aug_side_px
        if boring > boring_cutoff:
            continue

        # Base confidence - no idea what part of the image is cancer
        confidence = yolo_confidence_score  # YOLO
        if mask:
            mask_crop = make_crop(mask, chosen, pc_size)
            confidence = numpy.average(numpy.array(mask_crop)[:, :, 0]) / 255
        succesful += 1
        yield crop, confidence


def train_val_split(csv_data):
    print('Causing train-val split!')

    pictures = os.listdir(source_path)
    print(f'For {len(pictures)} pictures... {", ".join(pictures)}')
    tma = [p for p in pictures if any([csv_data.loc[csv_data['image_id'] == int(p.split('.')[0])] for p in pictures][0]['is_tma'])]
    non_tma = [p for p in pictures if p not in tma]

    (source_path / 'train').mkdir(parents=True, exist_ok=True)
    (source_path / 'val').mkdir(parents=True, exist_ok=True)

    def distribute(ptype):
        random.shuffle(ptype)
        for p in ptype[:int(len(ptype) * 0.83 + 0.99)]:
            os.rename(source_path / p, source_path / 'train' / p)
        for p in ptype[int(len(ptype) * 0.83 + 0.99):]:
            os.rename(source_path / p, source_path / 'val' / p)
    distribute(tma)
    distribute(non_tma)
    print('Split done!')


if __name__ == '__main__':
    done_crops = 0
    processed_images = 0
    last_reported = time.time()
    print(f'Cropper-chopper 2000 launched at {datetime.fromtimestamp(last_reported)}')
    print(f'tma_crops={tma_crops}, non_tma_crops={non_tma_crops}, '
          f'aug_side_px={aug_side_px}, aug_side_proportion={aug_side_proportion}')

    csv_data = pandas.read_csv(main_csv)
    if 'train' not in os.listdir(source_path):
        # Not yet divided
        train_val_split(csv_data)

    source_files = list(Path(source_path).rglob("*.[pP][nN][gG]"))
    print(f'Source files: {len(source_files)} in total')
    mask_files = os.listdir(mask_path)
    print(f'Mask files: {len(mask_files)} in total')
    to_path.mkdir(parents=True, exist_ok=True)

    for pat in source_files:
        try:
            p = pat.parts[-1]
            info = csv_data.loc[csv_data['image_id'] == int(p.split('.')[0])]
            tumor_class = list(info['label'])[0]
            is_tma = list(info['is_tma'])[0]
            n_crops = tma_crops if is_tma else non_tma_crops
            to_path_suffix = 'tma' if is_tma else 'wsi'

            img = PIL.Image.open(pat)
            mask: PIL.Image = None
            if p in mask_files:
                mask = PIL.Image.open(mask_path / p)
            i = 0
            for crop, confidence in augmenting_generator(picture=img, n_crops=20, mask=mask, it=to_path_suffix):
                save_to = to_path / to_path_suffix / Path(*pat.parts[2:-1])
                save_to.mkdir(parents=True, exist_ok=True)
                fn = f'crop_{str(p).split(".")[0]}_{i}'
                crop.save(save_to / str(fn + '.png'))
                crop.close()
                open(save_to / str(fn + '.txt'), 'w').write(f'{tumor_class} {confidence}')
                i += 1
                done_crops += 1
            img.close()
            if mask is not None:
                mask.close()
            processed_images += 1
        except Exception as e:
            print(f'Oh, no! Something broke!!! {traceback.format_exc()}')
        if time.time() > last_reported + 60:
            last_reported = time.time()
            print(f'{datetime.utcfromtimestamp(last_reported)}  Crops: {done_crops}, images: {processed_images}')
    print(f'{datetime.utcfromtimestamp(time.time())}  Crops: {done_crops}, images: {processed_images}')
    print(f'Cropper-chopper 2000 gracefully stopped at {datetime.fromtimestamp(time.time())}')
