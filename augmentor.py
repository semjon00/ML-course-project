import os
import random
import math
from pathlib import Path
import pandas
import PIL.Image
import PIL
import numpy
PIL.Image.MAX_IMAGE_PIXELS = None  # Otherwise it thinks that the images are oversized and may be compression bombs

# TODO: Soft target - crops should also contain marked targets.

tma_crops = 2000  # Per image, there are 25 TMA images in the training set
non_tma_crops = 300  # Per image, there are 513 non-TMA images

boring_cutoff = 0.3  # pictures with bigger proportion of boring pixels will be discarded
borders_expansion = 0.1  # Expand picture by how much on the sides? Useful for better representing borders
# TODO: We should totally account for different modes (20x vs 40x)
aug_side_proportion = 0.1  # Crop lenght as a fraction of longest side of input
aug_side_proportion_var = 0.1  # Crop length variation
aug_side_px = 786  # Crop will be downscaled to size (aug_side_px, aug_side_px)
do_random_flip = True  # Whether to allow augmenter to flip cuts.

# From-path
source_path = Path('data') / 'source'
mask_path = Path('data') / 'masks'
to_path = Path('data') / 'augmented'
main_csv = Path('data') / 'train.csv'

def augmenting_generator(picture: PIL.Image, n_crops: int, tumor_class=None, mask=None):
    max_side = max(picture.size)

    def p_to_px(p):
        return int(p * max_side)

    def pre_crop_size(angle, size):
        pc_size = angle % (math.tau / 4)
        pc_size = min(pc_size, math.tau / 4 - pc_size)
        pc_size = size * (math.sin(pc_size) + math.cos(pc_size))
        return pc_size

    succesful = 0
    total = 0
    while succesful < n_crops:
        # Failsafe: one black image shall never just indefinitely hang the entire code
        if total > 3 * n_crops:
            break

        angle = random.uniform(0, math.tau)
        size = aug_side_proportion * (1.0 + random.uniform(-aug_side_proportion_var, aug_side_proportion_var))
        pc_size = pre_crop_size(angle, size)

        chosen = []
        for side_prop in [x / max_side for x in picture.size]:
            chosen.append(random.uniform(-borders_expansion, side_prop + borders_expansion - pc_size))
        pre_crop = picture.crop((p_to_px(chosen[0]), p_to_px(chosen[1]),
                                 p_to_px(chosen[0]) + p_to_px(pc_size), p_to_px(chosen[1]) + p_to_px(pc_size)))

        pre_crop = pre_crop.rotate(angle / math.tau * 360.0)
        rot_off = (pre_crop.size[0] - p_to_px(size)) // 2
        pre_crop = pre_crop.crop((rot_off, rot_off, pre_crop.size[0] - rot_off, pre_crop.size[1] - rot_off))

        crop = pre_crop.resize((aug_side_px, aug_side_px), PIL.Image.Resampling.LANCZOS)
        if do_random_flip and random.randint(0, 1) == 1:
            crop = crop.transpose(PIL.Image.FLIP_TOP_BOTTOM)

        # Ouch, I don't like this code. Looks not performant at all.
        total += 1
        boring = numpy.sum(numpy.sum(numpy.array(crop), axis=2) == 0) / aug_side_px / aug_side_px
        if boring > boring_cutoff:
            continue

        succesful += 1
        yield crop, 1.0


if __name__ == '__main__':
    csv_data = pandas.read_csv(main_csv)
    source_files = list(Path(source_path).rglob("*.[pP][nN][gG]"))
    mask_files = os.listdir(mask_path)
    to_path.mkdir(parents=True, exist_ok=True)

    for pat in source_files:
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
        for crop, confidence in augmenting_generator(picture=img, n_crops=20, tumor_class=tumor_class, mask=mask):
            save_to = to_path / to_path_suffix / Path(*pat.parts[2:-1])
            save_to.mkdir(parents=True, exist_ok=True)
            fn = f'crop_{str(p).split(".")[0]}_{i}'
            crop.save(save_to / str(fn + '.png'))
            open(save_to / str(fn + '.txt'), 'w').write(f'{tumor_class} {confidence}')
            i += 1
        img.close()
        if mask is not None:
            mask.close()
