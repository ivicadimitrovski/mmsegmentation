# Copyright (c) OpenMMLab. All rights reserved.
import os
import rasterio
import numpy as np

from PIL import Image


def main():
    images_path = './flair_aerial_train'
    labels_path = './flair_labels_train'
    out_dir_images = './img_dir/'
    out_dir_labels = './ann_dir/'

    train_subfolders = ['D006', 'D007', 'D008', 'D009', 'D013', 'D016',
                        'D017', 'D021', 'D023', 'D030', 'D032', 'D033',
                        'D034', 'D035', 'D038', 'D041', 'D044', 'D046',
                        'D049', 'D051', 'D052', 'D055', 'D060', 'D063',
                        'D070', 'D072', 'D074', 'D078', 'D080', 'D081',
                        'D086', 'D091']
    val_subfolders = ['D004', 'D014', 'D029', 'D031', 'D058', 'D066',
                      'D067', 'D077']

    for path, _, files in os.walk(images_path):
        for name in files:
            tmp = path.split('/')
            print(tmp[-2], tmp[-3], path, name)

            # read image
            full_name = os.path.join(path, name)
            src_img = rasterio.open(full_name)
            array = src_img.read()
            array = array[0:3, :, :]
            array = np.transpose(array, (1, 2, 0))

            # read mask
            full_name_mask = os.path.join(labels_path, tmp[-3], tmp[-2], 'msk', name.replace('IMG', 'MSK'))
            src_mask = rasterio.open(full_name_mask)
            array_mask = src_mask.read()
            array_mask = np.transpose(array_mask, (1, 2, 0))
            array_mask = array_mask - 1
            array_mask[array_mask >= 12] = 255
            array_mask = array_mask.reshape(512, 512)
            #print(array_mask.shape)
            #print(np.unique(array_mask))

            if tmp[-3].split('_')[0] in train_subfolders:
                new_file_name = os.path.join(out_dir_images, 'train', tmp[-3] + "_" + tmp[-2] + "_" + name.replace('.tif', '.png'))
                new_file_name_mask = os.path.join(out_dir_labels, 'train', tmp[-3] + "_" + tmp[-2] + "_" + name.replace('.tif', '.png'))
            elif tmp[-3].split('_')[0] in val_subfolders:
                new_file_name = os.path.join(out_dir_images, 'val', tmp[-3] + "_" + tmp[-2] + "_" + name.replace('.tif', '.png'))
                new_file_name_mask = os.path.join(out_dir_labels, 'val', tmp[-3] + "_" + tmp[-2] + "_" + name.replace('.tif', '.png'))
            else:
                new_file_name = os.path.join(out_dir_images, 'test', tmp[-3] + "_" + tmp[-2] + "_" + name.replace('.tif', '.png'))
                new_file_name_mask = os.path.join(out_dir_labels, 'test', tmp[-3] + "_" + tmp[-2] + "_" + name.replace('.tif', '.png'))

            #im = Image.fromarray(array)
            #im.save(new_file_name)

            im = Image.fromarray(array_mask)
            im.save(new_file_name_mask)

    print('Done!')


if __name__ == '__main__':
    main()
