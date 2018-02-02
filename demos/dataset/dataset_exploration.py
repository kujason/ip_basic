import glob
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np


def main():

    input_depth_dir = os.path.expanduser(
        '~/Kitti/depth/val_selection_cropped/velodyne_raw')
    images_to_use = sorted(glob.glob(input_depth_dir + '/*'))

    npy_file_name = 'outputs/all_depths.npy'
    all_depths_saved = os.path.exists(npy_file_name)

    if not all_depths_saved:
        os.makedirs('./outputs', exist_ok=True)

        # Process depth images
        all_depths = np.array([])
        num_images = len(images_to_use)
        for i in range(num_images):

            # Print progress
            sys.stdout.write('\rProcessing {} / {}'.format(i, num_images - 1))
            sys.stdout.flush()

            depth_image_path = images_to_use[i]

            # Load depth from image
            depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)

            # Divide by 256
            depth_map = depth_image / 256.0

            valid_pixels = (depth_map > 0.0)
            all_depths = np.concatenate([all_depths, depth_map[valid_pixels]])

        np.save(npy_file_name, all_depths)

    else:
        # Load from npy
        all_depths = np.load(npy_file_name)

    print('')
    print('Depths')
    print('Min:   ', np.amin(all_depths))
    print('Max:   ', np.amax(all_depths))
    print('Mean:  ', np.mean(all_depths))
    print('Median:  ', np.median(all_depths))

    plt.hist(all_depths, bins=80)
    plt.show()


if __name__ == '__main__':
    main()
