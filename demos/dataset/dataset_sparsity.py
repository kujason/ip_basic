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

    # Process depth images
    num_images = len(images_to_use)
    all_sparsities = np.zeros(num_images)

    for i in range(num_images):

        # Print progress
        sys.stdout.write('\rProcessing index {} / {}'.format(i, num_images - 1))
        sys.stdout.flush()

        depth_image_path = images_to_use[i]

        # Load depth from image
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)

        # Divide by 256
        depth_map = depth_image / 256.0

        num_valid_pixels = len(np.where(depth_map > 0.0)[0])
        num_pixels = depth_image.shape[0] * depth_image.shape[1]

        sparsity = num_valid_pixels / (num_pixels * 2/3)
        all_sparsities[i] = sparsity

    print('')
    print('Sparsity')
    print('Min:   ', np.amin(all_sparsities))
    print('Max:   ', np.amax(all_sparsities))
    print('Mean:  ', np.mean(all_sparsities))
    print('Median:  ', np.median(all_sparsities))

    plt.hist(all_sparsities, bins=20)
    plt.show()


if __name__ == '__main__':
    main()
