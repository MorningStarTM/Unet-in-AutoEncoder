import cv2
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
import random
from glob import glob
import os
import matplotlib.pyplot as plt

WIDTH, HEIGHT = 256, 256
BATCH_SIZE = 16
lr = 1e-04
epochs = 100
image_size = [256,256]


def read_data(path):
    images = sorted(glob(os.path.join(path, "*", "*", "*")))
    
    return images, images

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, img_filenames,
                 mask_filenames,
                 batch_size=BATCH_SIZE,
                 shuffle=True):

        self.img_filenames = img_filenames
        self.mask_filenames = mask_filenames
        self.filenames = list(zip(img_filenames, mask_filenames))
        self.batch_size = batch_size
        self.shuffle= shuffle
        self.n = len(self.img_filenames)

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.filenames)

    def __get_data(self, batches):
        imgs=[]
        segs=[]
        for img_file, mask_file in batches:
            image = cv2.imread(img_file)
            image = cv2.resize(image, (WIDTH, HEIGHT))
            image = image / 255.
            
            mask = cv2.imread(mask_file)
            mask = cv2.resize(mask, (WIDTH, HEIGHT))#[:, :, 2]
            mask = mask / 255.

            imgs.append(image)
            segs.append(mask)

        return np.array(imgs), np.array(segs)
    
    def __getitem__(self, index):

        batches = self.filenames[index * self.batch_size:(index + 1) * self.batch_size]

        X, y = self.__get_data(batches)

        return (X, y)

    def __len__(self):

        return self.n // self.batch_size





def show_images_with_masks(image_dir, mask_dir):
    #image_files = sorted(os.listdir(image_dir))
    #mask_files = sorted(os.listdir(mask_dir))

    # Ensure equal number of images and masks
    num_files = min(len(image_dir), len(image_dir))

    # Display 5 images and masks
    num_display = min(num_files, 5)
    fig, axs = plt.subplots(num_display, 2, figsize=(10, 10))

    for i in range(num_display):

        image = cv2.imread(image_dir[i])
        mask = cv2.imread(mask_dir[i])#[:,:,2]
        

        # Display image
        axs[i, 0].imshow(image)
        axs[i, 0].axis('off')
        axs[i, 0].set_title('Image')

        # Display mask
        axs[i, 1].imshow(mask)
        axs[i, 1].axis('off')
        axs[i, 1].set_title('Mask')

    plt.tight_layout()
    plt.show()

