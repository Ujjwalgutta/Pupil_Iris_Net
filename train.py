import numpy as np
import pandas as pd
import os
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from keras_segmentation.models.segnet import vgg_segnet
from imgaug import augmenters as iaa
import argparse

def main(input_img_dir, input_mask_dir, test_img_dir, test_mask_dir, checkpoints_path, epochs, batch_size):

    input_img_paths = sorted(
        [
            os.path.join(input_img_dir, fname)
            for fname in os.listdir(input_img_dir)
            if fname.endswith(".png")
        ]
    )
    target_img_paths = sorted(
        [
            os.path.join(input_mask_dir, fname)
            for fname in os.listdir(input_mask_dir)
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )

    # verify the size of training images and masks
    input_img_paths = input_img_paths[:len(target_img_paths)]

    # Print the size of training set
    print("Number of samples:", len(input_img_paths))

    # Data Augmentation using Horizontal Flipping and Gaussian Blurring using a kernel between sized 0 and 3
    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontally flip 50% of the images
        iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
    ])

    # Define VGG-SegNet model (Encoder-Decoder) using pretrained ImageNet Weights
    def custom_augmentation():
        return  iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
            ])

    # Customize the model by defining image size and number of output classes.
    model_segnet = vgg_segnet(n_classes=3 ,  input_height=224, input_width=160)

    model_segnet.train(
        train_images =  input_img_dir,
        train_annotations = input_mask_dir,
        checkpoints_path = checkpoints_path, epochs=epochs,
        batch_size = batch_size,
        validate = True,
        val_images = test_img_dir,
        val_annotations = test_mask_dir,
        do_augment=True, # enable augmentation 
        custom_augmentation=custom_augmentation # sets the augmention function to use
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SegNet')
    parser.add_argument('--input_img_dir', default='data/train/images/')
    parser.add_argument('--input_mask_dir', default='data/train/masks/')
    parser.add_argument('--test_img_dir', default='data/test/images/')
    parser.add_argument('--test_mask_dir', default='data/test/masks/')
    parser.add_argument('--checkpoints_path', default="checkpoints/vgg_seg_net")
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    main(args.input_img_dir, args.input_mask_dir, args.test_img_dir, args.test_mask_dir, args.checkpoints_path, args.epochs, args.batch_size)
