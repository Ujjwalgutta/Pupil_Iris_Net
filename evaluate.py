import tensorflow as tf
from tensorflow import keras
from keras_segmentation.models.segnet import vgg_segnet
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def load_model():
    model_segnet1 = vgg_segnet(n_classes=3 ,  input_height=224, input_width=160)
    model_segnet1.load_weights("segnet_weights/vgg_seg_net.00011")

    return model_segnet1


# Extract test image paths
def extract_test_image_paths(test_img_dir, test_mask_dir):
    # Extract test image paths
    test_img_paths = sorted(
        [
            os.path.join(test_img_dir, fname)
            for fname in os.listdir(test_img_dir)
            if fname.endswith(".png")
        ]
    )

    test_mask_paths = sorted(
        [
            os.path.join(test_mask_dir, fname)
            for fname in os.listdir(test_mask_dir)
            if fname.endswith(".png")
        ]
    )

    return test_img_paths, test_mask_paths

# Inference, Ellipse Fitting
def predict_process_visualize(model_segnet1, test_img_paths):
    # Create lists to store images, masks and predicted pupil/iris parameters 
    pred_mask_list = []
    pupil_mask_list = []
    iris_mask_list = []
    overlayed_img_list = []
    iris_param_list = []
    pupil_param_list = []


    for i in range(len(test_img_paths)):
        pred_img = model_segnet1.predict_segmentation(
            inp=test_img_paths[i],
            out_fname=None
        )
        # Convert the image type for opencv processing
        pred_img = pred_img.astype(np.uint8)
        pred_img = cv2.resize(pred_img, (224,160), interpolation = cv2.INTER_CUBIC)
        pred_mask_list.append(pred_img)

        # Load the input image
        in_img = cv2.imread(test_img_paths[i])
        in_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB)

        # Create seperate Iris and Pupil Segmentation masks using predicted mask 
        iris_mask = np.zeros_like(pred_img)
        pupil_mask = np.zeros_like(pred_img)
        iris_ind = np.where(pred_img == 2)
        pupil_ind = np.where(pred_img == 1)
        iris_mask[iris_ind] = 2
        pupil_mask[pupil_ind] = 1
        kernel = np.ones((3,3),np.uint8)

        # Morphological Image Processing to eliminate any excessive pixels 
        pupil_mask_processed = cv2.morphologyEx(pupil_mask, cv2.MORPH_OPEN, kernel)
        pupil_mask_list.append(pupil_mask_processed)
        iris_mask_list.append(iris_mask)

        # Contour Detection Iris
        iris_cnts,hierarchy = cv2.findContours(iris_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Retain the contour with maximum area
        if len(iris_cnts) != 0:
            iris_c = max(iris_cnts, key = cv2.contourArea)
        # Fit ellipse using contour points
        ellipse_iris = cv2.fitEllipse(iris_c)
        # Calculate Iris Diameter and coordinates
        d_x_i,d_y_i = ellipse_iris[1]
        r_x_i,r_y_i = (d_x_i/2),(d_y_i/2)
        r_x_i = round(r_x_i,3)
        r_y_i = round(r_y_i,3)
        iris_param_list.append((r_x_i,r_y_i))

        # Contour Detection Pupil
        pupil_cnts,hierarchy = cv2.findContours(pupil_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Retain the contour with maximum area
        if len(pupil_cnts) != 0:
            pupil_c = max(pupil_cnts, key = cv2.contourArea)
        # Fit ellipse using contour points
        ellipse_pupil = cv2.fitEllipse(pupil_c)
        # Calculate Pupil Diameter and coordinates
        d_x_p,d_y_p = ellipse_pupil[1]
        r_x_p,r_y_p = (d_x_p/2),(d_y_p/2)
        r_x_p = round(r_x_p,3)
        r_y_p = round(r_y_p,3)
        pupil_param_list.append((r_x_p,r_y_p))

        # Overlay the predicted Ellipse on the input Image
        merg_img1 = cv2.ellipse(in_img, ellipse_iris, (255,0, 255), 1, cv2.LINE_AA)
        merg_img = cv2.ellipse(merg_img1, ellipse_pupil, (255,0, 255), 1, cv2.LINE_AA)
        overlayed_img_list.append(merg_img)
    
    # Visualize Results
    plt.figure()
    fig, axs = plt.subplots(10, 6, figsize=(30,30))
    for i in range(len(test_img_paths)):
        img = cv2.imread(test_img_paths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[i, 0].imshow(img)
        axs[i, 0].set_title('Input')
        img1 = cv2.imread(test_mask_paths[i],0)
        img1 = img1.astype(np.uint8)
        axs[i, 1].imshow(img1)
        axs[i, 1].set_title('Original Mask')
        axs[i, 2].imshow(pred_mask_list[i])
        axs[i, 2].set_title('Predicted Mask')
        axs[i, 3].imshow(iris_mask_list[i])
        axs[i, 3].set_title('Iris Mask')
        axs[i, 4].imshow(pupil_mask_list[i])
        axs[i, 4].set_title('Pupil Mask')
        axs[i, 5].imshow(overlayed_img_list[i])
        axs[i, 5].set_title('Overlayed')

    for ax in fig.get_axes():
        ax.label_outer()
    
    return iris_param_list, pupil_param_list


def calculate_mae(gt_dir, iris_param_list, pupil_param_list):
# gt_dir = "testing_set/groundtruth/"
    gt_paths = sorted(
        [
            os.path.join(gt_dir, fname)
            for fname in os.listdir(gt_dir)
            if fname.endswith(".csv")
        ]
    )
    print(gt_paths)


    gt_iris_list = []
    gt_pupil_list = []
    for i in range(len(gt_paths)):
        df = pd.read_csv(gt_paths[i])
        rad_x_i = df["radiusX_i_true"].values[0]
        rad_y_i = df["radiusY_i_true"].values[0]
        rad_x_p = df["radiusX_p_true"].values[0]
        rad_y_p = df["radiusY_p_true"].values[0]

        gt_iris_list.append((rad_x_i,rad_y_i))
        gt_pupil_list.append((rad_x_p,rad_y_p))

    error_x_i = 0
    error_y_i = 0
    error_x_p = 0
    error_y_p = 0
    for i in range(len(gt_iris_list)):
        error_x_i += (gt_iris_list[i][0] - iris_param_list[i][0])/gt_iris_list[i][0]
        error_y_i += (gt_iris_list[i][1] - iris_param_list[i][1])/gt_iris_list[i][1]
        error_x_p += (gt_pupil_list[i][0] - pupil_param_list[i][0])/gt_pupil_list[i][0]
        error_y_p += (gt_pupil_list[i][1] - pupil_param_list[i][1])/gt_pupil_list[i][1]

    mae_x_i = (abs(error_x_i)/len(gt_iris_list))*100
    mae_y_i = (abs(error_y_i)/len(gt_iris_list))*100
    mae_x_p = (abs(error_x_p)/len(gt_iris_list))*100
    mae_y_p = (abs(error_y_p)/len(gt_iris_list))*100
    return mae_x_i, mae_y_i, mae_x_p, mae_y_p, gt_iris_list, gt_pupil_list



if __name__ == "__main__":
    model_segnet1 = load_model()
    test_img_paths, test_mask_paths = extract_test_image_paths("data/test/images/", "data/test/masks/")
    iris_param_list, pupil_param_list = predict_process_visualize(model_segnet1, test_img_paths)
    mae_x_i, mae_y_i, mae_x_p, mae_y_p, gt_iris_list, gt_pupil_list = calculate_mae("data/test/groundtruth/", iris_param_list, pupil_param_list)

    # Print the metrics
    print("Mean absolute Percentage Error(IRIS)%:",mae_y_i)
    print("Mean absolute Percentage Error(PUPIL)%:",mae_x_p)


    # Display predicted and actual radii for test images
    data = pd.DataFrame({'Iris GroundTruth': gt_iris_list,
                        'Iris Predicted': iris_param_list,
                        'Pupil GroundTruth': gt_pupil_list,
                        'Pupil Predicted': pupil_param_list})

    data.head(10)