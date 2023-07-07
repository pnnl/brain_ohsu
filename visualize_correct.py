import os
import numpy as np
import cv2
import os
from os import listdir, makedirs
from os.path import join
from PIL import Image
import shutil
import sys
import logging
dim_offset = 14

def write_folder_stack(vol, path):
    if os.path.exists(path):
        print("Overwriting " + path)
        shutil.rmtree(path)

    makedirs(path)


    dim_offset = 14
    for i in range(vol.shape[0]):
        if i > 13 and i < 146:
            fname = os.path.join(path, "slice" + str(i).zfill(5) + ".tiff")
            cv2.imwrite(fname, vol[i][dim_offset:-dim_offset,dim_offset:-dim_offset])


def read_tiff_stack(path):
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        slice = np.array(img)

        images.append(slice)

    return np.array(images)


def smaller_folder_stack(output_folder, path):
    if os.path.exists(path):
        print("Overwriting " + path)
        shutil.rmtree(path)

    makedirs(path)

    fnames = get_dir(output_folder)
    vol = []
    for i in range(len(fnames)):
            img = cv2.imread(fnames[i], cv2.COLOR_BGR2GRAY)
            vol.append(img)
    y_pred = np.array(vol)
    # prediction has dim off set padding on x and y. only copied files from dim_off + on the z axis, so that's already alright
    y_pred = y_pred[:,dim_offset:-dim_offset,dim_offset:-dim_offset]
    for i in range(y_pred.shape[0]):
        fname = os.path.join(path, "slice" + str(i+14).zfill(5) + ".tiff")
        cv2.imwrite(fname, vol[i])
    return



def get_dir(path):
    tiffs = [os.path.join(path, f) for f in os.listdir(path) if f[0] != "."]

    return sorted(tiffs)


# 74 for testing
#
input_folder = "data/testing/testing-original/volumes"
file_names = get_dir(
    "/Users/oost464/Library/CloudStorage/OneDrive-PNNL/Desktop/projects/brain_ohsu/data/testing/testing-original/volumes/"
)
vol = read_tiff_stack(file_names[0])
write_folder_stack(vol, os.path.join(input_folder, f"slices_truth_volumes"))

input_folder = "data/testing/testing-original/labels"
file_names = get_dir(
    "/Users/oost464/Library/CloudStorage/OneDrive-PNNL/Desktop/projects/brain_ohsu/data/testing/testing-original/labels/"
)
vol = read_tiff_stack(file_names[0])
write_folder_stack(vol, os.path.join(input_folder, f"slices_truth_labels"))

smaller_folder_stack("data/testing/seg-_overlap_1.0_best_weights_checkpoint_all_changes_2X_20_aug_rot_normal_False.hdf5_guass_False_testing-original", os.path.join(input_folder, f"slices_pred_labels"))

pred_image_array = np.array(
    Image.open(
        "data/testing/seg-_overlap_1.0_best_weights_checkpoint_all_changes_2X_20_aug_rot_normal_False.hdf5_guass_False_testing-original/seg-slice00074.tiff"
    )
)
# /data/model-weights/best_weights_checkpoint_all_changes_2X_20_aug_rot_normal_False.hdf5
print(pred_image_array[50,50])
truth_array = np.array(
    Image.open("data/testing/testing-original/labels/slices_truth_labels/slice00074.tiff")
)
dim_offset = 14
#truth_array = truth_array[dim_offset:-dim_offset,dim_offset:-dim_offset]
pred_image_array= pred_image_array[dim_offset:-dim_offset,dim_offset:-dim_offset]
print(np.unique(truth_array, return_counts=True))

imgray_pred = np.empty((truth_array.shape[0], truth_array.shape[1], 3))
tp_locs = np.where((pred_image_array >= 0.5) & (truth_array == 2))
print(len(tp_locs[0]))
imgray_pred[tp_locs[0], tp_locs[1], :] = (0, 0, 255)  # blue
tn_locs = np.where((pred_image_array < 0.5) & (truth_array == 2))
print(len(tn_locs[0]))
imgray_pred[tn_locs[0], tn_locs[1], :] = (255, 255, 0)  # yellow false negative
fn_locs = np.where((pred_image_array >= 0.5) & (truth_array != 2))
print(len(fn_locs[0]))
# false positive
imgray_pred[fn_locs[0], fn_locs[1], :] = (255, 0, 0)  # red false


im = Image.fromarray(imgray_pred.astype(np.uint8))
im.save(
    "seg-_overlap_1.2_best_weights_checkpoint_all_changes_2X_20_aug_rot_normal_False.hdf5_guass_False_testing-original_fp_fn.tif"
)
