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

    for i in range(vol.shape[0]):
        fname = os.path.join(path, "slice" + str(i).zfill(5) + ".tiff")
        cv2.imwrite(fname, vol[i])

def read_tiff_stack(path):
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        slice = np.array(img)

        images.append(slice)

    return np.array(images)


def get_dir(path):
    tiffs = [os.path.join(path, f) for f in os.listdir(path) if f[0] != "."]

    return sorted(tiffs)

dim_offset = 14
file_name = "trailmap" # visualize/gauss_best_val_2_1 seg-_overlap_2.0_trailmap_model.hdf5_gauss_False_testing-original_test_2
# file_name = "trailmap_gauss" 
# file_name = "rot_gauss"
# file_name = "rot"
slice_num = 74
# pred starts at dim_offset in
pred_num = slice_num + dim_offset
input_folder = "visualize"


# get true labels
vol = read_tiff_stack("visualize/cube1_label.tif")
write_folder_stack(vol, os.path.join(input_folder, f"slices_truth_label"))

# get predicted volumes (already should be in 160 by 160 by 160)
pred_image_array = np.array(
    Image.open(
        f"visualize/{file_name}/seg-slice000{pred_num}.tiff"
    )
)
print(pred_image_array.shape)

# /data/model-weights/best_weights_checkpoint_all_changes_2X_20_aug_rot_normal_False.hdf5
truth_array = np.array(
    Image.open(f"visualize/slices_truth_label/slice000{slice_num}.tiff")
)

print(truth_array.shape)


print(np.unique(truth_array, return_counts=True))

imgray_pred = np.empty((truth_array.shape[0], truth_array.shape[1], 3))
tp_locs = np.where((pred_image_array >= 0.5) & (truth_array == 2))
print("true postives")
print(len(tp_locs[0]))
imgray_pred[tp_locs[0], tp_locs[1], :] = (0, 0, 255)  # blue
fn_locs = np.where((pred_image_array < 0.5) & (truth_array == 2))
print("falsenegatives")
print(len(fn_locs[0]))
imgray_pred[fn_locs[0], fn_locs[1], :] = (255, 102, 0) # orange false negative
fp_locs = np.where((pred_image_array >= 0.5) & (truth_array != 2))
print("false pos")
print(len(fp_locs[0]))

print("recall")
print(len(tp_locs[0])/(len(tp_locs[0]) + len(fn_locs[0]) ))
print("precision")
print(len(tp_locs[0])/(len(tp_locs[0]) + len(fp_locs[0])))
# false positive
imgray_pred[fp_locs[0], fp_locs[1], :] = (255, 0, 0)  # red false positive

edge_locs = np.where(truth_array == 4)
imgray_pred[edge_locs[0], edge_locs[1], :] = (255,255,255)  # white edges


im = Image.fromarray(imgray_pred.astype(np.uint8))
im.save(
    f"visualize/{file_name}_fp_fn.tif"
)
