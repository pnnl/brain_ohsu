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
    # dim_offset = 14
    # for i in range(vol.shape[0]):
    #     if i > 13 and i < vol.shape[0] -13:
    #         fname = os.path.join(path, "slice" + str(i).zfill(5) + ".tiff")
    #         cv2.imwrite(fname, vol[i][dim_offset:-dim_offset,dim_offset:-dim_offset])


# def smaller_folder_stack(output_folder, path):
#     if os.path.exists(path):
#         print("Overwriting " + path)
#         shutil.rmtree(path)

#     makedirs(path)

#     fnames = get_dir(output_folder)
#     vol = []
#     for i in range(len(fnames)):
#             if fnames[i][-4:] == "tiff":
#                 img = cv2.imread(fnames[i], cv2.COLOR_BGR2GRAY)
#                 vol.append(img)
#     y_pred = np.array(vol)
#     # prediction has dim off set padding on x and y. only copied files from dim_off + on the z axis, so that's already alright
#     y_pred = y_pred  #[:,dim_offset:-dim_offset,dim_offset:-dim_offset]
#     for i in range(y_pred.shape[0]):
#         fname = os.path.join(path, "slice" + str(i).zfill(5) + ".tiff")
#         cv2.imwrite(fname, vol[i])
#     return



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
file_name = "seg-_overlap_2.0_trailmap_model.hdf5_gauss_True_testing-original_test_1" # visualize/gauss_best_val_2_1 seg-_overlap_2.0_trailmap_model.hdf5_gauss_False_testing-original_test_2
# file_name = "normal_test_1" 
# file_name = "gauss_test_1"
#file_name = "seg-_overlap_2.0_best_weights_checkpoint_oversample_bol_True_aug_bol_True_lr_bol_True_flip_bol_False_el_1.0_rot_1.0__encode_last_layer_loss_0.0001_training__val_2_test_1__july23.hdf5_gauss_True_testing-original_test_1" 
slice_num = 74
# pred starts at dim_offset in
pred_num = slice_num + dim_offset
input_folder = "visualize"


# get true labels
vol = read_tiff_stack("visualize/cube1_labels.tiff")
write_folder_stack(vol, os.path.join(input_folder, f"slices_truth_labels"))

# smaller_folder_stack(f"brain_ohsu/data/testing/{file_name}", os.path.join(input_folder, f"slices_pred_labels"))

# get true volumes
vol = read_tiff_stack("visualize/cube1.tif")
write_folder_stack(vol, os.path.join(input_folder, f"slices_truth_volumes"))


# get predicted volumes (already should be in 160 by 160 by 160)
pred_image_array = np.array(
    Image.open(
        f"visualize/{file_name}/seg-slice000{pred_num}.tiff"
    )
)
print(pred_image_array.shape)

# /data/model-weights/best_weights_checkpoint_all_changes_2X_20_aug_rot_normal_False.hdf5
truth_array = np.array(
    Image.open(f"visualize/slices_truth_labels/slice000{slice_num}.tiff")
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

# compare_image =  np.array(
#     Image.open(
#         f"brain_ohsu/data/testing/{file_name2}/seg-slice00088.tiff"
#     )
# )

# compare_image= compare_image[dim_offset:-dim_offset,dim_offset:-dim_offset]
# imgray_pred = np.empty((truth_array.shape[0], truth_array.shape[1], 3))
# tp_locs = np.where((pred_image_array >= 0.5) & (compare_image >= 0.5))
# print("true postives")
# print(len(tp_locs[0]))
# imgray_pred[tp_locs[0], tp_locs[1], :] = (0, 0, 255)  # blue
# fn_locs = np.where((pred_image_array < 0.5) & (compare_image >= 0.5))
# print("falsenegatives")
# print(len(fn_locs[0]))
# imgray_pred[fn_locs[0], fn_locs[1], :] = (255, 255, 0)  # yellow false negative
# fp_locs = np.where((pred_image_array >= 0.5) & (compare_image < 0.5))
# print("false pos")
# print(len(fp_locs[0]))
# # false positive
# imgray_pred[fp_locs[0], fp_locs[1], :] = (255, 0, 0)  # red false positive



# im = Image.fromarray(imgray_pred.astype(np.uint8))
# im.save(
#     "compare.tif"
# )
