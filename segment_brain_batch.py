from inference import *
from models import *
import sys
import os
import shutil
import itertools

if __name__ == "__main__":


    base_path = os.path.abspath(__file__ + "/..")
        # Load the network
    model_weight_list = [
    
  
    #all changes variations, .5 background, 200 images per tif for all , nearest means using nearest for interpolation instead of 0 for labels
    '/data/model-weights/best_weights_checkpoint_all_changes_2X_20_aug_flip_normal_False.hdf5',
    '/data/model-weights/best_weights_checkpoint_all_changes_2X_20_aug_flip_rot_normal_False.hdf5',
    '/data/model-weights/best_weights_checkpoint_all_changes_2X_20_aug_nearest_normal_False.hdf5',
    '/data/model-weights/best_weights_checkpoint_all_changes_2X_20_aug_normal_False.hdf5',
    '/data/model-weights/best_weights_checkpoint_all_changes_2X_20_aug_rot_nearest_normal_False.hdf5',
    '/data/model-weights/best_weights_checkpoint_all_changes_2X_20_aug_rot_normal_False.hdf5', # best model 20 rot, no flip constant

    '/data/model-weights/best_weights_checkpoint_all_changes_2X_50_aug_flip_normal_False.hdf5', 
    '/data/model-weights/best_weights_checkpoint_all_changes_2X_50_aug_flip_rot_normal_False.hdf5',
    '/data/model-weights/best_weights_checkpoint_all_changes_2X_50_aug_nearest_normal_False.hdf5',
    '/data/model-weights/best_weights_checkpoint_all_changes_2X_50_aug_normal_False.hdf5',
    '/data/model-weights/best_weights_checkpoint_all_changes_2X_50_aug_rot_nearest_normal_False.hdf5',
    '/data/model-weights/best_weights_checkpoint_all_changes_2X_50_aug_rot_normal_False.hdf5',


    #only some changes at a time, .5 background for all

    '/data/model-weights/best_weights_checkpoint_orig_model_flip_1X_aug_200_50_percent_only_normal_False.hdf5', # includes rot, 200 images, not actually flipping
    '/data/model-weights/best_weights_checkpoint_orig_model_flip_1X_aug_200_only_normal_False.hdf5', # includes rot, 200 images, not actually flipping, 30
    '/data/model-weights/best_weights_checkpoint_aug_50_only_normal_False.hdf5', # includes rot
    '/data/model-weights/best_weights_checkpoint_aug_20_only_normal_False.hdf5', # includes rot
    '/data/model-weights/best_weights_checkpoint_lr_reduce_200_no_flip_normal_False.hdf5',
    '/data/model-weights/best_weights_checkpoint_oversample_200_no_flip_normal_False.hdf5',
      

    #best trailmap adjustments


    
    '/data/model-weights/best_weights_checkpoint_orig_model_200_preprocess_after_aug_normal_normal_True.hdf5', # not really aug
    '/data/model-weights/best_weights_checkpoint_orig_model_200_aug_background_decreased_normal_True.hdf5', # not really aug
    '/data/model-weights/best_weights_checkpoint_flip_orig_model_normal_True.hdf5',
    '/data/model-weights/best_weights_checkpoint_orig_model_flipped_background_decreased_normal_True.hdf5',
    '/data/model-weights/best_weights_checkpoint_orig_model_normal_True.hdf5',
    '/data/model-weights/best_weights_checkpoint_orig_model_background_decreased_normal_True.hdf5',
    "/data/model-weights/trailmap_model.hdf5",

    ]

    image_path  =  [base_path + f"/data/validation/validation-original",  base_path + f"/data/testing/testing-original"]
    combos = list(itertools.product(image_path, model_weight_list))

    input_batch, model_weight   = combos[int(sys.argv[-1])]

    base_path = os.path.abspath(__file__ + "/..")

    input_batch = [input_batch]
    # Verify each path is a directory
    for input_folder in input_batch:
        if not os.path.isdir(input_folder):
            raise Exception(input_folder + " is not a directory. Inputs must be a folder of files. Please refer to readme for more info")


    weights_path = base_path +  model_weight

    model = get_net()
    model.load_weights(weights_path)
    overlap_var = 1.2
    guass = False
    extra_name = ""
    name_folders = f'_overlap_{overlap_var}_{os.path.basename(weights_path)}_guass_{guass}_{extra_name}'

    for input_folder in input_batch:

        # Remove trailing slashes
        input_folder = os.path.normpath(input_folder)

        # Output folder name
        output_name = "seg-" + name_folders + os.path.basename(input_folder)
        output_dir = os.path.dirname(input_folder)

        output_folder = os.path.join(output_dir, output_name)


        # Create output directory. Overwrite if the directory exists
        if os.path.exists(output_folder):
            print(output_folder + " already exists. Will be overwritten")
            shutil.rmtree(output_folder)
            print(output_folder)

        os.makedirs(output_folder)

        print("input folder: " + input_folder)
        print("output_folder: " + output_folder)

        # Segment the brain
        if guass:
            segment_brain_guass(input_folder, output_folder, model, overlap_var, name_folders)
        else:
            segment_brain_normal(input_folder, output_folder, model, name_folders)

