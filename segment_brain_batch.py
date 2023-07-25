from inference import *
from models import *
import sys
import os
import shutil
import itertools

if __name__ == "__main__":


    gauss = sys.argv[1] == "True"
    add_suffix = sys.argv[2]

    base_path = os.path.abspath(__file__ + "/..")
        # Load the network
    model_weight_list = [
    "/data/model-weights/best_weights_checkpoint_oversample_bol_True_aug_bol_True_lr_bol_True_flip_bol_False_el_1.0_rot_1.0__encode_last_layer_loss_0.0001_training__val_1_test_5__july23.hdf5",
    "/data/model-weights/trailmap_model.hdf5",
    ]

    # data testing
    image_path  =  [ base_path + f"/data/testing/testing-original{add_suffix}"]
    combos = list(itertools.product(image_path, model_weight_list))

    input_batch, model_weight   = combos[int(sys.argv[-1])]
    print(type(gauss))
    base_path = os.path.abspath(__file__ + "/..")
    
    input_batch = [input_batch]
    # Verify each path is a directory
    for input_folder in input_batch:
        if not os.path.isdir(input_folder):
            raise Exception(input_folder + " is not a directory. Inputs must be a folder of files. Please refer to readme for more info")


    weights_path = base_path +  model_weight
 

    model = get_net()
    model.load_weights(weights_path)
    overlap_var = 2.0
    extra_name = ""
    name_folders = f'_overlap_{overlap_var}_{os.path.basename(weights_path)}_gauss_{gauss}_{extra_name}'

    for input_folder in input_batch:

        # Remove trailing slashes
        input_folder = os.path.normpath(input_folder)

        # Output folder name
        output_name = "seg-" + name_folders + os.path.basename(input_folder)
        # save in test/validation folder
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
        if gauss:
            print("gauss")
            print(gauss)
            segment_brain_gauss(input_folder, output_folder, model, overlap_var, name_folders)
        else:
            segment_brain_normal(input_folder, output_folder, model, name_folders)

