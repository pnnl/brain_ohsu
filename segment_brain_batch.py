from inference import *
from models import *
import sys
import os
import shutil

if __name__ == "__main__":

    base_path = os.path.abspath(__file__ + "/..")

    input_batch = sys.argv[1:]
    # Verify each path is a directory
    for input_folder in input_batch:
        if not os.path.isdir(input_folder):
            raise Exception(input_folder + " is not a directory. Inputs must be a folder of files. Please refer to readme for more info")

    # Load the network
    weights_path = base_path + "/data/model-weights/best_weights_checkpoint_normal_false.hdf5"

    model = get_net()
    model.load_weights(weights_path)
    overlap_var = 1.0
    guass = True
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

        # Segment the brain
        if guass:
            segment_brain_guass(input_folder, output_folder, model, overlap_var, name_folders)
        else:
            segment_brain_normal(input_folder, output_folder, model, name_folders)

