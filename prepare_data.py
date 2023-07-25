from training import *
from training.label_processor import process_labels
from utilities.utilities import *

if __name__ == "__main__":
    normal = sys.argv[3] == "True"
    add_suffix = sys.argv[2] 
    base_path = os.path.abspath(__file__ + "/..")
    print(sys.argv[1])
    print(sys.argv[2])
    print(sys.argv[3])
    # Must include operation being done.
    # e.g. generate_data_set
    if len(sys.argv) < 2:
        raise Exception("You must include the type of preparation. Please refer to readme")

    # Generate data set by cropping out 64 length cubes from larger chunks for validation
    if sys.argv[1] == "generate_validation_set":

        # Validation directories
        data_original_path = base_path + f"/data/validation/validation-original{add_suffix}"
        # change path here to 100 if not double
        data_set_path = base_path + f"/data/validation/validation-set_normal_{normal}{add_suffix}"

        # Default value is None
        nb_examples = None
        if len(sys.argv) > 4:
            nb_examples = int(sys.argv[4])

        print(data_original_path, data_set_path, normal, nb_examples)
        generate_data_set(data_original_path, data_set_path, normal = normal, nb_examples=nb_examples)

    # Generate data set by cropping out 64 length cubes from larger chunks for training
    elif sys.argv[1] == "generate_training_set":

        # Training directories
        data_original_path = base_path + f"/data/training/training-original{add_suffix}"
         # change path here to 100 if not double
        data_set_path = base_path + f"/data/training/training-set_normal_{normal}{add_suffix}"

        # Default value is None
        nb_examples = None
        if len(sys.argv) > 4:
            nb_examples = int(sys.argv[4])
            
        print(data_original_path, data_set_path, normal, nb_examples)
        generate_data_set(data_original_path, data_set_path, normal = normal,  nb_examples=nb_examples)


    # Add edge labels to your labeled data
    elif sys.argv[1] == "process_labels":

        if len(sys.argv) < 3:
            raise Exception("You must include the directory to the labels")

        labels_folder = sys.argv[2]
        output_name = "processed-labels"
        output_dir = os.path.dirname(labels_folder)
        output_folder = os.path.join(output_dir, output_name)

        if not os.path.isdir(labels_folder):
            raise Exception(labels_folder + " is not a directory. Inputs must be a folder of tiff files. Please refer to readme for more info")

        # Create output directory. Overwrite if the directory exists
        if os.path.exists(output_folder):
            print(output_folder + " already exists. Will be overwritten")
            shutil.rmtree(output_folder)

        os.makedirs(output_folder)

        for label_name in os.listdir(labels_folder):
            vol = read_tiff_stack(os.path.join(labels_folder, label_name))
            edge_vol = process_labels(vol)
            write_tiff_stack(edge_vol, os.path.join(output_folder, label_name))

    else:
        raise Exception("You must choose your type of preparation (generate_validation_set, generate_training_set, process_labels)")
