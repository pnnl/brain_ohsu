import random
from models import input_dim, normal
from utilities.utilities import *

oversample_foreground_percent = .9
# https://github.com/MIC-DKFZ/nnUNet/blob/6d02b5a4e2a7eae14361cde9599bbf4ccde2cd37/nnunet/training/dataloading/dataset_loading.py#L204
def get_do_oversample(i, nb_examples):
        return not i < round(nb_examples * (1 - oversample_foreground_percent))

def get_random_training(volume, label, i, nb_examples):
    # Get a random corner to cut out a chunk for the training-set

    # https://github.com/MIC-DKFZ/nnUNet/blob/6d02b5a4e2a7eae14361cde9599bbf4ccde2cd37/nnunet/training/dataloading/dataset_loading.py#L294
    if not get_do_oversample(i, nb_examples) or normal == True:

        # because the x y images are appended in list, the first order is z and the order in shape is z, x, y not x, y, z 
        z = random.randint(0, volume.shape[0] - input_dim)
        x = random.randint(0, volume.shape[1] - input_dim)
        y = random.randint(0, volume.shape[2] - input_dim)
    else:

        voxels_of_that_class = np.argwhere(label[:label.shape[0] -input_dim, :label.shape[1]-input_dim, :label.shape[2]-input_dim] == 2)

        if len(voxels_of_that_class) >0:
            selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
            # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
            # Make sure it is within the bounds of lb and ub
            z = selected_voxel[0]
            x = selected_voxel[1]
            y = selected_voxel[2]
        else:
            # If the image does not contain any foreground classes, we fall back to random cropping
            
            z = random.randint(0, volume.shape[0] - input_dim)
            x = random.randint(0, volume.shape[1] - input_dim)
            y = random.randint(0, volume.shape[2] - input_dim)

    volume_chunk = crop_cube(x, y, z, volume, input_dim)
    label_chunk = crop_cube(x, y, z, label, input_dim)

    return volume_chunk, label_chunk



def generate_data_set(data_original_path, data_set_path, nb_examples=None):
    # Get the directory for volumes and labels sorted
    volumes_path = sorted(get_dir(data_original_path + "/volumes"))
    labels_path = sorted(get_dir(data_original_path + "/labels"))

    if len(volumes_path) != len(labels_path):
        raise Exception("Volumes and labels folders must have the same number of items for there to be a 1 to 1 matching")

    # Contains the list of volumes and labels chunks in the training-original folder
    volumes = []
    labels = []

    # Read in the chunks from training-original
    for i in range(len(volumes_path)):
        volumes.append(read_tiff_stack(volumes_path[i]))
        labels.append(read_tiff_stack(labels_path[i]))

    if nb_examples is None:
        nb_examples = 100 * len(volumes_path)

    draw_progress_bar(0)
    for i in range(nb_examples):

        ind = i % len(volumes_path)
        volume_chunk, label_chunk = get_random_training(volumes[ind], labels[ind], i, nb_examples)

        write_tiff_stack(volume_chunk, data_set_path + "/volumes/volume-" + str(i) + ".tiff")
        write_tiff_stack(label_chunk, data_set_path + "/labels/label-" + str(i) + ".tiff")
 
        # Update bar every 5 percent
        if i % (nb_examples // 20) == 0:
            draw_progress_bar(i/nb_examples)

    draw_progress_bar(1)
    print("\n")
