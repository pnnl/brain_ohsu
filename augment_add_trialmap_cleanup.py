
import numpy as np
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.channel_selection_transforms import DataChannelSelectionTransform, \
    SegChannelSelectionTransform
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor

import os
cwd = os.getcwd()

from tensorflow.keras.utils import Sequence
import cv2
import numpy as np
import random
import random
input_dim = 100
output_dim = 100
validation_path = "data/validation/validation-set"
batch_size = 1
epochs = 1
oversample_foreground_percent = .8

import numpy as np
import cv2
import os
from os import listdir, makedirs
from os.path import join
from PIL import Image
import shutil
import sys

# https://github.com/MIC-DKFZ/nnUNet/blob/6d02b5a4e2a7eae14361cde9599bbf4ccde2cd37/nnunet/training/dataloading/dataset_loading.py#L204
def get_do_oversample(i, nb_examples):
        return not i < round(nb_examples * (1 - oversample_foreground_percent))


def crop_numpy(dim1, dim2, dim3, vol):
    if len(vol.shape) > 3:
        return vol[dim1:vol.shape[0] - dim1, dim2:vol.shape[1] - dim2, dim3:vol.shape[2] - dim3, :]
    else:
        return vol[dim1:vol.shape[0] - dim1, dim2:vol.shape[1] - dim2, dim3:vol.shape[2] - dim3]


def write_tiff_stack(vol, fname):
    im = Image.fromarray(vol[0])
    ims = []

    for i in range(1, vol.shape[0]):
        
        ims.append(Image.fromarray(vol[i]))

    im.save(fname, save_all=True, append_images=ims)


def get_dir(path):
    tiffs = [join(path, f) for f in listdir(path) if f[0] != '.']
    return sorted(tiffs)


def crop_cube(x, y, z, vol, cube_length=64):
    # Cube shape
    return crop_box(x, y, z, vol, (cube_length, cube_length, cube_length))


def crop_box(x, y, z, vol, shape):
    vol2 = vol.copy()
    return vol2[z:z + shape[2], x:x + shape[0], y:y + shape[1]]


"""
Read images from start_index to end_index from a folder

@param path: The path to the folder
@param start_index: The index of the image to start reading from inclusive
@param end_index: The end of the image to stop reading from exclusive

@raise FileNotFoundError: If the path to the folder cannot be found 
"""
def read_folder_section(path, start_index, end_index):
    fnames = get_dir(path)
    vol = []

    for f in fnames[start_index: end_index]:
        img = cv2.imread(f, cv2.COLOR_BGR2GRAY)
        vol.append(img)

    vol = np.array(vol)

    return vol


def read_folder_stack(path):
    fnames = get_dir(path)

    fnames.sort()
    vol = cv2.imread(fnames[0], cv2.COLOR_BGR2GRAY)

    if len(vol.shape) == 3:
        return vol

    vol = []

    for f in fnames:
        img = cv2.imread(f, cv2.COLOR_BGR2GRAY)
        vol.append(img)

    vol = np.array(vol)

    return vol

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
#         print(path)
#         plt.imshow(slice)
#         plt.show()
        images.append(slice)

    return np.array(images)


def coordinate_vol(coords, shape):
    vol = np.zeros(shape, dtype="uint16")
    for c in coords:
        vol[c[0], c[1], c[2]] = 1
    return vol


def preprocess(vol):
    return vol / 65535


def preprocess_batch(batch):
    assert len(batch.shape) == 5
    lst = []

    for i in range(batch.shape[0]):
        lst.append(preprocess(batch[i]))

    return np.array(lst)


def dist(p1, p2):
    sqr = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
    return sqr ** .5



"""
Progress bar to indicate status of the segment_brain function
"""

def draw_progress_bar(percent, eta="", bar_len = 40):
    # percent float from 0 to 1.
    sys.stdout.write("\r")
    sys.stdout.write("[{:<{}}] {:>3.0f}%       {:20}".format("=" * int(bar_len * percent), bar_len, percent * 100, eta))
    sys.stdout.flush()


import random

def get_random_training(volume, label, i, nb_examples):
    # Get a random corner to cut out a chunk for the training-set

    # https://github.com/MIC-DKFZ/nnUNet/blob/6d02b5a4e2a7eae14361cde9599bbf4ccde2cd37/nnunet/training/dataloading/dataset_loading.py#L294
    if not get_do_oversample(i, nb_examples):

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


def generate_data_set(data_original_path, data_set_path, nb_examples=10):
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
        # adding in to test
        # removing everything with  makes the scaling spread out into negatives
        # volume_chunk[:, :, :] = 0.0
        volume_chunk[:, 50:52, 50:52] = 10000.0
        # label_chunk[:, :, :] = 0
        label_chunk[:, 50:52, 50:52] =2
        write_tiff_stack(volume_chunk, data_set_path + "/volumes/volume-" + str(i) + ".tiff")
        write_tiff_stack(label_chunk, data_set_path + "/labels/label-" + str(i) + ".tiff")



data_original_path = "data/validation/validation-original"
data_set_path = "data/validation/validation-set"
##data_original_path = "orig2tiff_1"
nb_examples = None
generate_data_set(data_original_path, data_set_path, nb_examples=nb_examples)


# Load the data into
def load_data(data_path, nb_examples=10):
    volumes_folder_path = data_path + "/volumes"
    labels_folder_path = data_path + "/labels"

    volumes_path = get_dir(volumes_folder_path)
    labels_path = get_dir(labels_folder_path)

    assert len(labels_path) == len(volumes_path)

    total_length = len(labels_path)

    if nb_examples is None:
        nb_examples = total_length
    else:
        assert nb_examples <= total_length

    inds = list(range(len(volumes_path)))
    random.shuffle(inds)

    x = []
    y = []

    for i in range(nb_examples):
        rand_ind = inds[i]
        x.append(read_tiff_stack(volumes_path[rand_ind]))
        y.append(read_tiff_stack(labels_path[rand_ind]))

    inds = list(range(nb_examples))
    random.shuffle(inds)

    x_train = []
    y_train = []

    for i in inds:

        offset = (input_dim - output_dim)//2
        # keep offset zero until after spatial transformation
        offset = 0

        background = np.copy(crop_numpy(offset, offset, offset, y[i]))
        background[background == 2] = 0
        background[background == 3] = 0
        background[background == 4] = 0

        axons = np.copy(crop_numpy(offset, offset, offset, y[i]))
        axons[axons == 1] = 0
        axons[axons == 2] = 1
        axons[axons == 3] = 0
        axons[axons == 4] = 0

        artifact = np.copy(crop_numpy(offset, offset, offset, y[i]))
        artifact[artifact == 1] = 0
        artifact[artifact == 2] = 0
        artifact[artifact == 3] = 1
        artifact[artifact == 4] = 0

        edges = np.copy(crop_numpy(offset, offset, offset, y[i]))
        edges[edges == 1] = 0
        edges[edges == 2] = 0
        edges[edges == 3] = 0
        edges[edges == 4] = 1

        # 0 channel is segmentation
        # 1 channel is background
        # 2 channel is artifact
        # 3 channel is edge
        output = np.stack([axons, background, artifact, edges], axis=-1)

        input = x[i].reshape(x[i].shape + (1,))
        if np.count_nonzero(output == 1) > 0:
            x_train.append(input)
            y_train.append(output)

    x = np.array(x_train)
    y = np.array(y_train)


    return x, y



from builtins import range

import numpy as np
from batchgenerators.augmentations.utils import create_zero_centered_coordinate_mesh, elastic_deform_coordinates, \
    interpolate_img, \
    rotate_coords_2d, rotate_coords_3d, scale_coords, resize_segmentation, resize_multichannel_image, \
    elastic_deform_coordinates_2
from batchgenerators.augmentations.crop_and_pad_augmentations import random_crop as random_crop_aug
from batchgenerators.augmentations.crop_and_pad_augmentations import center_crop as center_crop_aug

#https://github.com/MIC-DKFZ/batchgenerators/blob/01f225d843992eec5467c109875accd6ea955155/batchgenerators/augmentations/spatial_transformations.py#L19

# patch_size is just full image in this case (already segmented)
# data and seg is just one
# modify channels so it's the last -1 number
def augment_spatial(data, seg, patch_size, patch_center_dist_from_border=30,
                    do_elastic_deform=True, alpha=( 1000., 1000.), sigma=(13., 13.),
                    do_rotation=False, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                    do_scale=False, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                      border_mode_seg='constant', border_cval_seg=0, order_seg=0,  random_crop=False, p_el_per_sample=1,
                    p_scale_per_sample=1, p_rot_per_sample=1, independent_scale_for_each_axis=False,
                    p_rot_per_axis: float = 1, p_independent_scale_per_axis: int = 1):
    dim = len(patch_size)
    
    seg_result = None
    if seg is not None:
        if dim == 2:
            seg_result = np.zeros((patch_size[0], patch_size[1], seg.shape[3]), dtype=np.float32)
        else:
            seg_result = np.zeros((patch_size[0], patch_size[1], patch_size[2], seg.shape[3]),
                                  dtype=np.float32)

    if dim == 2:
        data_result = np.zeros(( patch_size[0], patch_size[1], data.shape[3]), dtype=np.float32)
    else:
        data_result = np.zeros((  patch_size[0], patch_size[1], patch_size[2], data.shape[3]) ,
                               dtype=np.float32)

    if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
        patch_center_dist_from_border = dim * [patch_center_dist_from_border]


    coords = create_zero_centered_coordinate_mesh(patch_size)
    modified_coords = False

    if do_elastic_deform and np.random.uniform() < p_el_per_sample:
        a = np.random.uniform(alpha[0], alpha[1])
        s = np.random.uniform(sigma[0], sigma[1])
        coords = elastic_deform_coordinates(coords, a, s)
        modified_coords = True

    if do_rotation and np.random.uniform() < p_rot_per_sample:

        if np.random.uniform() <= p_rot_per_axis:
            a_x = np.random.uniform(angle_x[0], angle_x[1])
        else:
            a_x = 0

        if dim == 3:
            if np.random.uniform() <= p_rot_per_axis:
                a_y = np.random.uniform(angle_y[0], angle_y[1])
            else:
                a_y = 0

            if np.random.uniform() <= p_rot_per_axis:
                a_z = np.random.uniform(angle_z[0], angle_z[1])
            else:
                a_z = 0

            coords = rotate_coords_3d(coords, a_x, a_y, a_z)
        else:
            coords = rotate_coords_2d(coords, a_x)
        modified_coords = True

    if do_scale and np.random.uniform() < p_scale_per_sample:
        if independent_scale_for_each_axis and np.random.uniform() < p_independent_scale_per_axis:
            sc = []
            for _ in range(dim):
                if np.random.random() < 0.5 and scale[0] < 1:
                    sc.append(np.random.uniform(scale[0], 1))
                else:
                    sc.append(np.random.uniform(max(scale[0], 1), scale[1]))
        else:
            if np.random.random() < 0.5 and scale[0] < 1:
                sc = np.random.uniform(scale[0], 1)
            else:
                sc = np.random.uniform(max(scale[0], 1), scale[1])

        coords = scale_coords(coords, sc)
        modified_coords = True

    # now find a nice center location 
    if modified_coords:
        for d in range(dim):
            #don't add 2 to d since removed channel and number
            ctr = data.shape[d ] / 2. - 0.5
            coords[d] += ctr
        for channel_id in range(data.shape[3]):
            data_result[:, :, :,  channel_id] = interpolate_img(data[:, :, :, channel_id], coords, order_data,
                                                                    border_mode_data, cval=border_cval_data)
        if seg is not None:
            for channel_id in range(seg.shape[3]):
                
                seg_result[:, :, :,channel_id] = interpolate_img(seg[:, :, :, channel_id], coords, order_seg,
                                                                    border_mode_seg, cval=border_cval_seg,
                                                                    is_seg=True)

    else:

            data_result = data
            seg_result= seg
    

    #offset = (input_dim - output_dim)//2
    #seg_result = np.copy(crop_numpy(offset, offset, offset, seg_result))
    return data_result, seg_result

# %%
import numpy as np
from batchgenerators.augmentations.utils import get_range_val, mask_random_squares
from builtins import range
from scipy.ndimage import gaussian_filter

import random
from typing import Tuple

class VolumeDataGenerator(Sequence):
    def __init__(self,
                 samplewise_center=False,
                 samplewise_std_normalization=False,
                 min_max_normalization=False,
                 scale_constant_range=0.0,
                 scale_range=0.0,
                 rotation_range=0.0,
                 width_shift_range=0.0,
                 height_shift_range=0.0,
                 depth_shift_range=0.0,
                 zoom_range=0.0,
                 horizontal_flip=False,
                 vertical_flip=False,
                 depth_flip=False,
                 normal = False):

        self.scale_constant_range = scale_constant_range
        self.scale_range = scale_range
        self.samplewise_center = samplewise_center
        self.min_max_normalization = min_max_normalization
        self.samplewise_std_normalization = samplewise_std_normalization
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.depth_shift_range = depth_shift_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.depth_flip = depth_flip

    def _shift_img(self, image, dx, dy):

        if dx == 0 and dy == 0:
            return image

        rows = image.shape[0]
        cols = image.shape[1]
        dx = cols * dx
        dy = rows * dy

        M = np.float32([[1, 0, dx], [0, 1, dy]])
        result = cv2.warpAffine(image, M, (cols, rows))
        return result

    def _rotate_img(self, image, angle):

        if angle == 0:
            return image
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def _zoom_img(self, image, zoom_factor):
        if zoom_factor == 1:
            return image

        height, width = image.shape[:2]  # It's also the final desired shape
        new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

        ### Crop only the part that will remain in the result (more efficient)
        # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
        y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
        y2, x2 = y1 + height, x1 + width
        bbox = np.array([y1, x1, y2, x2])
        # Map back to original image coordinates
        bbox = (bbox / zoom_factor).astype(np.int)
        y1, x1, y2, x2 = bbox
        cropped_img = image[y1:y2, x1:x2]

        # Handle padding when downscaling
        resize_height, resize_width = min(new_height, height), min(new_width, width)
        pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) // 2
        pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
        pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0, 0)] * (image.ndim - 2)

        result = cv2.resize(cropped_img, (resize_width, resize_height))

        result = np.pad(result, pad_spec, mode='constant')
        assert result.shape[0] == height and result.shape[1] == width

        return result

    def _vflip_img(self, image, flip):
        if flip:

            return cv2.flip(image, 0)
        return image

    def _hflip_img(self, image, flip):
        if flip:
            return cv2.flip(image, 1)
        return image

    def _dflip_vol(self, vol, flip):
        if flip:
            zsize = vol.shape[0]
            for i in range(zsize // 2):
                vol[i], vol[zsize - i - 1] = vol[zsize - i - 1], vol[i]
        return vol

    def _scale_vol(self, vol, scale):
        if scale == 1:
            return vol

        return np.clip(vol * scale, 0, 2 ** 16 - 1)

    def _scale_constant_vol(self, vol, scale_constant):

        if scale_constant == 0:
            return vol

        mean = np.mean(vol)
        constant = scale_constant * mean

        return np.clip(vol + constant, 0, 2 ** 16 - 1)

    def _preprocess_vol(self, vol):

        trans_vol = vol

        if self.samplewise_center:
            trans_vol = trans_vol - np.mean(vol)

        elif self.samplewise_std_normalization:
            trans_vol = trans_vol / np.std(vol)


        elif self.min_max_normalization:
            trans_vol = trans_vol / 65535

        trans_vol = self._scale_vol(trans_vol, self.scale)
        trans_vol = self._scale_constant_vol(trans_vol, self.scale_constant)

        return trans_vol

    # tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    # tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
    #                                            p_per_channel=0.5))
    # tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))

    def augment_gaussian_blur(self, data_sample: np.ndarray, sigma_range: Tuple[float, float], per_channel: bool = True,
                            p_per_channel: float = 1, different_sigma_per_axis: bool = False,
                            p_isotropic: float = 0) -> np.ndarray:


        if not per_channel:
            sigma = get_range_val(sigma_range) if ((not different_sigma_per_axis) or
                                                ((np.random.uniform() < p_isotropic) and
                                                    different_sigma_per_axis)) \
                else [get_range_val(sigma_range) for _ in data_sample.shape[1:]]
        else:
            sigma = None
        for c in range(data_sample.shape[3]):
            if np.random.uniform() <= p_per_channel:
                if per_channel:
                    sigma = get_range_val(sigma_range) if ((not different_sigma_per_axis) or
                                                        ((np.random.uniform() < p_isotropic) and
                                                            different_sigma_per_axis)) \
                        else [get_range_val(sigma_range) for _ in data_sample.shape[1:]]
                data_sample[:,:, :, c] = gaussian_filter(data_sample[:, :, :, c], sigma, order=0)
        return data_sample

    def _set_params(self):

        self.rot_ang = random.randint(-self.rotation_range, self.rotation_range)
        self.width_shift = random.uniform(-self.width_shift_range, self.width_shift_range)
        self.vertical_shift = random.uniform(-self.height_shift_range, self.height_shift_range)
        self.zoom = random.uniform(1, 1 + self.zoom_range)
        self.hflip = random.choice([self.horizontal_flip, False])
        self.vflip = random.choice([self.vertical_flip, False])
        self.dflip = random.choice([self.depth_flip, False])
        self.scale = random.uniform(1 - self.scale_range, 1 + self.scale_range)
        self.scale_constant = random.uniform(0, self.scale_constant_range)

    def _transform_vol(self, vol):

        for i in range(vol.shape[0]):
            trans_img = vol[i]
            trans_img = self._rotate_img(trans_img, self.rot_ang)
            trans_img = self._shift_img(trans_img, self.width_shift, self.vertical_shift)
            trans_img = self._zoom_img(trans_img, self.zoom)
            trans_img = self._vflip_img(trans_img, self.vflip)
            trans_img = self._hflip_img(trans_img, self.hflip)



            if len(trans_img.shape) == 2:
                trans_img = trans_img.reshape(trans_img.shape + (1,))
            vol[i] = trans_img

        trans_vol = self._dflip_vol(vol, self.dflip)
        #trans_vol = self.augment_gaussian_blur(trans_vol, sigma_range=(0.5, 1.))

        return trans_vol
  
   

    def flow(self, x, y, batch_size):
        while True:
            x_gen = np.zeros((batch_size,) + x.shape[1:])
            offset = (input_dim - output_dim)//2
            y_gen = np.zeros((batch_size,) + crop_numpy(offset, offset, offset, y[0]).shape)
            inds = list(range(x.shape[0]))

            if len(inds) < batch_size:
                raise ValueError("Samples less than batch_size")

            random.shuffle(inds)

            counter = 0
            #for each image
            for i in inds[:batch_size]:
                print(i)
                ind = i % x.shape[0]
                self._set_params()
                x_copy = np.copy(x[ind])
                y_copy = np.copy(y[ind])
                
                preprocess_vol = self._preprocess_vol(x_copy)
                #preprocess_vol = self.augment_gaussian_blur(preprocess_vol)
                write_tiff_stack(preprocess_vol[:,:,:, 0].astype(float), data_set_path + "/look/processed-input" + str(i) + ".tiff")
                write_tiff_stack(y_copy[:,:,:, 0].astype(float), data_set_path + "/look/orig-label" + str(i) + ".tiff")
                write_tiff_stack(np.sum(y_copy[:,:,:, :].astype(float), -1), data_set_path + "/look/orig-label-all-info" + str(i) + ".tiff")
                print("min x")
                print(x_copy.min())
                print(x_copy.max())
                x2, y2 = augment_spatial(x_copy, y_copy, patch_size = preprocess_vol.squeeze().shape)
                print("augmnet")
                print(x2.min())
                print(x2.max())
                x2 = self._preprocess_vol(x2)
                print("final")
                print(x2.min())
                print(x2.max())
                print("preprcoess")
                print(preprocess_vol.min())
                print(preprocess_vol.max())
                write_tiff_stack(x2[:,:,:, 0], data_set_path + "/look/augment_spatial-input-" + str(i) + ".tiff")
                write_tiff_stack(y2[:,:,:, 0].astype(float), data_set_path + "/look/augment_spatial-label" + str(i) + ".tiff")
                write_tiff_stack(np.sum(y2[:,:,:, :].astype(float), -1), data_set_path + "/look/augment_spatial-label-all-info" + str(i) + ".tiff")
                # keep offset zero until after spatial transformation
                
                y2 = np.copy(crop_numpy(offset, offset, offset, y2))

                x_gen[counter] = x2
                y_gen[counter] = y2
                counter += 1
            print("")
            yield x_gen, y_gen

x_validation, y_validation = load_data(validation_path)

print("Loaded Data")

datagen = VolumeDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    depth_flip=True,
    min_max_normalization=True,
    scale_range=0.1,
    scale_constant_range=0.2
)

#train_generator = datagen.flow(x_train, y_train, batch_size)
validation_generator = datagen.flow(x_validation, y_validation, batch_size)

for i in range(1):
    x3, y3 =  next(validation_generator)
    print(np.unique(y3))
  