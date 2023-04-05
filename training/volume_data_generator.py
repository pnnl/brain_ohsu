from tensorflow.keras.utils import Sequence
import cv2
import numpy as np
import random
from builtins import range
from batchgenerators.augmentations.utils import create_zero_centered_coordinate_mesh, elastic_deform_coordinates, \
    interpolate_img, \
    rotate_coords_2d, rotate_coords_3d, scale_coords, resize_segmentation, resize_multichannel_image, \
    elastic_deform_coordinates_2
from batchgenerators.augmentations.crop_and_pad_augmentations import random_crop as random_crop_aug
from batchgenerators.augmentations.crop_and_pad_augmentations import center_crop as center_crop_augfrom 
from models.model import input_dim, output_dim
from utilities.utilities import *
offset = (input_dim - output_dim)//2


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
        self.normal = normal

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
        
    #https://github.com/MIC-DKFZ/batchgenerators/blob/01f225d843992eec5467c109875accd6ea955155/batchgenerators/augmentations/spatial_transformations.py#L19

    def augment_spatial(self, data, seg, patch_size, patch_center_dist_from_border=30,
                        do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
                        do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                        do_scale=False, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                        border_mode_seg='nearest', border_cval_seg=0, order_seg=3, random_crop=False, p_el_per_sample=.5,
                        p_scale_per_sample=0, p_rot_per_sample=.5, independent_scale_for_each_axis=False,
                        p_rot_per_axis: float = .5, p_independent_scale_per_axis: int = .5):
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
        return data_result, seg_result
        

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

        return trans_vol

    def flow(self, x, y, batch_size):
        while True:
            x_gen = np.zeros((batch_size,) + x.shape[1:])
            if self.normal == True:
                y_gen = np.zeros((batch_size,) + y.shape[1:])
            else:
                y_gen = np.zeros((batch_size,) + np.copy(crop_numpy_batch(offset, offset, offset, y)).shape[1:])
            inds = list(range(x.shape[0]))
            if len(inds) < batch_size:
                raise ValueError("Samples less than batch_size")

            random.shuffle(inds)
            counter = 0

            for i in inds[:batch_size]:
                ind = i % x.shape[0]
                self._set_params()
                x_copy = np.copy(x[ind])
                y_copy = np.copy(y[ind])
                preprocess_vol = self._preprocess_vol(x_copy)
                if self.normal == True:
                    x_gen[counter] = self._transform_vol(preprocess_vol)
                    y_gen[counter] = self._transform_vol(y_copy)
                else:
                    x2, y2 = self.augment_spatial(data= preprocess_vol, seg = y_copy, patch_size = preprocess_vol.squeeze().shape)
            
                    y2 = np.copy(crop_numpy(offset, offset, offset, y2))
                    x_gen[counter] = x2
                    y_gen[counter] = y2
                counter += 1

            yield x_gen, y_gen
