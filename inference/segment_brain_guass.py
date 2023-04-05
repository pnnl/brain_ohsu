from models.model import *
import numpy as np
import os
import time
import cv2
import sys
from PIL import Image
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
from utilities.utilities import *
import tensorflow as tf
import csv

# Will need to be adjusted depending on the GPU
batch_size = 15

# Don't run network on chunks which don't have a value above threshold
threshold = 0.00

# Edge width between the output and input volumes
dim_offset = (input_dim - output_dim) // 2

#0.9106592 1.1
# 0.8775488 2
# 0.9106592 1
# 0.93280274 original
"""
Progress bar to indicate status of the segment_brain function
"""

# https://github.com/MIC-DKFZ/nnUNet/blob/6d02b5a4e2a7eae14361cde9599bbf4ccde2cd37/nnunet/network_architecture/neural_network.py#L245
def get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map

def draw_progress_bar(percent, eta, bar_len = 40):
    # percent float from 0 to 1.
    sys.stdout.write("\r")
    sys.stdout.write("[{:<{}}] {:>3.0f}%       {:20}".format("=" * int(bar_len * percent), bar_len, percent * 100, eta))
    sys.stdout.flush()


def get_dir(path):
    tiffs = [os.path.join(path, f) for f in os.listdir(path) if f[0] != '.']

    return sorted(tiffs)

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
    for i in range(start_index, end_index):

        if i < 0:
            first_img = cv2.imread(fnames[0], cv2.COLOR_BGR2GRAY)
            vol.append(first_img)
        elif i >= len(fnames):
            last_img = cv2.imread(fnames[-1], cv2.COLOR_BGR2GRAY)
            vol.append(last_img)
        else:
            img = cv2.imread(fnames[i], cv2.COLOR_BGR2GRAY)
            vol.append(img)

    vol = np.array(vol)

    return vol



def loss_inference(input_folder, output_folder):
    

    file_names = get_dir(os.path.join(input_folder, "labels"))

    #only works for one tif stack
    y_true_flat = read_tiff_stack(file_names[0])

    background = np.copy(y_true_flat)
    background[background == 2] = 0
    background[background == 3] = 0
    background[background == 4] = 0

    axons = np.copy(y_true_flat)
    axons[axons == 1] = 0
    axons[axons == 2] = 1
    axons[axons == 3] = 0
    axons[axons == 4] = 0

    artifact = np.copy(y_true_flat)
    artifact[artifact == 1] = 0
    artifact[artifact == 2] = 0
    artifact[artifact == 3] = 1
    artifact[artifact == 4] = 0

    edges = np.copy(y_true_flat)
    edges[edges == 1] = 0
    edges[edges == 2] = 0
    edges[edges == 3] = 0
    edges[edges == 4] = 1

    # 0 channel is segmentation
    # 1 channel is background
    # 2 channel is artifact
    # 3 channel is edge
    y_true = np.stack([axons, background, artifact, edges], axis=-1)


    fnames = get_dir(output_folder)
    vol = []
    for i in range(len(fnames)):
            img = cv2.imread(fnames[i], cv2.COLOR_BGR2GRAY)
            vol.append(img)
    output_dict ={}
    y_pred = np.array(vol)
    tensor1 = tf.convert_to_tensor(np.expand_dims(y_true, axis=0))
    print(y_pred.shape)
    print(y_true.shape)
    tensor2 = tf.convert_to_tensor(np.expand_dims(y_pred, axis=(0, 4)))
    loss = adjusted_accuracy(tensor1, tensor2)
    output_dict["adjusted_accuracy"] = loss
    loss = axon_precision(tensor1, tensor2)
    output_dict["axon_precision"] = loss
    loss = axon_recall(tensor1, tensor2)
    output_dict["axon_recall"] = loss
    return output_dict


def write_folder_section(output_folder, file_names,  section_seg):
    # Write the segmentation into the output_folder
    for slice_index in range(section_seg.shape[0]):
        input_file_name = file_names[slice_index]

        output_file_name = "seg-" + os.path.basename(input_file_name)
        output_full_path = output_folder + "/" + output_file_name

        # If write fails print problem
        pil_image = Image.fromarray(section_seg[slice_index])
        pil_image.save(output_full_path)

def write_total(output_folder, file_names, output_total, agg_gauss_total):
    seg = output_total/agg_gauss_total
    #pd.DataFrame(seg[10:15,:,:].reshape(-1,  seg.shape[-1])).to_csv('values_'+ str(overlap_var) + '.csv')
    
    # Write the segmentation into the output_folder
    for slice_index in range(len(file_names)):
        input_file_name = file_names[slice_index]

        output_file_name = "seg-" + os.path.basename(input_file_name)
        output_full_path = output_folder + "/" + output_file_name

        # If write fails print problem
        pil_image = Image.fromarray(seg[slice_index])
        pil_image.save(output_full_path)
"""
Segment a brain by first cropping a list of chunks from the brain of the models's input size and executing the models on 
a batch of chunks. To conserve memory, the function ill load sections of the brain at once.   

@param input_folder: The input directory that is a folder of 2D tiff files of the brain 
@param output_folder: Directory to write the segmentation files

@raise FileNotFoundError: If input_folder cannot be found
@raise NotADirectoryError: If the input_folder is not a directory
"""


def segment_brain_guass(input_folder, output_folder, model, overlap_var, name, tif_input = True):
    gaussian_importance_map = get_gaussian((output_dim, output_dim, output_dim), sigma_scale=1. / 8)
    #if input is a tif file instead of stacked images, convert to stacked images
    if tif_input == True:
        file_names = get_dir(os.path.join(input_folder, f"volumes"))
        vol= read_tiff_stack(file_names[0])
        write_folder_stack(vol, os.path.join(input_folder, f"slices_{name}"))
    # Name of folder
    folder_name = os.path.basename(os.path.join(input_folder, f"slices_{name}"))
    # Get the list of tiff files
    file_names = get_dir(os.path.join(input_folder, f"slices_{name}"))

    print("overlap:" + str(overlap_var))

    if len(file_names) < 5:
        print("The Z direction must contain a minimum of 36 images")
        return
    
    first_img = cv2.imread(file_names[0], cv2.COLOR_BGR2GRAY)

    if first_img.shape[0] < 36 or first_img.shape[1] < 36:
        print("The X and Y direction must contain a minimum of 36 pixels")
        return

    agg_gauss_total = np.zeros((len(file_names), first_img.shape[0], first_img.shape[1]))
    output_total = np.zeros((len(file_names), first_img.shape[0], first_img.shape[1]))
    ones_total = np.zeros((len(file_names), first_img.shape[0], first_img.shape[1]))


    eta = "ETA: Pending"
    # Get start time in minutes. Needed to calculate ETA
    start_time = time.time()/60

    total_sections = (len(file_names) // output_dim) * output_dim
    print("Name: " + folder_name)

    draw_progress_bar(0, eta)
    # Each iteration of loop will cut a section from slices i to i + input_dim and run helper_segment_section

    section_index = -dim_offset
    while section_index <= len(file_names) - input_dim + dim_offset:

        # Read section of folder
        section = read_folder_section(os.path.join(input_folder, f"slices_{name}"), section_index, section_index + input_dim).astype('float32')

        # Make the volume pixel intensity between 0 and 1
        #use same pre processing as in volumne generator
        section_vol = section / (2 ** 16 - 1)
 
        # Get the segmentation of this chunk
        #section_seg = helper_segment_section(model, section_vol, gaussian_importance_map)
       # to overlap in the z direction, collect values from each section and the agg guassina map into a meta table
        orig_seg_cropped, aggregated_nb_of_predictions_cropped, aggregated_ones_cropped     = helper_segment_section(model, section_vol, gaussian_importance_map, overlap_var)
        
        output_total[section_index + dim_offset: section_index  + input_dim - dim_offset, :, :] += orig_seg_cropped[dim_offset: input_dim - dim_offset, :, :]
        agg_gauss_total[section_index + dim_offset: section_index  + input_dim - dim_offset, :, :] += aggregated_nb_of_predictions_cropped[dim_offset: input_dim - dim_offset, :, :]
        ones_total[section_index + dim_offset: section_index  + input_dim - dim_offset, :, :] += aggregated_ones_cropped[dim_offset: input_dim - dim_offset, :, :]
        # Write to output folder
        #write_folder_section(output_folder, file_names, section_index, section_seg)

        # Calculate ETA
        now_time = time.time()/60
        sections_left = ((total_sections - section_index)/output_dim) - 1
        time_per_section = (now_time - start_time)/(1 + section_index/output_dim)

        eta = "ETA: " + str(round(sections_left * time_per_section, 1)) + " mins"
        draw_progress_bar((section_index + dim_offset) / total_sections, eta)

        #section_index += output_dim
        #add by dim/overlap_var
        #to do: this should also be gaussian add ons, so don't write to folder until the end
        section_index += int(output_dim/overlap_var)

    # Fill in slices in the end
    end_aligned = len(file_names) - input_dim + dim_offset

    # Read section of folder
    section = read_folder_section(os.path.join(input_folder, f"slices_{name}"), end_aligned, end_aligned + input_dim).astype('float32')

    # Make the volume pixel intensity between 0 and 1
    section_vol = section / (2 ** 16 - 1)

    orig_seg_cropped, aggregated_nb_of_predictions_cropped, aggregated_ones_cropped   = helper_segment_section(model, section_vol, gaussian_importance_map, overlap_var)
    output_total[end_aligned + dim_offset: end_aligned  + input_dim - dim_offset, :, :] += orig_seg_cropped[dim_offset: input_dim - dim_offset, :, :]
    agg_gauss_total[end_aligned + dim_offset: end_aligned  + input_dim - dim_offset, :, :] += aggregated_nb_of_predictions_cropped[dim_offset: input_dim - dim_offset, :, :] 
    ones_total[end_aligned + dim_offset: end_aligned  + input_dim - dim_offset, :, :] += aggregated_ones_cropped[dim_offset: input_dim - dim_offset, :, :]
    #pd.DataFrame(ones_total[10,:,:]).to_csv('ones_total_'+ str(overlap_var) + '.csv')
    output_total /= agg_gauss_total
    write_folder_section(output_folder, file_names,  output_total)
    #pd.DataFrame(ones_total.reshape(-1,  ones_total.shape[-1])).to_csv('values_'+ str(overlap_var) + '.csv')
    # if labels exist, get the accuracy
    if tif_input == True:
        print(output_folder)
        print(input_folder)
        output_dict =loss_inference(input_folder, output_folder)

        with open(output_folder + "/" + f"dict{name}.csv", 'w') as csv_file:  
            writer = csv.writer(csv_file)
            for key, value in output_dict.items():
                writer.writerow([key, value])
    total_time = "Total: " + str(round((time.time()/60) - start_time, 1)) + " mins"
    draw_progress_bar(1, total_time)
    print("\n")

def write_tiff_stack(vol, fname):
    im = Image.fromarray(vol[0])
    ims = []

    for i in range(1, vol.shape[0]):
        ims.append(Image.fromarray(vol[i]))

    im.save(fname, save_all=True, append_images=ims)

"""
Helper function for segment_brain. Takes in a section of the brain 

*** Note: Due to the network output shape being smaller than the input shape, the edges will not be used  

@param models: network models
@param section: a section of the entire brain
"""


def helper_segment_section(model, section, gaussian_importance_map, overlap_var):
    # List of bottom left corner coordinate of all input chunks in the section
    coords = []

    # Pad the section to account for the output dimension being smaller the input dimension
    # temp_section = np.pad(section, ((0, 0), (dim_offset, dim_offset),
    #  
    #                                (dim_offset, dim_offset)), 'constant', constant_values=(0, 0))
    print(section.shape)
    temp_section = np.pad(section, ((0, 0), (dim_offset, dim_offset),
                                    (dim_offset, dim_offset)), 'edge')
    print(temp_section.shape)
    #keep track of sum of gaussian muliplier 
    #https://github.com/MIC-DKFZ/nnUNet/blob/6d02b5a4e2a7eae14361cde9599bbf4ccde2cd37/nnunet/network_architecture/neural_network.py#L363
    aggregated_nb_of_predictions = np.zeros_like(temp_section).astype('float32')
    aggregated_ones = np.zeros_like(temp_section).astype('float32')
    # Add most chunks aligned with top left corner
    #divide by 2 to get overlapping windows
    for x in range(0, temp_section.shape[1] - input_dim, int(output_dim/overlap_var)):
        for y in range(0, temp_section.shape[2] - input_dim, int(output_dim/overlap_var)):
            coords.append((0, x, y))

    # Add bottom side aligned with bottom side
    for x in range(0, temp_section.shape[1] - input_dim, int(output_dim/overlap_var)):
        coords.append((0, x, temp_section.shape[2]-input_dim))

    # Add right side aligned with right side
    for y in range(0, temp_section.shape[2] - input_dim, int(output_dim/overlap_var)):
        coords.append((0, temp_section.shape[1]-input_dim, y))

    # Add bottom right corner
    coords.append((0, temp_section.shape[1]-input_dim, temp_section.shape[2]-input_dim))

    coords = np.array(coords)
    # List of cropped volumes that the network will process
    batch_crops = np.zeros((batch_size, input_dim, input_dim, input_dim))
    # List of coordinates associated with each cropped volume
    batch_coords = np.zeros((batch_size, 3), dtype="int")

    # Keeps track of which coord we are at
    i = 0

    # Generate dummy segmentation
    seg = np.zeros(temp_section.shape).astype('float32')

    # Loop through each possible coordinate
    while i < len(coords):

        # Fill up the batch by skipping chunks below the threshold
        batch_count = 0
        while i < len(coords) and batch_count < batch_size:
            (z, x, y) = coords[i]

            # Get the chunk associated with the coordinate
            test_crop = temp_section[z:z + input_dim, x:x + input_dim, y:y + input_dim]

            # Only add chunk to batch if its max value is above threshold
            # (Avoid wasting time processing background chunks)
            if np.max(test_crop) > threshold:
                batch_coords[batch_count] = (z, x, y)
                batch_crops[batch_count] = test_crop
                batch_count += 1
            i += 1

        # Once the batch is filled up run the network on the chunks
        batch_input = np.reshape(batch_crops, batch_crops.shape + (1,))

        output = np.squeeze(model.predict(batch_input)[:, :, :, :, [0]])

        
        # Place the predictions in the segmentation
        for j in range(len(batch_coords)):
            (z, x, y) = batch_coords[j] + dim_offset
            #print(z,x,y)

            # multiple by guassian_importance map
            output[j] *=  gaussian_importance_map
            seg[z:z + output_dim, x:x + output_dim, y:y + output_dim] =+ output[j]
            #keep track of total sum of guassian_importance_map to normalize
            # https://github.com/MIC-DKFZ/nnUNet/blob/6d02b5a4e2a7eae14361cde9599bbf4ccde2cd37/nnunet/network_architecture/neural_network.py#L394
            aggregated_nb_of_predictions[z:z + output_dim, x:x + output_dim, y:y + output_dim]  += gaussian_importance_map
            aggregated_ones[z:z + output_dim, x:x + output_dim, y:y + output_dim]  += np.ones_like(gaussian_importance_map)
            #print(np.unique(aggregated_ones, return_counts = True))



    
    aggregated_nb_of_predictions_cropped = aggregated_nb_of_predictions[:, dim_offset: dim_offset + section.shape[1], dim_offset: dim_offset + section.shape[2]]
    
    aggregated_ones_cropped = aggregated_ones[:, dim_offset: dim_offset + section.shape[1], dim_offset: dim_offset + section.shape[2]]
    cropped_seg = seg[:, dim_offset: dim_offset + section.shape[1], dim_offset: dim_offset + section.shape[2]]
    orig_seg_cropped = seg[:, dim_offset: dim_offset + section.shape[1], dim_offset: dim_offset + section.shape[2]]
    # cropped_seg /= aggregated_nb_of_predictions_cropped 
    return orig_seg_cropped, aggregated_nb_of_predictions_cropped, aggregated_ones_cropped 


