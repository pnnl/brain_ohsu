
# TRAILMAP

This a modification of the code described in  [Mapping Mesoscale Axonal Projections in the Mouse Brain Using A 3D Convolutional Network](https://www.biorxiv.org/content/10.1101/812644v1.full) Friedmann D, Pun A, et al and located at [Github TrailMap](https://github.com/AlbertPun/TRAILMAP)

## Getting Started - Installation

Please follow the detailed instruction on [Github TrailMap](https://github.com/AlbertPun/TRAILMAP) and also pip install batchgenerators

```
pip install batchgenerators==0.25
```

## Inference

Please follow the instructions at [Github TrailMap](https://github.com/AlbertPun/TRAILMAP) with these modifications. 
* Add model weights path to segment_brain_batch.py at the model_weight_list variable.
* Add the path to the data (with label and volume folders) for the image_path variable. This script will use labels to calculate performance metrics for inference.
* Use the combination number of the desired model_weight and data combination as an argument for segment_brain_batch.py.
* If you wish to use guassian inference, set the variable guass to True under segment_brain.py. 

```
python3 segment_brain_batch.py {combination_number}

```

## Training

Please follow the instructions at [Github TrailMap](https://github.com/AlbertPun/TRAILMAP) with these modifications. 
* When preparing the data for training, the location of the  data is set directly with the prepare_data.py file at data_original_path and data_set_path for each of the functions.
* The data_set_path should match the name provided as the training data under train.py at line 21 and 22.
* The number of examples should be set directly within prepare_data with the nb_examples variable.
* The boolean argument indicates if the training should use normal setting or the settings adopted from nnU-net. 

```
python3 prepare_data.py "generate_training_set" True
python3 prepare_data.py "generate_validation_set" True
```

For training:
* The first argument is a boolean to indicate if the training should use normal setting or the settings adopted from nnU-net.
* The second argument indicates the name of the model. T
* The location of the training data is set directly with the train.py file at line 21 and 22. 

```
python3 train.py False "name_of_model"
```


## Authors
The work is adapted from code created by: 
* **Albert Pun**
* **Drew Friedmann**

## License
This work is adapted from a project licensed under the MIT License

## Acknowledgments
The work is adapted from work sponsored by: 
* Research sponsored by Liqun Luo's Lab

