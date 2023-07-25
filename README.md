
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
* Add the path to the data (with label and volume folders) for the image_path variable. This path can include an suffix variable. This script will use labels to calculate performance metrics for inference.
* The boolean argument indicates if you want to use guassian inference
* The string argument indicates training/validation/test data division suffix (leave '' if not using suffix)
* Use the combination number of the desired model_weight and image_path combination as an argument for segment_brain_batch.py.


```
python3 segment_brain_batch.py True "_test_1" {combination_number}

```

## Training

Please follow the instructions at [Github TrailMap](https://github.com/AlbertPun/TRAILMAP) with these modifications. 
* When preparing the data for training, the location of the  data is set directly with the prepare_data.py file at data_original_path (input) and data_set_path (output) for each of the functions.
* The data_set_path should match the name provided as the training data under train.py (training_path and validation_path).
* The number of examples should be set directly within prepare_data with the nb_examples variable.
* The second string argument indicates training/validation/test data division suffix (leave '' if not using suffix). 
* The boolean argument indicates if the training should oversample 

```
python3 prepare_data.py "generate_validation_set" "_val_2_test_1" True
python3 prepare_data.py "generate_training_set"  "_val_2_test_1" True

```

For training:

* The first string indicates the model name suffix. 
* The combination_number argument indicates the index of the combination of arguments set by variable combo under train.py
* The location of the training data is set directly with the train.py file  (training_path and validation_path) 
* The positive booleans under the argument list under variable combo indicate 1) no oversampling, 2) no rotation, 3) no learn scheduler, 4) flip, 5) elastic deformation percentage, 6) rotate deformation percentage, 7) layer settting (needs to be set in model.py), 8) learning rate (needs to be set in model.py), and 9) training/validation/test data division suffix (leave '' if not using suffix)

```
python3 train.py "_july23_test" {combination_number}
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

