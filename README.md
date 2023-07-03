
# TRAILMAP MODIFICATION

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
* The number of examples should be set directly within prepare_data with the nb_examples variable or as multiple of tif files in line 88 in generate_data_set.py
* The boolean argument indicates if the training should use normal setting or the settings adopted from nnU-net. 

```
python3 prepare_data.py "generate_training_set" True
python3 prepare_data.py "generate_validation_set" True
```

For training:
* The first argument is a boolean to indicate if the training should use normal setting or the settings adopted from nnU-net.
* The second argument indicates the name of the model.
* The location of the training data is set directly with the train.py file at line 21 and 22. 

```
python3 train.py False "name_of_model"
```


## Authors

Marjolein Oostrom

The work is adapted from code created by: 
* **Albert Pun**
* **Drew Friedmann**


## Disclaimer

This material was prepared as an account of work sponsored by an agency of the United States Government.  Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.
Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.
PACIFIC NORTHWEST NATIONAL LABORATORY
operated by
BATTELLE
for the
UNITED STATES DEPARTMENT OF ENERGY
under Contract DE-AC05-76RL01830


## Simplified BSD
____________________________________________
Copyright 2023 Battelle Memorial Institute

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Acknowledgments
The work is adapted from work sponsored by: 
* Research sponsored by Liqun Luo's Lab

