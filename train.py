from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from models.model import get_net
import tensorflow as tf
import os
from training import load_data, VolumeDataGenerator
from models import input_dim
from utilities.utilities import *

if __name__ == "__main__":

    base_path = os.path.abspath(__file__ + "/..")
    combo_number = int(sys.argv[-1])
    name_model = sys.argv[1]
    batch_size = 6
    epochs = 100
    print(name_model)
    combo = [
        # no oversampling, no rotation, no learn scheduler, flip, elastic deformation percentage, rotate deformation percentage, /
        # layer settting (set in model.py), learning rate (set in model.py), model name suffix
        (False, False, False, True, 1.0, 1.0, "last_layer", .0001, "_val_1_test_5"),
        (False, False, False, True, 1.0, 1.0, "last_layer", .0001, "_val_2_test_1"),
    ]

    oversample_bol, aug_bol, lr_bol, flip_bol, el_percentage, rot_percentage, encode_train, loss_start, training_data = combo[combo_number]
    name_model = f'oversample_bol_{oversample_bol}_aug_bol_{aug_bol}_lr_bol_{lr_bol}_flip_bol_{flip_bol}_el_{el_percentage}_rot_{rot_percentage}__encode_{encode_train}_loss_{loss_start}_training_{training_data}_{name_model}'
    print(name_model)

    training_path = base_path + f"/data/training/training-set_normal_{oversample_bol}{training_data}"
    validation_path = base_path + f"/data/validation/validation-set_normal_True{training_data}"

    print(training_path)
    print(validation_path)
    
    # load data needs to correspond to volumne generator
    x_train, y_train = load_data(training_path, normal = aug_bol)
    x_validation, y_validation = load_data(validation_path, normal = True)


    datagen = VolumeDataGenerator(
        horizontal_flip=flip_bol,
        vertical_flip=flip_bol,
        depth_flip=flip_bol,
        min_max_normalization=True,
        scale_range=0.1,
        scale_constant_range=0.2,
        normal =aug_bol,
         el_precentage = el_percentage,
         rot_precentage = rot_percentage
    )

    datagen_val = VolumeDataGenerator(
        horizontal_flip=False,
        vertical_flip=False,
        depth_flip=False,
        min_max_normalization=True,
        scale_range=0,
        scale_constant_range=0,
        normal = True
    )



    train_generator = datagen.flow(x_train, y_train, batch_size)
    validation_generator =  datagen_val.flow(x_validation, y_validation, batch_size)

    now = datetime.now()
    logdir = base_path + f"/data/tf-logs/{name_model}" +  now.strftime("%B-%d-%Y-%I:%M%p") + "/"

    tboard = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=False)

    current_checkpoint = ModelCheckpoint(filepath=base_path + f'/data/model-weights/latest_model_{epochs:03d}_{name_model}.hdf5', verbose=1)
    period_checkpoint = ModelCheckpoint(base_path + f'/data/model-weights/weights_{epochs:03d}_{name_model}.hdf5', period=20)
    best_weight_checkpoint = ModelCheckpoint(filepath=base_path + f'/data/model-weights/best_weights_checkpoint_{name_model}.hdf5',
                                             verbose=1, save_best_only=True)

    

    weights_path = base_path + "/data/model-weights/trailmap_model.hdf5"
    print(weights_path)

    model = get_net()
    # This will do transfer learning and start the model off with our current best model.
    # Remove the model.load_weight line below if you want to train from scratch
    model.load_weights(weights_path)
    if lr_bol == False:
        lr_scheduler = ReduceLROnPlateau()
        # use more steps in the epochs
        #https://stackoverflow.com/questions/39779710/setting-up-a-learningratescheduler-in-keras
        model.fit_generator(train_generator,
                            steps_per_epoch=700//batch_size,
                            epochs=epochs,
                            validation_data=validation_generator,
                            validation_steps=100//batch_size,
                            use_multiprocessing=False,
                            workers=1,
                            callbacks=[lr_scheduler, tboard, period_checkpoint, best_weight_checkpoint],
                            verbose=1)
        
    else:

        #https://stackoverflow.com/questions/39779710/setting-up-a-learningratescheduler-in-keras
        model.fit_generator(train_generator,
                            steps_per_epoch=700//batch_size,
                            epochs=epochs,
                            validation_data=validation_generator,
                            validation_steps=100//batch_size,
                            use_multiprocessing=False,
                            workers=1,
                            callbacks=[tboard, period_checkpoint, best_weight_checkpoint],
                            verbose=1)

    model_name = 'model_' + now.strftime("%B-%d-%Y-%I:%M%p")

