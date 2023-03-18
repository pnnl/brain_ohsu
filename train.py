from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from models.model import get_net
import tensorflow as tf
import os
from training import load_data, VolumeDataGenerator
from models import input_dim

if __name__ == "__main__":

    base_path = os.path.abspath(__file__ + "/..")
    normal = False
    batch_size = 6
    epochs = 50

    training_path = base_path + "/data/training/training-set"
    validation_path = base_path + "/data/validation/validation-set"
    test_path = base_path + "/data/test/test-set"

    x_train, y_train = load_data(training_path, normal = normal)
    x_validation, y_validation = load_data(validation_path, normal = True)
    x_test, y_test = load_data(test_path, normal = True)


    datagen = VolumeDataGenerator(
        horizontal_flip=False,
        vertical_flip=False,
        depth_flip=False,
        min_max_normalization=True,
        scale_range=0.1,
        scale_constant_range=0.2,
        normal = normal
    )

    datagen_no_flip = VolumeDataGenerator(
        horizontal_flip=False,
        vertical_flip=False,
        depth_flip=False,
        min_max_normalization=True,
        normal = True
    )


    train_generator = datagen.flow(x_train, y_train, batch_size)
    validation_generator = datagen_no_flip.flow(x_validation, y_validation, batch_size)
    test_generator = datagen_no_flip.flow(x_test, y_test, batch_size)

    now = datetime.now()
    logdir = base_path + "/data/tf-logs/" + now.strftime("%B-%d-%Y-%I:%M%p") + "/"

    tboard = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=False)

    current_checkpoint = ModelCheckpoint(filepath=base_path + '/data/model-weights/latest_model_false.hdf5', verbose=1)
    period_checkpoint = ModelCheckpoint(base_path + '/data/model-weights/weights{epochs:03d}_false.hdf5', period=20)
    best_weight_checkpoint = ModelCheckpoint(filepath=base_path + f'/data/model-weights/best_weights_checkpoint_normal_false.hdf5',
                                             verbose=1, save_best_only=True)

    

    weights_path = base_path + "/data/model-weights/original_best_weights.hdf5"

    model = get_net()
    # This will do transfer learning and start the model off with our current best model.
    # Remove the model.load_weight line below if you want to train from scratch
    model.load_weights(weights_path)
    if normal == False:
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                patience=5, min_lr=0.001)

        #https://stackoverflow.com/questions/39779710/setting-up-a-learningratescheduler-in-keras
        model.fit_generator(train_generator,
                            steps_per_epoch=120,
                            epochs=epochs,
                            validation_data=validation_generator,
                            validation_steps=30,
                            use_multiprocessing=False,
                            workers=1,
                            callbacks=[lr_scheduler, tboard, best_weight_checkpoint],
                            verbose=1)
        
    else:

        #https://stackoverflow.com/questions/39779710/setting-up-a-learningratescheduler-in-keras
        model.fit_generator(train_generator,
                            steps_per_epoch=120,
                            epochs=epochs,
                            validation_data=validation_generator,
                            validation_steps=30,
                            use_multiprocessing=False,
                            workers=1,
                            callbacks=[tboard,  best_weight_checkpoint],
                            verbose=1)

    model_name = 'model_' + now.strftime("%B-%d-%Y-%I:%M%p")
    # train test read_tiff_stack, get newest model, rename best_training for augmentation or not, add precision measurements, check aug variable
    scores = model.evaluate_generator(test_generator,
                            callbacks=[tboard,  best_weight_checkpoint],
                            verbose=1)