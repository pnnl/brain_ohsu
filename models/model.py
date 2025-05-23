import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, BatchNormalization, Conv3DTranspose, concatenate, \
    Cropping3D, Input
from tensorflow.keras.optimizers import Adam
# from keras.callbacks import ReduceLROnPlateau

input_dim = 64
output_dim = 36


def create_weighted_binary_crossentropy(axon_weight, background_weight, artifact_weight, edge_weight):
    def weighted_binary_crossentropy(y_true, y_pred):
        weights = tf.reduce_sum(y_true, axis=-1, keepdims=True)

        mask = tf.equal(weights, 1)

        axon_true = y_true[:, :, :, :, 0]
        axon_true = tf.expand_dims(axon_true, -1)
        axon_mask = tf.boolean_mask(axon_true, mask)

        background_true = y_true[:, :, :, :, 1]
        background_true = tf.expand_dims(background_true, -1)
        background_mask = tf.boolean_mask(background_true, mask)

        artifact_true = y_true[:, :, :, :, 2]
        artifact_true = tf.expand_dims(artifact_true, -1)
        artifact_mask = tf.boolean_mask(artifact_true, mask)

        edge_true = y_true[:, :, :, :, 3]
        edge_true = tf.expand_dims(edge_true, -1)
        edge_mask = tf.boolean_mask(edge_true, mask)

        mask_true = tf.boolean_mask(axon_true, mask)
        mask_pred = tf.boolean_mask(y_pred, mask)

        crossentropy = K.binary_crossentropy(mask_true, mask_pred)

        weight_vector = (axon_mask * axon_weight) + (background_mask * background_weight) + \
                        (artifact_mask * artifact_weight) + (edge_mask * edge_weight)

        weighted_crossentropy = weight_vector * crossentropy

        return K.mean(weighted_crossentropy)

    return weighted_binary_crossentropy


def weighted_binary_crossentropy(y_true, y_pred):
    loss = create_weighted_binary_crossentropy(1.5,1.0, 0.8, 0.05)(y_true, y_pred)
    return loss


def adjusted_accuracy(y_true, y_pred):
    #combine by taking sum accross label type (axons, artifacts, edge)
    #
    weights = tf.reduce_sum(y_true, axis=-1, keepdims=True)
    #mask where something is classified (background is not just 0, it's it's own catagory)
    mask = K.equal(weights, 1)

    # ones only by axons, not the other catagories
    axons_true = y_true[:, :, :, :, 0]
    axons_true = K.expand_dims(axons_true, -1)

    # get true and prediced axons over all labeled catagories
    mask_true = tf.cast(tf.boolean_mask(axons_true, mask), tf.float32)
    mask_pred = tf.cast(tf.boolean_mask(y_pred, mask), tf.float32)
    # anything predicted over .5 is good
    return K.mean(K.equal(mask_true, K.round(mask_pred)))


def axon_precision(y_true, y_pred):
    weights = tf.reduce_sum(y_true, axis=-1)

    mask = tf.equal(weights, 1)

    #will return list only including items that were true axons
    mask_true = tf.cast(tf.boolean_mask(y_true[:, :, :, :, 0], mask), tf.float32)
    mask_pred = tf.cast(tf.boolean_mask(y_pred[:, :, :, :, 0], mask), tf.float32)

    true_positives = K.sum(K.round(K.clip(mask_true * mask_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(mask_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision


def axon_recall(y_true, y_pred):
    weights = tf.reduce_sum(y_true, axis=-1)

    mask = tf.equal(weights, 1)

    mask_true = tf.cast(tf.boolean_mask(y_true[:, :, :, :, 0], mask), tf.float32)
    mask_pred = tf.cast(tf.boolean_mask(y_pred[:, :, :, :, 0], mask), tf.float32)

    true_positives = K.sum(K.round(K.clip(mask_true * mask_pred, 0, 1)))
    actual_positives = K.sum(K.round(K.clip(mask_true, 0, 1)))
    recall = true_positives / (actual_positives + K.epsilon())

    return recall


def artifact_precision(y_true, y_pred):
    weights = y_true[:, :, :, :, 2]

    mask = tf.equal(weights, 1)
    mask_true = tf.boolean_mask(y_true[:, :, :, :, 2], mask)
    mask_pred = tf.boolean_mask(1 - y_pred[:, :, :, :, 0], mask)

    true_positives = K.sum(K.round(K.clip(mask_true * mask_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(mask_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision

def f1_score(y_true, y_pred):

    precision = axon_precision(y_true, y_pred)
    recall = axon_recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def edge_axon_precision(y_true, y_pred):
    weights = tf.reduce_sum(y_true, axis=-1)

    mask = tf.equal(weights, 1)

    mask_true = tf.cast(tf.boolean_mask(y_true[:, :, :, :, 0], mask), tf.float32)
    mask_pred = tf.cast(tf.boolean_mask(y_pred[:, :, :, :, 0], mask), tf.float32)
    mask_edge_true = tf.cast(tf.boolean_mask(y_true[:, :, :, :, 3], mask), tf.float32)

    true_positives = K.sum(K.round(K.clip(mask_true * mask_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(mask_pred, 0, 1)))

    edge_count = K.sum(K.round(K.clip(mask_edge_true * mask_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon() - edge_count)

    return precision

def edge_f1_score(y_true, y_pred):

    precision = edge_axon_precision(y_true, y_pred)
    recall = axon_recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def get_net(encode_layer="full_layer", start_loss=0.001):
    # Level 1
    input = Input((input_dim, input_dim, input_dim, 1))
    conv1 = Conv3D(32, (3, 3, 3), activation="relu", padding="same")(input)
    batch1 = BatchNormalization()(conv1)
    conv1 = Conv3D(64, (3, 3, 3), activation="relu", padding="same")(batch1)
    batch1 = BatchNormalization()(conv1)

    # Level 2
    pool2 = MaxPooling3D((2, 2, 2))(batch1)
    conv2 = Conv3D(64, (3, 3, 3), activation="relu", padding="same")(pool2)
    batch2 = BatchNormalization()(conv2)
    conv2 = Conv3D(128, (3, 3, 3), activation="relu", padding="same")(batch2)
    batch2 = BatchNormalization()(conv2)

    # Level 3
    pool3 = MaxPooling3D((2, 2, 2))(batch2)
    conv3 = Conv3D(128, (3, 3, 3), activation="relu", padding="same")(pool3)
    batch3 = BatchNormalization()(conv3)
    conv3 = Conv3D(256, (3, 3, 3), activation="relu", padding="same")(batch3)
    batch3 = BatchNormalization()(conv3)

    # Level 4
    pool4 = MaxPooling3D((2, 2, 2))(batch3)
    conv4 = Conv3D(256, (3, 3, 3), activation="relu", padding="same")(pool4)
    batch4 = BatchNormalization()(conv4)
    conv4 = Conv3D(512, (3, 3, 3), activation="relu", padding="same")(batch4)
    batch4 = BatchNormalization()(conv4)

    # Level 3
    up5 = Conv3DTranspose(512, (2, 2, 2), strides=(2, 2, 2), padding="same", activation="relu")(batch4)
    merge5 = concatenate([up5, batch3])
    conv5 = Conv3D(256, (3, 3, 3), activation="relu")(merge5)
    batch5 = BatchNormalization()(conv5)
    conv5 = Conv3D(256, (3, 3, 3), activation="relu")(batch5)
    batch5 = BatchNormalization()(conv5)

    # Level 2
    up6 = Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), activation="relu")(batch5)
    merge6 = concatenate([up6, Cropping3D(cropping=((4, 4), (4, 4), (4, 4)))(batch2)])
    conv6 = Conv3D(128, (3, 3, 3), activation="relu")(merge6)
    batch6 = BatchNormalization()(conv6)
    conv6 = Conv3D(128, (3, 3, 3), activation="relu")(batch6)
    batch6 = BatchNormalization()(conv6)

    # Level 1
    up7 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding="same", activation="relu")(batch6)
    merge7 = concatenate([up7, Cropping3D(cropping=((12, 12), (12, 12), (12, 12)))(batch1)])
    conv7 = Conv3D(64, (3, 3, 3), activation="relu")(merge7)
    batch7 = BatchNormalization()(conv7)
    conv7 = Conv3D(64, (3, 3, 3), activation="relu")(batch7)
    batch7 = BatchNormalization()(conv7)

    preds = Conv3D(1, (1, 1, 1), activation="sigmoid")(batch7)
    model = Model(inputs=input, outputs=preds)


    if encode_layer == "full_layer":
        pass
    else:
        for layer in model.layers:
            if layer.name in  [['conv3d', 'conv3d_1'], ['conv3d_6', 'conv3d_7'], ['conv3d_13', 'conv3d_14']][int(encode_layer)]:
                layer.trainable = True
                print(layer.name)
                print(layer.trainable)
            else:
                layer.trainable = False
    print("loss used")
    print(float(start_loss))
    print("encode")
    print(encode_layer)
    model.compile(optimizer=Adam(lr=float(start_loss)), loss=weighted_binary_crossentropy,
                  metrics=[axon_precision, axon_recall, f1_score, artifact_precision, edge_axon_precision, adjusted_accuracy, edge_f1_score], run_eagerly=True)

    return model
