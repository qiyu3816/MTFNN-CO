import tensorflow as tf
import tensorflow.keras.backend as K

def custom_loss(y_true, y_pred):
    # mse
    mse = K.mean(K.square(y_true - y_pred), axis=-1)

    # classification penalty
    true_vals = tf.unstack(y_true, axis=-1)
    pred_vals = tf.unstack(y_pred, axis=-1)
    classification_loss = 0
    for t, p in zip(true_vals, pred_vals):
        classification_loss += tf.where(tf.math.less(t * 10, 1), tf.math.minimum(0.1 - p, 0), 0) * (-0.1)

    # loss that penalizes differences between sum(predictions) and sum(labels)
    sum_constraint = K.square(K.sum(y_pred, axis=-1) - K.sum(y_true, axis=-1))

    return (mse + classification_loss + sum_constraint)

def mtl_custom_loss(y_true, y_pred):
    # mse
    mse = K.mean(K.square(y_true - y_pred), axis=-1)

    # loss that penalizes differences between sum(predictions) and sum(labels)
    sum_constraint = K.square(K.sum(y_pred, axis=-1) - K.sum(y_true, axis=-1))

    return (mse + sum_constraint)