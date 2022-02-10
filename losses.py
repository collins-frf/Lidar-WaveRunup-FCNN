# -*- coding:utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
import settings


def absolute_error(y_true, y_pred):
    error = y_pred - y_true
    abs_error = tf.math.abs(error)
    ########################################
    temp_y_pred = tf.expand_dims(y_pred, axis=-1)
    temp_y_pred = tf.image.resize(temp_y_pred, [32, 32], method="area")
    continuity_error = tf.Variable(tf.zeros([1, 32, 32, 1]))
    plot = False
    for i in range(32):
        left_value = 0.0
        left_disc_count = 0.0
        first_disc = 0.0
        last_disc = 0.0
        for j in range(32):
            if temp_y_pred[0][i, j][0] > left_value:
                left_value = temp_y_pred[0][i, j][0]
            if (left_value > .5) and (temp_y_pred[0][i, j-1] < .5):
                if first_disc == 0.0:
                    first_disc = j
                else:
                    last_disc = j
                left_disc_count+=1.0
        if left_disc_count > 1:
            continuity_error[0, i, first_disc:last_disc, 0].assign(tf.ones(last_disc-first_disc))
            #plot = True
    final_cont_error = tf.image.resize(continuity_error, [512, 512], method="bicubic")
    final_error = abs_error + final_cont_error[:, :, :, 0]
    if plot == True:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 3, 1), plt.imshow(abs_error[0])
        ax2 = fig.add_subplot(1, 3, 2), plt.imshow(final_cont_error[0, :, :, 0])
        ax3 = fig.add_subplot(1, 3, 3), plt.imshow(final_error[0])
        plt.show()

    return tf.reduce_mean(final_error)
    loss.__name__ = "absolute_error"

def absolute_error_grad(y_true, y_pred):
    error = y_pred - y_true
    abs_error = tf.math.abs(error)
    ########################################
    temp_y_pred = tf.expand_dims(y_pred, axis=-1)
    temp_y_pred = tf.image.resize(temp_y_pred, [32, 32], method="area")
    continuity_error = tf.Variable(tf.zeros([settings.batch_size, 32, 32, 1]))
    plot = False
    y_pred_grad_y, y_pred_grad_x = tf.image.image_gradients(temp_y_pred)
    where_negative_x = tf.where(y_pred_grad_x<0.0, x=1.0, y=0.0)
    continuity_error.assign(where_negative_x)
    final_cont_error = tf.image.resize(continuity_error, [512, 512], method="bicubic")
    final_error = abs_error + final_cont_error[:, :, :, 0]
    if plot == True:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 3, 1), plt.imshow(abs_error[0])
        ax2 = fig.add_subplot(1, 3, 2), plt.imshow(final_cont_error[0, :, :, 0])
        ax3 = fig.add_subplot(1, 3, 3), plt.imshow(final_error[0])
        plt.show()

    return tf.reduce_mean(final_error)
    loss.__name__ = "absolute_error"