#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import pprint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# import tensorflow.compat.v2 as tf

# tf.enable_v2_behavior()

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import backend as K

from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import UpSampling1D

import pandas as pd

#### -------------------------- AMP -------------------------- ####
tf.keras.mixed_precision.experimental.set_policy('infer_float32_vars')
#### -------------------------- AMP -------------------------- ####

EPOCH_NUM = 5
BATCH_SIZE = 512
LEARNING_RATE = 0.0001

data_dir = './'

df_train_conv = pd.read_pickle(data_dir + 'df_train_conv.pkl')
df_eval_conv = pd.read_pickle(data_dir + 'df_validation_conv.pkl')

data_shape_train = df_train_conv.shape[1:]
data_shape_eval = df_eval_conv.shape[1:]

print("Input Train Shape: %s" % str(data_shape_train))
print("Input Train DType: %s\n" % type(df_train_conv))

print("Input Eval Shape: %s" % str(data_shape_train))
print("Input Eval DType: %s\n" % type(df_train_conv))

print("Input Train Samples: %d" % df_train_conv.shape[0])
print("Input Eval Samples: %d\n" % df_eval_conv.shape[0])

## Some parameters to change
top_filters = 40
middle_filters = 20
kernel_size_val = 3
reg_strength = 0.01

inputs = Input(shape=data_shape_train)
encoded = Conv1D(
    filters=top_filters,
    kernel_size=kernel_size_val,
    strides=1,
    padding='same',
    activation='relu',
    kernel_regularizer=regularizers.l1(reg_strength)
)(inputs)

for _ in range(2):
    encoded = MaxPooling1D(2, padding='same')(encoded)
    encoded = Conv1D(
        filters=middle_filters,
        kernel_size=kernel_size_val,
        strides=1,
        padding='same',
        activation='relu',
        kernel_regularizer=regularizers.l1(reg_strength)
    )(encoded)

decoded = Conv1D(
    filters=middle_filters,
    kernel_size=kernel_size_val,
    strides=1,
    padding='same',
    activation='relu',
    kernel_regularizer=regularizers.l1(reg_strength)
)(encoded)

decoded = UpSampling1D(2)(decoded)
decoded = Conv1D(
    filters=middle_filters,
    kernel_size=kernel_size_val,
    strides=1,
    padding='same',
    activation='relu',
    kernel_regularizer=regularizers.l1(reg_strength)
)(decoded)

decoded = UpSampling1D(2)(decoded)
decoded = Conv1D(
    filters=data_shape_train[1],
    kernel_size=kernel_size_val,
    strides=1,
    padding='same',
    activation='relu',
    kernel_regularizer=regularizers.l1(reg_strength)
)(decoded)

# ======================== LOSS SCALING ======================== #
# Instructions: Comment/Uncomment the loss scaler you want to test

# loss_scaler = tf.train.experimental.FixedLossScale(32768)

# loss_scaler = tf.train.experimental.DynamicLossScale(
#     initial_loss_scale=(2 ** 15),
#     increment_period=2000,
#     multiplier=2.0
# )

loss_scaler = "dynamic"

# ============================================================== #

optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE)
optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt=optimizer, loss_scale=loss_scaler)

print('Optimizer Configuration:')

print('\nOptimizer `get_slot_names()`: %s\n' % str(optimizer.get_slot_names()))

autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer=optimizer, loss='mae', metrics=['mae'])

# ============ Test Model Recreation from config ============ #
model_config = autoencoder.get_config()
autoencoder = Model.from_config(model_config)
autoencoder.compile(optimizer=optimizer, loss='mae', metrics=['mae'], run_eagerly=False)

# ============ Test Model Summary ============ #
autoencoder.summary()

from callback import ProgbarLogger

progbar_callback = ProgbarLogger(count_mode='samples', stateful_metrics=['mae'])

print("\nEvaluation Before training - At Initialization")
autoencoder.evaluate(x=df_eval_conv, y=df_eval_conv, batch_size=BATCH_SIZE, verbose=0, callbacks=[progbar_callback])
print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

for epoch in range(4):
    print("\nTraining Epoch: %d/4" % (epoch + 1))

    print("[*] Optimizer `learning_rate`: %f" % K.eval(optimizer.learning_rate))
    print("[*] Optimizer `lr`: %f" % K.eval(optimizer.lr))

    # print("\n[*] Optimizer `loss_scale_value`: %f" % K.eval(optimizer.loss_scale_value))
    #
    # try:
    #     print("[*] Optimizer `loss_scale_increment_period`: %f" % K.eval(optimizer.loss_scale_increment_period))
    #     print("[*] Optimizer `loss_scale_multiplier`: %f" % K.eval(optimizer.loss_scale_multiplier))
    #     print("[*] Optimizer `num_good_steps`: %f" % K.eval(optimizer.num_good_steps))
    # except ValueError:
    #     if not isinstance(loss_scaler, tf.train.experimental.DynamicLossScale):
    #         pass
    #     else:
    #         raise

    print('\n[*] Optimizer `variables()`', [var.name for var in optimizer.variables()])
    print('\n[*] Optimizer `get_weights()`', [weight.shape for weight in optimizer.get_weights()])
    print('\n[*] Optimizer `weights`', [weight.name for weight in optimizer.weights])
    print()

    autoencoder.fit(x=df_train_conv, y=df_train_conv, batch_size=BATCH_SIZE, verbose=0, callbacks=[progbar_callback])

    print("\nInference Epoch: %d/4" % (epoch + 1))
    autoencoder.evaluate(x=df_eval_conv, y=df_eval_conv, batch_size=BATCH_SIZE, verbose=0, callbacks=[progbar_callback])

    optimizer.lr = optimizer.lr / 2
    optimizer.learning_rate = optimizer.learning_rate / 2

    # Testing Model Saving
    print("\n\nSaving Model ...")
    autoencoder.save('my_model.h5')

    # Testing Model Saving
    print("Restoring Model ...")
    autoencoder = keras.models.load_model('my_model.h5')

    print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
