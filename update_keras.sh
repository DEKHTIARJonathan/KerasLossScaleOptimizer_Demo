#!/usr/bin/env bash

{ # try =>  TF2.x Container

    cp \
        tensorflow_sources/loss_scale_optimizer.py \
        /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/keras/mixed_precision/experimental/  2>/dev/null

    cp \
        tensorflow_sources/loss_scale.py \
        /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/keras/mixed_precision/experimental/  2>/dev/null

    cp \
        tensorflow_sources/optimizers.py \
        /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/keras/  2>/dev/null

} || { # catch =>  TF1.x Container

    cp \
        tensorflow_sources/loss_scale_optimizer.py \
        /usr/local/lib/python3.6/dist-packages/tensorflow/python/keras/mixed_precision/experimental/

    cp \
        tensorflow_sources/loss_scale.py \
        /usr/local/lib/python3.6/dist-packages/tensorflow/python/keras/mixed_precision/experimental/  2>/dev/null

    cp \
        tensorflow_sources/optimizers.py \
        /usr/local/lib/python3.6/dist-packages/tensorflow/python/keras/

}
