# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the loss scaling optimizer class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from six import add_metaclass

from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.framework import smart_cond
from tensorflow.python.keras import backend
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.utils.generic_utils import _GLOBAL_CUSTOM_OBJECTS
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training.experimental import loss_scale as loss_scale_module
from tensorflow.python.util.tf_export import keras_export


class _UnwrapPreventer(object):
    """Wrapper that DistributionStrategy will not unwrap.

    Typically, DistributionStrategy will unwrap values when going from a cross-
    replica context to a replica context via `call_for_each_replica`. This class
    is a wrapper that DistributionStrategy will not unwrap, so it can be used to
    prevent it from unwrapping a value.

    TODO(reedwm): Find/implement a better way of preventing values from being
    unwrapped by DistributionStrategy
    """

    def __init__(self, value):
        self.value = value


class OptimizerV2MetaFactory(type):

    def __call__(cls, opt, *args, **kwargs):

        if not isinstance(opt, optimizer_v2.OptimizerV2):
            raise ValueError('"opt" must be an instance of OptimizerV2, but got: %s'
                             % opt)

        if hasattr(opt, 'clipnorm'):
            raise ValueError('LossScaleOptimizer does not support wrapping '
                             'optimizers with a clipnorm. Optimizer %s has clipnorm '
                             '%s' % (opt, opt.clipnorm))

        if hasattr(opt, 'clipvalue'):
            raise ValueError('LossScaleOptimizer does not support wrapping '
                             'optimizers with a clipvalue. Optimizer %s has '
                             'clipvalue %s' % (opt, opt.clipvalue))

        return cls.__call__(opt=opt, *args, **kwargs)


@keras_export('keras.optimizers.LossScaleOptimizer')
@add_metaclass(OptimizerV2MetaFactory)
class LossScaleOptimizer(object):

    """
    Loss scaling is a process that multiplies the loss by a multiplier called the
    loss scale, and divides each gradient by the same multiplier. The pseudocode
    for this process is:

    ```
    loss = ...
    loss *= loss_scale
    grads = gradients(loss, vars)
    grads /= loss_scale
    ```

    Mathematically, loss scaling has no effect, but can help avoid numerical
    underflow in intermediate gradients when float16 tensors are used. By
    multiplying the loss, each intermediate gradient will have the same multiplier
    applied.

    The loss scale can either be a fixed constant, chosen by the user, or be
    dynamically determined. Dynamically determining the loss scale is convenient
    as a loss scale does not have to be explicitly chosen. However it reduces
    performance.

    This optimizer wraps another optimizer and applies loss scaling to it via a
    `LossScale`. Loss scaling is applied whenever gradients are
    computed, either through `minimize()` or `get_gradients()`. The loss scale is
    updated via `LossScale.update()` whenever gradients are applied, either
    through `minimize()` or `apply_gradients()`. For example:

    ```python
    opt = tf.keras.optimizers.SGD(0.1)
    opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")
    # 'minimize' applies loss scaling to the loss and updates the loss sale.
    opt.minimize(loss_fn)
    ```

    If a `tf.GradientTape` is used to compute gradients instead of
    `LossScaleOptimizer.minimize` or `LossScaleOptimizer.get_gradients`, the loss
    and gradients must be scaled manually. This can be done by calling
    `LossScaleOptimizer.get_scaled_loss` before passing the loss to
    `tf.GradientTape`, and `LossScaleOptimizer.get_unscaled_gradients` after
    computing the gradients with `tf.GradientTape`. For example:

    ```python
    opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(...)
    vars = ...
    with tf.GradientTape() as tape:
      loss = ...
      scaled_loss = opt.get_scaled_loss(loss)
    scaled_grads = tape.gradient(scaled_loss, vars)
    grads = opt.get_unscaled_gradients(scaled_grads)
    opt.apply_gradients(zip(grads, vars))  # Loss scale will be updated here
    ```
    """

    # TODO(reedwm): Maybe throw an error if mixed precision is used without this
    # optimizer being used.

    @staticmethod
    def __call__(opt, loss_scale):
        """Initializes a loss scaled optimizer.

        Args:
          opt: The Optimizer instance to wrap.
          loss_scale: The loss scale to scale the loss and gradients. This can
            either be an int/float to use a fixed loss scale, the string "dynamic"
            to use dynamic loss scaling, or an instance of a LossScale. The string
            "dynamic" equivalent to passing `DynamicLossScale()`, and passing an
            int/float is equivalent to passing a FixedLossScale with the given loss
            scale.
        Returns:
          Keras Optimizer with loss scaling
        """

        opt._loss_scale = loss_scale_module.get(loss_scale)

        for weight in loss_scale_module.get_loss_scale_weights(opt._loss_scale):
            # We cannot call `track_variable` in the LossScale class itself, because a
            # file outside of Keras cannot depend on a Keras file. Calling it here
            # instead is OK, because a variable only needs to be tracked if used with
            # a Keras class, and the only way to use LossScale with a Keras class is
            # through the LossScaleOptimizer.
            backend.track_variable(weight)

        opt._track_trackable(opt._loss_scale, 'loss_scale')

        class BaseOptimizer(object):
            _class = opt.__class__
            _classname = "%s.%s" % (opt.__module__, opt.__class__.__name__)
            _compute_gradients = opt._compute_gradients
            get_gradients = opt.get_gradients
            apply_gradients = opt.apply_gradients
            get_config = opt.get_config
            from_config = opt.from_config

        opt.loss_scale_base_opt = BaseOptimizer

        # Generate a fake class with name "LossScaleOptimizer"
        # Essential to avoid modifying the optimizer original class

        base_opt_class_dict = dict(opt.__class__.__dict__)
        base_opt_class_dict.update(dict(LossScaleOptimizer.__dict__))

        del base_opt_class_dict["__call__"]
        del base_opt_class_dict["__dict__"]
        del base_opt_class_dict["__weakref__"]

        opt.__class__ = type(
            LossScaleOptimizer.__name__,
            (opt.loss_scale_base_opt._class,),
            base_opt_class_dict
        )

        return opt

    def get_config(self):

        config = {
            'loss_scale_classname': self._loss_scale.__class__.__name__,
            'loss_scale_args': self._loss_scale.get_config(),
            'opt_base_classname': self.loss_scale_base_opt._classname,
            'opt_base_config': self.loss_scale_base_opt.get_config()
        }

        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):

        base_opt_classname = config["opt_base_classname"]

        opt_class = getattr(
            sys.modules[".".join(base_opt_classname.split(".")[:-1])],
            base_opt_classname.split(".")[-1]
        )
        opt = opt_class.from_config(config["opt_base_config"], custom_objects=custom_objects)

        for loss_scale_class in loss_scale_module.LossScale.__subclasses__():
            if config["loss_scale_classname"] == loss_scale_class.__name__:
                loss_scale = loss_scale_class.from_config(config["loss_scale_args"])
                break
        else:
            raise ValueError("Unsupported Loss Scale class: %s" % config["loss_scale_classname"])

        return LossScaleOptimizer(opt=opt, loss_scale=loss_scale)

    def _apply_gradients(self, grads, wrapped_vars, name):

        return self.loss_scale_base_opt.apply_gradients(list(zip(grads, wrapped_vars.value)), name)

    def _apply_gradients_cross_replica(self, distribution, grads_and_vars, name):
        grads = [g for g, _ in grads_and_vars]
        loss_scale_update_op, should_apply_grads = self._loss_scale.update(grads)

        def apply_fn():
            # We do not want DistributionStrategy to unwrap any MirroredVariables in
            # grads_and_vars, because even in a replica context, the wrapped optimizer
            # expects mirrored variables. So we wrap the variables with an
            # _UnwrapPreventer, preventing DistributionStrategy from unwrapping the
            # MirroredVariables.
            wrapped_vars = _UnwrapPreventer([v for _, v in grads_and_vars])

            return distribution.extended.call_for_each_replica(
                self._apply_gradients, args=(grads, wrapped_vars, name))

        # Note: We must call this cond() in a cross-replica context.
        # DistributionStrategy does not support having a cond in a replica context
        # with a branch that calls `merge_call`, and apply_gradients calls `merge_call`.
        maybe_apply_op = smart_cond.smart_cond(should_apply_grads,
                                               apply_fn,
                                               control_flow_ops.no_op)

        return control_flow_ops.group(maybe_apply_op, loss_scale_update_op)

    def _compute_gradients(self, loss, var_list, grad_loss=None):
        loss = self.get_scaled_loss(loss)
        grads_and_vars = self.loss_scale_base_opt._compute_gradients(loss, var_list, grad_loss)

        grads = [g for g, _ in grads_and_vars]
        variables = [v for _, v in grads_and_vars]
        unscaled_grads = self.get_unscaled_gradients(grads)
        return list(zip(unscaled_grads, variables))

    def get_gradients(self, loss, params):
        """
        Compute gradients of all trainable variables.
        See Optimizer.get_gradients() for more info.
        In _LossScaleOptimizer, get_gradients() is overriden to also
        apply loss scaling before computing the gradients.
        """
        loss = self.get_scaled_loss(loss)
        grads = self.loss_scale_base_opt.get_gradients(loss, params)
        return self.get_unscaled_gradients(grads)

    def apply_gradients(self, grads_and_vars=None, name=None):
        """Apply gradients to provided variables.
        See Optimizer.apply_gradients() for more info.
        """
        if distribution_strategy_context.in_cross_replica_context():
            raise ValueError('apply_gradients() must be called in a replica context.')

        grads_and_vars = tuple(grads_and_vars)

        return distribution_strategy_context.get_replica_context().merge_call(
            self._apply_gradients_cross_replica, args=(grads_and_vars, name)
        )

    def get_scaled_loss(self, loss):
        """Scales the loss by the loss scale.

        This method is only needed if you compute gradients manually, e.g. with
        `tf.GradientTape`. In that case, call this method to scale the loss before
        passing the loss to `tf.GradientTape`. If you use
        `LossScaleOptimizer.minimize` or `LossScaleOptimizer.get_gradients`, loss
        scaling is automatically applied and this method is unneeded.

        If this method is called, `get_unscaled_gradients` should also be called.
        See the `tf.keras.mixed_precision.experimental.LossScaleOptimizer` doc for
        an example.

        Args:
          loss: The loss, which will be multiplied by the loss scale. Can either be
            a tensor or a callable returning a tensor.

        Returns:
          `loss` multiplied by `LossScaleOptimizer.loss_scale()`.
        """
        loss_scale = self._loss_scale()

        if callable(loss):
            def new_loss():
                loss_val = loss()
                return loss_val * math_ops.cast(loss_scale, loss_val.dtype)

            return new_loss

        else:
           return loss * math_ops.cast(loss_scale, loss.dtype)

    def get_unscaled_gradients(self, grads):
        """Unscales the gradients by the loss scale.

        This method is only needed if you compute gradients manually, e.g. with
        `tf.GradientTape`. In that case, call this method to unscale the gradients
        after computing them with `tf.GradientTape`. If you use
        `LossScaleOptimizer.minimize` or `LossScaleOptimizer.get_gradients`, loss
        scaling is automatically applied and this method is unneeded.

        If this method is called, `get_scaled_loss` should also be called. See
        the `tf.keras.mixed_precision.experimental.LossScaleOptimizer` doc for an
        example.

        Args:
          grads: A list of tensors, each which will be divided by the loss scale.
            Can have None values, which are ignored.

        Returns:
          A new list the same size as `grads`, where every non-None value in `grads`
          is divided by `LossScaleOptimizer.loss_scale()`.
        """
        loss_scale = self._loss_scale()
        loss_scale_reciprocal = 1. / loss_scale
        return [g * math_ops.cast(loss_scale_reciprocal, g.dtype) if g is not None else None for g in grads]

    @property
    def loss_scale(self):
        """The `LossScale` instance associated with this optimizer."""
        return self._loss_scale

    @property
    def loss_scale_increment_period(self):
        """The `LossScale` instance associated with this optimizer."""
        try:
            return self._loss_scale._increment_period
        except AttributeError:
            raise ValueError("Loss Scale does not have an attribute `_increment_period`")

    @property
    def loss_scale_multiplier(self):
        """The `LossScale` instance associated with this optimizer."""
        try:
            return self._loss_scale._multiplier
        except AttributeError:
            raise ValueError("Loss Scale does not have an attribute `_multiplier`")

    @property
    def loss_scale_value(self):
        """The `LossScale` instance associated with this optimizer."""

        loss_scale_values_attr_names = ["_current_loss_scale", "_loss_scale_value"]

        for attr_name in loss_scale_values_attr_names:
            if hasattr(self._loss_scale, attr_name):
                return getattr(self._loss_scale, attr_name)
        else:
            raise ValueError(
                "Loss Scale does not have any attribute with name: `%s`" % str(loss_scale_values_attr_names)
            )

    @property
    def num_good_steps(self):
        """The `LossScale` instance associated with this optimizer."""
        try:
            return self._loss_scale._num_good_steps
        except AttributeError:
            raise ValueError("Loss Scale does not have an attribute `_num_good_steps`")


_GLOBAL_CUSTOM_OBJECTS["LossScaleOptimizer"] = LossScaleOptimizer
