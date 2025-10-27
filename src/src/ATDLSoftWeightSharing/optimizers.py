"""
A modified implementation of custom optimizers for the neural network compression tutorial.

Author: Karen Ullrich, Sep 2016
Updated for TensorFlow 2.x compatibility
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np


class AdamWithParamGroups(keras.optimizers.Optimizer):
    """
    Adam optimizer with different learning rates for different parameter groups.
    This is an extended version that allows training parameters with different hyperparameters.
    """
    def __init__(
        self,
        learning_rates=None,
        beta_1=None,
        beta_2=None,
        epsilon=1e-7,
        amsgrad=False,
        name="AdamWithParamGroups",
        param_types_dict=None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        # Set defaults
        if learning_rates is None:
            learning_rates = [0.001]
        if param_types_dict is None:
            param_types_dict = []
        
        # Convert learning rates to list if single value
        if isinstance(learning_rates, (int, float)):
            learning_rates = [learning_rates]
        
        # Store learning rates for different parameter types
        self._set_hyper("learning_rate", learning_rates[0])  # Primary learning rate
        self.learning_rates = learning_rates
        self.param_types_dict = ['other'] + param_types_dict  # Add default 'other' type
        
        # Set up beta parameters
        self._beta_1 = beta_1 if beta_1 else [0.9] * len(learning_rates)
        self._beta_2 = beta_2 if beta_2 else [0.999] * len(learning_rates)
        
        # Set hyperparameters
        self._set_hyper("beta_1", self._beta_1[0])
        self._set_hyper("beta_2", self._beta_2[0])
        self._set_hyper("epsilon", epsilon)
        self.amsgrad = amsgrad

        # Store individual learning rates as variables
        self.lr_variables = {}
        self.beta1_variables = {}
        self.beta2_variables = {}
        
        for i, param_type in enumerate(self.param_types_dict):
            lr_val = learning_rates[i] if i < len(learning_rates) else learning_rates[-1]
            beta1_val = self._beta_1[i] if i < len(self._beta_1) else self._beta_1[-1]
            beta2_val = self._beta_2[i] if i < len(self._beta_2) else self._beta_2[-1]
            
            self.lr_variables[param_type] = tf.Variable(
                lr_val, name=f"learning_rate_{param_type}", trainable=False
            )
            self.beta1_variables[param_type] = tf.Variable(
                beta1_val, name=f"beta1_{param_type}", trainable=False
            )
            self.beta2_variables[param_type] = tf.Variable(
                beta2_val, name=f"beta2_{param_type}", trainable=False
            )

    def _create_slots(self, var_list):
        # Create slots for momentum and velocity
        for var in var_list:
            self.add_slot(var, "m")
            self.add_slot(var, "v")
            if self.amsgrad:
                self.add_slot(var, "vhat")

    def _resource_apply_dense(self, grad, var):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((self._beta1_variables[var_device],
                         self._beta2_variables[var_device]),
                        self._epsilon)
        
        # Determine parameter type based on variable name
        param_type = 'other'
        var_name = var.name.lower()
        for pt in self.param_types_dict:
            if pt in var_name and pt != 'other':
                param_type = pt
                break
        
        # Get the learning rate for this parameter type
        lr = self.lr_variables[param_type]
        beta1 = self.beta1_variables[param_type]
        beta2 = self.beta2_variables[param_type]
        
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        
        alpha_t = lr * tf.math.rsqrt(beta2 * (1.0 - tf.pow(beta2, tf.cast(self.iterations + 1, tf.float32)))) / (
            1.0 - tf.pow(beta1, tf.cast(self.iterations + 1, tf.float32))
        )
        
        m_t = beta1 * m + (1.0 - beta1) * grad
        v_t = beta2 * v + (1.0 - beta2) * tf.square(grad)
        
        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat_t = tf.maximum(vhat, v_t)
            var_update = var - alpha_t * m_t / (tf.sqrt(vhat_t) + self._get_hyper("epsilon", var_dtype))
            return tf.group(*[var.assign(var_update),
                              m.assign(m_t),
                              v.assign(v_t),
                              vhat.assign(vhat_t)])
        else:
            var_update = var - alpha_t * m_t / (tf.sqrt(v_t) + self._get_hyper("epsilon", var_dtype))
            return tf.group(*[var.assign(var_update),
                              m.assign(m_t),
                              v.assign(v_t)])

    def _resource_apply_sparse(self, grad, var, indices):
        # Similar to dense but for sparse gradients
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((self._beta1_variables[var_device],
                         self._beta2_variables[var_device]),
                        self._epsilon)
        
        # Determine parameter type based on variable name
        param_type = 'other'
        var_name = var.name.lower()
        for pt in self.param_types_dict:
            if pt in var_name and pt != 'other':
                param_type = pt
                break
        
        # Get the learning rate for this parameter type
        lr = self.lr_variables[param_type]
        beta1 = self.beta1_variables[param_type]
        beta2 = self.beta2_variables[param_type]
        
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        
        alpha_t = lr * tf.math.rsqrt(beta2 * (1.0 - tf.pow(beta2, tf.cast(self.iterations + 1, tf.float32)))) / (
            1.0 - tf.pow(beta1, tf.cast(self.iterations + 1, tf.float32))
        )
        
        # m_t = beta1 * m + (1.0 - beta1) * grad
        m_scaled_g_values = grad * (1 - beta1)
        m_t = tf.scatter_update(m, indices, beta1 * tf.gather(m, indices) + m_scaled_g_values)
        
        # v_t = beta2 * v + (1.0 - beta2) * (grad * grad)
        v_scaled_g_values = (grad * grad) * (1 - beta2)
        v_t = tf.scatter_update(v, indices, beta2 * tf.gather(v, indices) + v_scaled_g_values)
        
        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat_t = tf.scatter_update(vhat, indices, tf.maximum(tf.gather(vhat, indices), tf.gather(v_t, indices)))
            var_update = var - alpha_t * tf.gather(m_t, indices) / (tf.sqrt(tf.gather(vhat_t, indices)) + self._get_hyper("epsilon", var_dtype))
        else:
            var_update = var - alpha_t * tf.gather(m_t, indices) / (tf.sqrt(tf.gather(v_t, indices)) + self._get_hyper("epsilon", var_dtype))
            
        return tf.group(*[var.assign(var_update),
                          m.assign(m_t),
                          v.assign(v_t)])

    def get_config(self):
        config = super().get_config()
        config.update({
            "learning_rates": self.learning_rates,
            "param_types_dict": self.param_types_dict,
            "beta_1": self._beta_1,
            "beta_2": self._beta_2,
            "epsilon": self._get_hyper("epsilon"),
            "amsgrad": self.amsgrad,
        })
        return config