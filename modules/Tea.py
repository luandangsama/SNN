# Import libraries

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import Model, activations, initializers, regularizers, constraints
from tensorflow.keras.layers import Layer, InputSpec, Dropout, Flatten, Activation, Input, Lambda, Concatenate,Average, Permute, concatenate
import tensorflow as tf


@tf.RegisterGradient("CustomRound")
def _const_round_grad(unused_op, grad):
    return grad

class Tea(Layer):
    def __init__(self,
                 units):
        """Initializes a new TeaLayer.

        Arguments:
            units -- The number of neurons to use for this layer."""
        super().__init__()
        self.units = units
        # Needs to be set to `True` to use the `K.in_train_phase` function.
        self.uses_learning_phase = True
        # super(Tea, self).__init__(**kwargs)
    def get_config(self):

        config = super().get_config()
        config.update({
            'units': self.units
        })
        return config

def tea_weight_initializer(shape, dtype=np.float32):
    """Returns a tensor of alternating 1s and -1s, which is (kind of like)
    how IBM initializes their weight matrix in their TeaLearning
    literature.

    Arguments:
        shape -- The shape of the weights to intialize.

    Keyword Arguments:
        dtype -- The data type to use to initialize the weights.
                 (default: {np.float32})"""
    num_axons = shape[0]
    num_neurons = shape[1]
    ret_array = np.zeros((int(num_axons), int(num_neurons)), dtype=np.float32)
    for axon_num, axon in enumerate(ret_array):
        if axon_num % 2 == 0:
            for i in range(len(axon)):
                ret_array[axon_num][i] = 1
        else:
            for i in range(len(axon)):
                ret_array[axon_num][i] = -1
    return tf.convert_to_tensor(ret_array)

def build(self, input_shape):
    assert len(input_shape) >= 2
    shape = (input_shape[-1], self.units)
    self.static_weights = self.add_weight(
        name='weights',
        shape=shape,
        initializer=tea_weight_initializer,
        trainable=False)
    # Intialize connections around 0.5 because they represent probabilities.
    self.connections = self.add_weight(
        name='connections',
        initializer=initializers.TruncatedNormal(mean=0.5),
        shape=shape)
    self.biases = self.add_weight(
        name='biases',
        initializer='zeros',
        shape=(self.units,))
    super(Tea, self).build(input_shape)

# Bind the method to our class
Tea.build = build

def call(self, x):
    with tf.compat.v1.get_default_graph().gradient_override_map(
        {"Round":"CustomRound"}):
        # Constrain input
        x = tf.round(x)
        # Constrain connections
        connections = self.connections
        connections = tf.round(connections)
        connections = K.clip(connections, 0, 1)
        # Multiply connections with weights
        weighted_connections = connections * self.static_weights
        # Dot input with weighted connections
        output = K.dot(x, weighted_connections)
        # Constrain biases
        biases = tf.round(self.biases)
        output = K.bias_add(
            output,
            biases,
            data_format='channels_last'
        )
        # Apply activation / spike
        output = K.in_train_phase(
            K.sigmoid(output),
            tf.cast(tf.greater_equal(output, 0.0), tf.float32)
        )
    return output
    
# Bind the method to our class
Tea.call = call

def compute_output_shape(self, input_shape):
    assert input_shape and len(input_shape) >= 2
    assert input_shape[-1]
    output_shape = list(input_shape)
    output_shape[-1] = self.units
    return tuple(output_shape)
    
# Bind the method to our class
Tea.compute_output_shape = compute_output_shape


class AdditivePooling(Layer):
  
    """A helper layer designed to format data for output during TeaLearning.
    If the data input to the layer has multiple spikes per classification, the
    spikes for each tick are summed up. Then, all neurons that correspond to a
    certain class are summed up so that the output is the number of spikes for
    each class. Neurons are assumed to be arranged such that each
    `num_classes` neurons represent a guess for each of the classes. For
    example, if the guesses correspond to number from 0 to 9, the nuerons are
    arranged as such:

        neuron_num: 0  1  2  3  4  5  6  7  8  9  10 11 12  ...
        guess:      0  1  2  3  4  5  6  7  8  9  0  1  2   ..."""

    def __init__(self,
                 num_classes,
                 **kwargs):
        """Initializes a new `AdditivePooling` layer.

        Arguments:
            num_classes -- The number of classes to output.
        """
        self.num_classes = num_classes
        self.num_inputs = None
        super(AdditivePooling, self).__init__(**kwargs)

    def get_config(self):

        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'num_inputs': self.num_inputs
        })
        return config

    def build(self, input_shape):
        assert len(input_shape) >= 2
        # The number of neurons must be collapsable into the number of classes
        assert input_shape[-1] % self.num_classes == 0
        self.num_inputs = input_shape[-1]

    def call(self, x):
        # Sum up ticks if there are ticks
        if len(x.shape) >= 3:
            output = K.sum(x, axis=1)
        else:
            output = x
        # Reshape output
        output = tf.reshape(
            output,
            [-1, int(self.num_inputs // self.num_classes), self.num_classes]
        )
        # Sum up neurons
        output = tf.reduce_sum(output, 1)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        # Last dimension will be number of classes
        output_shape[-1] = self.num_classes
        # Ticks were summed, so delete tick dimension if exists
        if len(output_shape) >= 3:
            del output_shape[1]
        return tuple(output_shape)