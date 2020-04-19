from tensorflow.python.framework import ops
from tensorflow.python.framework import common_shapes
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.eager import context
import tensorflow as tf
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as K


class LeakyDense2D(Dense):
    """ A dense layer which gives the output of length based on `mask_array`.
    The output tensor obtained from dense operation is sliced based on `mask_array`.
    If `mask_array` is not provided, it should act as normal Dense layer.
    `leak_inputs`: boolean, if False, the layer will act as normal Dense layer.
    `mask_array` is an array with some of its values as zeros while some non zeros.
    First a dense operation is applied and then if `mask_array` is provided, only those outputs will be returned where `mask_array`>0.0.
    ```python
    tf.reset_default_graph()
    dense_outputs = np.random.random((16,3))    # random 2d array where second dimension shows number
                                                # of features to be predicted
    y_of_int_all = {}
    sparcity = [0.4, 0.5, 0.6]
    for dim in range(dense_outputs.shape[1]):
        _y = dense_outputs[:,dim]
        y_masked = _y.copy()  # masked copy will have some values equal to zero, where zero values means
                              # we don't have observations at those points
        ind = np.random.choice(np.arange(_y.size), replace=False, size=int(_y.size * random.choice(sparcity)))
        y_masked[ind] = 0
        y_of_int = y_masked[np.where(y_masked>0.0)].reshape(-1,1)
        y_of_int_all['val_'+str(dim)] = y_of_int
        y_of_int_all['key_'+str(dim)] = ind   # this will be masked_array
    # tensorflow version
    y_of_int_all_ten = {}  # container to hold all output arrays which are to be compared with observed
    for dim in range(dense_outputs.shape[1]):
        _y = dense_outputs[:,dim]          # y will be the output from dense operation/layer or full_outputs
        y_masked = _y.copy()
        ind =y_of_int_all['key_' + str(dim)]
        y_masked[ind] = 0
        mask_ph = tf.convert_to_tensor(y_masked, dtype=tf.float32)
        full_outputs = tf.convert_to_tensor(_y, dtype=tf.float32)     # full_outputs would be output after dense operation
        outputs_1d = tf.reshape(full_outputs, [-1,], name='outputs_1d')
        mask_ph_1d = tf.reshape(mask_ph, [-1,], name='mask_ph_1d')
        mask = tf.math.greater(mask_ph, tf.constant(0.0, dtype=tf.float32), name='masking_op')
        outputs = full_outputs[mask] #outputs_1d[mask]
        outputs = tf.expand_dims(outputs, axis=1)    # These are out outputs for loss calculation only
        y_of_interest_tensor = K.eval(outputs)
        y_of_int_all_ten['val_'+str(dim)] = y_of_interest_tensor
        y_of_int_all_ten['key_'+str(dim)] = ind
    # comparing numpy vs tensorflow solution
    for val_a, val_t in zip(y_of_int_all.values(), y_of_int_all_ten.values()):
        np.testing.assert_array_almost_equal(val_a,val_t)
    ```
    """

    def __init__(self,
                 leaky_inputs=False,
                 mask_array=None,
                 name='LeakyDense2D',
                 verbose=1,
                 **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(LeakyDense2D, self).__init__(name=name,
                                           **kwargs)
        self.leaky_inputs = leaky_inputs
        self.mask_array = mask_array
        self.verbose=verbose

    def build(self, input_shape):

        if self.leaky_inputs:
            if self.verbose>0: print('performing mask ope')
            if self.mask_array is None:
                raise ValueError('provide mask array')

        #             if not isinstance(self.mask_array, dict):
        #                 raise TypeError("mask array must be a dictionary consisting of masked array for all observations")

        #             if len(self.mask_array) != self.units:
        #                 raise ValueError("dictionary mask_array must contain mask array for all observations")

        super(LeakyDense2D, self).build(input_shape)

    def call(self, inputs):

        inputs = ops.convert_to_tensor(inputs)
        rank = common_shapes.rank(inputs)
        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, self.kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.get_shape().as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            outputs = gen_math_ops.mat_mul(inputs, self.kernel)

        if self.activation is not None:
            outputs = self.activation(outputs)  # pylint: disable=not-callable

        if self.verbose>0: print(outputs.get_shape(), 'outputs before masking')

        if self.leaky_inputs:
            if self.verbose>0: print('performing mask op')
            self.full_outputs = outputs
            # outputs_d = {}
            # for dim in range(outputs.get_shape()[1]):

            outputs_1d = tf.reshape(outputs, [-1, ], name='outputs_1d_')

            # mask_array = self.mask_array['key_' + str(dim)]
            if self.verbose>0: print('mask_array', self.mask_array.get_shape())

            mask_array_1d = tf.reshape(self.mask_array, [-1, ], name='mask_ph_1d_')

            if self.verbose>0: print('mask_array_1d', mask_array_1d.get_shape())

            mask = tf.math.greater(mask_array_1d, tf.constant(0.0), name='masking_op_')

            outputs = outputs_1d[mask]
            outputs = tf.expand_dims(outputs, axis=1)
            # outputs_d['val_' + str(dim)] = outputs

            if self.verbose > 0: print(outputs.get_shape(), 'shape of output after masking')

            return outputs
        else:
            self.full_outputs = outputs

            if self.verbose>0: print(outputs.get_shape(), 'shape of output without masking')

            return outputs