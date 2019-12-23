# TensorRT Custom layer implementation
Tensorflow.keras.layers.masking

class Masking(Layer):
  Masks a sequence by using a mask value to skip timesteps.
  For each timestep in the input tensor (dimension #1 in the tensor),
  if all values in the input tensor at that timestep
  are equal to `mask_value`, then the timestep will be masked (skipped)
  in all downstream layers (as long as they support masking).
  If any downstream layer does not support masking yet receives such
  an input mask, an exception will be raised.
  Example:
  Consider a Numpy data array `x` of shape `(samples, timesteps, features)`,
  to be fed to an LSTM layer. You want to mask timestep #3 and #5 because you
  lack data for these timesteps. You can:
  - Set `x[:, 3, :] = 0.` and `x[:, 5, :] = 0.`
  - Insert a `Masking` layer with `mask_value=0.` before the LSTM layer:
  ```python
  samples, timesteps, features = 32, 10, 8
  inputs = np.random.random([samples, timesteps, features]).astype(np.float32)
  inputs[:, 3, :] = 0.
  inputs[:, 5, :] = 0.
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Masking(mask_value=0.,
                                    input_shape=(timesteps, features)))
  model.add(tf.keras.layers.LSTM(32))
  output = model(inputs)
  # The time step 3 and 5 will be skipped from LSTM calculation.
  ```
  See [the masking and padding
  guide](https://www.tensorflow.org/guide/keras/masking_and_padding)
  for more details.
  
  
  
  
  
  
  
