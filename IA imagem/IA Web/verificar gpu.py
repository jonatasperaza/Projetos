import tensorflow as tf
print("Número de GPUs disponíveis: ", len(tf.config.experimental.list_physical_devices('GPU')))
