import tensorflow as tf

# One Hot implementation with multiple labels
def multi_one_hot(x, depth):
    one_hot = tf.reduce_sum(tf.one_hot(x, depth, dtype=tf.float32), axis=-2)
    # Handles duplicate entries in the original vector
    return tf.cast(one_hot > 0, dtype=tf.float32)

# Deterministic reduce_sum from https://www.twosigma.com/articles/a-workaround-for-non-determinism-in-tensorflow/
def reduce_sum_det(x):
    v = tf.reshape(x, [1, -1])
    return tf.reshape(tf.matmul(v, tf.ones_like(v), transpose_b=True), [])

