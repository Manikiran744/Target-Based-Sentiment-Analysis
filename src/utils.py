import random
import tensorflow as tf

def shuffle(inputs_fw , inputs_bw , tr_y , length):
    indices = tf.range(start=0, limit=4000, dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)

    shuffled_fw = tf.gather(inputs_fw, shuffled_indices)
    shuffled_bw = tf.gather(inputs_bw, shuffled_indices)
    shuffled_y   = tf.gather(tr_y , shuffled_indices)

    return shuffled_fw , shuffled_bw , shuffled_y

def get_batch_data(inputs_fw , inputs_bw , y , batch_size):
      
      length = len(y)
      i = 0
      while(True):
           if i+ 64 > len(y):
             i = 0
           i+=64
           yield (inputs_fw[i % length : (i+batch_size) %length ] , inputs_bw[i % length : (i+batch_size) %length ] ), y[i % length : (i+batch_size) %length ]
       