import tensorflow as tf
from inception_v4 import inception_v4, inception_v4_arg_scope
from PIL import Image
from imagenet_labels import imagenet_labels
import numpy as np
slim = tf.contrib.slim

height = 299
width = 299
channels = 3

with tf.Session() as sess:
  X = tf.placeholder(tf.float32, shape=[None, height, width, channels])
  filename_queue = tf.train.string_input_producer(['./grouse.jpg']) #  list of files to read
  
  reader = tf.WholeFileReader()
  key, value = reader.read(filename_queue)
  
  images = tf.image.decode_jpeg(value)
  init_op = tf.global_variables_initializer()

  with slim.arg_scope(inception_v4_arg_scope()):
    #Create inception net
    logits, end_points = inception_v4(X, is_training=False,
                                     create_aux_logits=False)
    #Get softmax predictions
    predictions = end_points['Predictions']
    
    #Load pre-trained parameters
    saver = tf.train.Saver()
    saver.restore(sess, 'inception_v4.ckpt')
    
    #Initialize variables
    init_op = tf.global_variables_initializer()
    
    #Get index of prediction
    index = tf.argmax(predictions, axis = 1)
    
    #Run variable initialization
    sess.run(init_op)

    #Load image
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    test_image = np.asarray(images.eval())
    Image.fromarray(test_image).show()

    coord.request_stop()
    coord.join(threads)

    #Run initial prediction
    results = sess.run([predictions, index], {X: np.reshape(test_image, [1, 299, 299, 3])})
    print('Original Classification: {}'.format(results[1]))

