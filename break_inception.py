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
    x_adv = tf.identity(X)
    #Create inception net
    logits, end_points = inception_v4(x_adv, is_training=False,
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

    #FGSM
    clip_max = 1
    clip_min = 0

    yshape = tf.shape(predictions)
    ydim = yshape[1]
    target = tf.one_hot(index, ydim, on_value=clip_max,
                        off_value=clip_min)

    eps = 0.01 
    epochs = 10
    def _cond(x_adv, i):
        return tf.less(i, epochs)

    def _body(x_adv, i):
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=target, logits=logits)
        dy_dx, = tf.gradients(loss, x_adv)
        x_adv = tf.stop_gradient(x_adv + eps*tf.sign(dy_dx))
        x_adv = tf.clip_by_value(x_adv, clip_min, clip_max)
        return x_adv, i+1

    x_adv, i = tf.while_loop(_cond, _body, (x_adv, 0),
                             back_prop=False, name='fgsm')
    
    perturbed = sess.run(x_adv, {X: test_image})
