import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tfd=tf.contrib.distributions

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

dataset = tf.data.Dataset.from_tensor_slices((mnist.train.images, mnist.train.labels)).batch(32)

iterator = dataset.make_initializable_iterator()
iterator.make_initializer(dataset)
images_flat, labels = iterator.get_next()
images = tf.reshape(images_flat, shape=[-1,28,28,1])
it_init_op = iterator.initializer

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape,positive=True):
    initial = tf.truncated_normal(shape, stddev=.05)
    if positive:
        initial = tf.math.abs(initial)
    return tf.Variable(initial)

def autoencoder(autoencodee, latent_dims):
    with tf.variable_scope("encoder"):
        w_conv1_e = weight_variable([5,5,1,16])
        b_conv1_e = bias_variable([16])
        h_conv1_e = tf.nn.conv2d(autoencodee, filter=w_conv1_e, strides=(1,2,2,1), padding="VALID")
        h_conv1_e = tf.nn.leaky_relu(tf.nn.bias_add(h_conv1_e, b_conv1_e))

        w_conv2_e = weight_variable([3,3,16,32])
        b_conv2_e = bias_variable([32])
        h_conv2_e = tf.nn.conv2d(h_conv1_e, filter=w_conv2_e, strides=(1,2,2,1), padding="VALID")
        h_conv2_e = tf.nn.leaky_relu(tf.nn.bias_add(h_conv2_e, b_conv2_e))

        print('e1:',h_conv1_e.get_shape())
        print('e2:',h_conv2_e.get_shape())
        #h_conv2_e = tf.nn.max_pool(h_conv2_e, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")

    latent_size = np.prod(h_conv2_e.get_shape().as_list()[1:])
    latent_1 = tf.reshape(h_conv2_e, [-1,latent_size])
    w_fc1 = weight_variable([latent_size, latent_size//3])
    b_fc1 = bias_variable([latent_size//3])
    latent_2 = tf.nn.leaky_relu(tf.matmul(latent_1, w_fc1) + b_fc1)

    w_fc2 = weight_variable([latent_size//3, 4])
    b_fc2 = bias_variable([4])
    latent_3 = tf.nn.leaky_relu(tf.matmul(latent_2, w_fc2) + b_fc2)

    w_fc3 = weight_variable([4, latent_size//3])
    b_fc3 = bias_variable([latent_size//3])
    latent_4 = tf.nn.leaky_relu(tf.matmul(latent_3, w_fc3) + b_fc3)

    w_fc4 = weight_variable([latent_size//3, latent_size])
    b_fc4 = bias_variable([latent_size])
    latent_5 = tf.nn.leaky_relu(tf.matmul(latent_4, w_fc4) + b_fc4)

    print('l1:',latent_1.get_shape())
    print('l2:',latent_2.get_shape())
    print('l3:',latent_3.get_shape())
    print('l4:',latent_4.get_shape())
    print('l5:',latent_5.get_shape())

    fc_out = tf.reshape(latent_5, [-1] + h_conv2_e.get_shape().as_list()[1:])

    print('hidden:', fc_out.get_shape())

    with tf.variable_scope("decoder"):
        w_conv1_d = weight_variable([3,3,16,32])
        b_conv1_d = bias_variable([16])
        h_conv1_d = tf.nn.conv2d_transpose(fc_out, filter=w_conv1_d, output_shape=tf.shape(h_conv1_e), strides=(1,2,2,1), padding="VALID")
        h_conv1_d = tf.nn.leaky_relu(tf.nn.bias_add(h_conv1_d, b_conv1_d))

        w_conv2_d = weight_variable([5,5,1,16])
        b_conv2_d = bias_variable([1])
        h_conv2_d = tf.nn.conv2d_transpose(h_conv1_d, filter=w_conv2_d, output_shape=tf.shape(images), strides=(1,2,2,1), padding="VALID")
        h_conv2_d = tf.nn.leaky_relu(tf.nn.bias_add(h_conv2_d,b_conv2_d))

        return h_conv2_d

out = autoencoder(images, 4)
out = tf.reshape(out, shape=[-1,784])
unsprv_loss= tf.reduce_mean(tf.losses.absolute_difference(out,images_flat))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
unsprv_train_step = optimizer.minimize(unsprv_loss)

saver = tf.train.Saver()
print("GPU:",tf.test.gpu_device_name())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    #saver.restore(sess, "/tmp/model.ckpt")
    for epoch in range(1):
        sess.run(it_init_op)

        print(epoch)
        step = 0
        while True:
            try:
                step += 1
                imgs_in, imgs_out, new_loss, __ =sess.run([images,out,unsprv_loss, unsprv_train_step])
                if step % 100 == 0:
                    print(new_loss)
            except (tf.errors.OutOfRangeError):
                break
    for i in range(5):
        imgs_out = imgs_out.reshape(-1,28,28,1)
        img_in = imgs_in[i,:,:,0]
        img_out = imgs_out[i,:,:,0]
        print(imgs_in.shape)
        print(imgs_out.shape)
        print(img_out.shape)
        fig,ax = plt.subplots()

        plt.imshow(img_in)
        plt.show()
        plt.imshow(img_out)
        plt.show()
    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in path {}".format(save_path))
    
