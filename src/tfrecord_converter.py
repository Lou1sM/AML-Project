import numpy as np
import tensorflow as tf

def array_to_tfrecords(D, Q, A, output_file):
    feature = {
        'D': tf.train.Feature(float_list=tf.train.FloatList(value=D.flatten())),
        'Q': tf.train.Feature(float_list=tf.train.FloatList(value=Q.flatten())),
        'A': tf.train.Feature(float_list=tf.train.FloatList(value=A.flatten()))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    serialized = example.SerializeToString()
    
    writer = tf.python_io.TFRecordWriter(output_file)
    writer.write(serialized)
    writer.close()


def parse_proto(example_proto, d_shape=(640,766,50), q_shape=(640,60,50), a_shape=(640,2)):
    features = {
        'D': tf.FixedLenFeature((d_shape), tf.float32),
        'Q': tf.FixedLenFeature((q_shape), tf.float32),
        'A': tf.FixedLenFeature((a_shape), tf.float32),
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features['D'], parsed_features['Q'], parsed_features['A']

def read_tfrecords(file_names,
                   img_shapes,
                   buffer_size=100,
                   batch_size=32,
                   ):
    dataset = tf.data.TFRecordDataset(file_names)
    dataset = dataset.map(lambda x: parse_proto(x))
    dataset = dataset.shuffle(buffer_size)
    #dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    return dataset
# Allows same dataset object to be initialized with different datasets, could maybe
# be useful in future
#return tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)


if __name__ == "__main__":
    #data = np.load('/home/louis/datasets/moving_mnist/mnist_test_seq.npy')
    data = np.load('/home/louis/AML-Project/data/padded_train_data1.0.npy')
    
    #data = np.load('/home/louis/datasets/moving_mnist/small1.npy')
    print(data.shape)
    D = data[0][0]
    Q = data[0][1]
    A = data[0][2]
    print(D.shape)
    print(Q.shape)
    array_to_tfrecords(D=D, Q=Q, A=A, output_file="test.tfrecord")
    """
    for i in range(100):
        sliced_data = data[:,100*i:100*(i+1),:,:]
        print(sliced_data.flatten().shape)
        array_to_tfrecords(sliced_data, "file{}.tfrecord".format(i+1))
    """
    #filelist = ["/home/louis/datasets/moving_mnist/tfrecords/file{}.tfrecord".format(i) for i in range(1,101)]
    recovered = read_tfrecords(file_names=("test.tfrecord"), buffer_size=100, img_shapes=(8192000,))
    iter_ = recovered.make_initializable_iterator()
    d_tensor, q_tensor, a_tensor = iter_.get_next()
    print(d_tensor.get_shape())
    init = iter_.initializer

    with tf.Session() as sess:
        sess.run(init)
        d_val = sess.run(d_tensor)
        print(d_val)
        print(d_val.shape)
        print(d_val[0,:,:,:].all() == D.all())
