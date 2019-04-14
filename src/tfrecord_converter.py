import shutil
import numpy as np
import os
import tensorflow as tf

def array_to_tfrecords(D, Q, A, DL, QL, ID, output_file):
    feature = {
        'D': tf.train.Feature(float_list=tf.train.FloatList(value=D.flatten())),
        'Q': tf.train.Feature(float_list=tf.train.FloatList(value=Q.flatten())),
        'A': tf.train.Feature(int64_list=tf.train.Int64List(value=A.flatten())),
        'DL': tf.train.Feature(int64_list=tf.train.Int64List(value=DL.flatten())),
        'QL': tf.train.Feature(int64_list=tf.train.Int64List(value=QL.flatten())),
        'ID': tf.train.Feature(bytes_list=tf.train.BytesList(value=[d.encode() for d in ID]))

    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    serialized = example.SerializeToString()
    
    writer = tf.python_io.TFRecordWriter(output_file)
    writer.write(serialized)
    writer.close()


def parse_proto(example_proto, d_shape=(640,766,50), q_shape=(640,60,50), a_shape=(640,2), l=640):
    #print(d_shape, q_shape, a_shape, l)
    features = {
        'D': tf.FixedLenFeature((d_shape), tf.float32),
        'Q': tf.FixedLenFeature((q_shape), tf.float32),
        'A': tf.FixedLenFeature((a_shape), tf.int64),
        'DL': tf.FixedLenFeature((l), tf.int64),
        'QL': tf.FixedLenFeature((l), tf.int64),
        'ID': tf.FixedLenFeature((l), tf.string)
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features
    return parsed_features['D'], parsed_features['Q'], parsed_features['A'], parsed_features['DL'], parsed_features['QL'], parsed_features['ID']

def read_tfrecords(file_names,
                   img_shapes=None,
                   buffer_size=100,
                   batch_size=32,
                   d_shape=None,
                   q_shape=None,
                   a_shape=None,
                   l=None
                   ):
    dataset = tf.data.TFRecordDataset(file_names)
    dataset = dataset.map(lambda x: parse_proto(x, d_shape, q_shape, a_shape, l))
    dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
    #dataset = dataset.shuffle(buffer_size)
    #dataset = dataset.repeat()
    #dataset = dataset.batch(batch_size)
    return dataset
# Allows same dataset object to be initialized with different datasets, could maybe
# be useful in future
#return tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)


if __name__ == "__main__":
    #data = np.load('/home/anonymous/datasets/moving_mnist/mnist_test_seq.npy')
    #data = np.load('/home/anonymous/AML-Project/padded_train_data1.0.npz')
    
    """
    data = data.f.arr_0
    #data = np.load('/home/anonymous/datasets/moving_mnist/small1.npy')
    #print(data.shape)
    D = data[0][0]
    Q = data[0][1]
    A = data[0][2]
    ID = data[0][3]
    #print(A[0])
    DL = data[1][0]
    QL = data[1][1]
    #print(D.shape)
    #print(Q.shape)
    #print(ID)
    #print(len(DL))
    #print(DL.shape)
    D = data[0][0]
    Q = data[0][1]
    A = data[0][2]
    print(A[0])
    DL = data[1][0]
    QL = data[1][1]
    print(type(QL[0]))
    print(D.shape)
    print(Q.shape)
        
    array_to_tfrecords(D=D, Q=Q, A=A, DL=DL, QL=QL, ID=ID, output_file="test.tfrecord")
    print('s', np.prod(D.shape))
    for i in range(100):
        sliced_data = data[:,100*i:100*(i+1),:,:]
        print(sliced_data.flatten().shape)
        array_to_tfrecords(sliced_data, "file{}.tfrecord".format(i+1))
    #filelist = ["/home/anonymous/datasets/moving_mnist/tfrecords/file{}.tfrecord".format(i) for i in range(1,101)]
    #recovered = read_tfrecords(file_names=("test.tfrecord"), buffer_size=100, img_shapes=(8192000,))
    #recovered = read_tfrecords(file_names=("test.tfrecord"), buffer_size=100, img_shapes=(1225600,))
    recovered = read_tfrecords(file_names=("test.tfrecord"), buffer_size=100, d_shape=D.shape, q_shape=Q.shape, a_shape=A.shape, l=len(DL))
    """
    
    it = 135 
    for file_name in os.listdir('/home/shared/data/batched_data/'):
        file_path = os.path.join('/home/shared/data/batched_data/', file_name)
        np_data = np.load(file_path)
        data = np_data.f.arr_0
        D = data[0][0][:,:600,:]
        D = np.float32(D)
        Q = data[0][1]
        A = data[0][2]
        Q = np.float32(Q)
        ID = data[0][3]
        DL = data[1][0]
        QL = data[1][1]
        out_file_path = '/home/shared/data/tfrecords/file{}.tfrecord'.format(it) 


        try:
            assert (not os.path.isfile(out_file_path))
            array_to_tfrecords(D=D, Q=Q, A=A, DL=DL, QL=QL, ID=ID, output_file=out_file_path)
            print('Writing to file {}'.format(out_file_path))
            #recovered = read_tfrecords(file_names=("/home/shared/data/tfrecords/file0.tfrecord"), buffer_size=100, d_shape=D.shape, q_shape=Q.shape, a_shape=A.shape, l=len(DL))
            recovered = read_tfrecords(file_names=(out_file_path), buffer_size=100, d_shape=D.shape, q_shape=Q.shape, a_shape=A.shape, l=len(DL))
            iter_ = recovered.make_initializable_iterator()
            tensor_dict = iter_.get_next()

            d_tensor = tensor_dict['D']
            id_tensor = tensor_dict['ID']
            init = iter_.initializer

            with tf.Session() as sess:
                sess.run(init)
                #sess.run(d_tensor)
                d_val, id_val = sess.run([d_tensor, id_tensor])
                id_val = id_val.decode()
                assert( id_val == ID[0])
                assert(d_val.all() == D[0,:,:].all())
                print('removing file {}'.format(file_path))
                #os.chmod(file_path, 0o777)
                os.remove(file_path)
                #shutil.rmtree(file_path)
        except Exception as e:
            print(e)
            print('ERROR: unable to write data from file {}'.format(file_path))
        it += 1
        






    iter_ = recovered.make_initializable_iterator()
    tensor_dict = iter_.get_next()

    d_tensor = tensor_dict['D']
    id_tensor = tensor_dict['ID']
    print(d_tensor)
    print(d_tensor.get_shape())
    init = iter_.initializer

    with tf.Session() as sess:
        sess.run(init)
        #sess.run(d_tensor)
        d_val, id_val = sess.run([d_tensor, id_tensor])
        print(D.shape)
        print(d_val.shape)
        print(type(D[0,0,0]))
        print(type(d_val[0,0]))
        print(d_val[:,:].all() == D[0,:,:].all())
        #id_val = sess.run(id_tensor)
        id_val = id_val.decode()
        print(id_val)
        print(ID[0])
        print(id_val == ID[0])
