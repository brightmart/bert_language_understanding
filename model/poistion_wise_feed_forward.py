# -*- coding: utf-8 -*-
import tensorflow as tf
import time
"""
Position-wise Feed-Forward Networks
In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully
connected feed-forward network, which is applied to each position separately and identically. This
consists of two linear transformations with a ReLU activation in between.
FFN(x) = max(0,xW1+b1)W2+b2
While the linear transformations are the same across different positions, they use different parameters
from layer to layer. Another way of describing this is as two convolutions with kernel size 1.
The dimensionality of input and output is d_model= 512, and the inner-layer has dimensionalityd_ff= 2048.
"""
class PositionWiseFeedFoward(object):
    """
    position-wise feed forward networks. formula as below:
    FFN(x)=max(0,xW1+b1)W2+b2
    """
    def __init__(self,x,layer_index,d_model=512,d_ff=2048):
        """
        :param x: shape should be:[batch,sequence_length,d_model]
        :param layer_index:  index of layer
        :return: shape:[sequence_length,d_model]
        """
        shape_list=x.get_shape().as_list()
        assert(len(shape_list)==3)
        self.x=x
        self.layer_index=layer_index
        self.d_model=d_model
        self.d_ff=d_ff
        self.initializer = tf.random_normal_initializer(stddev=0.1)

    def position_wise_feed_forward_fn(self):
        """
        positional wise fully connected feed forward implement as two layers of cnn
        x:       [batch,sequence_length,d_model]
        :return: [batch,sequence_length,d_model]
        """
        # 1.conv layer 1
        input=tf.expand_dims(self.x,axis=3) # [batch,sequence_length,d_model,1]
        # conv2d.input: [batch,sentence_length,embed_size,1]. filter=[filter_size,self.embed_size,1,self.num_filters]
        output_conv1=tf.layers.conv2d(  # output_conv1: [batch_size,sequence_length,1,d_ff]
            input,filters=self.d_ff,kernel_size=[1,self.d_model],padding="VALID",
            name='conv1',kernel_initializer=self.initializer,activation=tf.nn.relu
        )
        output_conv1 = tf.transpose(output_conv1, [0,1,3,2])  #output_conv1:[batch_size,sequence_length,d_ff,1]
        # print("output_conv1:",output_conv1)

        # 2.conv layer 2
        output_conv2 = tf.layers.conv2d( # output_conv2:[batch_size, sequence_length,1,d_model]
            output_conv1,filters=self.d_model,kernel_size=[1,self.d_ff],padding="VALID",
            name='conv2',kernel_initializer=self.initializer,activation=None
        )
        output=tf.squeeze(output_conv2) #[batch,sequence_length,d_model]
        return output #[batch,sequence_length,d_model]

    def position_wise_feed_forward_fc_fn(self):
        """
        positional wise fully connected feed forward implement as original version.
        FFN(x) = max(0,xW1+b1)W2+b2
        this function provide you as an alternative if you want to use original version, or you don't want to use two layers of cnn,
        but may be less efficient as sequence become longer.
        x:       [batch,sequence_length,d_model]
        :return: [batch,sequence_length,d_model]
        """
        # 0. pre-process input x
        _,sequence_length,d_model=self.x.get_shape().as_list()

        element_list = tf.split(self.x, sequence_length,axis=1)  # it is a list,length is sequence_length, each element is [batch_size,1,d_model]
        element_list = [tf.squeeze(element, axis=1) for element in element_list]  # it is a list,length is sequence_length, each element is [batch_size,d_model]
        output_list=[]
        for i, element in enumerate(element_list):
            with tf.variable_scope("foo", reuse=True if i>0 else False):
                # 1. layer 1
                W1 = tf.get_variable("ff_layer1", shape=[self.d_model, self.d_ff], initializer=self.initializer)
                z1=tf.nn.relu(tf.matmul(element,W1)) # z1:[batch_size,d_ff]<--------tf.matmul([batch_size,d_model],[d_model, d_ff])
                # 2. layer 2
                W2 = tf.get_variable("ff_layer2", shape=[self.d_ff, self.d_model], initializer=self.initializer)
                output_element=tf.matmul(z1,W2) # output:[batch_size,d_model]<----------tf.matmul([batch_size,d_ff],[d_ff, d_model])
                output_list.append(output_element) # a list, each element is [batch_size,d_model]
        output=tf.stack(output_list,axis=1) # [batch,sequence_length,d_model]
        return output # [batch,sequence_length,d_model]

#test function of position_wise_feed_forward_fn
#time spent:OLD VERSION(FC): length=1000,time spent:2.04 s; NEW VERSION(CNN):0.03s, speed up as 68x.
def test_position_wise_feed_forward_fn():
    start=time.time()
    x=tf.ones((8,1000,512)) #batch_size=8,sequence_length=10 ;
    layer_index=0
    postion_wise_feed_forward=PositionWiseFeedFoward(x,layer_index)
    output=postion_wise_feed_forward.position_wise_feed_forward_fn()
    end=time.time()
    print("x:",x.shape,";output:",output.shape)
    print("time spent:",(end-start))
    return output

def test():
    with tf.Session() as sess:
        result=test_position_wise_feed_forward_fn()
        sess.run(tf.global_variables_initializer())
        result_=sess.run(result)
        print("result_.shape:",result_.shape)

#test()