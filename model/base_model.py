# -*- coding: utf-8 -*-
import tensorflow as tf
from  model.multi_head_attention import MultiHeadAttention
from model.poistion_wise_feed_forward import PositionWiseFeedFoward
from model.layer_norm_residual_conn import LayerNormResidualConnection
class BaseClass(object):
    """
    base class has some common fields and functions.
    """
    def __init__(self,d_model,d_k,d_v,sequence_length,h,batch_size,num_layer=6,decoder_sent_length=None):
        """
        :param d_model:
        :param d_k:
        :param d_v:
        :param sequence_length:
        :param h:
        :param batch_size:
        :param embedded_words: shape:[batch_size,sequence_length,embed_size]
        """
        self.d_model=d_model
        self.d_k=d_k
        self.d_v=d_v
        self.sequence_length=sequence_length
        self.h=h
        self.num_layer=num_layer
        self.batch_size=batch_size
        self.decoder_sent_length=decoder_sent_length

    def sub_layer_postion_wise_feed_forward(self, x, layer_index)  :# COMMON FUNCTION
        """
        position-wise feed forward. you can implement it as feed forward network, or two layers of CNN.
        :param x: shape should be:[batch_size,sequence_length,d_model]
        :param layer_index: index of layer number
        :return: [batch_size,sequence_length,d_model]
        """
        # use variable scope here with input of layer index, to make sure each layer has different parameters.
        with tf.variable_scope("sub_layer_postion_wise_feed_forward"  + str(layer_index)):
            postion_wise_feed_forward = PositionWiseFeedFoward(x, layer_index,d_model=self.d_model,d_ff=self.d_model*4)
            postion_wise_feed_forward_output = postion_wise_feed_forward.position_wise_feed_forward_fn()
        return postion_wise_feed_forward_output

    def sub_layer_multi_head_attention(self ,layer_index ,Q ,K_s,V_s,mask=None,is_training=None,dropout_keep_prob=0.9)  :# COMMON FUNCTION
        """
        multi head attention as sub layer
        :param layer_index: index of layer number
        :param Q: shape should be: [batch_size,sequence_length,embed_size]
        :param k_s: shape should be: [batch_size,sequence_length,embed_size]
        :param mask: when use mask,illegal connection will be mask as huge big negative value.so it's possiblitity will become zero.
        :return: output of multi head attention.shape:[batch_size,sequence_length,d_model]
        """
        #print("sub_layer_multi_head_attention.",";layer_index:",layer_index)
        with tf.variable_scope("base_mode_sub_layer_multi_head_attention_" +str(layer_index)):
            #2. call function of multi head attention to get result
            multi_head_attention_class = MultiHeadAttention(Q, K_s, V_s, self.d_model, self.d_k, self.d_v, self.sequence_length,self.h,
                                                            is_training=is_training,mask=mask,dropout_rate=(1.0-dropout_keep_prob))
            sub_layer_multi_head_attention_output = multi_head_attention_class.multi_head_attention_fn()  # [batch_size*sequence_length,d_model]
        return sub_layer_multi_head_attention_output  # [batch_size,sequence_length,d_model]

    def sub_layer_layer_norm_residual_connection(self,layer_input ,layer_output,layer_index,dropout_keep_prob=0.9,use_residual_conn=True,sub_layer_name='layer1'): # COMMON FUNCTION
        """
        layer norm & residual connection
        :param input: [batch_size,equence_length,d_model]
        :param output:[batch_size,sequence_length,d_model]
        :return:
        """
        #print("sub_layer_layer_norm_residual_connection.layer_input:",layer_input,";layer_output:",layer_output,";dropout_keep_prob:",dropout_keep_prob)
        #assert layer_input.get_shape().as_list()==layer_output.get_shape().as_list()
        #layer_output_new= layer_input+ layer_output
        variable_scope="sub_layer_layer_norm_residual_connection_" +str(layer_index)+'_'+sub_layer_name
        #print("######sub_layer_layer_norm_residual_connection.variable_scope:",variable_scope)
        with tf.variable_scope(variable_scope):
            layer_norm_residual_conn=LayerNormResidualConnection(layer_input,layer_output,layer_index,residual_dropout=(1-dropout_keep_prob),use_residual_conn=use_residual_conn)
            output = layer_norm_residual_conn.layer_norm_residual_connection()
        return output  # [batch_size,sequence_length,d_model]