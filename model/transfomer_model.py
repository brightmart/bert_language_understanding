# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
 main idea:  based on multiple layer self-attention model(encoder of Transformer), pretrain two tasks( masked language model and next sentence prediction task)
             on large scale of corpus, then fine-tuning by add a single classification layer.
"""

import tensorflow as tf
import numpy as np
from model.encoder import Encoder
from model.config_transformer import Config
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

class TransformerModel:
    def __init__(self,config):
        """
        init all hyperparameter with config class, define placeholder, computation graph
        """
        self.num_classes = config.num_classes
        print("BertModel.num_classes:",self.num_classes)
        self.batch_size = config.batch_size
        self.sequence_length = config.sequence_length
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.learning_rate = tf.Variable(config.learning_rate, trainable=False, name="learning_rate")
        self.clip_gradients=config.clip_gradients
        self.decay_steps=config.decay_steps
        self.decay_rate=config.decay_rate
        self.d_k=config.d_k
        self.d_model=config.d_model
        self.h=config.h
        self.d_v=config.d_v
        self.num_layer=config.num_layer
        self.use_residual_conn=True
        self.is_training=config.is_training

        # place holder(X,y)
        self.input_x= tf.placeholder(tf.int32, [self.batch_size, self.sequence_length], name="input_x")  # e.g.is a sequence, input='the man [mask1] to [mask2] store'
        self.input_y=tf.placeholder(tf.float32, [self.batch_size, self.num_classes],name="input_y")

        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate *config.decay_rate)
        self.initializer=tf.random_normal_initializer(stddev=0.1)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

        self.instantiate_weights()
        self.logits =self.inference() # shape:[None,self.num_classes]
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")  # shape:[None,]

        if not self.is_training:
            return
        self.loss_val = self.loss()
        self.train_op = self.train()

    def inference(self):
        """
        main inference logic here: invoke transformer model to do inference. input is a sequence, output is also a sequence.
        input representation-->
        :return:
        """
        # 1. input representation(input embedding, positional encoding, segment encoding)
        token_embeddings = tf.nn.embedding_lookup(self.embedding,self.input_x)  # [batch_size,sequence_length,embed_size]
        self.input_representation=tf.add(tf.add(token_embeddings,self.segment_embeddings),self.position_embeddings)  # [batch_size,sequence_length,embed_size]

        # 2. repeat Nx times of building block( multi-head attention followed by Add & Norm; feed forward followed by Add & Norm)
        encoder_class=Encoder(self.d_model,self.d_k,self.d_v,self.sequence_length,self.h,self.batch_size,self.num_layer,self.input_representation,
                              self.input_representation,dropout_keep_prob=self.dropout_keep_prob,use_residual_conn=self.use_residual_conn)
        h = encoder_class.encoder_fn() # [batch_size,sequence_length,d_model]

        # 3. get logits for different tasks by applying projection layer
        logits=self.project_tasks(h) # shape:[None,self.num_classes]
        return logits # shape:[None,self.num_classes]

    def project_tasks(self,h):
        """
        project the representation, then to do classification.
        :param h: [batch_size,sequence_length,d_model]
        :return: logits: [batch_size, num_classes]
        transoform each sub task using one-layer MLP ,then get logits.
        get some insights from densely connected layers from recently development
        """
        cls_representation = h[:, 0, :] # [CLS] token's information: classification task's representation
        logits = tf.layers.dense(cls_representation, self.num_classes)   # shape:[None,self.num_classes]
        logits = tf.nn.dropout(logits,keep_prob=self.dropout_keep_prob)  # shape:[None,self.num_classes]
        return logits

    def loss(self,l2_lambda=0.0001*3,epislon=0.000001):
        # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
        # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
        # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        losses= tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)  #[batch_size,num_classes]
        self.losses = tf.reduce_mean((tf.reduce_sum(losses,axis=1)))  # shape=(?,)-->(). loss for all data in the batch-->single loss
        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda

        loss=self.losses+self.l2_loss
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding"):  # embedding matrix
            self.embedding = tf.get_variable("embedding", shape=[self.vocab_size, self.d_model],initializer=self.initializer)  # [vocab_size,embed_size]
            self.segment_embeddings = tf.get_variable("segment_embeddings", [self.d_model],initializer=tf.constant_initializer(1.0))  # a learned sequence embedding
            self.position_embeddings = tf.get_variable("position_embeddings", [self.sequence_length, self.d_model],initializer=tf.constant_initializer(1.0))  # sequence_length,1]


# train the model on toy task: learn to count,sum up all inputs, and distinct whether the total value of input is below or greater than a threshold.
# usage: first run train () to train the model, it will save checkpoint to file system. then run predict() to make a prediction based on checkpoint.
def train():
    # 1.init config and model
    config=Config()
    threshold=(config.sequence_length/2)+1
    model = TransformerModel(config)
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    saver = tf.train.Saver()
    save_path = config.ckpt_dir + "model.ckpt"
    #if not os.path.exists(config.ckpt_dir):
    #    os.makedirs(config.ckpt_dir)
    with tf.Session(config=gpu_config) as sess:
        sess.run(tf.global_variables_initializer())
        if os.path.exists(config.ckpt_dir): # 如果存在，则加载预训练过的模型
            saver.restore(sess, tf.train.latest_checkpoint(save_path))
        for i in range(100000):
            # 2.feed data
            input_x = np.random.randn(config.batch_size, config.sequence_length)  # [None, self.sequence_length]
            input_x[input_x >= 0] = 1
            input_x[input_x < 0] = 0
            input_y = generate_label(input_x,threshold)
            # 3.run session to train the model, print some logs.
            loss, _ = sess.run([model.loss_val,  model.train_op],feed_dict={model.input_x: input_x, model.input_y: input_y,model.dropout_keep_prob: config.dropout_keep_prob})
            print(i, "loss:", loss, "-------------------------------------------------------")
            if i==300:
                print("label[0]:", input_y[0]);print("input_x:",input_x)
            if i % 500 == 0:
                saver.save(sess, save_path, global_step=i)

# use saved checkpoint from model to make prediction, and print it, to see whether it is able to do toy task successfully.
def predict():
    config=Config()
    threshold=(config.sequence_length/2)+1
    config.batch_size=1
    model = TransformerModel(config)
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    saver = tf.train.Saver()
    ckpt_dir = config.ckpt_dir
    print("ckpt_dir:",ckpt_dir)
    with tf.Session(config=gpu_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
        for i in range(100):
            # 2.feed data
            input_x = np.random.randn(config.batch_size, config.sequence_length)  # [None, self.sequence_length]
            input_x[input_x >= 0] = 1
            input_x[input_x < 0] = 0
            target_label = generate_label(input_x,threshold)
            input_sum=np.sum(input_x)
            # 3.run session to train the model, print some logs.
            logit,prediction = sess.run([model.logits, model.predictions],feed_dict={model.input_x: input_x ,model.dropout_keep_prob: config.dropout_keep_prob})
            print("target_label:", target_label,";input_sum:",input_sum,"threshold:",threshold,";prediction:",prediction);
            print("input_x:",input_x,";logit:",logit)


def generate_label(input_x,threshold):
    """
    generate label with input
    :param input_x: shape of [batch_size, sequence_length]
    :return: y:[batch_size]
    """
    batch_size,sequence_length=input_x.shape
    y=np.zeros((batch_size,2))
    for i in range(batch_size):
        input_single=input_x[i]
        sum=np.sum(input_single)
        if i == 0:print("sum:",sum,";threshold:",threshold)
        y_single=1 if sum>threshold else 0
        if y_single==1:
            y[i]=[0,1]
        else: # y_single=0
            y[i]=[1,0]
    return y

#train()
#predict()