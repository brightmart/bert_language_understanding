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
from model.config import Config
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"

class BertCNNModel:
    def __init__(self,config):
        """
        init all hyperparameter with config class, define placeholder, computation graph
        """
        self.num_classes = config.num_classes
        self.num_classes_lm = config.num_classes_lm
        self.sequence_length_lm = config.sequence_length_lm # sentence sequence length for training masked langauge mdoel.
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
        self.is_pretrain=config.is_pretrain
        self.is_fine_tuning=config.is_fine_tuning

        #################
        self.filter_sizes = [2, 3, 4, 5]
        self.embed_size = self.d_model
        self.num_filters = 128
        self.is_training_flag = self.is_training
        self.stride_length = 1
        #################

        # below is for fine-tuning stage
        self.input_x= tf.placeholder(tf.int32, [self.batch_size, self.sequence_length], name="input_x")  # e.g.is a sequence, input='the man [mask1] to [mask2] store'
        print("bert_model.num_classes:",self.num_classes)
        self.input_y=tf.placeholder(tf.float32, [self.batch_size, self.num_classes],name="input_y")
        # below is pre-trained task 1: masked language model
        self.x_mask_lm=tf.placeholder(tf.int32, [self.batch_size, self.sequence_length_lm], name="x_mask_lm")
        self.y_mask_lm=tf.placeholder(tf.int32, [self.batch_size],name="y_mask_lm")
        self.p_mask_lm=tf.placeholder(tf.int32, [self.batch_size],name="p_mask_lm")

        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate *config.decay_rate)
        self.initializer=tf.random_normal_initializer(stddev=0.1)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

        self.instantiate_weights()

        if not self.is_fine_tuning:
            self.logits_lm =self.inference_lm() # shape:[None,self.num_classes]
            self.predictions_lm = tf.argmax(self.logits_lm, axis=1, name="predictions")       # shape:[None,]<---[ None,num_classes]
            accuary_tensor =tf.equal(tf.cast(self.predictions_lm, tf.int32),self.y_mask_lm) # shape:()
            self.accuracy_lm=tf.reduce_mean(tf.cast(accuary_tensor,tf.float32))
            self.loss_val_lm = self.loss_lm()  # train masked language model
        else: # fine tuning
            self.logits=self.inference()
            self.loss_val = self.loss() # fine tuning

        if not self.is_training:
            return
        if not self.is_fine_tuning:
            self.train_op_lm = self.train_lm()
        else:  # fine tuning
            self.train_op = self.train()

    def inference_lm(self):
        """
        this is for pre-trained language model.
        main inference logic here: invoke transformer model to do inference,input is a sequence, output is also a sequence, get representation of masked token(s) and use a classifier
        to train the model.
        # idea of the hidden state of masked position(s):
        #   1) a batch of position index,
            2) one hot it, multiply with total sequence represenation,
            3)every where is 0 for the second dimension(sequence_length),
            4) only one place is 1,
            5) thus we can sum up without loss any information.
        :return:
        """
        # 1. input representation(input embedding, positional encoding, segment encoding)
        token_embeddings = tf.nn.embedding_lookup(self.embedding,self.x_mask_lm)  # [batch_size,sequence_length,embed_size]
        self.input_representation_lm=tf.add(tf.add(token_embeddings,self.segment_embeddings_lm),self.position_embeddings_lm)  # [batch_size,sequence_length,embed_size]
        #########################################################
        self.total_sequence_length = self.sequence_length_lm
        h_lm = self.conv_layers_return_2layers(self.input_representation_lm, 'conv_layer',reuse_flag=False)  # [batch_size,sequence_length-filter_size + 1,num_filters]
        # 2. repeat Nx times of building block( multi-head attention followed by Add & Norm; feed forward followed by Add & Norm)
        #encoder_class=Encoder(self.d_model,self.d_k,self.d_v,self.sequence_length_lm,self.h,self.batch_size,self.num_layer,self.input_representation_lm,
        #                      self.input_representation_lm,dropout_keep_prob=self.dropout_keep_prob,use_residual_conn=self.use_residual_conn)
        #h_lm = encoder_class.encoder_fn() # [batch_size,sequence_length,d_model]

        # 3. get last hidden state of the masked position(s), and project it to make a predict.
        #p_mask_lm_onehot=tf.one_hot(self.p_mask_lm,self.sequence_length_lm) # [batch_size, sequence_length_lm]
        #p_mask_lm_expand=tf.expand_dims(p_mask_lm_onehot,axis=-1) #  # [batch_size, sequence_length_lm,1]
        #h_lm_multiply=tf.multiply(h_lm,p_mask_lm_expand)     # [batch_size,sequence_length,d_model]
        #h_lm_representation=tf.reduce_sum(h_lm_multiply,axis=1) # batch_size,d_model].
        ##########################################################

        # 4. project representation of masked token(s) to vocab size
        with tf.variable_scope("pre_training"):
            logits_lm = tf.layers.dense(h_lm, self.vocab_size)   # shape:[None,self.vocab_size]
            logits_lm = tf.nn.dropout(logits_lm,keep_prob=self.dropout_keep_prob)  # shape:[None,self.num_classes]
        return logits_lm # shape:[None,self.num_classes]

    def inference(self):
        """
        this is for fine-tuning.
        main inference logic here: invoke transformer model to do inference,input is a sequence, output is also a sequence, get representation of masked token(s) and use a classifier
        to train the model.
        # idea of the hidden state of masked position(s):
        #   1) a batch of position index,
            2) one hot it, multiply with total sequence represenation,
            3)every where is 0 for the second dimension(sequence_length),
            4) only one place is 1,
            5) thus we can sum up without loss any information.
        :return:
        """
        # 1. input representation(input embedding, positional encoding, segment encoding)
        token_embeddings = tf.nn.embedding_lookup(self.embedding,self.input_x)  # [batch_size,sequence_length,embed_size]
        self.input_representation=tf.add(tf.add(token_embeddings,self.segment_embeddings_lm),self.position_embeddings)  # [batch_size,sequence_length,embed_size]
        #############################
        self.total_sequence_length = self.sequence_length
        # 2. repeat Nx times of building block( multi-head attention followed by Add & Norm; feed forward followed by Add & Norm)
        #encoder_class=Encoder(self.d_model,self.d_k,self.d_v,self.sequence_length,self.h,self.batch_size,self.num_layer,self.input_representation,
        #                      self.input_representation,dropout_keep_prob=self.dropout_keep_prob,use_residual_conn=self.use_residual_conn)
        #h= encoder_class.encoder_fn() # [batch_size,sequence_length,d_model]
        h = self.conv_layers_return_2layers(self.input_representation, 'conv_layer',reuse_flag=False)  # [batch_size,sequence_length-filter_size + 1,num_filters]
        # 3. get hidden state of token of [cls], and project it to make a predict.
        #h_cls=h[:,0,:] # [batch_size,d_model]
        ############################

        # 4. project representation of masked token(s) to vocab size
        with tf.variable_scope("fine_tuning"):
            logits = tf.layers.dense(h, self.num_classes)   # shape:[None,self.vocab_size]
            logits = tf.nn.dropout(logits,keep_prob=self.dropout_keep_prob)  # shape:[None,self.num_classes]
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

    def loss_lm(self,l2_lambda=0.0001*3):
        # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
        # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
        # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        y_mask_lm_onehot=tf.one_hot(self.y_mask_lm,self.vocab_size)
        print("#loss_lm.y_mask_lm_onehot:",y_mask_lm_onehot,";logits_lm:",self.logits_lm)
        losses= tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_mask_lm_onehot,logits=self.logits_lm)  #[batch_size,num_classes]
        print("#loss_lm.losses:",losses)
        lm_loss = tf.reduce_mean(losses)
        self.l2_loss_lm = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        loss=lm_loss+self.l2_loss_lm
        return loss

    def loss(self,l2_lambda=0.0001*3):
        # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
        # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
        # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        losses= tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)  #[batch_size,num_classes]
        self.losses = tf.reduce_mean((tf.reduce_sum(losses,axis=1)))  # shape=(?,)-->(). loss for all data in the batch-->single loss
        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda

        loss=self.losses+self.l2_loss
        return loss

    def conv_layers_return_2layers(self, input_x, name_scope, reuse_flag=False):  # great 81.3
        """main computation graph here: 1.embedding-->2.CONV-RELU-MAX_POOLING-->3.linear classifier"""
        # 1.=====>get emebedding of words in the sentence
        sentence_embeddings_expanded = tf.expand_dims(input_x,
                                                      -1)  # [None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv

        # 2.=====>loop each filter size. for each filter, do:convolution-pooling layer(a.create filters,b.conv,c.apply nolinearity,d.max-pooling)--->
        # you can use:tf.nn.conv2d;tf.nn.relu;tf.nn.max_pool; feature shape is 4-d. feature is a new variable
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope(str(name_scope) + "convolution-pooling-%s" % filter_size, reuse=reuse_flag):
                # 1) CNN->BN->relu
                filter = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1, self.num_filters],
                                         initializer=self.initializer)
                conv = tf.nn.conv2d(sentence_embeddings_expanded, filter, strides=[1, self.stride_length, 1, 1],padding="VALID",name="conv")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                conv = tf.contrib.layers.batch_norm(conv, is_training=self.is_training_flag, scope='cnn1')
                print(i, "conv1:", conv)
                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])  # ADD 2017-06-09
                h = tf.nn.relu(tf.nn.bias_add(conv, b),"relu")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`

                # 2) CNN->BN->relu
                h = tf.reshape(h, [-1, self.total_sequence_length - filter_size + 1, self.num_filters,1])  # shape:[batch_size,sequence_length-filter_size+1,num_filters,1]
                # Layer2:CONV-RELU
                filter2 = tf.get_variable("filter2-%s" % filter_size,[filter_size, self.num_filters, 1, self.num_filters],initializer=self.initializer)
                conv2 = tf.nn.conv2d(h, filter2, strides=[1, 1, 1, 1], padding="VALID",name="conv2")  # shape:[batch_size,sequence_length-filter_size*2+2,1,num_filters]
                conv2 = tf.contrib.layers.batch_norm(conv2, is_training=self.is_training_flag, scope='cnn2')
                print(i, "conv2:", conv2)
                b2 = tf.get_variable("b2-%s" % filter_size, [self.num_filters])  # ADD 2017-06-09
                h = tf.nn.relu(tf.nn.bias_add(conv2, b2),"relu2")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`

                # 3. Max-pooling
                pooling_max = tf.squeeze(tf.nn.max_pool(h, ksize=[1, (self.total_sequence_length - filter_size * 2 + 2), 1, 1],strides=[1, 1, 1, 1], padding='VALID', name="pool"))
                # pooling_avg=tf.squeeze(tf.reduce_mean(h,axis=1)) #[batch_size,num_filters]
                print(i, "pooling:", pooling_max)
                # pooling=tf.concat([pooling_max,pooling_avg],axis=1) #[batch_size,num_filters*2]
                pooled_outputs.append(pooling_max)  # h:[batch_size,sequence_length - filter_size + 1,1,num_filters]
        # concat
        print("#####pooled_outputs:", pooled_outputs)
        h = tf.concat(pooled_outputs, axis=-1)  # [batch_size,num_total_filters]
        print("h.concat:", h)

        with tf.name_scope("dropout"):
            h = tf.nn.dropout(h,
                              keep_prob=self.dropout_keep_prob)  # [batch_size,sequence_length - filter_size + 1,num_filters]
        return h  # [batch_size,sequence_length - filter_size + 1,num_filters]

    def train_lm_old(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val_lm, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op

    def train_lm(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        self.learning_rate_=learning_rate
        #noise_std_dev = tf.constant(0.3) / (tf.sqrt(tf.cast(tf.constant(1) + self.global_step, tf.float32))) #gradient_noise_scale=noise_std_dev

        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss_val_lm))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #ADD 2018.06.01
        with tf.control_dependencies(update_ops):  #ADD 2018.06.01
            train_op = optimizer.apply_gradients(zip(gradients, variables))
        #train_op = tf_contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op

    def train_old(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        self.learning_rate_=learning_rate
        #noise_std_dev = tf.constant(0.3) / (tf.sqrt(tf.cast(tf.constant(1) + self.global_step, tf.float32))) #gradient_noise_scale=noise_std_dev

        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss_val))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #ADD 2018.06.01
        with tf.control_dependencies(update_ops):  #ADD 2018.06.01
            train_op = optimizer.apply_gradients(zip(gradients, variables))
        #train_op = tf_contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op
    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding"):  # embedding matrix
            self.embedding = tf.get_variable("embedding", shape=[self.vocab_size, self.d_model],initializer=self.initializer)  # [vocab_size,embed_size]
            self.segment_embeddings_lm = tf.get_variable("segment_embeddings_lm", [self.d_model],initializer=tf.constant_initializer(1.0))  # a learned sequence embedding
            self.position_embeddings_lm = tf.get_variable("position_embeddings_lm", [self.sequence_length_lm, self.d_model],initializer=tf.constant_initializer(1.0))  # sequence_length,1]

            #self.segment_embeddings = tf.get_variable("segment_embeddings", [self.d_model],initializer=tf.constant_initializer(1.0))  # a learned sequence embedding
            self.position_embeddings = tf.get_variable("position_embeddings", [self.sequence_length,self.d_model],initializer=tf.constant_initializer(1.0))  # [sequence_length,self.d_model]


# train the model on toy task: learn to count,sum up input, and distinct whether the total value of input is below or greater than a threshold.
# usage: use train() to train the model, and it will save checkpoint to file system; use predict() to make a prediction based on the trained model.

def train():
    # 1.init config and model
    config=Config()
    threshold=(config.sequence_length/2)+1
    model = BertCNNModel(config)
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    saver = tf.train.Saver()
    save_path = config.ckpt_dir + "model.ckpt"
    #if not os.path.exists(config.ckpt_dir):
    #    os.makedirs(config.ckpt_dir)
    batch_size = 8
    with tf.Session(config=gpu_config) as sess:
        sess.run(tf.global_variables_initializer())
        if os.path.exists(config.ckpt_dir): #
            saver.restore(sess, tf.train.latest_checkpoint(save_path))
        for i in range(10000):
            # 2.feed data
            input_x = np.random.randn(config.batch_size, config.sequence_length)  # [None, self.sequence_length]
            input_x[input_x >= 0] = 1
            input_x[input_x < 0] = 0
            input_y = generate_label(input_x,threshold)
            p_mask_lm=[i for i in range(batch_size)]
            # 3.run session to train the model, print some logs.
            loss, _ = sess.run([model.loss_val,  model.train_op],feed_dict={model.x_mask_lm: input_x, model.y_mask_lm: input_y,model.p_mask_lm:p_mask_lm,
                                                                            model.dropout_keep_prob: config.dropout_keep_prob})
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
    model = BertCNNModel(config)
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