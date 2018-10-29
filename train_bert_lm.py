# -*- coding: utf-8 -*-
#process--->1.load data(X,y). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)

"""
 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
 main idea:  based on multiple layer self-attention model(encoder of Transformer), pretrain two tasks( masked language model and next sentence prediction task)
             on large scale of corpus, then fine-tuning by add a single classification layer.
train the model(transformer) with data enhanced by pre-training of two tasks.
default hyperparameter is d_model=512,h=8,d_k=d_v=64(big). if you have a small data set or want to train a
small model, use d_model=128,h=8,d_k=d_v=16(small), or d_model=64,h=8,d_k=d_v=8(tiny).
"""
import tensorflow as tf
import numpy as np
#from model.bert_model import BertModel # TODO TODO TODO test whether pretrain can boost perofrmance with other model
from model.bert_cnn_model import BertCNNModel as BertModel
from data_util_hdf5 import create_or_load_vocabulary,load_data_multilabel,assign_pretrained_word_embedding,set_config
import os
from evaluation_matrix import *
from pretrain_task import mask_language_model,mask_language_model_multi_processing
from model.config import Config
import random

#configuration
FLAGS=tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean("test_mode",True,"whether it is test mode. if it is test mode, only small percentage of data will be used")
tf.app.flags.DEFINE_string("data_path","./data/","path of traning data.")
tf.app.flags.DEFINE_string("mask_lm_source_file","./data/bert_train2.txt","path of traning data.")
tf.app.flags.DEFINE_string("ckpt_dir","./checkpoint_lm/","checkpoint location for the model") #save to here, so make it easy to upload for test
tf.app.flags.DEFINE_integer("vocab_size",60000,"maximum vocab size.")
tf.app.flags.DEFINE_integer("d_model", 64, "dimension of model") # 512-->128
tf.app.flags.DEFINE_integer("num_layer", 6, "number of layer")
tf.app.flags.DEFINE_integer("num_header", 8, "number of header")
tf.app.flags.DEFINE_integer("d_k", 8, "dimension of k") # 64
tf.app.flags.DEFINE_integer("d_v", 8, "dimension of v") # 64

tf.app.flags.DEFINE_string("tokenize_style","word","checkpoint location for the model")
tf.app.flags.DEFINE_integer("max_allow_sentence_length",10,"max length of allowed sentence for masked language model")
tf.app.flags.DEFINE_float("learning_rate",0.0001,"learning rate") #0.001
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.9, "percentage to keep when using dropout.")
tf.app.flags.DEFINE_integer("sequence_length",200,"max sentence length")#400
tf.app.flags.DEFINE_integer("sequence_length_lm",10,"max sentence length for masked language model")
tf.app.flags.DEFINE_boolean("is_training",True,"is training.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_boolean("is_fine_tuning",False,"is_finetuning.ture:this is fine-tuning stage")
tf.app.flags.DEFINE_integer("num_epochs",30,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")
tf.app.flags.DEFINE_boolean("use_pretrained_embedding",False,"whether to use embedding or not.")#
tf.app.flags.DEFINE_string("word2vec_model_path","./data/Tencent_AILab_ChineseEmbedding_100w.txt","word2vec's vocabulary and vectors") # data/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5--->data/news_12g_baidubaike_20g_novel_90g_embedding_64.bin--->sgns.merge.char
tf.app.flags.DEFINE_integer("process_num",20,"number of cpu process")

def main(_):
    vocab_word2index, _= create_or_load_vocabulary(FLAGS.data_path,FLAGS.mask_lm_source_file,FLAGS.vocab_size,test_mode=FLAGS.test_mode,tokenize_style=FLAGS.tokenize_style)
    vocab_size = len(vocab_word2index);print("bert_pertrain_lm.vocab_size:",vocab_size)
    index2word={v:k for k,v in vocab_word2index.items()}
    #train,valid,test=mask_language_model(FLAGS.mask_lm_source_file,FLAGS.data_path,index2word,max_allow_sentence_length=FLAGS.max_allow_sentence_length,test_mode=FLAGS.test_mode)
    train, valid, test = mask_language_model(FLAGS.mask_lm_source_file, FLAGS.data_path, index2word, max_allow_sentence_length=FLAGS.max_allow_sentence_length,test_mode=FLAGS.test_mode, process_num=FLAGS.process_num)

    train_X, train_y,train_p = train
    valid_X, valid_y,valid_p = valid
    test_X,test_y,test_p = test

    print("length of training data:",train_X.shape,";train_Y:",train_y.shape,";train_p:",train_p.shape,";valid data:",valid_X.shape,";test data:",test_X.shape)
    # 1.create session.
    gpu_config=tf.ConfigProto()
    gpu_config.gpu_options.allow_growth=True
    with tf.Session(config=gpu_config) as sess:
        #Instantiate Model
        config=set_config(FLAGS,vocab_size,vocab_size)
        model=BertModel(config)
        #Initialize Save
        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint.")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
            for i in range(2): #decay learning rate if necessary.
                print(i,"Going to decay learning rate by half.")
                sess.run(model.learning_rate_decay_half_op)
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_pretrained_embedding:
                vocabulary_index2word={index:word for word,index in vocab_word2index.items()}
                assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size,FLAGS.word2vec_model_path,model.embedding,config.d_model) # assign pretrained word embeddings
        curr_epoch=sess.run(model.epoch_step)

        # 2.feed data & training
        number_of_training_data=len(train_X)
        print("number_of_training_data:",number_of_training_data)
        batch_size=FLAGS.batch_size
        iteration=0
        score_best=-100
        for epoch in range(curr_epoch,FLAGS.num_epochs):
            loss_total_lm, counter =  0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)):
                iteration=iteration+1
                if epoch==0 and counter==0:
                    print("trainX[start:end]:",train_X[start:end],"train_X.shape:",train_X.shape)
                feed_dict = {model.x_mask_lm: train_X[start:end],model.y_mask_lm: train_y[start:end],model.p_mask_lm:train_p[start:end],
                             model.dropout_keep_prob: FLAGS.dropout_keep_prob}
                current_loss_lm,lr,l2_loss,_=sess.run([model.loss_val_lm,model.learning_rate,model.l2_loss_lm,model.train_op_lm],feed_dict)
                loss_total_lm,counter=loss_total_lm+current_loss_lm,counter+1
                if counter %30==0:
                    print("%d\t%d\tLearning rate:%.5f\tLoss_lm:%.3f\tCurrent_loss_lm:%.3f\tL2_loss:%.3f\t"%(epoch,counter,lr,float(loss_total_lm)/float(counter),current_loss_lm,l2_loss))
                if start!=0 and start%(4000*FLAGS.batch_size)==0: # epoch!=0
                    loss_valid, acc_valid= do_eval(sess, model, valid,batch_size)
                    print("%d\tValid.Epoch %d ValidLoss:%.3f\tAcc_valid:%.3f\t" % (counter,epoch, loss_valid, acc_valid*100))
                    # save model to checkpoint
                    if acc_valid>score_best:
                        save_path = FLAGS.ckpt_dir + "model.ckpt"
                        print("going to save check point.")
                        saver.save(sess, save_path, global_step=epoch)
                        score_best=acc_valid
            sess.run(model.epoch_increment)

validation_size=100*FLAGS.batch_size
def do_eval(sess,model,valid,batch_size):
    """
    do evaluation using validation set, and report loss, and f1 score.
    :param sess:
    :param model:
    :param valid:
    :param num_classes:
    :return:
    """
    valid_X,valid_y,valid_p=valid
    number_examples=valid_X.shape[0]
    if number_examples>10000:
        number_examples=validation_size
    print("do_eval.valid.number_examples:",number_examples)
    if number_examples>validation_size: valid_X,valid_y,valid_p=valid_X[0:validation_size],valid_y[0:validation_size],valid_p[0:validation_size]
    eval_loss,eval_counter,eval_acc=0.0,0,0.0
    for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples,batch_size)):
        feed_dict = {model.x_mask_lm: valid_X[start:end],model.y_mask_lm: valid_y[start:end],model.p_mask_lm:valid_p[start:end],
                     model.dropout_keep_prob: 1.0} # FLAGS.dropout_keep_prob
        curr_eval_loss, logits_lm, accuracy_lm= sess.run([model.loss_val_lm,model.logits_lm,model.accuracy_lm],feed_dict) # logits：[batch_size,label_size]
        eval_loss=eval_loss+curr_eval_loss
        eval_acc=eval_acc+accuracy_lm
        eval_counter=eval_counter+1
    return eval_loss/float(eval_counter+small_value), eval_acc/float(eval_counter+small_value)

if __name__ == "__main__":
    tf.app.run()