# -*- coding: utf-8 -*-
import codecs
import random
import numpy as np
import multiprocessing
from collections import Counter
import os
import pickle
#import gensim
#from gensim.models import KeyedVectors
import h5py
import time
import json
import jieba
import tensorflow as tf
from model.config import Config


PAD_ID = 0
UNK_ID=1
CLS_ID=2
MASK_ID=3
_PAD="_PAD"
_UNK="UNK"
_CLS="CLS"
_MASK="MASK"

LABEL_SPLITTER='__label__'

def build_chunk(lines, chunk_num=10):
    """
    split list into sub lists:分块
    :param lines: total thing
    :param chunk_num: num of chunks
    :return: return chunks but the last chunk may not be equal to chunk_size
    """
    total = len(lines)
    chunk_size = float(total) / float(chunk_num + 1)
    chunks = []
    for i in range(chunk_num + 1):
        if i == chunk_num:
            chunks.append(lines[int(i * chunk_size):])
        else:
            chunks.append(lines[int(i * chunk_size):int((i + 1) * chunk_size)])
    return chunks

def load_data_multilabel(data_path,traning_data_path,valid_data_path,test_data_path,vocab_word2index,label2index,sentence_len,process_num=20,test_mode=False,tokenize_style='word',model_name=None):
    """
    convert data as indexes using word2index dicts.
    1) use cache file if exist; 2) read source files; 3)transform to train/valid data to standardized format; 4)save to file system if data not exists
    :param traning_data_path:
    :param vocab_word2index:
    :param vocab_label2index:
    :return:
    """
    t1 = time.clock()
    print("###load_data_multilabel.data_path:",data_path,";traning_data_path:",traning_data_path,";valid_data_path:",valid_data_path,";test_data_path:",test_data_path)
    print("###vocab_word2index:",len(vocab_word2index),";label2index:",len(label2index),";sentence_len:",sentence_len)
    # 1. use cache file if exist
    if model_name is not None:
        cache_file =data_path+"/"+model_name+'train_valid_test.h5'
    else:
        cache_file =data_path+"/"+'train_valid_test.h5'

    cache_file_exist_flag=os.path.exists(cache_file)
    print("cache_path:",cache_file,"train_valid_test_file_exists:",cache_file_exist_flag,";traning_data_path:",traning_data_path,";valid_data_path:",valid_data_path)
    if cache_file_exist_flag:
        print("load_data_multilabel.going to load cache file from file system and return")
        train, valid, test=load_cache_from_hdf5(cache_file)
        t2 = time.clock()
        print('load_data_multilabel.ended.time spent:', (t2 - t1))
        return train,valid,test

    # 2. read source file (training,valid,test set)
    train_lines=read_file(traning_data_path)
    valid_lines = read_file(valid_data_path)
    test_lines = read_file(test_data_path)
    if test_mode:
        train_lines = train_lines[0:10000]
        valid_lines = valid_lines[0:1000]
        test_lines = test_lines[0:1000]

    number_examples=len(train_lines)
    print("load_data_multilabel.length of train_lines:",number_examples,";valid_lines:",len(valid_lines),";test_lines:",len(test_lines))

    # 3. transform to train/valid data to standardized format
    ############## below is for multi-processing ########################################################################################################
    # 3.1 get chunks as list.
    chunks = build_chunk(train_lines, chunk_num=process_num - 1)
    pool = multiprocessing.Pool(processes=process_num)
    # 3.2 use multiprocessing to handle different chunk. each chunk will be transformed and save to file system.
    for chunk_id, each_chunk in enumerate(chunks):
        file_name= data_path+"training_data_temp_" + str(chunk_id) # ".npy" #data_path +
        print("#start multi-processing:",chunk_id,file_name)
        # 3.3 apply_async
        print("chunk:",len(each_chunk),";file_name:",file_name,";")
        pool.apply_async(transform_data_to_index,args=(each_chunk, file_name, vocab_word2index, label2index,sentence_len,'train',tokenize_style))
    pool.close()
    pool.join()

    print("load_data_multilabel.finish all of sub tasks.")

    # 3.4 merge sub file to final file.
    X, Y=[],[]
    for chunk_id in range(process_num):
        file_name_X =data_path+"training_data_temp_" + str(chunk_id)+'X.npy'
        file_name_Y =data_path+"training_data_temp_" + str(chunk_id)+'Y.npy'
        x_sub=np.load(file_name_X)
        y_sub=np.load(file_name_Y)
        X.extend(x_sub)
        Y.extend(y_sub)
        command = 'rm ' + file_name_X+" "+file_name_Y
        os.system(command)
    ############## above is for multi-processing ##########################################################################################################

    train= np.array(X),np.array(Y)
    valid=transform_data_to_index(valid_lines,None,vocab_word2index, label2index,sentence_len,'valid',tokenize_style)
    test=transform_data_to_index(test_lines,None,vocab_word2index, label2index,sentence_len,'test',tokenize_style)

    # 4. save to file system if data not exists
    if not os.path.exists(cache_file):
        print("going to dump train/valid/test data to file sytem!") #pickle.dump((train,valid,test),data_f,protocol=pickle.HIGHEST_PROTOCOL) #TEMP REMOVED. ,protocol=2
        dump_cache_to_hdf5(cache_file, train, valid, test) # move some code to function 2018-07-12

    t2 = time.clock()
    print('load_data_multilabel.ended.time spent:', (t2 - t1))
    return train ,valid,test

def read_file(file_path):
    """
    read file, and return lines
    :param file_path: path of ifle
    :return: lines, a list
    """
    file_object = codecs.open(file_path, mode='r', encoding='utf-8')
    lines = file_object.readlines()
    random.shuffle(lines)
    file_object.close()
    return lines

def transform_data_to_index(lines,target_file_path,vocab_word2index,label2index,sentence_len,data_type,tokenize_style):
    """
    transform data to index using vocab and label dict.
    :param lines:
    :param vocab_word2index:
    :param accusation_label2index:
    :param article_label2index:
    :param deathpenalty_label2index:
    :param lifeimprisonment_label2index:
    :param sentence_len: max sentence length
    :return:
    """
    print("###data_type:",data_type,"transform_data_to_index.start.target_file_path:",target_file_path)
    X = []
    Y= []
    label_size=len(label2index)
    print("###################label2index:",label2index)
    for i, line in enumerate(lines):
        try:
            # 1. transform input string to x
            input_list,input_labels=get_input_strings_and_labels(line, tokenize_style=tokenize_style)
            #input_list = token_string_as_list(input_strings,tokenize_style=tokenize_style)
            x_list = [vocab_word2index.get(x, UNK_ID) for x in input_list if x.strip()]  # transform input to index
            x_list.insert(0,CLS_ID) # INSERT SPECIAL TOKEN:[CLS]. it will be used for classificaiton.
            x_list=pad_truncate_list(x_list, sentence_len)

            # 2. transform label to y
            label_list = [label2index[label] for label in input_labels]
            y = transform_multilabel_as_multihot(label_list, label_size)

            X.append(x_list)
            Y.append(y)
            if i % 100 == 0:
                print(data_type,i,"transform_data_to_index.line:",line,";x_list:",x_list)
                print(data_type,i,"transform_data_to_index.input_labels:",input_labels,";label_list:",label_list,";y:",y)
        except Exception as e:
            if random.randint(0, 10) == 1:print("ignore line. you may be in test_model=True, label may not exist.",line,e)
    X=np.array(X)
    Y = np.array(Y)

    data = (X,Y)
    #dump to target file if and only if it is training data.
    print("###data_type:",data_type,"transform_data_to_index.finished")
    if data_type=='train':
        #with open(target_file_path, 'ab') as target_file:
        print(data_type,"transform_data_to_index.dump file.target_file_path:",target_file_path)
            #pickle.dump(data, target_file,protocol=pickle.HIGHEST_PROTOCOL)
        np.save(target_file_path+'X.npy', X) # np.save(outfile, x)
        np.save(target_file_path+'Y.npy', Y) # np.save(outfile, x)

    else:
        print("###:data_type:",data_type,";going to return data.")
        return data

def transform_multilabel_as_multihot(label_list,label_size):
    """
    convert to multi-hot style
    :param label_list: e.g.[0,1,4], here 4 means in the 4th position it is true value(as indicate by'1')
    :param label_size: e.g.199
    :return:e.g.[1,1,0,1,0,0,........]
    """
    result=np.zeros(label_size)
    #set those location as 1, all else place as 0.
    result[label_list] = 1
    return result

def transform_mulitihot_as_dense_list(multihot_list):
    length=len(multihot_list)
    result_list=[i for i in range(length) if multihot_list[i] > 0]
    return result_list

def create_or_load_vocabulary(data_path,training_data_path,vocab_size,test_mode=False,tokenize_style='word',fine_tuning_stage=False,model_name=None):
    """
    create or load vocabulary and label using training data.
    process as: load from cache if exist; load data, count and get vocabularies and labels, save to file.
    :param data_path: folder of data
    :param training_data_path: path of training data
    :param vocab_size: size of word vocabulary
    :param test_mode: if True only select few to test functional, else use all data
    :param tokenize_style: tokenize input as word(default) or character.
    :return: vocab_word2index, label2index
    """
    print("create_or_load_vocabulary.data_path:",data_path,";training_data_path:",training_data_path,";vocab_size:",vocab_size,";test_mode:",test_mode,";tokenize_style:",tokenize_style)
    t1 = time.clock()
    if not os.path.isdir(data_path): # create folder if not exists.
        os.makedirs(data_path)

    # 1.if cache exists,load it; otherwise create it.
    if model_name is not None:
        cache_path =data_path+model_name+'vocab_label.pik'
    else:
        cache_path =data_path+'vocab_label.pik'
    print("cache_path:",cache_path,"file_exists:",os.path.exists(cache_path))
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as data_f:
            print("going to load cache file.vocab of words and labels")
            return pickle.load(data_f)

    # 2.load and shuffle raw data
    file_object = codecs.open(training_data_path, mode='r', encoding='utf-8')
    lines=file_object.readlines()
    file_object.close()

    random.shuffle(lines)
    if test_mode:
       lines=lines[0:20000]
    else:
        lines = lines[0:200*1000] # to make create vocabulary process more quicker, we only random select 200k lines.

    # 3.loop each line,put to counter
    c_inputs=Counter()
    c_labels=Counter()
    for i,line in enumerate(lines):
        input_list,input_label=get_input_strings_and_labels(line, tokenize_style=tokenize_style)
        c_inputs.update(input_list)
        c_labels.update(input_label)
        if i % 1000 == 0: # print some information for debug purpose
            print(i,"create_or_load_vocabulary.line:",line)
            print(i,"create_or_load_vocabulary.input_label:",input_label,";input_list:",input_list)

    # 4.get most frequency words and all labels
    if tokenize_style=='char':vocab_size=6000 # if we are using character instead of word, then use small vocabulary size.
    vocab_list=c_inputs.most_common(vocab_size)
    vocab_word2index={}
    vocab_word2index[_PAD]=PAD_ID
    vocab_word2index[_UNK]=UNK_ID
    vocab_word2index[_CLS]=CLS_ID
    vocab_word2index[_MASK]=MASK_ID
    for i,tuplee in enumerate(vocab_list):
        word,freq=tuplee
        vocab_word2index[word]=i+4

    label2index={}
    label_list=c_labels.most_common()
    for i,tuplee in enumerate(label_list):
        label_name, freq = tuplee
        label_name=label_name.strip()
        label2index[label_name]=i

    # 5.save to file system if vocabulary of words not exists.
    if not os.path.exists(cache_path):
        with open(cache_path, 'ab') as data_f:
            print("going to save cache file of vocab of words and labels")
            pickle.dump((vocab_word2index, label2index), data_f)

    t2 = time.clock()
    print('create_vocabulary.ended.time spent for generate training data:', (t2 - t1))
    return vocab_word2index, label2index

def get_lable2index(data_path,training_data_path,tokenize_style='word'):
    """
    get dict of lable to index.
    :param lines: lines from input file
    :param tokenize_style:
    :return:
    """
    cache_file =data_path+"/"+'fine_tuning_label.pik'
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as data_f:
            print("going to load cache file of label for fine-tuning.")
            return pickle.load(data_f)
    file_object = codecs.open(training_data_path, mode='r', encoding='utf-8')
    lines=file_object.readlines()
    random.shuffle(lines)
    lines=lines[0:60000] # only read 100k lines to make training fast
    c_labels=Counter()
    for i,line in enumerate(lines):
        _,input_label=get_input_strings_and_labels(line, tokenize_style=tokenize_style)
        c_labels.update(input_label)
        if i % 1000 == 0: # print some information for debug purpose
            print(i,"get_lable2index.line:",line);print(i,"get_lable2index.input_label:",input_label)
    label2index={}
    label_list=c_labels.most_common()
    for i,tuplee in enumerate(label_list):
        label_name, freq = tuplee
        label_name=label_name.strip()
        label2index[label_name]=i
    if not os.path.exists(cache_file):
        with open(cache_file, 'ab') as data_f:
            print("going to save label dict for fine-tuning.")
            pickle.dump(label2index, data_f)
    return label2index

def get_input_strings_and_labels(line,tokenize_style='word'):
    """
    get input strings and labels by passing a line of raw input.
    :param line:
    :return:
    """
    element_list = line.strip().split(LABEL_SPLITTER)
    input_strings = element_list[0]
    input_list = token_string_as_list(input_strings, tokenize_style=tokenize_style)
    input_labels = element_list[1:]
    input_labels=[str(label).strip() for label in input_labels if label.strip()]
    #print("get_input_strings_and_labels.line:",line,";element_list:",element_list,";input_labels:",input_labels) # input_labels: ['1']
    return input_list,input_labels

def token_string_as_list(string,tokenize_style='word'):
    if random.randint(0, 500) == 1:print("token_string_as_list.string:",string,"tokenize_style:",tokenize_style)
    length=len(string)
    if tokenize_style=='char':
        listt=[string[i] for i in range(length)]
    elif tokenize_style=='word':
        listt=jieba.lcut(string)
    listt=[x for x in listt if x.strip()]
    return listt

def get_part_validation_data(valid,num_valid=6000*20):#6000
    valid_X, valid_X_feature, valid_Y_accusation, valid_Y_article, valid_Y_deathpenalty, valid_Y_lifeimprisonment, valid_Y_imprisonment,weight_accusations,weight_artilces=valid
    number_examples=len(valid_X)
    permutation = np.random.permutation(number_examples)[0:num_valid]
    valid_X2, valid_X2_feature,valid_Y_accusation2, valid_Y_article2, valid_Y_deathpenalty2, valid_Y_lifeimprisonment2, valid_Y_imprisonment2,weight_accusations2,weight_artilces=[],[],[],[],[],[],[],[],[]
    for index in permutation :
        valid_X2.append(valid_X[index])
        valid_X2_feature.append(valid_X_feature[index])
        valid_Y_accusation2.append(valid_Y_accusation[index])
        valid_Y_article2.append(valid_Y_article[index])
        valid_Y_deathpenalty2.append(valid_Y_deathpenalty[index])
        valid_Y_lifeimprisonment2.append(valid_Y_lifeimprisonment[index])
        valid_Y_imprisonment2.append(valid_Y_imprisonment[index])
    return valid_X2,valid_X2_feature,valid_Y_accusation2,valid_Y_article2,valid_Y_deathpenalty2,valid_Y_lifeimprisonment2,valid_Y_imprisonment2,weight_accusations2,weight_artilces

def dump_cache_to_hdf5(cache_file, train, valid, test):
    """
    dump cache to h5
    :param cache_file:
    :param train: a tuple
    :param valid: a tuple
    :param test: a tuple
    :return: return nothing
    """
    # 1.get elements
    X,Y= train
    X_valid,Y_valid=valid
    X_test,Y_test=test

    # 2.save to h5 file
    f = h5py.File(cache_file, 'w')
    f['X'] = X
    f['Y'] = Y

    f['X_valid'] = X_valid
    f['Y_valid'] = Y_valid

    f['X_test'] = X_test
    f['Y_test'] = Y_test
    f.close()

def load_cache_from_hdf5(cache_file):
    """
    load cache from h5
    :param cache_file:
    :return: train,valid, test
    """
    f = h5py.File(cache_file, 'r')
    X = f['X']
    Y = f['Y']
    train = np.array(X),np.array(Y)

    X_valid = f['X_valid']
    Y_valid = f['Y_valid']
    valid = np.array(X_valid),np.array(Y_valid)

    X_test = f['X_test']
    Y_test = f['Y_test']
    test = np.array(X_test),np.array(Y_test)
    f.close()

    return train, valid, test
def pad_truncate_list(x_list, maxlen):
    """
    pad and truncate input to maxlen based on trucating and padding strategy
    :param x_list:e.g. [1,10,3,5,...]
    :return:result_list:a new list,length is maxlen
    """
    result_list=[0 for i in range(maxlen)]
    length_input=len(x_list)
    if length_input>maxlen:
        x_list = x_list[0:maxlen]
    for i, element in enumerate(x_list):
        result_list[i] = element
    return result_list

def assign_pretrained_word_embedding(sess,vocabulary_index2word,vocab_size,word2vec_model_path,embedding_instance,embed_size):
    """
    assign pretrained word embedding
    :param sess:
    :param vocabulary_index2word:
    :param vocab_size:
    :param model:
    :param word2vec_model_path:
    :param embedding_instance:
    :return:
    """
    print("using pre-trained word emebedding.started.word2vec_model_path:",word2vec_model_path,";vocab_size:",vocab_size,";embed_size:",embed_size)
    word2vec_dict=load_word2vec(word2vec_model_path,embed_size)
    word_embedding_2dlist = [[]] * vocab_size        # create an empty word_embedding list.
    word_embedding_2dlist[0] = np.zeros(embed_size)  # assign empty for first word:'PAD'
    word_embedding_2dlist[1] = np.zeros(embed_size)  # assign empty for second word:'UNK'
    word_embedding_2dlist[2] = np.zeros(embed_size)  # assign empty for third word:'CLS'
    word_embedding_2dlist[3] = np.zeros(embed_size)  # assign empty for third word:'MASK'

    bound = np.sqrt(0.3) / np.sqrt(vocab_size)  # bound for random variables.3.0
    count_exist = 0;
    count_not_exist = 0
    for i in range(4, vocab_size):  # loop each word
        word = vocabulary_index2word[i]  # get a word
        embedding = None
        try:
            embedding = word2vec_dict[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding;
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, embed_size);
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(embedding_instance,word_embedding)
    sess.run(t_assign_embedding)
    print("====>>>>word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")

def load_word2vec(word2vec_model_path,embed_size):
    """
    load pretrained word2vec in txt format
    :param word2vec_model_path:
    :return: word2vec_dict. word2vec_dict[word]=vector
    """
    #word2vec_object = codecs.open(word2vec_model_path,'r','utf-8') #open(word2vec_model_path,'r')
    #lines=word2vec_object.readlines()
    #word2vec_dict={}
    #for i,line in enumerate(lines):
    #    if i==0: continue
    #    string_list=line.strip().split(" ")
    #    word=string_list[0]
    #    vector=string_list[1:][0:embed_size]
    #    word2vec_dict[word]=vector
    ######################
    word2vec_dict = {}
    with open(word2vec_model_path, errors='ignore') as f:
        meta = f.readline()
        for line in f.readlines():
            items = line.split(' ')
            #if len(items[0]) > 1 and items[0] in vocab:
            word2vec_dict[items[0]] = np.fromiter(items[1:][0:embed_size], dtype=float)
    return word2vec_dict

def set_config(FLAGS,num_classes,vocab_size):
    config=Config()

    config.learning_rate = FLAGS.learning_rate
    config.batch_size = FLAGS.batch_size
    config.sequence_length = FLAGS.sequence_length
    config.vocab_size = vocab_size
    config.dropout_keep_prob = FLAGS.dropout_keep_prob
    config.num_classes=num_classes
    config.is_training = FLAGS.is_training
    config.d_model=FLAGS.d_model
    config.num_layer=FLAGS.num_layer
    config.h=FLAGS.num_header
    config.d_k=FLAGS.d_k
    config.d_v=FLAGS.d_v

    config.clip_gradients = 5.0
    config.decay_steps = 1000
    config.decay_rate = 0.9
    config.ckpt_dir = 'checkpoint/dummy_test/'
    #config.sequence_length_lm=FLAGS.max_allow_sentence_length
    config.num_classes_lm=vocab_size
    #config.is_pretrain=FLAGS.is_pretrain
    config.sequence_length_lm=FLAGS.sequence_length_lm
    config.is_fine_tuning=FLAGS.is_fine_tuning

    return config

# below is for testing create_or_load_vocabulary,load_data_multilabel/
data_path='./data/'
traning_data_path=data_path+'xxx_20181022_train.txt'
valid_data_path=data_path+'xxx_20181022_test.txt'
test_data_path=valid_data_path
vocab_size=50000
process_num=5
test_mode=True
sentence_len=200
#vocab_word2index, label2index=create_or_load_vocabulary(data_path,traning_data_path,vocab_size,test_mode=False)
#print(";vocab_word2index:",len(vocab_word2index),";label2index:",label2index)
#train, valid, test = load_data_multilabel(data_path, traning_data_path,valid_data_path,test_data_path, vocab_word2index,
 #                                         label2index, sentence_len,process_num=process_num, test_mode=test_mode)
#print("train.shape:",train[0].shape,train[1].shape,";valid.shape:",valid[0].shape,valid[1].shape,";test.shape:",test[0].shape,test[1].shape)