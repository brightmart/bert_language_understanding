# -*- coding: utf-8 -*-
import jieba
from data_util_hdf5 import PAD_ID,UNK_ID,MASK_ID,_PAD,_UNK,_MASK,create_or_load_vocabulary
import random
import re
import numpy as np
import os
import time
import pickle
import multiprocessing

splitter='|&|'
eighty_percentage=0.8
nighty_percentage=0.9

"""
data generator for two tasks: 1. masked language mode; 2. predict next sentence 
"""
def mask_language_model(source_file,target_file,index2word,max_allow_sentence_length=10,test_mode=False,process_num=10):
    """
    generate data for perform mask language model.
    :parameter source_file: source file where raw data come from
    :parameter target_file: save processed data to target file
    :parameter index2word: a dict, it is a dictionary of word, given a index, you can get a word
    :parameter max_allow_sentence_length

    we feed the input through a deep Transformer encoder and then use the final hidden states corresponding to the masked positions to
    predict what word was masked, exactly like we would train a language model.

    source_file each line is a sequence of token, can be a sentence.
    Input Sequence  : The man went to [MASK] store with [MASK] dog
    Target Sequence :                  the                his

    try prototype first, that is only select one word

    the training data generator chooses 15% of tokens at random,e.g., in the sentence 'my dog is hairy it chooses hairy'. It then performs the following procedure:

    80% of the time: Replace the word with the [MASK] token, e.g., my dog is hairy → my dog is [MASK]
    10% of the time: Replace the word with a random word,e.g.,my dog is hairy → my dog is apple
    10% of the time: Keep the word un- changed, e.g., my dog is hairy → my dog is hairy.
    The purpose of this is to bias the representation towards the actual observed word.

    :parameter sentence_length: sentence length. all the sentence will be pad or truncate to this length.
    :return: list of tuple. each tuple has a input_sequence, and target_sequence: (input_sequence, target_sequence)
    """
    # 1. read source file
    t1 = time.clock()
    cache_file=target_file + 'mask_lm.pik'
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as data_f:
            print("going to load cache file of masked language model.")
            train,valid,test=pickle.load(data_f)
            return train,valid,test

    X_mask_lm=[]
    y_mask_lm=[]
    p_mask_lm=[]
    source_object=open(source_file,'r')
    source_lines=source_object.readlines()
    if test_mode:
        source_lines=source_lines[0:1000]
    list_indices = [i for i in range(max_allow_sentence_length)]
    count=0
    vocab_size=len(index2word)
    word2index={v:k for k,v in index2word.items()}
    # 2. loop each line, split document into sentence. for each sentence, try generate a input for mask language model.
    for i, line in enumerate(source_lines):
        sentence_list = re.sub('[,.。；，]', splitter, line).split(splitter)
        for j,sentence in enumerate(sentence_list):
            sentence=sentence.lower().strip()
            string_list=[x for x in jieba.lcut(sentence.strip()) if x and x not in ["\"","：","、","，","）","（"]]
            sentence_length=len(string_list)
            if sentence_length>max_allow_sentence_length: # string list is longer then sentence_length
                string_list=string_list[0:max_allow_sentence_length]
            else: # ignore short sentence currently, temporary
                continue
            # the training data generator chooses 15% of tokens at random,e.g., in the sentence my dog is hairy it chooses hairy.
            index=random.sample(list_indices, 1)
            index=index[0]
            mask_word=string_list[index]
            ########### 3.THREE DIFFERENT TYPES when generate sentence.#################################################
            random_number=random.random()
            if random_number<=eighty_percentage: #  80% of the time: Replace the word with the [MASK] token, e.g., my dog is hairy → my dog is [MASK]
                string_list[index]=_MASK
            elif random_number<=nighty_percentage: # 10% of the time: Replace the word with a random word,e.g.,my dog is hairy → my dog is apple
                random_word = random.randint(3, vocab_size)
                string_list[index]=index2word.get(random_word,_UNK)
            else: # 10% of the time: Keep the word un- changed, e.g., my dog is hairy → my dog is hairy.
                string_list[index]=mask_word
            ###########THREE DIFFERENT TYPES when generate sentence.####################################################
            # 4. process sentence/word as indices/index
            string_list_indexed=[word2index.get(x,UNK_ID) for x in string_list]
            mask_word_indexed=word2index.get(mask_word,UNK_ID)
            X_mask_lm.append(string_list_indexed) # input(x) to list
            y_mask_lm.append(mask_word_indexed) # input(y) to list
            p_mask_lm.append(index)

            count=count+1
            if i%1000==0:
                print(count,"index:",index,"i:",i,"j:",j,";mask_word_1:",mask_word,";string_list:",string_list)
                print(count,"index:",index,"i:",i,"j:",j,";mask_word_indexed:",mask_word_indexed,";string_list_indexed:",string_list_indexed)
            #if count%101==0: break

    # 5. split data into train/valid/test, and save to file system
    num_examples=len(X_mask_lm)
    num_train_index=int(0.9*num_examples)
    num_valid_index=int(0.95*num_examples)
    X_mask_lm_train,y_mask_lm_train,p_mask_lm_train=X_mask_lm[0:num_train_index],y_mask_lm[0:num_train_index],p_mask_lm[0:num_train_index]
    X_mask_lm_valid,y_mask_lm_valid,p_mask_lm_valid=X_mask_lm[num_train_index:num_valid_index],y_mask_lm[num_train_index:num_valid_index],p_mask_lm[num_train_index:num_valid_index]
    X_mask_lm_test,y_mask_lm_test,p_mask_lm_test=X_mask_lm[num_valid_index:],y_mask_lm[num_valid_index:],p_mask_lm[num_valid_index:]
    train=get_data_as_array(X_mask_lm_train,y_mask_lm_train,p_mask_lm_train)
    valid=get_data_as_array(X_mask_lm_valid,y_mask_lm_valid,p_mask_lm_valid)
    test=get_data_as_array(X_mask_lm_test,y_mask_lm_test,p_mask_lm_test)

    if not os.path.exists(cache_file):
        with open(cache_file, 'ab') as data_f:
            print("going to save cache file of masked langauge model.")
            pickle.dump((train,valid,test), data_f)

    t2 = time.clock()
    print('mask_language_model.ended.time spent:', (t2 - t1))
    return train ,valid,test

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

def mask_language_model_multi_processing(source_file,target_file,index2word,max_allow_sentence_length=10,test_mode=False,process_num=10):
    """
    generate data for perform mask language model.
    :parameter source_file: source file where raw data come from
    :parameter target_file: save processed data to target file
    :parameter index2word: a dict, it is a dictionary of word, given a index, you can get a word
    :parameter max_allow_sentence_length

    :parameter sentence_length: sentence length. all the sentence will be pad or truncate to this length.
    :return: list of tuple. each tuple has a input_sequence, and target_sequence: (input_sequence, target_sequence)
    """
    # 1. read source file
    t1 = time.clock()
    cache_file=target_file + 'mask_lm.pik'
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as data_f:
            print("going to load cache file of masked language model.")
            train,valid,test=pickle.load(data_f)
            return train,valid,test

    source_object=open(source_file,'r')
    source_lines=source_object.readlines()
    if test_mode:
        source_lines=source_lines[0:1000]

    # 2.multi-processing to process lines #############################################################################
    # 2.1 get chunks as list.
    chunks = build_chunk(source_lines, chunk_num=process_num - 1)
    pool = multiprocessing.Pool(processes=process_num)
    # 2.2 use multiprocessing to handle different chunk. each chunk will be transformed and save to file system.
    for chunk_id, each_chunk in enumerate(chunks):
        file_name= data_path+"training_data_temp_lm_" + str(chunk_id)
        # 3.3 apply_async
        print("#mask_language_model_multi_processing.length of chunk:",len(each_chunk),";file_name:",file_name,";chunk_id:",chunk_id)
        pool.apply_async(process_one_chunk_lm,args=(each_chunk,max_allow_sentence_length,index2word,file_name)) # apply_async
    pool.close()
    pool.join()
    print("mask_language_model_multi_processing.load_data_multilabel.finish all of sub tasks.")

    # 2.3 merge sub file to final file, and remove.
    X_mask_lm=[]
    y_mask_lm=[]
    p_mask_lm=[]
    for chunk_id in range(process_num):
        file_name_X =data_path+"training_data_temp_lm_" + str(chunk_id)+'_X_mask_lm.npy'
        file_name_y =data_path+"training_data_temp_lm_" + str(chunk_id)+'_y_mask_lm.npy'
        file_name_p =data_path+"training_data_temp_lm_" + str(chunk_id)+'_p_mask_lm.npy'
        x_sub=np.load(file_name_X)
        y_sub=np.load(file_name_y)
        p_sub=np.load(file_name_p)
        X_mask_lm.extend(x_sub)
        y_mask_lm.extend(y_sub)
        p_mask_lm.extend(p_sub)
        command = 'rm ' + file_name_X+" "+file_name_y+" "+file_name_p
        os.system(command)
    # 2.multi-processing to process lines #############################################################################

    # 3. split to train, valid, test
    num_examples=len(X_mask_lm)
    num_train_index=int(0.9*num_examples)
    num_valid_index=int(0.95*num_examples)
    X_mask_lm_train,y_mask_lm_train,p_mask_lm_train=X_mask_lm[0:num_train_index],y_mask_lm[0:num_train_index],p_mask_lm[0:num_train_index]
    X_mask_lm_valid,y_mask_lm_valid,p_mask_lm_valid=X_mask_lm[num_train_index:num_valid_index],y_mask_lm[num_train_index:num_valid_index],p_mask_lm[num_train_index:num_valid_index]
    X_mask_lm_test,y_mask_lm_test,p_mask_lm_test=X_mask_lm[num_valid_index:],y_mask_lm[num_valid_index:],p_mask_lm[num_valid_index:]
    train=get_data_as_array(X_mask_lm_train,y_mask_lm_train,p_mask_lm_train)
    valid=get_data_as_array(X_mask_lm_valid,y_mask_lm_valid,p_mask_lm_valid)
    test=get_data_as_array(X_mask_lm_test,y_mask_lm_test,p_mask_lm_test)

    # 4. save to file system as cache file if not exist.
    if not os.path.exists(cache_file):
        with open(cache_file, 'ab') as data_f:
            print("going to save cache file of masked langauge model.")
            pickle.dump((train,valid,test), data_f)

    return train ,valid,test

def process_one_chunk_lm(lines,max_allow_sentence_length,index2word,sub_target_file_path):
    """
    process one chunk for generate data of language model
    :return:
    """
    print("process_one_chunk_lm.started...")
    list_indices = [i for i in range(max_allow_sentence_length)]
    word2index={v:k for k,v in index2word.items()}
    X_mask_lm=[]
    y_mask_lm=[]
    p_mask_lm=[]
    for i, line in enumerate(lines):
        sentence_list = re.sub('[,.。；，]', splitter, line).split(splitter)
        for j, sentence in enumerate(sentence_list):
            sentence = sentence.lower().strip()
            string_list = [x for x in jieba.lcut(sentence.strip()) if x and x not in ["\"", "：", "、", "，", "）", "（"]]
            sentence_length = len(string_list)
            if sentence_length > max_allow_sentence_length:  # string list is longer then sentence_length
                string_list = string_list[0:max_allow_sentence_length]
            else:  # ignore short sentence currently, temporary
                continue
            # the training data generator chooses 15% of tokens at random,e.g., in the sentence my dog is hairy it chooses hairy.
            index = random.sample(list_indices, 1)
            index = index[0]
            mask_word = string_list[index]
            ########### 3.THREE DIFFERENT TYPES when generate sentence.#################################################################
            random_number = random.random()
            if random_number <= eighty_percentage:  # 80% of the time: Replace the word with the [MASK] token, e.g., my dog is hairy → my dog is [MASK]
                string_list[index] = _MASK
            elif random_number <= nighty_percentage:  # 10% of the time: Replace the word with a random word,e.g.,my dog is hairy → my dog is apple
                random_word = random.randint(3, vocab_size)
                string_list[index] = index2word.get(random_word, _UNK)
            else:  # 10% of the time: Keep the word un- changed, e.g., my dog is hairy → my dog is hairy.
                string_list[index] = mask_word
            ###########THREE DIFFERENT TYPES when generate sentence.#####################################################################
            # process sentence/word as indices/index
            string_list_indexed = [word2index.get(x, UNK_ID) for x in string_list]
            mask_word_indexed = word2index.get(mask_word, UNK_ID)
            # append to list
            X_mask_lm.append(string_list_indexed)  # input(x) to list
            y_mask_lm.append(mask_word_indexed)  # input(y) to list
            p_mask_lm.append(index)
            # print some log
            if i % 1000 == 0:
                print("index:", index, "i:", i, "j:", j, ";mask_word_1:", mask_word, ";string_list:",string_list)
                print( "index:", index, "i:", i, "j:", j, ";mask_word_indexed:", mask_word_indexed,";string_list_indexed:", string_list_indexed)
    # save to file system
    X_mask_lm=np.array(X_mask_lm)
    y_mask_lm=np.array(y_mask_lm)
    p_mask_lm=np.array(p_mask_lm)
    np.save(sub_target_file_path+'_X_mask_lm.npy',X_mask_lm) # file_name_X =data_path+"training_data_temp_lm_" + str(chunk_id)+'_X_mask_lm.npy'
    np.save(sub_target_file_path+'_y_mask_lm.npy',y_mask_lm)
    np.save(sub_target_file_path+'_p_mask_lm.npy',p_mask_lm)
    print("process_one_chunk_lm.ended.file saved:",sub_target_file_path+'_X_mask_lm.npy')

def get_data_as_array(X_mask_lm_train,y_mask_lm_train,p_mask_lm_train):
    """
    get data as array
    :param X_mask_lm_train:
    :param y_mask_lm_train:
    :param p_mask_lm_train:
    :return:
    """
    return np.array(X_mask_lm_train),np.array(y_mask_lm_train),np.array(p_mask_lm_train)

source_file='./data/l_20181024_union.txt'
data_path='./data/'
traning_data_path=data_path+'l_20181024_union.txt'
valid_data_path=data_path+'l_20181024_union_valid.txt'
test_data_path=valid_data_path
vocab_size=50000
process_num=5
test_mode=True
sentence_len=200
#vocab_word2index, label2index=create_or_load_vocabulary(data_path,traning_data_path,vocab_size,test_mode=False)
#index2word={v:k for k,v in vocab_word2index.items()}
#X_mask_lm,y_mask_lm,p_mask_lm=mask_language_model(source_file,data_path,index2word,max_allow_sentence_length=10)#print("X_mask_lm:",X_mask_lm)
#print("X_mask_lm:",X_mask_lm.shape,";y_mask_lm:",y_mask_lm,";p_mask_lm:",p_mask_lm)
