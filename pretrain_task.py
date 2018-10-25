# -*- coding: utf-8 -*-
import jieba
from data_util_hdf5 import PAD_ID,UNK_ID,MASK_ID,_PAD,_UNK,_MASK,create_or_load_vocabulary
import random
import re
import numpy as np
import os
import time
import pickle
splitter='|&|'
eighty_percentage=0.8
nighty_percentage=0.9

"""
data generator for two tasks: 1. masked language mode; 2. predict next sentence 
"""
def mask_language_model(source_file,target_file,index2word,max_allow_sentence_length=10,test_mode=False):
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
