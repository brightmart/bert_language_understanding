# -*- coding: utf-8 -*-
import numpy as np
import random
import codecs
"""
compute single evaulation matrix for task1,task2 and task3:
compute f1 score(micro,macro) for accusation & relevant article, and score for pentaly
"""

small_value=0.00001
random_number=500
def compute_confuse_matrix_batch(y_targetlabel_list,y_logits_array,label_dict,name='default'):
    """
    compute confuse matrix for a batch
    :param y_targetlabel_list: a list; each element is a mulit-hot,e.g. [1,0,0,1,...]
    :param y_logits_array: a 2-d array. [batch_size,num_class]
    :param label_dict:{label:(TP, FP, FN)}
    :param name: a string for debug purpose
    :return:label_dict:{label:(TP, FP, FN)}
    """
    for i,y_targetlabel_list_single in enumerate(y_targetlabel_list):
        label_dict=compute_confuse_matrix(y_targetlabel_list_single,y_logits_array[i],label_dict,name=name)
    return label_dict

def compute_confuse_matrix(y_targetlabel_list_single,y_logit_array_single,label_dict,name='default'):
    """
    compute true postive(TP), false postive(FP), false negative(FN) given target lable and predict label
    :param y_targetlabel_list: a list. length is batch_size(e.g.1). each element is a multi-hot,like '[0,0,1,0,1,...]'
    :param y_logit_array: an numpy array. shape is:[batch_size,num_classes]
    :param label_dict {label:(TP,FP,FN)}
    :return: macro_f1(a scalar),micro_f1(a scalar)
    """
    #1.get target label and predict label
    y_target_labels=get_target_label_short(y_targetlabel_list_single) #e.g. y_targetlabel_list[0]=[2,12,88]
    #y_logit=y_logit_array_single #y_logit_array[0] #[202]
    y_predict_labels=[i for i in range(len(y_logit_array_single)) if y_logit_array_single[i]>=0.50] #TODO 0.5PW e.g.[2,12,13,10]
    if len(y_predict_labels) < 1: y_predict_labels = [np.argmax(y_logit_array_single)]

    #if len(y_predict_labels)<1:    y_predict_labels=[np.argmax(y_logit_array_single)] #TODO ADD 2018.05.29
    if random.choice([x for x in range(random_number)]) ==1:print(name+".y_target_labels:",y_target_labels,";y_predict_labels:",y_predict_labels) #debug purpose

    #2.count number of TP,FP,FN for each class
    y_labels_unique=[]
    y_labels_unique.extend(y_target_labels)
    y_labels_unique.extend(y_predict_labels)
    y_labels_unique=list(set(y_labels_unique))
    for i,label in enumerate(y_labels_unique): #e.g. label=2
        TP, FP, FN = label_dict[label]
        if label in y_predict_labels and label in y_target_labels:#predict=1,truth=1 (TP)
            TP=TP+1
        elif label in y_predict_labels and label not in y_target_labels:#predict=1,truth=0(FP)
            FP=FP+1
        elif label not in y_predict_labels and label in y_target_labels:#predict=0,truth=1(FN)
            FN=FN+1
        label_dict[label] = (TP, FP, FN)
    return label_dict


def compute_penalty_score_batch(target_deaths, predict_deaths,target_lifeimprisons, predict_lifeimprisons,target_imprsions, predict_imprisons):
    """
    compute penalty score(task 3) for a batch.
    :param target_deaths: a list. each element is a mulit-hot list
    :param predict_deaths: a 2-d array. [batch_size,num_class]
    :param target_lifeimprisons: a list. each element is a mulit-hot list
    :param predict_lifeimprisons: a 2-d array. [batch_size,num_class]
    :param target_imprsions: a list. each element is a mulit-hot list
    :param predict_imprisons: a 2-d array. [batch_size,num_class]
    :return: score_batch: a scalar, average score for that batch
    """
    length=len(target_deaths)
    score_total=0.0
    for i in range(length):
        score=compute_penalty_score(target_deaths[i], predict_deaths[i], target_lifeimprisons[i],predict_lifeimprisons[i],target_imprsions[i], predict_imprisons[i])
        score_total=score_total+score
    score_batch=score_total/float(length)
    return score_batch

def compute_penalty_score(target_death, predict_death,target_lifeimprison, predict_lifeimprison,target_imprsion, predict_imprison):
    """
    compute penalty score(task 3) for a single data
    :param target_death:  a mulit-hot list. e.g. [1,0,0,1,...]
    :param predict_death: [num_class]
    :param target_lifeimprison: a mulit-hot list. e.g. [1,0,0,1,...]
    :param predict_lifeimprison: [num_class]
    :param target_imprsion: a mulit-hot list. e.g. [1,0,0,1,...]
    :param predict_imprison:[num_class]
    :return: score: a scalar,score for this data
    """
    score_death=compute_death_lifeimprisonment_score(target_death, predict_death)
    score_lifeimprisonment=compute_death_lifeimprisonment_score(target_lifeimprison, predict_lifeimprison)
    score_imprisonment=compute_imprisonment_score(target_imprsion, predict_imprison)
    score=((score_death+score_lifeimprisonment+score_imprisonment)/3.0)*(100.0)
    return score

def compute_death_lifeimprisonment_score(target,predict):
    """
    compute score for death or life imprisonment
    :param target: a list
    :param predict: an array
    :return: score: a scalar
    """

    score=0.0
    target=np.argmax(target)
    predict=np.argmax(predict)
    if random.choice([x for x in range(random_number)]) == 1:print("death_lifeimprisonment_score.target:", target, ";predict:", predict)
    if target==predict:
        score=1.0
    if random.choice([x for x in range(random_number)]) == 1:print("death_lifeimprisonment_score:",score)
    return score

def compute_imprisonment_score(target_value,predict_value):
    """
    compute imprisonment score
    :param target_value: a scalar
    :param predict_value:a scalar
    :return: score: a scalar
    """
    if random.choice([x for x in range(random_number)]) ==1:print("x.imprisonment_score.target_value:",target_value,";predict_value:",predict_value)
    score=0.0
    v=np.abs(np.log(predict_value+1.0)-np.log(target_value+1.0))
    if v<=0.2:
        score=1.0
    elif v<=0.4:
        score=0.8
    elif v<=0.6:
        score=0.6
    elif v<=0.8:
        score=0.4
    elif v<=1.0:
        score=0.2
    else:
        score=0.0
    if random.choice([x for x in range(random_number)]) ==1:print("imprisonment_score:",score)
    return score

def compute_micro_macro(label_dict):
    """
    compute f1 of micro and macro
    :param label_dict:
    :return: f1_micro,f1_macro: scalar, scalar
    """
    f1_micro = compute_f1_micro_use_TFFPFN(label_dict)
    f1_macro= compute_f1_macro_use_TFFPFN(label_dict)
    return f1_micro,f1_macro

def compute_f1_micro_use_TFFPFN(label_dict):
    """
    compute f1_micro
    :param label_dict: {label:(TP,FP,FN)}
    :return: f1_micro: a scalar
    """
    TF_micro_accusation, FP_micro_accusation, FN_micro_accusation =compute_TF_FP_FN_micro(label_dict)
    f1_micro_accusation = compute_f1(TF_micro_accusation, FP_micro_accusation, FN_micro_accusation,'micro')
    return f1_micro_accusation

def compute_f1_macro_use_TFFPFN(label_dict):
    """
    compute f1_macro
    :param label_dict: {label:(TP,FP,FN)}
    :return: f1_macro
    """
    f1_dict= {}
    num_classes=len(label_dict)
    for label, tuplee in label_dict.items():
        TP,FP,FN=tuplee
        f1_score_onelabel=compute_f1(TP,FP,FN,'macro')
        f1_dict[label]=f1_score_onelabel
    f1_score_sum=0.0
    for label,f1_score in f1_dict.items():
        f1_score_sum=f1_score_sum+f1_score
    f1_score=f1_score_sum/float(num_classes)
    return f1_score

#[this function is for debug purpose only]
def compute_f1_score_write_for_debug(label_dict,label2index):
    """
    compute f1 score. basicly you can also use other function to get result
    :param label_dict: {label:(TP,FP,FN)}
    :return: a dict. key is label name, value is f1 score.
    """
    f1score_dict={}
    # 1. compute f1 score for each accusation.
    for label, tuplee in label_dict.items():
        TP, FP, FN = tuplee
        f1_score_single = compute_f1(TP, FP, FN, 'normal_f1_score')
        accusation_index2label = {kv[1]: kv[0] for kv in label2index.items()}
        label_name=accusation_index2label[label]
        f1score_dict[label_name]=f1_score_single

    # 2. each to file system for debug purpose.
    f1score_file='debug_accuracy.txt'
    write_object = codecs.open(f1score_file, mode='a', encoding='utf-8')
    write_object.write("\n\n")

    #tuple_list = sorted(f1score_dict.items(), lambda x, y: cmp(x[1], y[1]), reverse=False)
    tuple_list = sorted(f1score_dict.items(), key=lambda x: x[1], reverse=False)

    for tuplee in tuple_list:
        label_name,f1_score=tuplee
        write_object.write(label_name+":"+str(f1_score)+"\n")
    write_object.close()
    return f1score_dict

def compute_f1(TP,FP,FN,compute_type):
    """
    compute f1
    :param TP_micro: number.e.g. 200
    :param FP_micro: number.e.g. 200
    :param FN_micro: number.e.g. 200
    :return: f1_score: a scalar
    """
    precison=TP/(TP+FP+small_value)
    recall=TP/(TP+FN+small_value)
    f1_score=(2*precison*recall)/(precison+recall+small_value)

    if random.choice([x for x in range(500)]) == 1:print(compute_type,"precison:",str(precison),";recall:",str(recall),";f1_score:",f1_score)

    return f1_score

def compute_TF_FP_FN_micro(label_dict):
    """
    compute micro FP,FP,FN
    :param label_dict_accusation: a dict. {label:(TP, FP, FN)}
    :return:TP_micro,FP_micro,FN_micro
    """
    TP_micro,FP_micro,FN_micro=0.0,0.0,0.0
    for label,tuplee in label_dict.items():
        TP,FP,FN=tuplee
        TP_micro=TP_micro+TP
        FP_micro=FP_micro+FP
        FN_micro=FN_micro+FN
    return TP_micro,FP_micro,FN_micro

def init_label_dict(num_classes):
    """
    init label dict. this dict will be used to save TP,FP,FN
    :param num_classes:
    :return: label_dict: a dict. {label_index:(0,0,0)}
    """
    label_dict={}
    for i in range(num_classes):
        label_dict[i]=(0,0,0)
    return label_dict

def get_target_label_short(y_mulitihot):
    """
    get target label.
    :param y_mulitihot: [0,0,1,0,1,0,...]
    :return: taget_list.e.g. [3,5,100]
    """
    taget_list = [];
    for i, element in enumerate(y_mulitihot):
        if element == 1:
            taget_list.append(i)
    return taget_list