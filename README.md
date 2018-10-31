# bert for language understanding

## Table of Contents

#### 1.Introduction

#### 2.Performance

#### 3.Usage

#### 4.Sample Data, Data Format 

#### 5.Suggestion to User

#### 6.Short Description of BERT

#### 7.Long Description of BERT from author

#### 8.Pretrain Language Understanding Task 

#### 9.Environment

#### 10.Implementation Details

#### 11.Questions for Better Understanding of Transformer and BERT

#### 12.Toy Task

#### 13.Multi-label Classification Task

#### 14.TODO List

#### 15.Conclusion

#### 16.References



## Introduction
##### Pre-train is all you need!

BERT achieve new state of art result on more than 10 nlp tasks recently.

This is an tensorflow implementation of Pre-training of Deep Bidirectional Transformers for Language Understanding

(Bert) and Attention is all you need(Transformer). 

Update: The majority part of replicate main ideas of these two papers was done, there is a apparent performance gain 
  
for pre-train a model & fine-tuning compare to train the model from scratch.


##### Experiment with pre-trian and fine-tuning 
We have done experiment to replace backbone network of bert from Transformer to TextCNN, and the result is that 

pre-train the model with masked language model using lots of raw data can boost performance in a notable amount. 

More generally, we believe that pre-train and fine-tuning strategy is model independent and pre-train task independent. 

with that being said, you can replace backbone network as you like. and add more pre-train tasks or define some new pre-train tasks as 

you can, pre-train will not be limited to masked language model and or predict next sentence task. What surprise us is that,

with a middle size data set that say, one million, even without use external data, with the help of pre-train task 

like masked language model, performance can be boost in a big margin, and the model can converge even fast. sometime 

training can be in a only need a few epoch in fine-tuning stage.

##### Intention
While there is an open source(<a href='https://github.com/tensorflow/tensor2tensor'>tensor2tensor</a>) and official

implementation of Transformer and BERT official implementation coming soon, but there are/may hard to read, not easy to understand. 

We are not intent to replicate original papers entirely, but to apply the main ideas and solve nlp problem in a better way.

The majority part fo work here was done by another repository last year: <a href='https://github.com/brightmart/text_classification'>text classification</a>

## Performance 

MIDDLE SIZE DATASET(<a href='https://pan.baidu.com/s/1HUzBXB_-zzqv-abWZ74w2Q'>cail2018</a>, 450k)

Model                        | TextCNN(No-pretrain)| TextCNN(Pretrain-Finetuning)| Gain from pre-train 
---                          | ---                 | ---                         | -----------     
F1 Score after 1 epoch       |  0.09               | 0.58                        |  0.49        
F1 Score after 5 epoch       |  0.40               | 0.74                        |  0.35    
F1 Score after 7 epoch       |  0.44               | 0.75                        |  0.31                         
F1 Score after 35 epoch      |  0.58               | 0.75                        |  0.27                                                
Training Loss at beginning   |  284.0              | 84.3                        |  199.7             
Validation Loss after 1 epoch|  13.3               | 1.9                         |  11.4                 
Validation Loss after 5 epoch|  6.7                | 1.3                         |  5.4                              
----------------------------------------------------------------------------------------------
Notice: 

  a.pre-train model completed training after just running 7 epoch as max epoch reached to 35. 

     in fact, pre-train model start training from epoch 27 where pre-train stage ended.

  c.f1 Score reported here is on validation set, an average of micro and macro of f1 score. 

  d.f1 score after 35 epoch is reported on test set.


SMALL SIZE DATASET(private, 100k)

Model                        | TextCNN(No-pretrain) | TextCNN(Pretrain-Finetuning) | Gain from pre-train 
---                          | ---                  | ---                          | -----------                
F1 Score after 1 epoch       |  0.44                | 0.57                         | 10%+  
Validation Loss after 1 epoch|  55.1                | 1.0                          |  54.1                
Training Loss at beginning    |  68.5                | 8.2                         | 60.3                
            
------------------------------------------------------------------------------------------------



## Usage

if you want to try BERT with pre-train of masked language model and fine-tuning. take two steps:

  ##### [step 1]  pre-train masked language with BERT: 
     
     python train_bert_lm.py [DONE]
 
  ##### [step 2] fine-tuning:  
   
     python train_bert_fine_tuning.py [Done]
    
   as you can see, even at the start point of fine-tuning, just after restore parameters from pre-trained model, the loss of model is smaller
   
   than training from completely new, and f1 score is also higher while new model may start from 0.
   
   Notice: to help you try new idea first, you can set hyper-paramater test_mode to True. it will only load few data, and start to training quickly.
  
  
   ##### [basic step] to handle a classification problem with transform(optional): 
    
     python train_transform.py [DONE, but a bug exist prevent it from converge, welcome you to fix, email: brightmart@hotmail.com]
        
  #### Optional hyper-parameters
  
  d_model: dimension of model.   [512]
  
  num_layer: number of layers. [6]
  
  num_header: number of headers of self-attention [8]
  
  d_k: dimension of Key(K). dimension of Query(Q) is the same. [64]
  
  d_v: dimension of V. [64]
  
    default hyperparameter is d_model=512,h=8,d_k=d_v=64(big). if you have want to train the model fast, or has a small data set 
    
    or want to train a small model, use d_model=128,h=8,d_k=d_v=16(small), or d_model=64,h=8,d_k=d_v=8(tiny).
  
 
## Sample Data & Data Format

##### for pre-train stage 
each line is document(several sentences) or a sentence. that is free-text you can get easily.

check data/bert_train.txt or bert_train2.txt in the zip file.

##### for data used on fine-tuning stage:

input and output is in the same line, each label is start with '__label__'. 

there is a space between input string and the first label, each label is also splitted by a space.

e.g. 
token1 token2 token3 __label__l1 __label__l5 __label__l3

token1 token2 token3 __label__l2 __label__l4

check data/bert_train.txt or bert_train2.txt in the zip file.

check 'data' folder for sample data. <a href='https://pan.baidu.com/s/1HUzBXB_-zzqv-abWZ74w2Q'>down load a middle size data set here

</a>with 450k 206 classes, each input is a document, average length is around 300, one or multi-label associate with input.

## Suggestion for User

1. things can be easy: 
           
      1) download data set(around 200M, 450k data, with some cache file), unzip it and put it in data/ folder,
           
      2) run step 1 for pre-train, 
           
      3) and run step 2 for fine-tuning.

2. i finish above three steps, and want to have a better performance, how can i do further. do i need to find a big dataset?

     No. you can generate a big data set yourself for pre-train stage by downloading some free-text, make sure each line 
     
     is a document or sentence then replace data/bert_train2.txt with your new data file.

3. what's more?

     try some big hyper-parameter or big model(by replacing backbone network) util you can observe all your pre-train data.

     play around with model:model/bert_cnn_model.py, or check pre-process with data_util_hdf5.py.



## Short Description of BERT:
Pretrain mashed language model and next sentence prediction task on large scale of corpus, 

based on multiple layer self-attetion model, then fine tuning by add a classification layer.

As BERT model is based on Transformer, currently we are working on add pretrain task to the model.

<img src="https://github.com/brightmart/bert_language_understanding/blob/master/data/aa3.jpeg"  width="60%" height="60%" />

<img src="https://github.com/brightmart/bert_language_understanding/blob/master/data/aa4.jpeg"  width="65%" height="65%" />


Notice: 
 cail2018 is around 450k as link above.

 training size of private data set is around 100k, number of classes is 9, for each input there exist one or more label(s).
 
 f1 score for cail2018 is reported as micro f1 score.

## Long Description of BERT from author
The basic idea is very simple. For several years, people have been getting very good results "pre-training" DNNs as a language model 

and then fine-tuning on some downstream NLP task (question answering, natural language inference, sentiment analysis, etc.).

Language models are typically left-to-right, e.g.:

    "the man went to a store"

     P(the | <s>)*P(man|<s> the)*P(went|<s> the man)*…

The problem is that for the downstream task you usually don't want a language model, you want a the best possible contextual representation of 

each word. If each word can only see context to its left, clearly a lot is missing. So one trick that people have done is to also train a 

right-to-left model, e.g.:

     P(store|</s>)*P(a|store </s>)*…

Now you have Â two representations of each word, one left-to-right and one right-to-left, and you can concatenate them together for your downstream task.

But intuitively, it would be much better if we could train a single model that wasÂ deeply bidirectional.

It's unfortunately impossible to train a deep bidirectional model like a normal LM, because that would create cycles where words can indirectly
 
"see themselves," and the predictions become trivial.

What we can do instead is the very simple trick that's used in de-noising auto-encoders, where we mask some percent of words from the input and 

have to reconstruct those words from context. We call this a "masked LM" but it is often called a Cloze task.


## Pretrain Language Understanding Task 

### task 1: masked language model
 
  we feed the input through a deep Transformer encoder and then use the final hidden states corresponding to the masked positions to
    
   predict what word was masked, exactly like we would train a language model.

    source_file each line is a sequence of token, can be a sentence.
    Input Sequence  : The man went to [MASK] store with [MASK] dog
    Target Sequence :                  the                his
    
   how to get last hidden state of masked position(s)?
   
     1) we keep a batch of position index,
     2) one hot it, multiply with represenation of sequences,
     3) everywhere is 0 for the second dimension(sequence_length), only one place is 1,
     4) thus we can sum up without loss any information.
            
   for more detail, check method of mask_language_model from pretrain_task.py and train_vert_lm.py

### task 2: next sentence prediction
  
  many language understanding task, like question answering,inference, need understand relationship
  
  between sentence. however, language model is only able to understand without a sentence. next sentence
  
  prediction is a sample task to help model understand better in these kinds of task.
  
  50% of chance the second sentence is tbe next sentence of the first one, 50% of not the next one.
   
  given two sentence, the model is asked to predict whether the second sentence is real next sentence of 
  
  the first one.
  
    Input : [CLS] the man went to the store [SEP] he bought a gallon of milk [SEP]
    Label : Is Next

    Input = [CLS] the man heading to the store [SEP] penguin [MASK] are flight ##less birds [SEP]
    Label = NotNext
    
<img src="https://github.com/brightmart/bert_language_understanding/blob/master/data/aa1.jpeg"  width="65%" height="65%" />

<img src="https://github.com/brightmart/bert_language_understanding/blob/master/data/aa2.jpeg"  width="65%" height="65%" />


## Environment
python 3+ tensorflow 1.10

## Implementation Details
1. what share and not share between pre-train and fine-tuning stages?

   1).basically, all of parameters of backbone network used by pre-train and fine-tuning stages are shared each other.
   
   2).as we can to share parameters as much as possible, so that during fine-tuning stage we need to learn as few 
   
   parameter as possible, we also shared word embedding for these two stages.
   
   3).therefore most of parameters were already learned at the beginning of fine-tuning stage.
   
2. how we implement masked language model?
   
   to make things easily, we generate sentences from documents, split them into sentences. for each sentence
   
   we truncate and padding it to same length, and randomly select a word, then replace it with [MASK], its self and a random 
   
   word.
   
3. how to make fine-tuning stage more efficient, while not break result and knowledge we learned from pre-train stage?
   
   we use a small learning rate during fine-tuning, so that adjust was done in a tiny extent.

## Questions for Better Understanding of Transformer and BERT

1. why we need self-attention?

     self-attention a new type of network recently gain more and more attention. traditionally we use 

     rnn or cnn to solve problem. however rnn has a problem in parallel, and cnn is not good at model position sensitive tasks.

     self-attention can run in parallel, while able to model long distance dependency.


2. what is multi-heads self-attention, what does q,k,v stand for? add something here.

     mulit-heads self-attention is a self-attention, while it divide and project q and k into several different subspace,

     then do attention. 

     q stand for question, k stand for keys. for machine translation task, q is previous hidden state of decodes, k represent 

     hidden states of encoder. each of element of k will compute a similarity score with q. and then softmax will be used

     to do normalize score, we will get weights. finally a weighted sum is computed by using weights apply to v.

     but in self-attention scenario, q,k,v are all the same, as the representation of input sequences of a task.

3. what is position-wise feedfoward?

     it is a feed forward layer, also called fully connected(FC) layer. but since in Transformer, all the input and output of 

     layers are sequence of vectors:[sequence_length, d_model]. we usually do FC to a vector of input. so we do it again,

     but different time step has its own FC.

4. what is the main contribution of BERT?

     while pre-train task  already exist for many years, it introduce a new way(so called bi-directional) to do language model 

     and use it for down stream task. as data for language model is everywhere. it proved to be powerful, and hence it reshape 

     nlp world.

5. why author use three different types of tokens when generating training data of masked language model?

   the authors believe that in fine-tuning stage there is no [MASK] token. so it mismatch between pre-train and fine-tuning.

   it also force the model to attention all the context information in a sentence.

6. what made BERT model to achieve new state of art result in language understanding tasks?

   Big model, Big computation, and most importantly--New algorithm Pre-train the model using free-text data.

## Toy Task

toy task is used to check whether model can work properly without depend on real data.

it ask the model to count numbers, and sum up of all inputs. and a threshold is used, 

if summation is greater(or less) than a threshold, then the model need to predict it as 1( or 0).

inside model/transform_model.py, there is a train and predict method. 

first you can run train() to start training, then run predict() to start prediction using trained model. 

as the model is pretty big, with default hyperparamter(d_model=512, h=8,d_v=d_k=64,num_layer=6), it require lots of data before it can converge.

at least 10k steps is need, before loss become less than 0.1. if you want to train it fast with small

data, you can use small set of hyperparmeter(d_model=128, h=8,d_v=d_k=16, num_layer=6)


## Multi-label Classification Task with transformer and BERT
you can use it two solve binary classification, multi-class classification or multi-label classification problem.

it will print loss during training,  and print f1 score for each epoch during validation.

## TODO List
1. fix a bug in Transformer [IMPORTANT,recruit a team member and need a merge request] 

   ( Transformer: why loss of pre-train stage is decrease for early stage, but loss is still not so small(e.g. loss=8.0)? even with

   more pre-train data, loss is still not small)

2. support sentence pair task [IMPORTANT,recruit a team member and need a merge request]

3. add pre-train task of next sentence prediction [IMPORTANT,recruit a team member and need a merge request]

4. need a data set for sentiment analysis or text classification in english [IMPORTANT,recruit a team member and need a merge request]

5. special handle first token [cls] as input and classification [DONE]

6. position embedding is not shared between with pre-train and fine-tuning yet. since here on pre-train stage length may 

     shorter than fine-tuning stage.

7. pre-train with fine_tuning: need load vocabulary of tokens from pre-train stage, but labels from real task. [DONE]

8. learning rate should be smaller when fine-tuning. [Done]


## Conclusions

1. pre-train is all you need. while using transformer or some other complex deep model can help you achieve top performance

   in some tasks, pretrain with other model like textcnn using huge amount of raw data then fine-tuning your model on task specific data set, 

   will always help you gain additional performance.

2. add more here.

Add suggestion, problem, or want to make a contribution, welcome to contact with me: brightmart@hotmail.com

## References
1. <a href='https://arxiv.org/abs/1706.03762'>Attention Is All You Need</a>

2. <a href='https://arxiv.org/abs/1810.04805'>BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding</a>

3. <a href='https://github.com/tensorflow/tensor2tensor'>Tensor2Tensor for Neural Machine Translation</a>

4. <a href='https://arxiv.org/abs/1408.5882'>Convolutional Neural Networks for Sentence Classification</a>

5. <a href='https://arxiv.org/pdf/1807.02478.pdf'>CAIL2018: A Large-Scale Legal Dataset for Judgment Prediction</a>

