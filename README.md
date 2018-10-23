# bert_language_understanding
Pre-training of Deep Bidirectional Transformers for Language Understanding

## Short Description:
Pretrain mashed language model and next sentence prediction task on large scale of corpus, 

based on multiple layer self-attetion model, then fine tuning by add a classification layer.

As BERT model is based on Transformer, currently we are working on add pretrain task to the model.

## Usage
to train with transform: python train_transform.py [DONE]

to train with BERT: python train_bert.py [WIP]

## Data Format

input and output is in the same line, each label is start with '__label__'. 

there is a space between input string and the first label, each label is also splitted by a space.

you can have two class( binary problem), multi-class, or multi-label.
e.g. 
token1 token2 token3 __label__l1 __label__l5 __label__l3
token1 token2 token3 __label__l2 __label__l4


## Environment
python 3+ tensorflow 1.10

## Toy Task

toy task is used to check whether model can work properly without using or depend on really data.

the task is ask model to count numbers, and sum up of all inputs, if it is greater( or less )than

a threshold, then the model need to predict 1( or 0).

inside model/transform_model.py, there is a train and predict method. run train() to start training,

run predict() to start prediction using trained model. 

as the model is pretty big, with default hyperparamter(d_model=512, h=8,d_v=d_k=64,num_layer=6), it require lots of data before it can converage.

at least 10k steps is need, before loss become less than 0.1. if you want to train it fast with small

data, you can use small set of hyperparmeter(d_model=128, h=8,d_v=d_k=16, num_layer=6)


## Multi-label Classification Task with transformer and BERT

## Reference
<a href='https://arxiv.org/abs/1706.03762'>Attention Is All You Need</a>
<a href='https://arxiv.org/abs/1810.04805'>BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding</a>



