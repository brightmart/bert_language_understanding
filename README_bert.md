【使用bert预训练过的中文模型:最短教程】

【使用自带数据集】

    1.下载模型和数据集合：https://github.com/google-research/bert
    
    2. 使用命令run_classifier.py,带上参数

【如何使用自定义数据集?】

    1.在run_classifier.py中添加一个Processor，告诉processor怎么取输入和标签；并加该processor到main中的processors
    
    2.将自己的数据集放入到特定目录。每行是一个数据，包括输入和标签，中间用"\t"隔开。
    
    3.运行命令run_classifier.py,带上参数
 
【session-feed方式使用bert模型;使用bert做在线预测】

1.<a href='https://github.com/brightmart/bert_language_understanding/blob/master/run_classifier_predict_online.py'>简明例子</a>

【目前支持的任务类型】

    1.文本分类(二分类或多分类)；
    
    2.句子对分类Sentence Pair Classificaiton(输入两个句子，输出一个标签)
