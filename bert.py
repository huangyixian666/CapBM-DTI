import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb        #pip install transformer-pytorch
# import transformershuggingface as ppbhuggingface
import warnings
warnings.filterwarnings('ignore')
import numpy
from transformers import BertModel, BertTokenizer
import re

def bert(protein_list,):
    #download model from https://github.com/agemagician/ProtTrans
    model = BertModel.from_pretrained("Rostlab/prot_bert")
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
    # load pretein sequences
    # df=pd.read_csv(protein_file, header=None)
    # get protein-BERT feature
    x=len(protein_list)
    list2=[]
    for i in range(x):
    #    print(df.iloc[1,2])
        print(i)
    #    print(len(df.iloc[i,1]))
        sequence_Example=protein_list[i]
        #         想要的长度*2
        # print(len(df.iloc[i,1])) 
        if len(protein_list[i])>2000:  
        #         想要的长度
            sequence_Example=' '.join(protein_list[i].split()[:1000])
        print('protein-length:',len(sequence_Example))
        sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
        encoded_input = tokenizer(sequence_Example, return_tensors='pt')
        
        output = model(**encoded_input)
        # print(output)
        s = output[0].data.cpu().numpy() # 数据类型转换
        list2.append(s)
        # numpy.save("./data/"+str(i)+'.npy',s)
        # print(s)
        encoded_input=0
        sequence_Example=0
        s=0
        output=0
        # get BERT_Mean feature
    # df=pd.read_csv('./data/space gold protein dataset.csv')
    list1=[]
    for i in range(x):
        # data = np.load("./data/"+str(i)+'.npy')
        data=list2[i]
        # print(data)
        d=data.mean(axis=1)
        # print(d)
        feat=d[0].tolist()
        # print(feat)
        list1.append(feat)
    #print(list1)
    features=pd.DataFrame(list1)
    return features