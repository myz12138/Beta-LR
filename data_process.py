import numpy as np
import json
import torch.nn  as nn
import torch
from transformers import AutoTokenizer,AutoModel
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
import re
from utils import *
def read_ReClor_data(option=''):
    context_list,question_list,answers_list,option_label_list=[],[],[],[]
    if option=='train':
        with open(args.data_path+args.data_name+'/train/train.json','r',encoding='utf-8') as js:
            json_data=json.load(js)
            for _ in json_data:
                context_list.append(_['context'])
                question_list.append(_['question'])
                answers_list.append(_['answers'])
                option_label_list.append(_['label'])
    elif option=='val':
        with open(args.data_path+args.data_name+'/val/val.json','r',encoding='utf-8') as js:
            json_data=json.load(js)
            for _ in json_data:
                context_list.append(_['context'])
                question_list.append(_['question'])
                answers_list.append(_['answers'])
                option_label_list.append(_['label'])
    elif option=='test':
        with open(args.data_path+args.data_name+'/test/test.json','r',encoding='utf-8') as js:
            json_data=json.load(js)
            for _ in json_data:
                context_list.append(_['context'])
                question_list.append(_['question'])
                answers_list.append(_['answers'])
                option_label_list.append(0)
    else:
        print("check the input of option")
        return 
    return context_list,answers_list,question_list,option_label_list

def read_LogiQA_data(option=''):
    context_list=[]
    question_list=[]
    answers_list=[]
    option_label_list=[]
    dic={'a':0,'b':1,'c':2,'d':3}
 
    if option=='train':
        with open(args.data_path+args.data_name+'/Train.txt','r',encoding='utf-8') as txt:
            txt_data=txt.readlines()
            for i in range(len(txt_data)):
                txt_data[i]=txt_data[i].strip('\n')
                if i%8==1:
                    option_label_list.append(dic[txt_data[i]])
                elif i%8==2:
                    context_list.append(txt_data[i])
                elif i%8==3:  
                    question_list.append(txt_data[i])
                elif i%8==4:
                    answers_list.append([txt_data[i][2:],txt_data[i+1][2:],txt_data[i+2][2:],txt_data[i+3][2:]])
                
    elif option=='val':
        with open(args.data_path+args.data_name+'/Eval.txt','r',encoding='utf-8') as txt:
            txt_data=txt.readlines()
            for i in range(len(txt_data)):
                txt_data[i]=txt_data[i].strip('\n')
                if i%8==1:
                    option_label_list.append(dic[txt_data[i]])
                elif i%8==2:
                    context_list.append(txt_data[i])
                elif i%8==3:  
                    question_list.append(txt_data[i])
                elif i%8==4:
                    answers_list.append([txt_data[i][2:],txt_data[i+1][2:],txt_data[i+2][2:],txt_data[i+3][2:]])
    elif option=='test':
        with open(args.data_path+args.data_name+'/Test.txt','r',encoding='utf-8') as txt:
            txt_data=txt.readlines()
            for i in range(len(txt_data)):
                txt_data[i]=txt_data[i].strip('\n')
                if i%8==1:
                    option_label_list.append(dic[txt_data[i]])
                elif i%8==2:
                    context_list.append(txt_data[i])
                elif i%8==3:  
                    question_list.append(txt_data[i])
                elif i%8==4:
                    answers_list.append([txt_data[i][2:],txt_data[i+1][2:],txt_data[i+2][2:],txt_data[i+3][2:]])
    else:
        print("check the input of option")
        return 
    return context_list,answers_list,question_list,option_label_list


def sentence_data(contexts,options,questions):
    new_texts=[]
    for i in range(len(contexts)):
        contexts_i=re.split('[.|,|!|?]',contexts[i])
        for j in contexts_i:
            if len(j)<=1:
                contexts_i.remove(j)
        contexts_cat=''.join([contexts_i[j]+'||' for j in range(len(contexts_i)-1)])
        contexts_cat=contexts_cat+contexts_i[-1]
        new_texts.append([contexts_cat+'</s>'+questions[i]+'</s>'+j for j in options[i]])
    return new_texts

def encode_sentence_texts(texts):
    token_ids,token_mask,idx=[],[],[]
    for i in range(len(texts)):
        token_i=tokenizer(texts[i],return_tensors='pt',padding=True,truncation=True).to(device)
        idx_mid,idx_end=(token_i['input_ids']==49085).nonzero(),(token_i['input_ids']==2).nonzero()
        idx_i=torch.cat([idx_mid[:int(idx_mid.shape[0]/4),:],idx_end[:int(idx_end.shape[0]/4),:]],dim=0)
        token_ids.append(token_i['input_ids'])
        token_mask.append(token_i['attention_mask'])
        idx.append(idx_i)
    return token_ids,token_mask,idx#500*4*x,500*4*x,500*2*2

def encode_texts(texts):#为model3的编码，idx将c和q+o剥离
    token_ids,token_mask,idx=[],[],[]
    for i in range(len(texts)):
        token_i=tokenizer(texts[i],return_tensors='pt',padding=True,truncation=True).to(device)
        idx_i=(token_i['input_ids']==2).nonzero()[:2]
        token_ids.append(token_i['input_ids'])
        token_mask.append(token_i['attention_mask'])
        idx.append(idx_i)
    return token_ids,token_mask,torch.stack(idx)#500*4*x,500*4*x,500*2*2

class data_Loader(Dataset):
    def __init__(self,input_ids,attention_mask,idx,label):
        self.input_ids=input_ids
        self.attention_mask=attention_mask
        self.idx=idx
        self.label=label
        self.len=len(label)

    def __getitem__(self, index) :
        return self.input_ids[index],self.attention_mask[index],self.idx[index],self.label[index]
    def __len__(self):
        return self.len

