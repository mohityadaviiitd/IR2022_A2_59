#!/usr/bin/env python
# coding: utf-8

# In[87]:


import nltk
import os
import string
import numpy as np
import pandas as pd
from operator import itemgetter
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import math
import matplotlib.pyplot as plt
lemmatizer=WordNetLemmatizer()
stemmer= PorterStemmer()
from random import shuffle
import re
import pandas as pd
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


# In[88]:


folder="Humor,Hist,Media,Food"


# In[89]:


all_files_url = []
paths=[]
folder_url=str(os.getcwd())+"\\"+folder+"\\"
docn=1
for root, dirs, files in os.walk(folder_url, topdown=False):
    for i in files:
        paths.append(folder_url+i)
        docn+=1


# In[91]:


#data extraction and preprocessing
postings={}
postings2={}
doc=1
for path in paths:
    newl=[]
    file=open(path, 'r')
    try:
        data=file.read().strip() #stripping white spaces
    except UnicodeDecodeError:
        a=1
        
    file.close()
    data=data.lower() #lower case
    data=re.sub(r'\d+', '', data) #removing numbers
    data=data.translate(str.maketrans("","", string.punctuation)) #removing punctuation
    stop_words = set(stopwords.words('english')) # for removing stopwords
    tokens=word_tokenize(data) #tokenization
    for i in tokens:
        j=stemmer.stem(i) # for stemming
        if j not in stop_words and len(j)>1:
            newl.append(j)
            if j not in postings2.keys():
                postings2[j]=[doc]
            else:
                li=postings2[j]
                if doc not in li:
                    li.append(doc)
    postings[doc]=(newl.copy())

    doc+=1


# In[92]:


# print(postings[0])


# In[93]:


def union(d1, d2):
    d1=set(d1)
    d2=set(d2)
    return list(d1 | d2)


# In[94]:


def intersection(d1, d2):
    d1=set(d1)
    d2=set(d2)
    return list(d1 & d2)


# In[95]:


def jc_fn(qry, postings):
    jc={}
    ans1=[]
    ans2=[]
    for i, k in enumerate(postings):
        doc_words=postings[k]
        ans1=union(doc_words, qry)
        ans2=intersection(doc_words, qry)
        jc[k]=len(ans2)/len(ans1)
    jc=dict(sorted(jc.items(), key=lambda item: item[1], reverse=True))
    return jc
        
        
        


# In[97]:


inp_query=input("Enter String Query: ")
nums=int(input("Number of documents: "))
data=inp_query.strip()
data=data.lower()
data=data.translate(str.maketrans("","", string.punctuation)) #removing punctuation
tokens=word_tokenize(data)
stop_words = set(stopwords.words('english'))
final_tokens=[]

for i in tokens:
    if i not in stop_words:
        final_tokens.append(i) 

if len(final_tokens)<1:
    print("no valid tokens")
else:
    jc=jc_fn(final_tokens, postings)
    for i, k in enumerate(jc):
        if i==nums:
            break
        ans=paths[k].split("\\")
        print(ans[-1])


# In[130]:


#tf
def tf_matrix(variant, postings):
    ans={}
    for i, k in enumerate(postings):
        tf={}
        words=postings[k]
        for j in words:
            if j not in tf.keys():
                tf[j]=1
            else:
                tf[j]+=1
        ans[k]=tf
    if variant=="raw":
        return ans
    ans2={}
    for i, k in enumerate(ans):
        tf={}
        words=ans[k]
        for j in words:
            fre=words[j]
            if variant=="log":
                score=np.log(1+fre)
            elif variant=="double_norm":
                maxim=max(words.values())
                score=0.5+(0.5*(fre/maxim))
            elif variant=="term":
                summation=sum(words.values())
                score=fre/summation
            elif variant=="binary":
                if fre>0:
                    score=1
                else:
                    score=0    
            tf[j]=score
        ans2[k]=tf
    return ans2

# print(len(ans))


# In[131]:


# idd=0
# for i, k in enumerate(ans):
#     print(k, ans[k])
#     if idd==5:
#         break
#     idd+=1


# In[132]:


#idf
idf={}
total_docs=len(postings)
for i, k in enumerate(postings2):
    word_docs=len(postings2[k])
    value=np.log(total_docs/word_docs)
    idf[k]=value
    


# In[133]:


#tf-idf
def tf_idf_fn(ans):
    tf_final={}
    for i, k in enumerate(ans):
        tf_idf_dict={}
        all_words=ans[k]
        for i, word in enumerate(all_words):
            cur_f1=all_words[word]
            cur_idf=idf[word]
            f=cur_f1*cur_idf
            tf_idf_dict[word]=f
        tf_final[k]=tf_idf_dict
    return tf_final

# idd=0
# for i, k in enumerate(tf_final):
#     print(k, tf_final[k])
#     if idd==5:
#         break
#     idd+=1
        


# In[134]:


def sort_docs(ans, qry):
    tf_idf_sort={}
    for word in qry:
        for i, k in enumerate(ans):
            a_doc=ans[k]
            if word in a_doc.keys():
                score=a_doc[word]
                if k in tf_idf_sort.keys():
                    tf_idf_sort[k]+=score
                else:
                    tf_idf_sort[k]=score
                
    tf_idf_sort=dict(sorted(tf_idf_sort.items(), key=lambda item: item[1], reverse=True))
    return tf_idf_sort


# In[139]:


def final_fn(qry_list, variant, postings,paths):
    print(variant, end=":\n")
    ans=tf_matrix(variant, postings)
    tf_ans=tf_idf_fn(ans)
    final_ans=sort_docs(tf_ans, qry_list)
    for i, doc_k in enumerate(final_ans):
        if i==5:
            break
        pat=paths[doc_k].split("\\")
        print(pat[-1])


# In[140]:



qry_list=postings2.keys()
final_fn(qry_list, "binary", postings, paths)
final_fn(qry_list, "raw", postings, paths)
final_fn(qry_list, "term", postings, paths)
final_fn(qry_list, "log", postings,paths)
final_fn(qry_list, "double_norm", postings, paths)


# In[ ]:


#Q2


# In[79]:


f=open("IR-assignment-2-data.txt", "r")
data=f.readlines()
qid4=[]
for line in data:
    tokens=list(line.split())
    token=tokens[0]
    tokens[0]=int(token)
    token1=tokens[1]
    label=list(token1.split(":"))
    label=label[1]
    if label=='4':
        qid4.append(tokens)
# print(len(qid4))


# In[175]:


qid44=qid4.copy()
qid44.sort(key=lambda x: x[0], reverse=True)
newf=open("maxDCG", "w")
final_t=""
for i in qid44:
    text=""
    text+= ' '.join([str(elem) for elem in i])
    final_t+=text+"\n"
newf.write(final_t)


# In[220]:


n=len(qid4)
rel=[]
for i in qid4:
    rel.append(i[0])
idx=0
dcg=0
tot=0
for i in rel:
    tot+=1
    dg=2**i- 1
    dn=np.log2(idx+2)
    dcg+=dg/dn
    idx+=1


# In[221]:


print(dcg)


# In[222]:


n=len(qid4)
rel=[]
for i in qid4:
    rel.append(i[0])
idx=0
dcg=0
tot=0
for i in rel:
    tot+=1
    dg=2**i- 1
    dn=np.log2(idx+2)
    dcg+=dg/dn
    idx+=1
    if(tot==50):
        break


# In[223]:


print(dcg)


# In[ ]:




