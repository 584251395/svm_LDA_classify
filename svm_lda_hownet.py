import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets,linear_model
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection  import cross_val_predict
from sklearn import metrics
from scipy.sparse import csr_matrix
from sklearn import svm
import os,re,time,logging,math
import thulac
from gensim import corpora,models
from pprint import pprint
#coding: utf-8
import urllib.request
import urllib.parse
import json
thu=thulac.thulac()
#去除文本中指定字符chara
def rm_char(text,chara='\u3000'):
    text = re.sub(chara,'',text)
    return text
def get_stop_words(path='data/stopword.txt'):
    file = open(path,'rb').read().decode('utf8').split('\r\n')
    return set(file)
def rm_tokens(words): # 去掉一些停用次和数字
    words_list = list(words)
    stop_words = get_stop_words()
    for i in range(words_list.__len__())[::-1]:
        if words_list[i] in stop_words: # 去除停用词
            words_list.pop(i)
        elif words_list[i].isdigit():
            words_list.pop(i)
    return words_list
#print(rm_tokens(['12','35','你好']))
#清华THULAC分词软件,去除指定词性list,seg_only是否进行词性标注
def cutAndsplit(sentence,cxl=['d','u','w','c'],seg_only=False):
    text=thu.cut(sentence,text=False)
    cutlist=[]
    for index in range(len(text)):
        if  text[index][1] not in set(cxl):
            if seg_only==False:
                cutlist.append(text[index])
            else:
                cutlist.append(text[index][0])
    #print(cutary)
    return cutlist

#print(cutAndsplit('将指定列的文本分词,并去除停顿词',seg_only=True))
#将指定列的表列分词并去除停顿词XTable:DataFrame
def Sep_rm_stopword(XTable,col,label='label'):
    handellist1=[]
    #print(XTable.shape)
    for row in XTable.itertuples(index=True, name='Pandas'):
        txt=getattr(row, col)
        txt=rm_char(txt)
        cutlist=rm_tokens(cutAndsplit(txt,seg_only=True))
        templist2=[]
        templist2.append(cutlist)
        templist2.append(getattr(row, label))
        handellist1.append(templist2)
        #XTable.ix[[getattr(row, 'Index')],[col]]=cutlist        
    return handellist1
#df4 = pd.DataFrame({'col1':['小米手机有毒!2222','手机不错!666'],'col2':[0,1]},index=['a','b'])
#hl=Sep_rm_stopword(df4,'col1','col2')
#hl[0].append(5)
#print(hl)
#XTable是总的数据：genDictionary(X,"commentcontent")
#生成预料词典并持久化
def genDictionary(XTable,col,label='label'):
    hlist=Sep_rm_stopword(XTable,col,label)
    diclist=[item[0] for item in hlist]
    #print(diclist)
    dictionary=corpora.Dictionary(diclist)
    #no_below=15表示去除只出现在15个及以下文本中的词
    #no_above=0.5表示去除出现在超过50%以上文档中的词
    #keep_n=100000 表示最后只保留频率排序在前100000的词
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=1000000)
    dictionary.save_as_text('data/handledic.txt')
    dictionary.save('data/handledic.dict')
#genDictionary(df4,'col1','col2')
#加载词典
def loadDictionary():
    return corpora.Dictionary.load('data/handledic.dict')
#生成语料库
def genCorpus(dic,XTable,col,label='label'):
    hlist=Sep_rm_stopword(XTable,col,label)
    for item in hlist:
        item[0]=dic.doc2bow(item[0])        
    return hlist
#划分训练集和测试集
def train_test_sep(hlist,test_size=0.2):
    trainsize=math.floor(len(hlist)*(1-test_size))
    train_set=hlist[0:trainsize]
    test_set=hlist[trainsize:]
    return train_set,test_set
#生成语料库tfidf
def genTfidf(XList):
    hlist=[item[0] for item in XList]
    tfidf = models.TfidfModel(hlist)#统计tfidf
    corpus_tfidf=[]
    #得到每个文本的tfidf向量，稀疏矩阵
    for item in XList:
        corpus_tfidf.append([tfidf[item[0]],item[1]])
    return corpus_tfidf
    #corpora.MmCorpus.serialize('data/tfidfcorpus.mm', corpus_tfidf)
#根据预料库生成ida模型
def genIdaModel(XList,dic,num):
    hlist=[item[0] for item in XList]
    lda = models.ldamodel.LdaModel(hlist, id2word = dic,num_topics=num,eval_every=5,per_word_topics=True)
    lda.save('data/ida.model')
 #加载ida模型
def loadIdaModel():
    return models.ldamodel.LdaModel.load('data/ida.model')
#将XList转换为svm输入格式
def Convert_XList_svmmatrix(XList):
    data=[]
    rows=[]
    cols=[]
    tags=[]
    line_c=0
    for item in XList:
        for elem in item[0]:
            rows.append(line_c)
            cols.append(elem[0])
            data.append(elem[1])
        line_c=line_c+1
        tags.append(item[1])
    svm_matrix = csr_matrix((data,(rows,cols))).toarray()
    return svm_matrix,tags
    
def svm_classify(train_set,train_set_tag,test_set):

    clf = svm.LinearSVC()
    clf_res = clf.fit(train_set,train_set_tag)
    train_pred  = clf_res.predict(train_set)
    test_pred = clf_res.predict(test_set)
    return train_pred,test_pred

    return clf_res
def cal_index(tags,pretags,class_num=[0,1,2]):
    classdic={}
    for item in class_num:
        classdic['T_'+str(item)]=0
        classdic['F_'+str(item)]=0
        classdic['Tag_'+str(item)]=0
    #print(classdic)
    for index in range(len(tags)):
        tagv=tags[index]
        prev=pretags[index]
        classdic['Tag_'+str(tagv)]=classdic['Tag_'+str(tagv)]+1
        if prev==tagv:
            classdic['T_'+str(prev)]=classdic['T_'+str(prev)]+1
        else:
            classdic['F_'+str(prev)]=classdic['F_'+str(prev)]+1
    prec=0.0
    #print(classdic)
    i=0
    for item in class_num:
        if (classdic['F_'+str(item)]+classdic['T_'+str(item)])==0:
            i=i+1
            continue
        prec=prec+(classdic['T_'+str(item)]+0.0)/(classdic['F_'+str(item)]+classdic['T_'+str(item)]+0.0)
    if (len(class_num)-i)==0:
        prec=0.0
    else:
        prec=prec/(len(class_num)-i)
    recall=0.0
    i=0
    for item in class_num:
        if (classdic['Tag_'+str(item)])==0:
            i=i+1
            continue
        recall=recall+(classdic['T_'+str(item)]+0.0)/(classdic['Tag_'+str(item)]+0.0)
    if (len(class_num)-i)==0:
        recall=0.0
    else:
        recall=recall/(len(class_num)-i) 
    #print('recall'+str(recall))
    if (prec+recall)==0:
        f_measure_index=0
    else:
        f_measure_index=2*prec*recall/(prec+recall)
    #print('f_measure_index'+str(f_measure_index))
    return prec,recall,f_measure_index
def relsvmclssify(corpusXList):
    xl=genTfidf(corpusXList)
    svmlist,tags=Convert_XList_svmmatrix(xl)
    train_set,test_set=train_test_sep(svmlist,test_size=0.5)
    train_set_tag,test_set_tag=train_test_sep(tags,test_size=0.5)
    trainpre,testpre=svm_classify(train_set,train_set_tag,test_set)
    prec,recall,f_measure_index=cal_index(test_set_tag,testpre,[0,1,2])
    print('precision:'+str(prec))
    print('recall:'+str(recall))
    print('f_measure_index:'+str(f_measure_index))
data=pd.read_csv('data/handletest.csv')
#print(data.head())
#切分数据得到样本特征X，样本输出Y
genDictionary(data,"commentcontent",'credit')
dic=loadDictionary()
corpusXList=genCorpus(dic,data,"commentcontent",'credit')
relsvmclssify(corpusXList)
print('lda')
def  getPerplexity(lda,XList):
    hlist=[item[0] for item in XList]
    return lda.bound(hlist)
#训练LDA模型选择最优主题数量值
def  get_lda_by_num_topics(train_set,test_set,dic,num=5,iternum=1000):
    Perplexity=[];
    genIdaModel(train_set,dic,num)
    lda=loadIdaModel()
    for i in range(iternum):
          genIdaModel(train_set,dic,num)
          lda=loadIdaModel()
          Perplexitytmp=getPerplexity(lda,test_set)
          if len(Perplexity)>1 and (Perplexitytmp-Perplexity[-1])>=100:
              break
          Perplexity.append(Perplexitytmp)
          num=num+5
    print('Perplexity: ')
    print(Perplexity)
    print('number of topics')
    print(num)
    return lda
train_set_ldaXList,test_set_ldaXList=train_test_sep(corpusXList,test_size=0.5)
lda=get_lda_by_num_topics(train_set_ldaXList,test_set_ldaXList,dic)
#通过情感词典获取词语情感极性和情感强度
def load_senDic_data():
    dic={}
    sendicdata=pd.read_csv('data/sendic.csv')
    i=0
    for index, row in sendicdata.iterrows():
        dic[row['word']]=[row['intensity'],row['polarity']]
    return dic
sendic=load_senDic_data()
def getPolrity(word):
    for item in sendic:
        if word in item:
            return sendic[item][0],sendic[item][1]
    return 0.0,0.0
intensity,polarity=getPolrity('最好')
print('jixing'+str(intensity)+str(polarity))
#获取dic中键值的情感倾向
def getPolrityDic(dic={}):
    positive=0.0
    negative=0.0
    for item in dic:
        intense,polar=getPolrity(item)
        if polar==1:
            positive=positive+dic[item]
        if polar==2:
            negative=negative+dic[item]
    polartyde=positive-negative
    return polartyde
#主题预测并更新模型
def predictThemebow(dic,txtbow,model):
    model_pred=model[txtbow[0]]
    themedic={}
    for topic in model_pred:
        for item in topic:
            wordspair=model.get_topic_terms(item[0], topn=10)
            for worditem in wordspair:
                if dic.get(worditem[0]) in themedic:
                    themedic[dic.get(worditem[0])]=themedic[dic.get(worditem[0])]+worditem[1]
                else:
                    themedic[dic.get(worditem[0])]=worditem[1]
    
    #print(themedic)
    pol=getPolrityDic(themedic)
    if pol>0:
        return 1
    elif pol<0:
        return 2
    else:
        return 0
def predictXListPolarity(dic,XList,model):
    pred=[]
    for item in XList:
        pred.append(predictThemebow(dic,item,model))
    return pred
def getTags(XList):
    tags=[item[1] for item in XList]
    return tags
def pred_lda_sen_index(dic,ldaXList,model):
    tags=getTags(ldaXList)
    print(tags)
    pol=predictXListPolarity(dic,ldaXList,model)
    print(pol)
    prec,recall,f_measure_index=cal_index(tags,pol,[0,1,2])
    print('precision:'+str(prec))
    print('recall:'+str(recall))
    print('f_measure_index:'+str(f_measure_index))
    
#lda=loadIdaModel()
pred_lda_sen_index(dic,test_set_ldaXList,lda)
#print(pol)
# 创建一个空的 DataFrame
#df_empty = pd.DataFrame(columns=['id'])
#for index in lda.show_topics(formatted=False,num_topics=5,num_words=10):
#    print(index)
#    ary=[]
#    for i in index[1]:
#        ary.append(i[0]+' '+str(i[1]))
#    df_empty[index[0]]=ary
#print(df_empty)
#df_empty.to_csv('data/csvresult.csv')
#predictTheme(dic,'你好，客服态度差，快递服务好',lda)
#print(lda.print_topics(20))#打印前20个topic的词分布
#print(lda.print_topic(20))#打印id为20的topic的词分布
#corpus_lda = lda[corpus_tfidf] #每个文本对应的LDA向量，稀疏的，元素值是隶属与对应序数类的权重


#划分数据测试集和训练级，
#test_size指定划分百分比，
#random_state固定为整数时每次划分结果相同

#print('xtrain:'+str(X_train.shape)+' ytrain:'+str(y_train.shape)+'Xtest:'+str(X_test.shape)+'ytest:'+str(y_test.shape))
#训练得到线性回归模型参数
linreg=LinearRegression()
linreg.fit(X_train,y_train)
print(linreg.intercept_)
print(linreg.coef_)
#评估我们的模型的好坏程度，对于线性回归来说，
#我们一般用均方差（Mean Squared Error, MSE）或
#者均方根差(Root Mean Squared Error,
#RMSE)在测试集上的表现来评价模型的好坏。
y_pred=linreg.predict(X_test)
print(metrics.mean_squared_error(y_test,y_pred))
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
predicted =cross_val_predict(linreg, X, y, cv=10)
fig,ax=plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
