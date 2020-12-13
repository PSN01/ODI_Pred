#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yaml as y
import os
import numpy as np
from sklearn.svm import SVC
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import random
from keras.models import Sequential
from keras.layers import Dense
np.random.seed(7)
path="/home/psn/Documents/neolen_internship/odis/"
team1=input('Country-1 as India/Australia/England')
team2=input('Country-2 as Srilanka/Australia/England')

c1=0
c2=0
total=0
no_outcome=0

India_win=[]
toss_winner=[]
batfirst_decision=[]
runs_gte300=[]
runs_250to299=[]
runs_200to249=[]
runs_lt200=[]
wickets0=[]
wickets1t4=[]
wickets5t7=[]
wicketsgte8=[]

c=0
d={}
winner_of_match=""
def classify(team_total,wickets):
    if wickets>=8:
        wicketsgte8.append(1)
        wickets5t7.append(0)
        wickets1t4.append(0)
        wickets0.append(0)
    elif wickets>=5 and wickets<8:
        wicketsgte8.append(0)
        wickets5t7.append(1)
        wickets1t4.append(0)
        wickets0.append(0)
    elif wickets>=1 and wickets<5:
        wicketsgte8.append(0)
        wickets5t7.append(0)
        wickets1t4.append(1)
        wickets0.append(0)
    else:
        wicketsgte8.append(0)
        wickets5t7.append(0)
        wickets1t4.append(0)
        wickets0.append(1)

    if team_total>=300:
        runs_gte300.append(1)
        runs_250to299.append(0)
        runs_200to249.append(0)
        runs_lt200.append(0)
    elif team_total>=250 and team_total<300:
        runs_gte300.append(0)
        runs_250to299.append(1)
        runs_200to249.append(0)
        runs_lt200.append(0)
    elif team_total>=200 and team_total<250:
        runs_gte300.append(0)
        runs_250to299.append(0)
        runs_200to249.append(1)
        runs_lt200.append(0)
    else:
        runs_gte300.append(0)
        runs_250to299.append(0)
        runs_200to249.append(0)
        runs_lt200.append(1)
        
fn_set=[]
for root,dirs,files in os.walk(path):
    for file in files:
        if file.endswith(".yaml"):
            fn_set.append(file)

f=[]
if team1=='India':
    fn_set=["1022353.yaml", "1022361.yaml", "1022361.yaml", "1022367.yaml", "1022375.yaml", "1030219.yaml", "1030221.yaml",
              "1030227.yaml", "238188.yaml", "238189.yaml", "238190.yaml", "238193.yaml", "238194.yaml", "244510.yaml", 
              "256607.yaml", "256612.yaml", "256665.yaml", "256665.yaml", "267708.yaml", "267709.yaml", "267710.yaml",
              "267710.yaml", "267712.yaml", "267712.yaml", "267713.yaml", "267713.yaml", "267715.yaml", "267715.yaml",
              "291366.yaml", "291366.yaml", "291369.yaml", "291369.yaml", "293076.yaml", "293077.yaml", "297795.yaml", 
              "297796.yaml", "297797.yaml", "297798.yaml", "297799.yaml", "297801.yaml", "297802.yaml", "297803.yaml", 
              "297804.yaml", "297805.yaml", "335356.yaml", "335356.yaml", "335358.yaml", "335358.yaml", "343732.yaml", 
              "343732.yaml", "343733.yaml", "343733.yaml", "343734.yaml", "343734.yaml", "343736.yaml", "343736.yaml", 
              "345471.yaml", "361043.yaml", "361044.yaml", "361045.yaml", "361046.yaml", "361047.yaml", "366341.yaml", 
              "366341.yaml", "386530.yaml", "386530.yaml", "386531.yaml", "386531.yaml", "386533.yaml", "386534.yaml"]
    f=["467883.yaml", "345469.yaml", "792297.yaml", "903593.yaml", "293078.yaml", "297793.yaml", "297794.yaml",
         "434259.yaml", "434259.yaml", "434262.yaml", "434262.yaml", "1034819.yaml", "1034821.yaml", "1034823.yaml",
         "249753.yaml", "249756.yaml", "647253.yaml", "647255.yaml", "647259.yaml", "386532.yaml", "386532.yaml",
         "386533.yaml", "244511.yaml", "247476.yaml", "247476.yaml", "249745.yaml", "291360.yaml", "291360.yaml", 
         "518957.yaml", "518957.yaml", "1030223.yaml", "1030225.yaml", "770123.yaml", "770127.yaml", "792289.yaml"]

elif team1=='Australia':
    fn_set=["1000887.yaml","1000889.yaml","1000891.yaml","1000893.yaml","1000895.yaml","1001371.yaml","1001373.yaml",
              "1001375.yaml","1022349.yaml","1022355.yaml","226375.yaml","226376.yaml","226377.yaml","226378.yaml",
              "226380.yaml","226381.yaml","226382.yaml","236358.yaml","236595.yaml","236963.yaml","247458.yaml",
              "247466.yaml","247478.yaml","247485.yaml","247491.yaml","247496.yaml","247499.yaml","247503.yaml",
              "247506.yaml","247507.yaml","249228.yaml","249229.yaml","249230.yaml","249231.yaml","249232.yaml",
              "249233.yaml","249234.yaml","249235.yaml","249240.yaml","249241.yaml","249748.yaml","249750.yaml",
              "249757.yaml","249759.yaml","256608.yaml","256610.yaml","256614.yaml","256615.yaml","291345.yaml",
              "291346.yaml","291347.yaml","291359.yaml","291361.yaml","291362.yaml","291364.yaml","291365.yaml",
              "291367.yaml","291368.yaml","291370.yaml","291371.yaml","291372.yaml","336201.yaml","336202.yaml",
              "336204.yaml","351684.yaml","351685.yaml","351686.yaml","351687.yaml","351688.yaml","351689.yaml"]
        
elif team1=='England':
    fn_set= ["1022347.yaml","1022357.yaml","1022365.yaml","1022371.yaml","1029001.yaml","1029003.yaml","1031425.yaml",
              "1031427.yaml","1031429.yaml","225245.yaml","225246.yaml","225247.yaml","225248.yaml","225249.yaml",
              "225250.yaml","225251.yaml","225252.yaml","225253.yaml","225254.yaml","247463.yaml","247479.yaml",
              "247484.yaml","247489.yaml","247500.yaml","249236.yaml","249237.yaml","249238.yaml","249239.yaml",
              "249755.yaml","258465.yaml","258466.yaml","258467.yaml","258471.yaml","258472.yaml","258473.yaml",
              "258474.yaml","258475.yaml","258476.yaml","258477.yaml","291217.yaml","291218.yaml","291220.yaml",
              "291221.yaml","296904.yaml","296905.yaml","296906.yaml","296907.yaml","296908.yaml","296914.yaml",
              "296915.yaml","296916.yaml","296917.yaml","296918.yaml","350043.yaml","350044.yaml","350045.yaml",
              "350046.yaml","350047.yaml","350048.yaml","350049.yaml","380715.yaml","380716.yaml","415276.yaml",
              "415282.yaml","426387.yaml","426388.yaml","426389.yaml","426390.yaml","426391.yaml","426403.yaml"]

random.shuffle(fn_set)
    
m=fn_set[:35]
fn_set=fn_set[35:]

for root,dirs,files in os.walk(path):
    for file in files:
        if file.endswith(".yaml"):
            d=y.load(open(path+file))
            teams=d['info']['teams']
            c=c+1
            print("Reading file",c)
            s=set(teams)
            if team1 in s:
                for t in s:
                    m.append(t)
                if 'winner' in d['info']['outcome']:
                    match_winner=d['info']['outcome']['winner']
                    if match_winner==team1:
                        India_win.append(1)
                    else:
                        India_win.append(0)
                else:
                    no_outcome=no_outcome+1
                    continue
                
                twinner=d['info']['toss']['winner']
                if twinner==team1:
                    toss_winner.append(1)
                    toss=d['info']['toss']['winner']
                    if toss=='bat':
                        batfirst_decision.append(1)
                    else:
                        batfirst_decision.append(0)
                else:
                    toss_winner.append(0)
                    toss=d['info']['toss']['decision']
                    if toss=='bat':
                        batfirst_decision.append(0)
                    else:
                        batfirst_decision.append(1)
                total+=1
                team_total=1
                wickets=0
                
                #condition:- India playing 1st
                if d['innings'][0]['1st innings']['team']== team1:
                    for i in range(len(d['innings'][0]['1st innings']['deliveries'])):
                        new_dic=d['innings'][0]['1st innings']['deliveries'][i]
                        for j in new_dic:
                            team_total=team_total+new_dic[j]['runs']['total']
                            try:
                                if new_dic[j]['wicket']:
                                    wickets=wickets+1
                            except:
                                pass
                try:
                    for i in range(len(d['innings'][1]['2nd innings']['deliveries'])):
                        new_dic=d['innings'][1]['2nd innings']['deliveries'][i]
                        for j in new_dic:
                            if d['innings'][1]['2nd innings']['team']==team1:
                                team_total=team_total+new_dic[j]['runs']['total']
                                try:
                                    if new_dic[j]['wicket']:
                                        wickets=wickets+1
                                except KeyError:
                                    pass
                except IndexError:
                    print("No second Innings")
                    continue
                classify(team_total,wickets)
            else:
                continue

toss_win=np.array(toss_winner)
batfirst=np.array(batfirst_decision)
gte300=np.array(runs_gte300)
btw250_299=np.array(runs_250to299)
btw200_250=np.array(runs_200to249)
btw1_200=np.array(runs_lt200)
gte8=np.array(wicketsgte8);
btw5_7=np.array(wickets5t7);
btw1_4=np.array(wickets1t4);
w0=np.array(wickets0)
features=np.array([toss_win,batfirst,gte300,btw1_200,btw200_250,btw250_299,gte8,btw5_7,btw1_4,w0])
class_label_X=np.array(India_win)

features=features.reshape((features.shape[1],-1))
clf=SVC(gamma='auto')
clf.fit(features,class_label_X)
clf_dt=tree.DecisionTreeClassifier(max_depth=5)
clf_dt.fit(features,class_label_X)
clf_rf=RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
clf_rf.fit(features, class_label_X)

clf_ab=AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1),algorithm="SAMME",n_estimators=200)
clf_ab.fit(features,class_label_X)
clf_lin=SGDClassifier()
clf_lin.fit(features,class_label_X)
clf_glb=GaussianNB()
clf_glb.fit(features,class_label_X)

c1=0
c2=0
total=0
no_outcome=0

India_win=[]
toss_winner=[]
batfirst_decision=[]
runs_gte300=[]
runs_250to299=[]
runs_200to249=[]
runs_lt200=[]
wickets0=[]
wickets1t4=[]
wickets5t7=[]
wicketsgte8=[]

for root,dirs,files in os.walk(path):
    for file in files:
        if file.endswith(".yaml") and file in m:
            d=y.load(open(path+file))
            teams=d['info']['teams']
            c=c+1
            print("Reading file",c)
            s=set(teams)
            if team1 in s:
                if 'winner' in d['info']['outcome']:
                    match_winner=d['info']['outcome']['winner']
                    if match_winner==team1:
                        India_win.append(1)
                    else:
                        India_win.append(0)
                else:
                    no_outcome=no_outcome+1
                    continue
                twinner=d['info']['toss']['winner']
                if twinner==team1:
                    toss_winner.append(1)
                    toss=d['info']['toss']['decision']
                    if toss=='bat':
                        batfirst_decision.append(1)
                    else:
                        batfirst_decision.append(0)
                else:
                    toss_winner.append(0)
                    toss=d['info']['toss']['decision']
                    if toss=='bat':
                        batfirst_decision.append(0)
                    else:
                        batfirst_decision.append(1)
                total+=1
                team_total=0
                wickets=0
                
                #Condition:2:- India playing in 1st
                if d['innings'][0]['1st innings']['team']==team1:
                    for i in range(len(d['innings'][0]['1st innings']['deliveries'])):
                        new_dic=d['innings'][0]['1st innings']['deliveries'][i]
                        for j in new_dic:
                            team_total=team_total+new_dic[j]['runs']['total']
                            try:
                                if new_dic[j]['wicket']:
                                    wickets=wickets+1
                            except:
                                pass
                try:
                    #Condition 3: India playing 2nd innings
                    for i in range(len(d['innings'][1]['2nd innings']['deliveries'])):
                        new_dic=d['innings'][1]['2nd innings']['deliveries'][i]
                        for j in new_dic:
                            if d['innings'][1]['2nd innings']['team']==team1:
                                team_total=team_total+new_dic[j]['runs']['total']
                                try:
                                    if new_dic[j]['wicket']:
                                        wickets=wickets+1
                                except KeyError:
                                    pass
                except IndexError:
                    print("No 2nd Innings")
                    continue
                    
                classify(team_total,wickets)
            else:
                continue

toss_win=np.array(toss_winner)
batfirst=np.array(batfirst_decision)
gte300=np.array(runs_gte300)
btw250_299=np.array(runs_250to299)
btw200_250=np.array(runs_200to249)
btw1_200=np.array(runs_lt200)

gte8=np.array(wicketsgte8)
btw5_7=np.array(wickets5t7)
btw1_4=np.array(wickets1t4)
w0=np.array(wickets0)
features_y=np.array([toss_win,batfirst,gte300,btw1_200,btw200_250,btw250_299,gte8,btw5_7,btw1_4,w0])
class_label_Y=np.array(India_win)

features_y=features_y.reshape((features_y.shape[1],-1))
pred=clf.predict(features_y)
pred_dt=clf_dt.predict(features_y)
pred_rf=clf_rf.predict(features_y)
pred_ab=clf_ab.predict(features_y)
pred_ln=clf_lin.predict(features_y)
pred_glb=clf_glb.predict(features_y)

print('Decision Tree Predictions',pred_dt)
print('Adaboost Predicitions',pred_rf)
print('Naive Bayes Classifier',pred_glb)
print('Random Forest Predicitions',pred_rf)
correct=0
for i in range(len(class_label_Y)):
    if class_label_Y[i]==pred[i]:
        correct+=1
print("SVM accuracy is {}:".format(correct/len(pred)))
correct=0
for i in range(len(class_label_Y)):
    if class_label_Y[i]==pred_dt[i]:
        correct+=1
print("Decision Tree accuracy is {}:".format(correct/len(pred_dt)))
correct=0
for i in range(len(class_label_Y)):
    if class_label_Y[i]==pred_rf[i]:
        correct+=1
print("Random Forest accuracy is {}:".format(correct/len(pred_rf)))
correct=0
for i in range(len(class_label_Y)):
    if class_label_Y[i]==pred_ab[i]:
        correct+=1
print("Adaboost accuracy is {}:".format(correct/len(pred_ab)))
correct=0
for i in range(len(class_label_Y)):
    if class_label_Y[i]==pred_ln[i]:
        correct+=1
print("Linear Classifier is {}:".format(correct/len(pred_ln)))
correct=0           
for i in range(len(class_label_Y)):
    if class_label_Y[i]==pred_glb[i]:
        correct+=1
print("Naive Bayes Classifier is {}:".format(correct/len(pred_glb)))          
                
        


# In[2]:


#Model Architecture
model=Sequential()
model.add(Dense(48, input_dim=10,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(features,class_label_X,epochs=1000,batch_size=10)
pred_NN=model.predict(features_y)
for x in range(len(pred_NN)):
    if x<0.5:
        pred_NN[x]=0
    else:
        pred_NN[x]=1
correct=0
for i in range(len(pred_NN)):
    if pred_NN[i]==class_label_Y[i]:
        correct+=1
acc=correct/len(class_label_Y)
print("Accuracy of NN is ", acc)


# In[ ]:




