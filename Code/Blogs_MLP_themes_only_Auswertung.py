
# coding: utf-8

# # Auswertung der Klassifikation mit dem MLPClassifier (themes_only)
# 
# Autorin: Maria Hartmann

# In[1]:


# read predicitons
f = open('../MLP/MLP_themes_only_predictions.txt', "r", encoding="utf-8")
lines = f.readlines()
f.close()
trues = []
preds = []
idents = []
labellist = []
for line in lines:
    if line.startswith("TRUE:"):
        line = line.replace("TRUE: ", "").strip()
        #trues.append(line.split(", "))
        labels = line.split(", ")
        tmp_line=[]
        for label in labels:
            label = label.strip().replace(",", "").strip()
            tmp_line.append(label)
        trues.append(tmp_line)
        labellist.append(tmp_line)
        
    elif line.startswith("PRED:"):
        line = line.replace("PRED: ", "").strip()
        #preds.append(line.split(","))
        labels = line.split(", ")
        tmp_line=[]
        for label in labels:
            label = label.strip().replace(",", "").strip()
            tmp_line.append(label)
        preds.append(tmp_line)
        if len(line) != 0:
            labellist.append(tmp_line)
    elif ".txt" in line:
        idents.append(line)
        
    else:
        continue

labelset = sorted(set([item for sublist in labellist for item in sublist]))


# In[2]:


pred_count=0 #Anzal der Vorhersagen insgesamt
label_count=0 # Anzahl der Labels im Testset
false_count=0 # Anzal der falschen Vorhersagen (falsche Vorhersagen sind Vorhersagen, die nicht in den richtigen Labels enthalten sind)
right_count=0 # Anzal der richtigen Vorhersagen
not_pred_count=0 # Anzahl der nicht vorhergesagten Labels
pred_dic=dict(zip(labelset, [0]*len(labelset))) # Vorkommen der Labels in den Vorhersagen
label_dic=dict(zip(labelset, [0]*len(labelset))) # Vorkommen der Labels im Testset
correct_predicted=dict(zip(labelset, [0]*len(labelset))) # Richtige Vorhersagen und ihre Vorkommen
false_predicted=dict(zip(labelset, [0]*len(labelset))) # Falsche Vorhersagen und ihre Vorkommen
not_predicted=dict(zip(labelset, [0]*len(labelset))) # nicht vorhergesagt Labels und die Anzahl ihrer nicht Vorhersagen

for ident, true, pred in zip(idents, trues, preds):
    for t in true:
        label_count+=1
        if t in label_dic:
            label_dic[t]+=1
        else:
            #label_dic[t]=1
            print("Label im Dictionary nicht gefunden")
        if t not in pred:
            not_pred_count+=1
            if t in not_predicted:
                not_predicted[t]+=1
            else:
                #not_predicted[t]=1
                print("Label im Dictionary nicht gefunden")
        else:
            right_count+=1
            if t in correct_predicted:
                correct_predicted[t]+=1
            else:
                #false_predicted[t]=1
                print("Label im Dictionary nicht gefunden")
                
    for p in pred:
        if len(p) != 0:
            pred_count+=1
            if p in pred_dic:
                pred_dic[p]+=1
            else:
                #pred_dic[t]=1
                print("Label im Dictionary nicht gefunden")
            if p not in true:
                false_count+=1
                if p in false_predicted:
                    false_predicted[p]+=1
                else:
                    #false_predicted[p]=1
                    print("Label im Dictionary nicht gefunden")
        else:
            continue


# In[3]:


# Gesamte Labels im Testset
print(len(label_dic))
for k, v in label_dic.items():
    print("%s: %s" % (k, v))
    
#labels = [(k, label_dic[k]) for k in sorted(label_dic, key=label_dic.get, reverse=True)]
#for k, v in labels:
    #print("%s: %s" % (k, v))


# In[4]:


# Gesamte Vorhersagen im Testset
print(len(pred_dic))
for k, v in pred_dic.items():
    print("%s: %s" % (k, v))


# In[5]:


# Korrekte Vorhersagen je Label im Korpus
print(len(correct_predicted))
for k, v in correct_predicted.items():
    print("%s: %s" % (k, v))


# In[6]:


# Falsche Vorhersagen je Label im Korpus
print(len(false_predicted))
for k, v in false_predicted.items():
    print("%s: %s" % (k, v))


# In[7]:


# nicht vorhergesagte Labels im Korpus
print(len(not_predicted))
for k, v in not_predicted.items():
    print("%s: %s" % (k, v))


# PROZENTUALE AUSWERTUNG:
# 
# Prozentsatz = (Prozentwert/Grundwert)*100
# 
# 
# label_dic = correct_predicted + not_predicted
# 
# pred_dic = correct_predicted + false_predicted

# In[8]:


# nicht vorhergesagte Labels in Prozent
not_predicted_prozent={}
for key, value in label_dic.items():
    for not_k, not_v in not_predicted.items():
        if not_k == key:
            prozent=round(((not_v/value)*100),2)
            not_predicted_prozent[not_k]=prozent
            
for k, v in not_predicted_prozent.items():
    print("%s: %s%%" % (k, v))


# In[9]:


# Fehlklassifizierungsrate je Label in Prozent
false_predicted_prozent={}
for key, value in pred_dic.items():
    for false_k, false_v in false_predicted.items():
        if false_k == key:
            if value != 0:
                prozent=round(((false_v/value)*100),2)
                false_predicted_prozent[false_k]=prozent
            else:
                false_predicted_prozent[false_k]=0
for k, v in false_predicted_prozent.items():
    print("%s: %s%%" % (k, v))


# In[10]:


# Korrektklassifizierungsrate je Label in Prozent
correct_predicted_prozent={}
for key, value in pred_dic.items():
    for correct_k, correct_v in correct_predicted.items():
        if correct_k == key:
            if value != 0:
                prozent=round(((correct_v/value)*100),2)
                correct_predicted_prozent[correct_k]=prozent
            else:
                correct_predicted_prozent[correct_k]=0
for k, v in correct_predicted_prozent.items():
    print("%s: %s%%" % (k, v))


# Ergebnisse in Datei schreiben

# In[11]:


f = open('../Preprocessing/blogposts_per_themes_only_reduced.txt', "r", encoding="utf-8")
lines = f.readlines()
f.close()

label_train_dic=dict(zip(labelset, [0]*len(labelset))) # Vorkommen der Labels im Testset

for line in lines:
    label, count = line.split(":")
    label = label.strip()
    count = count.strip()
    
    for k,v in label_train_dic.items():
        if k == label:
            label_train_dic[k]=count


# In[12]:


for k, v in label_train_dic.items():
    print("%s: %s" % (k, v))


# In[13]:


label_keys = label_dic.keys()
label_train_count = label_train_dic.values()
label_count = label_dic.values()
pred_count = pred_dic.values()
correct_count = correct_predicted.values()
correct_percent = correct_predicted_prozent.values()
false_count = false_predicted.values()
false_percent = false_predicted_prozent.values()
not_pred_count = not_predicted.values()
not_pred_percent = not_predicted_prozent.values()

with open('../MLP/MLP_themes_only_prozentuale_auswertung.txt', 'w+', encoding="utf-8") as file:
    file.write("Label (themes_only), Trainingsset (#), Testset (#), Vorhersagen (#), Korrekte Vorhersagen (#), Korrekte Vorhersagen (%), Falsche Vorhersagen (#), Falsche Vorhersagen (%), Nicht vorhergesagte Labels (#), Nicht vorhergesagte Labels (%)\n")
    for key, train_count, test_count, pred_count, correct_count, correct_percent, false_count, false_percent, not_pred_count, not_pred_percent in zip(label_keys, label_train_count, label_count, pred_count, correct_count, correct_percent, false_count, false_percent, not_pred_count, not_pred_percent):
        print("%s, %s, %s, %s, %s, %s%%, %s, %s%%, %s, %s%%" % (key, train_count, test_count, pred_count, correct_count, correct_percent, false_count, false_percent, not_pred_count, not_pred_percent))
        file.write("%s, %s, %s, %s, %s, %s%%, %s, %s%%, %s, %s%%\n" % (key, train_count, test_count, pred_count, correct_count, correct_percent, false_count, false_percent, not_pred_count, not_pred_percent))

