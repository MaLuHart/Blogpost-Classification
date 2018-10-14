
# coding: utf-8

# # Erstellung des Datensatzes 
# Texte: Blogbeiträge der Blogplattform Hypotheses.org
# Labels: Die von den Wissenschaftlern gewählten Themen und Disziplinen
# 
# Autorin: Maria Hartmann

# In[2]:


# Import libraries
import numpy as np
import csv # for csv output
import requests # HTTP for humans
from bs4 import BeautifulSoup # module for web scraping
import _thread 
from threading import Thread # to start parallel threads
import time # to get the processing time
import os
import shutil # to move files
from collections import Counter # to count element appearences in a list


# Einlesen der Ausgangsdatei metadata.csv

# In[4]:


# metadata.csv einlesen
folder = '../Preprocessing'
file = folder+'/metadata.csv'
lines = [] # all lines from metadata
de_lines = [] # german lines from metadata

with open(file, 'r', encoding='utf-8') as openfile:
    metadata = openfile.readlines()
    openfile.close()
    for i, line in enumerate(metadata):
        lines.append(line.replace('\n', '').split(";"))
        if lines[i][1] == "de":
            de_lines.append(lines[i])
        else:
            continue

    
# de_lines in numpy_array umgewandelt, weil Zugriff schneller geht, kann aber nicht verändert werden
np_lines = np.array(de_lines)
print(type(np_lines))

# Blogs ohne Disziplinen aus metadata.csv rausfiltern und in error_lines.csv schreiben
# Fehlerhafte Blogs (z.B. nicht mehr verfügbar oder einmalig andere Sprache) übergehen (wegschmeißen)
# die restlichen deutschen Daten in de_lines.csv schreiben
with open(folder+'/de_lines.csv', 'w', newline='', encoding="utf-8") as decsv, open(folder+'/unlabeled_lines.csv', 'w', newline='', encoding="utf-8") as unlabeledcsv, open(folder+'/error_lines.csv', 'w', newline='', encoding="utf-8") as errorcsv: 
    de = csv.writer(decsv, delimiter = ";")
    unlabeled = csv.writer(unlabeledcsv, delimiter = ";")
    errors = csv.writer(errorcsv, delimiter = ";")
    for i, line in enumerate(np_lines):
        if (np_lines[i][7] == "marginalie") or (np_lines[i][7] == "ciera"):
            # keine Disziplinen zugeordnet, 
            unlabeled.writerow(line)
        elif (np_lines[i][7] == "holocaustwebsites"):
        # holocaustwebsites rausgefiltert, weil diese Website  nicht mehr verfügbar ist
        # alles andere wird über den Blogpost-per-Blog-Index gefiltert
        #elif (np_lines[i][7] == "holocaustwebsites") or (np_lines[i][7] == "aleesp") or (np_lines[i][7] == "filstoria") or (np_lines[i][7] == "atb"):
            # aleesp rausgefiltert, weil es eigentlich ein spanischer Blog ist und im deutschen Korpus nur 1x vorkommt
            # filstoria rausgefiltert, weil es eigentlich ein italienischer Blog ist und im deutschen Korpus nur 1x vorkommt
            # atb rausgefiltert, weil Disciplines und Themes fehlerhaft sind (mit Doppelpunkt) und der Blog mehrheitlich englisch ist
            errors.writerow(line)
        else:
            de.writerow(line)


# de_lines.csv in data einlesen, um die Fehler nicht mit einlesen zu müssen 
data = [] # alle lines aus de_lines, ohne errors
bloglist = [] # alle blogs, die in de_lines vorkommen
with open(folder+'/de_lines.csv', 'r', encoding='utf-8') as openfile:
    de_csv = openfile.readlines()
    openfile.close()
    for i, line in enumerate(de_csv):
        data.append(line.replace('\n', '').split(";")) 
        bloglist.append(data[i][7])



# In[5]:


# remove blogs with less than 10 posts, damit anderssprachige blogs mit einzelnen deutschen Posts rausgefiltert werden

c = Counter(bloglist)
blog_select = []
counter = 0
for key in sorted(c): 
    if c[key] < 10:
        #print("%s: %s" % (key, c[key]))
        blog_select.append(key)
        counter += c[key]


trainset = [x for x in data if x[7] not in blog_select]


print(len(data))
print(len(trainset))
print(len(bloglist))
print(len(blog_select))
print(counter)


# Auslesen der Klassenbezeichnungen 

# In[ ]:


# crawl subjects and themes
errorlist = []
def get_disciplines(index):
    #print("\nline 1:", line)
    #print(i+1)
    #print(line)
    url = trainset[index][9]
    #print(i, "\npage:", url)
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "html.parser")

    if(soup.find(title="Zum Hypotheses-Katalogeintrag")):
        element = soup.find(title="Zum Hypotheses-Katalogeintrag")
        link = element.get("href")
        #print("link:", link)
        
    elif(soup.find(title="Zum OpenEdition-Katalogeintrag")):
        element = soup.find(title="Zum OpenEdition-Katalogeintrag")
        link = element.get("href")
        #print("link:", link)

    elif(soup.find(title="Ce carnet dans le catalogue d'Hypothèses")):
        element = soup.find(title="Ce carnet dans le catalogue d'Hypothèses")
        link = element.get("href")
        #print("link:", link)
        
    elif(soup.find(title="Ce carnet dans le catalogue d'OpenEdition")):
        element = soup.find(title="Ce carnet dans le catalogue d'OpenEdition")
        link = element.get("href")
        #print("link:", link)    

    elif(soup.find(title="This blog in Hypotheses catalogue")):
        element = soup.find(title="This blog in Hypotheses catalogue")
        link = element.get("href")
        #print("link:", link)
        
    elif(soup.find(title="This blog in OpenEdition catalogue")):
        element = soup.find(title="This blog in OpenEdition catalogue")
        link = element.get("href")
        #print("link:", link)

    else:
        print("Kein Open-Edition-Link gefunden!", index, trainset[index])
        trainset[index].append("Kein Open-Edition-Link gefunden!")
        errorlist.append(line)
        return

    subpage = requests.get(link)
    #print(subpage)
    subsoup = BeautifulSoup(subpage.text, "html.parser")
    morelinks = subsoup.find(class_="more-links")
    disciplines = []
    for i, child in enumerate(morelinks.children):
        #print("disciplines:", i, child)
        disciplines.append(child)

    #print(disciplines[9])
    #print(disciplines[14])
    if len(disciplines) > 13:
        trainset[index].append(disciplines[9].replace("\n", "").strip())
        trainset[index].append(disciplines[14].replace("\n", "").strip())
    elif len(disciplines) > 8:
        trainset[index].append(disciplines[9].replace("\n", "").replace('"', '').strip())
    else:
        print("Keine Disziplinen gefunden!", index, trainset[index])
        trainset[index].append("Keine Disziplinen gefunden!")
        errorlist.append(trainset[index])
        
    #print("\nline 2:", line)
    #print("trainset[i]:", trainset[i])
    #print("FERTIG")


start = time.time()
# Create two threads as follows
threads = []
for i in range(0,len(trainset)):
    if (i % 100 == 0):
                print("Schon wieder 100 Threads gestartet:", i)
    try:
        t = Thread(target = get_disciplines, args=(i, ))
        t.start()
        threads.append(t)
    except:
        print ("Error: unable to start thread")
        
for t in threads:
    #  join() stellt sicher, dass das Hauptprogramm wartet, bis alle Threads terminiert haben
    t.join()

print("Laufzeit in Minuten:", (time.time() - start) / 60)


# In[5]:


# show errors
print(len(errorlist))
print(errorlist)


# Speicherung der deutschen Blogbeiträge und ihrer Metadaten

# In[6]:


# add ; subjects ; themes to de_labeled_metadata.csv
print(type(trainset))

trainset.sort()
np_lines = np.array(trainset)
#print(np_lines)

with open(folder+'/de_labeled_metadata.csv', 'w', newline='', encoding="utf-8") as labeledcsv:
    labeled = csv.writer(labeledcsv, delimiter = ";")
    labeled.writerow(["filename", "language", "author", "numwords", "category", "date", "licence", "blog", "post", "url", "title", "disciplines", "themes"])
    
    for i, line in enumerate(np_lines):
        labeled.writerow(line)
        

    


# In[8]:


# move all german files to folder txt_de
newfolder = folder+'/txt_de'
if not os.path.exists(newfolder):
    os.makedirs(newfolder)


newfilelist = os.listdir(newfolder)
newfilelist.sort()
oldfolder = folder+'/txt'
filelist = os.listdir(oldfolder)
filelist.sort()
#print(trainset[0])
#print(len(trainset))
#trainset.sort()

for line in trainset:
    file = line[0] + '.txt'
    if (file in filelist) and (file not in newfilelist):
        shutil.copy2((oldfolder+'/'+file), (newfolder+'/'+file))
        #print("deutsch: ", (oldfolder+'/'+file))
    else:
        #print("Nicht deutsch")
        continue


# In[9]:


# 100 missing files in folder 'txt'
missing = []
filelist = os.listdir(newfolder)
filelist.sort()
#trainset.sort()
for line in trainset:
    file = line[0] + '.txt'
    if file not in filelist:
        missing.append(file)
        #print("deutsch: ", (directory+'/'+file))
    else:
        #print("Nicht deutsch")
        continue
    
print(missing)
print(len(missing))


# In[10]:


# open german metadata file: de_labeled_metadata.csv
# and every Blogpost

filelist = os.listdir(newfolder)
filelist.sort()
lines = [] # alle in de_labeled_metadata.csv verzeichnete Blogposts
corpus = [] # deutschsprachige Blogposts
labels = [] # zugehörige Labels
errors = [] # in metadata.csv verzeichnete, aber in hypoposts-txt.zip nicht enthaltene Blogposts
filenames = [] # Blogposts ohne Fehler
onelabel = [] # Blogposts mit nur einer Art von Label (Thema oder Disziplin)

with open(folder+'/de_labeled_metadata.csv', 'r', encoding='utf-8') as openfile:
    metadata = openfile.readlines()
    openfile.close()
    #print(metadata[0])
    for i, line in enumerate(metadata[1:]):
        #print(i)
        lines.append(line.split(";"))
        #print("\nFile:", lines[i][0])
        #print("Themes:", lines[i][11])
        #print("Disciplines:", lines[i][12])
        file = (lines[i][0] + '.txt')
        #print("Filename:", file)
        
        if file in filelist:
            
            with open((newfolder+'/'+file), 'r', encoding='utf-8') as textfile:
                text = textfile.read()
                textfile.close()
                filenames.append(file)
                corpus.append(text)
              
            if len(lines[i]) > 12:
                labels.append(lines[i][11] + "; " + lines[i][12])
            elif len(lines[i]) > 10:
                labels.append(lines[i][11])
                onelabel.append(file)
            else:
                print("keine Disziplin gefunden!", lines[i])
                
        else:
            print("File nicht gefunden!", file)
            errors.append(file)
            continue

print("\n")
print(len(corpus))
print(len(labels))
print(len(filenames))
print(len(errors))
print(len(onelabel))

for blog in onelabel:
    print(blog)


# Erstellung der Datenbasis: Dateinamen, Blogbeiträge und zugehörige Klassen (Themen und Disziplinen)

# In[11]:


# write csv-file de_labeled_corpus.csv: filename, classes, text

with open(folder+'/de_labeled_corpus.csv', 'w', newline='', encoding="utf-8") as labeledcsv:
    labeled = csv.writer(labeledcsv, delimiter = ";")
    labeled.writerow(["filename", "classes", "text"])
    
    for file, label, line in zip(filenames, labels, corpus):
        labeled.writerow([file.replace('\n', ' '), label.replace('\n', ''), line.replace('\n', ' ')])
        

