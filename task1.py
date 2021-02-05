'''
    I download the word vectors from https://github.com/Embedding/Chinese-Word-Vectors
'''
import fire
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def splitAllWord(typeOfDataset="dev"):
    from nltk.tokenize.stanford_segmenter import StanfordSegmenter

    segmenter = StanfordSegmenter()
    segmenter.default_config('zh')

    maxCount = 2000000

    pathOfDev = "dataset/task1/%s.tsv" % typeOfDataset
    dfOfDev = pd.read_csv(pathOfDev, delimiter="\t")

    pathOfNewDev = "%s_split.tsv" % typeOfDataset

    count = 0
    with open(pathOfNewDev, "w", encoding='utf-8') as fw:
        for row in dfOfDev.iterrows():
            if count >= maxCount:
                break
            if count % 100 == 0:
                print ("[%s]count = %s" % (typeOfDataset, count))

            label = row[1]['label']
            fw.write(str(label))
            fw.write("\t")
            sentence = row[1]['text_a']
            
            segmentOfSentence = segmenter.segment(sentence)
            for word in segmentOfSentence.split():
                fw.write(word)
                fw.write(" ")
            fw.write("\n")

            count += 1

def readAllWord():
    listOfAllWords = []

    maxCount = 1000000

    for typeOfDataset in ["dev", "test", "train"]:
        pathOfDataset = "%s_split.tsv" % typeOfDataset
        dfOfDataset = pd.read_csv(pathOfDataset, delimiter="\t", header=None)
        print (dfOfDataset)
        
        count = 0
        for sentence in dfOfDataset[1]:
            if count >= maxCount:
                break
            if count % 100 == 0:
                print ("[%s]count = %s" % (typeOfDataset, count))
                
            for word in sentence.split():
                if word not in listOfAllWords:
                    listOfAllWords.append(word)
            count += 1

    print (len(listOfAllWords))
    with open("allWord.txt", "w", encoding='utf-8') as fw:
        for word in listOfAllWords:
            fw.write(word + "\n")

def readModel():
    pathOfModel = "sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5"
    #pathOfModel = "small_word_vector.csv"
    
    dfOfModel = pd.read_csv(pathOfModel, delimiter=" ", error_bad_lines=False, engine="python", header=None, encoding='utf-8', index_col=0)
    print (dfOfModel)
    labelOfModel = dfOfModel.index.tolist()
    
    listOfWord = []
    count = 0
    with open("allWord.txt", "r", encoding='utf-8') as fr:
        for line in fr.readlines():
            word = line.split("\n")[0]

            if word in labelOfModel:
                listOfWord.append(word)

            count += 1
    print ("count = %s" % count)

    vectorWord = dfOfModel.loc[listOfWord]
    print (vectorWord)
    vectorWord.to_csv("vectorWord.csv")

def plot_with_names(low_dim_embs, labels, filename='visualize_task1.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(100, 100))
    for i, label in enumerate(labels):
        if i % 100 == 0:
            print ("i = %s\tlabel = %s" % (i, label))
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom', fontproperties="SimSun")

    plt.savefig(filename)

def visualizeVector():
    pathOfVector = "vectorWord.csv"
    dfOfVector = pd.read_csv(pathOfVector)
    print (dfOfVector)

    dfOfVector = dfOfVector[:50000]

    dfData = dfOfVector.drop(labels='301', axis=1).drop(labels='0', axis=1)
    print (dfData)

    label = dfOfVector["0"]
    print (label)

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=500)
    low_dim_embs = tsne.fit_transform(dfData)
    plot_with_names(low_dim_embs, label)

def prepareDataset(typeOfDataset="dev"):
    pathOfVector = "vectorWord.csv"
    dfOfVector = pd.read_csv(pathOfVector, index_col=0).drop(labels='301', axis=1)
    print (dfOfVector)
    labelOfModel = dfOfVector.index.tolist()

    pathOfDataset = "%s_split.tsv" % typeOfDataset
    dfOfDataset = pd.read_csv(pathOfDataset, delimiter="\t", header=None)
    print (dfOfDataset)

    count = 0
    with open("%s_vector.tsv" % typeOfDataset, "w", encoding='utf-8') as fw:
        for row in dfOfDataset.iterrows():
            if count % 100 == 0:
                print ("[%s]count = %s\trow = %s" % (typeOfDataset, count, row))

            label = row[1][0]
            fw.write(str(label))
            fw.write("\t")

            text = row[1][1]
            listOfWord = []
            for word in text.split(" "):
                if word in labelOfModel:
                    listOfWord.append(word)

            vectorWord = dfOfVector.loc[listOfWord]
            vectorWord = vectorWord[:1]
            meanVectorWord = vectorWord.mean(axis=0)

            for i in meanVectorWord:
                fw.write(str(i))
                fw.write("\t")
            fw.write("\n")

            count += 1

def plot_with_labels(low_dim_embs, labels, filename='visualize_task2.png', number_label=2):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"

    listOfColor = ["#000000", "#FFFF00", "#FF00FF", "#00FFFF", "#FF0000", "#0000FF", "#00FF00"]
    assert number_label <= len(listOfColor), "Two many labels, must less than %s" % len(listOfColor)

    plt.figure(figsize=(10, 10))
    for i, label in enumerate(labels):
        if i % 100 == 0:
            print ("i = %s\tlabel = %s" % (i, label))
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y, c=listOfColor[label])

    plt.savefig(filename)

def visualizeDataset(typeOfDataset="dev"):
    dfOfVector = pd.read_csv("%s_vector.tsv" % typeOfDataset, delimiter="\t", index_col=0, header=None).drop(labels=301, axis=1)
    print (dfOfVector)

    dfData = np.array(dfOfVector)
    print (dfData)
    m, n = dfData.shape
    #print ("n = %s" % n)    # 300

    label = dfOfVector.index
    print (label)

    w1 = [1 for i in range(n)]
    w2 = [0 for i in range(n)]
    w2[0] = 1
    w2[1] = -w1[0]/w1[1]
    w = np.array([w1, w2]).T
    print (w.shape) # (300, 2)

    low_dim_embs = np.matmul(dfData, w)
    print ("low_dim_embs = %s" % low_dim_embs)
    plot_with_labels(low_dim_embs, label, 'visualize_task2_%s.png' % typeOfDataset, 2)

if __name__ == "__main__":
    fire.Fire()