'''
    https://www.jiqizhixin.com/articles/2018-05-15-10
    https://github.com/Embedding/Chinese-Word-Vectors
    https://github.com/brightmart/nlp_chinese_corpus
    https://github.com/to-shimo/chinese-word2vec
    https://www.dazhuanlan.com/2020/01/21/5e264dc6220b4/
    https://mmchiou.gitbooks.io/ai_gc_methodology_2018_v1-private/content/zhong-wen-word2vector/ke-ji-da-lei-tai-jie-shao-wen-jian-wordvector-jeiba.html
'''
import pandas as pd
from nltk.tokenize.stanford_segmenter import StanfordSegmenter

def splitAllWord():
    segmenter = StanfordSegmenter()
    segmenter.default_config('zh')

    maxCount = 2000000

    pathOfDev = "dataset/task1/dev.tsv"
    dfOfDev = pd.read_csv(pathOfDev, delimiter="\t")

    pathOfNewDev = "dev_split.tsv"

    count = 0
    with open(pathOfNewDev, "w", encoding='utf-8') as fw:
        for row in dfOfDev.iterrows():
            if count >= maxCount:
                break
            if count % 100 == 0:
                print ("[dev]count = %s" % count)

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
    segmenter = StanfordSegmenter()
    segmenter.default_config('zh')

    listOfAllWords = []

    maxCount = 1000000

    pathOfTrain = "dataset/task1/train.tsv"
    dfOfTrain = pd.read_csv(pathOfTrain, delimiter="\t")
    count = 0
    for sentence in dfOfTrain["text_a"]:
        if count >= maxCount:
            break
        if count % 100 == 0:
            print ("[train]count = %s" % count)
        segmentOfSentence = segmenter.segment(sentence)
        for word in segmentOfSentence.split():
            if word not in listOfAllWords:
                listOfAllWords.append(word)
        count += 1

    pathOfTest = "dataset/task1/test.tsv"
    dfOfTest = pd.read_csv(pathOfTest, delimiter="\t")
    count = 0
    for sentence in dfOfTest["text_a"]:
        if count >= maxCount:
            break
        if count % 100 == 0:
            print ("[test]count = %s" % count)
        segmentOfSentence = segmenter.segment(sentence)
        for word in segmentOfSentence.split():
            if word not in listOfAllWords:
                listOfAllWords.append(word)
        count += 1

    pathOfDev = "dataset/task1/dev.tsv"
    dfOfDev = pd.read_csv(pathOfDev, delimiter="\t")
    count = 0
    for sentence in dfOfDev["text_a"]:
        if count >= maxCount:
            break
        if count % 100 == 0:
            print ("[dev]count = %s" % count)
        segmentOfSentence = segmenter.segment(sentence)
        for word in segmentOfSentence.split():
            if word not in listOfAllWords:
                listOfAllWords.append(word)
        count += 1

    print (listOfAllWords)
    print (len(listOfAllWords))
    with open("allWord.txt", "w", encoding='utf-8') as fw:
        for word in listOfAllWords:
            fw.write(word + "\n")

def readModel():
    pathOfModel = "sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5"
    #dfOfModel = pd.read_csv(pathOfModel, delimiter=" ", error_bad_lines=False, engine="python")
    header = ["word"]
    for i in range(300):
        header.append(str(i))
    dfOfModel = pd.read_csv(pathOfModel, delimiter=" ", error_bad_lines=False, header=None)
    print (dfOfModel)

def main():
    splitAllWord()
    #readAllWord()
    #readModel()

if __name__ == "__main__":
    main()