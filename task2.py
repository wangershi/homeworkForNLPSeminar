'''
    follow the instruction in https://blog.csdn.net/zrx1024/article/details/87826531?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-2.control
'''
from nltk.tokenize.stanford_segmenter import StanfordSegmenter
from nltk.parse.stanford import StanfordParser

segmenter = StanfordSegmenter()
segmenter.default_config('zh')
chi_parser = StanfordParser()

def getParser(sentence):
    segmentOfSentence = segmenter.segment(sentence)
    return list(chi_parser.parse(segmentOfSentence.split()))

def main():
    listOfParserResult = []
    count = 0
    maxLen = 300
    dictOfLen = {}
    with open("dataset/task2.txt", "r", encoding='utf-8') as fr:
        for line in fr.readlines():
            count += 1

            if count >= 400:
                break
            
            line = line.split("\n")[0]
            if len(line) > maxLen:
                line = line[:maxLen]
            print (len(line))
            print (type(line))
            print ("count = %s:\t%s" % (count, line))

            lenOfLine = len(line)
            lenOfLine = lenOfLine//10*10
            if lenOfLine not in dictOfLen:
                dictOfLen[lenOfLine] = 1
            else:
                dictOfLen[lenOfLine] += 1

            parserResult = getParser(line)
            print (parserResult)
            listOfParserResult.extend(parserResult)
    #print (dictOfLen)   # {100: 3, 50: 15, 20: 10, 60: 11, 70: 5, 40: 15, 120: 4, 30: 7, 90: 6, 160: 2, 190: 3, 180: 6, 360: 1, 220: 1, 110: 1, 80: 3, 130: 1, 210: 1, 170: 1, 200: 3, 720: 1}
    
    with open("daoz/task2.txt", "w", encoding='utf-8') as fw:
        for parserResult in listOfParserResult:
            fw.write("%s" % parserResult)
            fw.write("\n")
    
if __name__ == "__main__":
    main()