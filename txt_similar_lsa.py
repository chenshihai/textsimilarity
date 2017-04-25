# -*- coding: utf-8 -*-
#@author chenshihai
#@date 2017-04-24
from scipy.sparse import linalg
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
import numpy
import time
import math
import sys
default_encoding = "utf-8"
if default_encoding != sys.getdefaultencoding():
    reload(sys)
    sys.setdefaultencoding(default_encoding)


def pre_process(inputfile):  # 文件预处理，切分成文章名称前15位，然后文章关键词
    docnamelist = []  # 文档名称列表
    doc = []
    docwordslist = [] # 文档文章列表，每篇文章内含docwords（dict类型）
    docwordsall = {}  # 总词袋库 词-出现文章数
    stopword = ['c', 'e', 'm', 'p', 'u', 'w', 'y']  # 停用词合集
    try:
        source = open(inputfile, "r")  # 读取文件
        line = source.readline()  # 读取的每一行
        docname = ''
        while line:
            if line != '\n':
                docname = line[0:15]
                if len(docnamelist) != 0:
                    if docname == docnamelist[-1]:  # 如果等于最近的一个文章编号，说明是一篇
                        line = line.rstrip('\n')  # 删除字符串末尾换行
                        words = line.split("  ")  # 以两个空格分隔开的词表
                        #print words
                        for word in words:
                            flag = True
                            for s in stopword:
                                if word.find(s) != -1 or word == "":
                                    flag = False  # 标志位，如果包含停用词则为False，跳出循环
                                    break
                            if flag:
                                wordtmp = word.split("/")  # 将单词分割开，只取斜杠前面的部分
                                wordtmp = wordtmp[0]
                                if wordtmp in docwords:
                                    docwords[wordtmp] = docwords[wordtmp] + 1
                                else:
                                    docwords[wordtmp] = 1  # 该篇文章第一次出现这个词
                                    if wordtmp in docwordsall:
                                        docwordsall[wordtmp] = docwordsall[wordtmp] + 1  # 如果这个词在总词袋中则出现文章数加1
                                    else:
                                        docwordsall[wordtmp] = 1#否则记为1
                    else:  # 如果不是一篇，添加文章编号
                        docwordslist.append(docwords)  # 将之前文章词列表放入文档集合
                        docnamelist.append(docname)
                        docwords = {}  # 清空一篇文章的单词-数量
                        continue
                else:
                    docnamelist.append(docname)  # 列表没有元素，直接放入，第一次循环进入
                    docwords = {}  # 声明一篇文章的单词-数量
                    continue
            line = source.readline()  # 继续读取一行
        docwordslist.append(docwords)  # 将最后一篇文章词表放入


        docwordsall_save = docwordsall.copy()   # docwordsall_save 保存去掉停用词的词袋数量的副本
        for word in docwordsall_save:  # docwordsall去除全文只出现一次的单词后的词袋
            if docwordsall[word] == 1:
                docwordsall.pop(word)

        docwordsall_index =docwordsall.copy()
        # 将总词袋库进行编号，从0开始
        index = 0
        for word in docwordsall_index:
            docwordsall_index[word] = index;
            index = index + 1;
        # for word in docwordsall:
        #     print word, docwordsall[word]
        # print "+++++++++++++++"
        # print len(docnamelist)
        # print len(docwordslist)
        # i = 0  # 打印测试
        # for docwords in docwordslist:
        #     print docnamelist[i]
        #     i = i + 1
        #     for d in docwords:
        #         print d, docwords[d]
        #
        #     print "---------------------------------------------"

    except Exception, e:
            print 'Error:', e
    finally:
        source.close()
        print 'END fileprocess \n\n'
    weightMatrix = lil_matrix((len(docwordsall_index), len(docwordslist)), dtype=numpy.float)

    i = 0
    for docwords in docwordslist:  # 计算tf-idf值
        for word in docwords:
            if word in docwordsall:  # 计算只考虑去掉一次词的词，但是频率计算算的是原来总词数
                tf = float(docwords[word]) / len(docwords)
                idf = math.log(len(docwordslist)/docwordsall[word])
                tf_idf = tf * idf
                j = docwordsall_index[word]
                weightMatrix[j, i] = tf_idf
        i = i + 1

    weightMatrixF = weightMatrix.asformat("csr")
    ## SVD降维处理
    singularValue = 300
    setValue = 300
    U, S, V = linalg.svds(weightMatrixF, k=singularValue)  # U =U.shape[0] *300;S = 300*300;V = 300*V.shape[1]
    UN = U[:, 0:setValue]
    SN = S[::-1][0:setValue][::-1]
    VN = V[0:setValue, :]

    weightMatrixN = numpy.dot(numpy.dot(UN, numpy.diag(SN)), VN)

    similarity = numpy.dot(weightMatrixN.T, weightMatrixN)
    similarity = similarity - numpy.diag(numpy.diag(similarity))  # 连续使用两个diag构建对角阵 相当于去掉文章的自比较


    print "去掉停用词词袋长度", len(docwordsall_save)
    print "再去掉一次词词袋长度", len(docwordsall)
    print "文章名数量", len(docwordslist)
    print "文章内容数量", len(docnamelist)
    print
    for i in range(5):
        postion = numpy.argmax(similarity)  # 获得最大值，既最相似的两篇文章
        raw, column = similarity.shape
        m, n = divmod(postion, raw)  # position // raw,position % raw
        print "Top", i+1, ": 第", m, "篇和第", n, "篇"
        print "文章编号", docnamelist[m], "---", docnamelist[n]
        for word in docwordslist[m]:
            print word,
        print
        for word in docwordslist[n]:
            print word,
        print
        similarity[m][n] = 0
def main():
    start_time = time.time()
    pre_process("199801_clear.txt")
    end_time = time.time()
    print("---The total time---", end_time - start_time)
if __name__ == '__main__':
    main()
