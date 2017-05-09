#-*- coding: utf-8 -*-
from pyspark import SparkContext, SparkConf
import math,time

start_time = time.time()

conf = SparkConf().setAppName("SOCPMI")
sc = SparkContext(conf = conf)
window_size = 5	#左右各5個詞
delta = 2.0		#corpus越大，delta要越大，12個例句用0.7，我先暫時用2
gama = 3

def window(origin):
	origin = origin.split(" ")
	output = []
	for Hitword in origin:
		index = origin.index(Hitword)	
		start = (index - 5) if (index - 5) > 0 else 0
		leng = len(origin) - start
		if leng >= 11:	#array夠長，可以直接11個長度
			tmp = origin[start:(start + window_size * 2 + 1)]		
		else:	#不夠11的，則是剩餘的長度
			tmp = origin[start:]
		tmp = list(set(tmp))
		for x in tmp:
			if x == Hitword:
				continue
			output.append(((Hitword,x),1))
	return output

def log2(number):
	log2 = math.log(number)/math.log(2)
	return log2

def Beta(word):
	beta = int(((log2(int(word[1])+1))**2) * (log2(NumberOfWords)/delta))
	return (word[0],[beta])

#將WordNeighbors與其對應的wordCount組成(WordNeighbors,wordCount)的pair
#Example of word : (u'consists', [(u'more', u'consists', '1'), (u'first', u'consists', '2'), (u'consists', 3)])
def PreProcessingPMI_part1(word):
	output = []
	for WNpair in word[1]:
		if len(WNpair) == 2:
			neighborWC = WNpair
			break
	for WNpair in word[1]:
		if len(WNpair) == 2:
			continue
		output.append((WNpair[0],[(WNpair,neighborWC)]))	
	return output

#取出Word與其對應的wordCount組成(Word,wordCount)的pair
#Example of word : #(u'nutritious', [((u'nutritious', u'delicious', '1'), (u'delicious', 18)), ((u'nutritious', u'remain', '1'), (u'remain', 2)), (u'nutritious', 1)])
def PreProcessingPMI_part2(word):
	output = []
	for WNpair in word[1]:
		if WNpair[0] == word[0]:
			WordWC = WNpair
			break
	for WNpair in word[1]:
		if WNpair[0] == word[0]:
			continue
		output.append((WordWC,)+WNpair)	
	return output

#Example of word : ((u'nutritious', 1), (u'nutritious', u'delicious', '1'), (u'delicious', 18))
def PMI(word):
	W_N = 1.0		#W_N 為 Word 與 Neighbor各自出現的次數
	for value in word:
		if len(value) == 3:
			WN = float(value[2])*NumberOfWordsInArticle	#WN 為 Word 與 Neighbor 共同出現的次數
			output = value
		else:
			W_N=W_N*float(value[1])
	PMI = log2(WN/W_N)
	output = (output[0],[(output[1],PMI)])
	return output

#取出 Word 的 beta 組成(Word,beta,[ Neighbor , PMI ]])的 pair
#Example of word : (u'nutritious', [(u'delicious', 11.09755666411615), (u'remain', 14.26748166555846), (u'rice', 2.900066914311634), 6])
def SOCPMI_preprocess(word):
	if isinstance(word[1], list):	#確認 Word 有沒有 Neighbor
		for q in word[1]:
			if not isinstance(q, tuple):	#若不是 tuple 型態則表示該值即為 Word 的 beta
				beta = q
				word[1].remove(q)
				break
		word[1].sort(key=lambda tup: tup[1],reverse=True)	#將 Word 的 Neighbor 按照 PMI 由大到小作排序
		output = (word[0],beta,word[1])	
	else:
		output = word #若無 Neighbor 即輸出 (Word,beta)
	return output

#Example of word : (u'nutritious', [(u'delicious', 11.09755666411615), (u'remain', 14.26748166555846), (u'rice', 2.900066914311634), 6])
def SOCPMI(word):
	similarity = 0
	word1 = word[0][0]
	word2 = word[1][0]
	beta1 = word[0][1]
	beta2 = word[1][1]
	if len(word[0]) == 3:
		word1Neighbor = dict(word[0][2])
		for key in word1Neighbor:
			word1Neighbor[key] = float(word1Neighbor[key])
	else:
		word1Neighbor = {}
	if len(word[1]) == 3:
		word2Neighbor = dict(word[1][2])
		for key in word2Neighbor:
			word2Neighbor[key] = float(word2Neighbor[key])
	else:
		word2Neighbor = {}
	sum1 = 0
	sum2 = 0
	word1NeighborKey = []
	word2NeighborKey = []
	if beta1 != 0:
		if len(word1Neighbor) < beta1 :
			beta1 = len(word1Neighbor)

		for tup in word[0][2]:
			word1NeighborKey.append(tup[0])					#一旦將list of tuple轉成dict，順序就會亂掉

		for q in xrange(0,beta1):					#只計算top beta1
			if word1NeighborKey[q] in word2Neighbor and word2Neighbor[word1NeighborKey[q]] > 0:
				sum1 += word2Neighbor[word1NeighborKey[q]]**gama

		#計算 top beta1 後跟最後一個word 的 pmi 值相同的字
		if len(word1Neighbor) > beta1 :
			for q in xrange(beta1,len(word1Neighbor)):
				if abs(word1Neighbor[word1NeighborKey[beta1-1]] - word1Neighbor[word1NeighborKey[q]]) > 0.1:
					break
				if word1NeighborKey[q] in word2Neighbor and word2Neighbor[word1NeighborKey[q]] > 0:
					sum1 += word2Neighbor[word1NeighborKey[q]]**gama
		if sum1!=0:
			sum1 = sum1/beta1

	if beta2 != 0 and len(word2Neighbor) > 0:
		if len(word2Neighbor) < beta2 :
			beta2 = len(word2Neighbor)

		for tup in word[1][2]:
			word2NeighborKey.append(tup[0])						#一旦將list of tuple轉成dict，順序就會亂掉

		for q in xrange(0,beta2):					#只計算top beta2
			if word2NeighborKey[q] in word1Neighbor and word1Neighbor[word2NeighborKey[q]] > 0:
				sum2 += word1Neighbor[word2NeighborKey[q]]**gama

		#計算 top beta2 後跟最後一個word 的 pmi 值相同的字
		if len(word2Neighbor) > beta2 :
			for q in xrange(beta2,len(word2Neighbor)):
				if abs(word2Neighbor[word2NeighborKey[beta2-1]] - word2Neighbor[word2NeighborKey[q]]) > 0.1:
					break
				if word2NeighborKey[q] in word1Neighbor and word1Neighbor[word2NeighborKey[q]] > 0:
					sum2 += word1Neighbor[word2NeighborKey[q]]**gama
		if sum2!=0:
			sum2 = sum2/beta2

	similarity = sum1 + sum2;

	output = (word1,word2,similarity)	
	return output

#輸入原始文字檔
text_file = sc.textFile("./SOCPMIInput/CleanCorpus.txt")

#WordCount
WordCounts = text_file.flatMap(lambda line: line.split(" ")).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)

#文章共有幾個不同的詞
NumberOfWords = len(WordCounts.collect())

#文章總詞數
NumberOfWordsInArticle = WordCounts.map(lambda word:int(word[1])).reduce(lambda a,b:a+b)

#抓出 Word 的 Neighbor 以及共同出現次數
WordNeighbors = text_file.map(window).flatMap(lambda q:q).reduceByKey(lambda a,b:a+b).map(lambda q : (q[0][0],(q[0][1],str(q[1]))))

WordNeighborsKeyList = WordNeighbors.map(lambda a:a[0]).distinct().collect()

WordsBeta = WordCounts.map(Beta)

#開始計算PMI

#只取出 CandidateNeighborsKeyList 的 WordCount
WordNeighbors_WordCounts = WordCounts.filter(lambda word: word[0] in WordNeighborsKeyList).map(lambda word : (word[0],[word]))

#將 Neighbor 當 key 以透過 Reduce 取得 Neighbor 的 WordCount
NeighborCandidate = WordNeighbors.map(lambda q : (q[1][0],[(q[0],q[1][0],q[1][1])])).union(WordCounts.map(lambda word : (word[0],[word]))).reduceByKey(lambda a,b:a+b).filter(lambda word:len(word[1])>1)

pmi = NeighborCandidate.map(PreProcessingPMI_part1).flatMap(lambda line:line).union(WordNeighbors_WordCounts).reduceByKey(lambda a,b:a+b).map(PreProcessingPMI_part2).flatMap(lambda a:a).map(PMI)

#開始計算SOCPMI

socpmiPreProcess = pmi.union(WordsBeta).reduceByKey(lambda a,b:a+b).map(SOCPMI_preprocess)

socpmi = socpmiPreProcess.cartesian(socpmiPreProcess).map(SOCPMI).filter(lambda word: word[2] != 0)

#map from tuple to String and charset to Utf8
socpmi = socpmi.map(lambda word:u'\t'.join(unicode(s) for s in word).encode("utf-8").strip())

socpmi.saveAsTextFile("Word_SOCPMI_Similarity")

#f = open("./SOCPMIOutput/Word_SOCPMI_Similarity.txt","w")
#for x in socpmi.collect(): 
#	f.write(x + '\n')
#f.close()

#輸出Debug用
# for x in socpmi.take(10):
# 	print x

sc.stop()

print("--- %s minutes ---" % ((time.time() - start_time)/60))
