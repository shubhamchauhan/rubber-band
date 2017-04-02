from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from NlpTools.Pos.pos import PosTagger
from NlpTools.Normalizer.PosInputWrapper import preprocess
from gensim import corpora, models
import gensim
import os
import logging

file_name = "politics_sports.txt" #write the file name that you want to train with

here = os.getcwd()
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)

en_stop = stopwords.words('english')
tokenizer = RegexpTokenizer(r'\w+')
p_stemmer = PorterStemmer()

f = open('../media/{}'.format(file_name), "r")
vocab = f.readlines()
sentences = []
texts = []

def GetKeywords(text, tagged):
	#print(tagged)
	text = text.split(' ')
	processed_tweet = []
	count = 0
	if len(text) != len(tagged):
		print(text,tagged)
	for i in tagged:
		if i == 'noun' or i == 'verb':
			processed_tweet.append(text[count-1])
		count = count + 1
	#print(processed_tweet)
	return processed_tweet

pos = PosTagger()


for i in vocab:
	i = i.replace("\n","")
	raw = i.lower()
	if len(i.split())>3:
		print('yes')
		raw = preprocess(raw)
		#print(raw)
		raw_tagged = pos.pos(raw)
		raw = GetKeywords(raw, raw_tagged)
		if len(raw)> 0:
			stopped_tokens = [i for i in raw if not i in en_stop]
			stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
			texts.append(stemmed_tokens)
	else:
		print('no')

dictionary = corpora.Dictionary(texts)
dictionary.save('questions.dict')
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('questions.mm', corpus)
mm = corpora.MmCorpus('questions.mm')
ldamodel = gensim.models.ldamodel.LdaModel(corpus = mm, num_topics=10, id2word = dictionary, passes=4, chunksize = 100)
ldamodel.save('models/lda.sframe')
print(ldamodel.print_topics(10))

'''model = gensim.models.ldamodel.LdaModel(vocab, num_topics = 10)
model.save('{}/models/model_{}'.format(here, file_name))'''