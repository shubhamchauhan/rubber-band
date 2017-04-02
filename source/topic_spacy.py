
# coding: utf-8

# In[1]:

import spacy
import gensim
import logging
from gensim import corpora, models


# In[2]:

nlp = spacy.load('en')


# In[38]:

class TopicExtrator(object):

    def __init__(self):
        self.tokens = []
        self.name = 'temp'

    def makeTokens(self, hashtags):
        for i in hashtags:
            stemmed_tokens = []
            for token in nlp(i):
                if token.is_stop == False:
                    if token.pos_ in ['NOUN','ADJ', 'VBZ']:
                        stemmed_tokens.append(token.text)
            
            self.tokens.append(stemmed_tokens)
            print(stemmed_tokens)

    def ldaModel(self):
        dictionary = corpora.Dictionary(self.tokens)
        dictionary.save('{}.dict'.format(self.name))
        #logging.basicConfig(ormat f= '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)
        corpus = [dictionary.doc2bow(text) for text in self.tokens]
        corpora.MmCorpus.serialize('{}.mm'.format(self.name), corpus)
        mm = corpora.MmCorpus('{}.mm'.format(self.name))

        lda = gensim.models.ldamodel.LdaModel(corpus = mm, num_topics=3, id2word = dictionary, passes=10, update_every= 2, chunksize = 10000) 
        return(lda.print_topics(3))


# In[37]:

if __name__ == "__main__":
    k = TopicExtrator()
    k.makeTokens(['liverpool is the best english team ever', 
    	'Go liverpool, I want you guys to win this time', 
    	'if they didnt have oil money chelsea would have been a shitty club', 
    	'what a match cannot wait to watch how chelsea and liverpool with take onto each other',
    	'Simran wanted play in this match but Klopp didnt allow it to happen',
    	'Shivam is such a shitty player that he cannot even kick a ball'])
    print(k.ldaModel())


# In[ ]:



