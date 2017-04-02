from RnpMat import MatTable
import _pickle as pickle
import numpy
from scipy.optimize import minimize
import os
'''import MySQLdb as mdb
db = mdb.connect("localhost","root","agent47","bubble")
curo = db.cursor()
curi = db.cursor()
curo1 = db.cursor()
curi1 = db.cursor()
curo.execute("""SELECT * FROM stream where status = 0;""")
row = curo.fetchone()
'''

## Examples on usage of Pos class
"""
Example:

Processing:

pos = Pos()
pos.pos("This is a sentence") #returns a list of pos tags

Training:

pos= Pos(mode="train")
pos.train()

Benchmarking:

pos.Pos()
pos.benchmark()

"""

class Pos(object):

    def __init__(self,mode="process"):
        
        self.cur_dir =os.getcwd()
        
        self.previous_pos = MatTable()
        self.current_pos = MatTable()
        self.next_pos = MatTable()
        self.curr_0 = .5
        self.prev_0 = .5
        self.curr_1 = .8
        self.next_1 = .2
        self.curr_re = .4
        self.prev_re = .3
        self.next_re = .3
        
        self.mode = mode
        if mode == "process":
            self.load()
        print("__init__")

    def __train_process__(self,data_dict):

        """
        Input:
            data_dict
                Type: dictionary
                Format: {"word":[prev_word_pos,word_pos,next_word_pos]}
                Example: {"word":[None,"Noun","Verb"]}
        Returns:
            None
        """

        for ele in data_dict:
            self.previous_pos.map([ele,data_dict[ele][0]])
            self.current_pos.map([ele,data_dict[ele][1]])
            self.next_pos.map([ele,data_dict[ele][2]])

        print("__train_process__")

    def __process__(self,data):
        processed = []
        for i in xrange(len(data)):
            
            if i == 0:
                a = self.previous_pos.get(data[i+1])
                b = self.current_pos.get(data[i])
                s_a = numpy.sum(a)+1
                s_b = numpy.sum(b)+1
                processed.append(self.prev_0/s_a*a+self.curr_0/s_b*b)
                
            elif i == len(self.data)-1:
                a = self.current_pos.get(data[i])
                b = self.next_pos.get(data[i-1])
                s_a = numpy.sum(a)+1
                s_b = numpy.sum(b)+1
                processed.append(self.curr_1/s_a*a+self.next_1/s_b*b)
                
            else:
                a = self.previous_pos.get(data[i+1])
                b = self.current_pos.get(data[i])
                c = self.next_pos.get(data[i-1])
                s_a = numpy.sum(a)+1
                s_b = numpy.sum(b)+1
                s_c = numpy.sum(c)+1
                processed.append(self.prev_re/s_a*a+self.curr_re/s_b*b+self.next_re/s_c*c)
                
        processed = [x.tolist() for x in processed]
        processed_ids = [x.index(max(x)) for x in processed]
        return_data = [self.columns_[i] for i in processed_ids]
        return return_data
        
        print("__process__")

    def pos(self,sent):
        if type(sent) == str:
            self.data = sent.lower().split()
        elif hasattr(sent,'__iter__'):
            self.data = list[sent]
        else:
            raise TypeError

        processed = self.__process__(self.data)
        return processed
        

    def __dump__(self):
        data_to_dump = [self.previous_pos,self.current_pos,self.next_pos]
        pos_trained = open(self.cur_dir+"/data_pickles/pos_trained.pkl","wb")
        pickle.dump(data_to_dump,pos_trained,-1)
        pos_trained.close()

        print("__dump__")

    def load(self):
        pos_trained_data = open(self.cur_dir+"/data_pickles/pos_trained.pkl","rb")
        lst = pickle.load(pos_trained_data)
        self.previous_pos = lst[0]
        self.current_pos = lst[1]
        self.next_pos = lst[2]
        self.columns_ = self.current_pos.get_columns()

        print("load")

    def train(self):
        if not self.mode == "process":
            pkl_file = open(self.cur_dir+"/data_pickles/pos_data.pkl","rb")
            lst = pickle.load(pkl_file)
            pkl_file.close()
            self.current_pos.map(["",None])
            for ele in lst:
                self.__train_process__(ele)
            self.__dump__()
            print("train complete")
            return True
        else:
            print("Not Allowed")
            return False

        print("train")

    def check_rows(self):
        for i in range(len(self.next_pos.get_columns())):
            print(self.previous_pos.get_columns()[i],self.current_pos.get_columns()[i],self.next_pos.get_columns()[i])

        print("check_rows")

    def values(self,word):
        print(self.previous_pos.get_columns())
        print(self.previous_pos.get(word))
        print(self.current_pos.get(word))
        print(self.next_pos.get(word))
        
        print("values")

    def benchmark(self):
        f = os.getcwd()+"/data_pickles/benchmark_data.pkl"
        correct_words = 0.0
        wrong_words = 0.0
        correct_sents = 0.0
        wrong_sents = 0.0
        
        import time
        start_time = time.time()
        with open(f,"rb") as handle:
            lines = pickle.load(handle)
        valid_pos = set(self.current_pos.get_columns())
        open("output.txt","w").close()
        output_file = open("wrong_sent_tags.txt","a")
        parsing_start_time = time.time()
        for line in lines:
            if set(line[1].lower().split()) <= valid_pos:
                sent_correct = True
                pred_pos = self.pos(line[0])
                act_pos = line[1].lower().split()
                
                for idx,ele in enumerate(pred_pos):
                    if ele == act_pos[idx]:
                        correct_words += 1
                    else:
                        wrong_words += 1
                        sent_correct = False
                                
                if sent_correct:
                    correct_sents +=1

                else:
                    wrong_sents +=1
                    output_file.write(line[0]+"\n")
                    output_file.write(str(line[1].lower())+"\n")
                    output_file.write(" ".join(pred_pos)+"\n\n")
                                    
                    
        output_file.close()
        total_time = time.time()-start_time
        parsing_time = time.time()-parsing_start_time
        print("Benchmark completed in total time(sec): ", total_time)
        print("Parsing completed in total time(sec): ", parsing_time)
        print("Total words: ", correct_words+wrong_words)
        print("Average words/sec: ", round((correct_words+wrong_words)*1.0/total_time))
        print("Word Accuracy: ", 1.0*correct_words/(correct_words+wrong_words))
        print("Sentence Accuracy: ", 1.0*correct_sents/(correct_sents+wrong_sents))
        
        print("benchmarked")

#pos =Pos()
pos= Pos(mode="train")
pos.train()

'''while row is not None:

    if row[1] is not None:
        try:
            if row[7] == 'en':
                k = pos.pos(row[1])
                words = row[1].split(" ")
                print(words)
                nouns = []
                print(k)
                for i in k:
                    if i == 'noun':
                        nouns.append(i)
                print(nouns)



        except:
            pass
    count = count + 1



    row = curo.fetchone()
    print(count)


curo.close()
curi.close()
curo1.close()
curi1.close()
db.close()'''