import numpy,copy
from collections import OrderedDict, deque


class MatTable(object):

    def __init__(self,columns=['adp','det','noun','verb','adj','conj','prt','.','adv','num','pron'],dtype=int,mode="training"):
        self.row_idx_dict = OrderedDict()
        self.col_idx_dict = OrderedDict()
    
        self.row_new_items_process = deque()
        self.dtype = dtype
        self.mode = mode
        self.mat = numpy.zeros((0,len(columns)),dtype=self.dtype)

        for idx,ele in enumerate(columns):
            self.col_idx_dict[ele] = idx
        
    def map(self,items):
                 
        if items[0] in self.row_idx_dict and items[1] in self.col_idx_dict:
            i = self.row_idx_dict[items[0]]
            j = self.col_idx_dict[items[1]]
            self.mat[i,j]  += 1
            
        elif items[1] in self.col_idx_dict:
            self.row_idx_dict[items[0]] = len(self.row_idx_dict)
            self.row_new_items_process.append(items)
            
        else:
            pass

        if len(self.row_new_items_process) >= 1:
            self.__processQueue__()
            
    def __processQueue__(self):
        tempMat = copy.deepcopy(self.mat)
        new_dim = (len(self.row_idx_dict),len(self.col_idx_dict))
        self.mat = numpy.zeros(new_dim,dtype=self.dtype)
        self.mat[:tempMat.shape[0],:] = tempMat
        
        while len(self.row_new_items_process)>0:
            self.map(self.row_new_items_process.popleft())

    def get_rows (self):
        return self.row_idx_dict.keys()

    def get_columns(self):
        return self.col_idx_dict.keys()

    def get_table(self):
        return self.mat

    def get(self,item):
        try:
            return self.mat[self.row_idx_dict[item],:]
        except KeyError:
            return numpy.asarray([0]*len(self.col_idx_dict))
            

