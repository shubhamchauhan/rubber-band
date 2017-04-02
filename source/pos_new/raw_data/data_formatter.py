import pickle


def training_data():
    data_file = open("trainingdata.txt")
    lines = data_file.readlines()
    data_list = []

    for line in lines:
        formatted_line = [x.split('/') for x in line.lower().split()]
        for idx,ele in enumerate(formatted_line):
            data_dict = {}
            if idx ==0:
                data_dict[ele[0]] = [None,ele[1],formatted_line[idx+1][1]]
            elif idx == len(formatted_line)-1:
                data_dict[ele[0]] = [formatted_line[idx-1][1],ele[1],None]
            else:
                data_dict[ele[0]] = [formatted_line[idx-1][1],ele[1],formatted_line[idx+1][1]]
            data_list.append(data_dict)

    pkl_file = open("pos_data.pkl","wb")
    pickle.dump(data_list,pkl_file,-1)
    pkl_file.close()

def benchmarking_data():
    data_file = open("benchmarkingdata.txt")
    lines = data_file.readlines()
    success = 0
    failed = 0
    final = []
    for line in lines:
        sent = []
        tags = []
        temp_line = line.split()
        for ele in temp_line:
            sent.append(ele.split('/')[0])
            tags.append(ele.split('/')[1])
        final.append([' '.join(sent),' '.join(tags)])

    with open("benchmark_data.pkl","wb") as handle:
        pickle.dump(final,handle,-1)

    print "BenchMark data Processing complete :D"

    with open("benchmark_data.pkl","rb") as handle:
        x = len(pickle.load(handle))

        if x == len(lines):
            print "valid File"
        else:
            print "something went wrong"
        
    
        
def optminizing():
    data_file = open("trainingdata.txt")
    lines = data_file.readlines()
    success = 0
    failed = 0
    final = []
    for i in xrange(1000):
        sent = []
        tags = []
        temp_line = lines[i].split()
        for ele in temp_line:
            sent.append(ele.split('/')[0])
            tags.append(ele.split('/')[1])
        final.append([' '.join(sent),' '.join(tags)])

    with open("opt_data.pkl","wb") as handle:
        pickle.dump(final,handle,-1)

    print "BenchMark data Processing complete :D"

    with open("opt_data.pkl","rb") as handle:
        x = len(pickle.load(handle))

        if x == 1000:
            print "valid File"
        else:
            print "something went wrong"
    
    


training_data()
benchmarking_data()
optminizing()
            
        
