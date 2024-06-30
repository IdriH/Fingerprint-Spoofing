import numpy 

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))



def load(fname): 
    DList = []
    labelsList = []


    with open(fname) as f: 
        for line in f: 
            try: 
                attrs = line.split(',')[0:-1]
                attrs = vcol(numpy.array([float(i) for i in attrs]))
                label = line.split(',')[-1]
                DList.append(attrs)
                labelsList.append(label)
            
            except: 
                pass
    
    return numpy.hstack(DList),numpy.array(labelsList, dtype = numpy.int32)