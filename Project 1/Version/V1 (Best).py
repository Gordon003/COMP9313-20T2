## import modules here
from time import time

########## Question 1 ##########
# do not change the heading of the function
def c2lsh(data_hashes, query_hashes, alpha_m, beta_n):
    
    
    print("Start")
    start_time = time()
    
    rdd = data_hashes.map(lambda x: data_map( x, query_hashes, 10 ) )
    print(rdd.getNumPartitions())
    
    rdd = rdd.filter(lambda x: x[1] >= alpha_m)
    print(rdd.getNumPartitions())
    
    rdd = rdd.reduceByKey(lambda x,y: x)
    print(rdd.getNumPartitions())

    
    print("End")
    end_time = time()
    print('C2LSH Time:', end_time - start_time)
    
    offset = 1
    rdd = data_hashes.filter( lambda x: count(x[1], query_hashes, offset) >= alpha_m )
    
    return rdd.keys()

def data_map(data_hashes, query_hashes, offset):
    counter = 0
    id, hashes = data_hashes
    for pos in range(len(data_hashes)):
        if abs(hashes[pos] - query_hashes[pos]) <= offset:
            counter += 1
    return counter

def count(hashes_1, hashes_2, offset):
    counter = 0
    pos = 0
    while pos != len(hashes_1):
        if abs(hashes_1[pos] - hashes_2[pos]) <= offset:
            counter += 1
        pos += 1
    return counter