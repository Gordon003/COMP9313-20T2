## import modules here

########## Question 1 ##########
# do not change the heading of the function
def c2lsh(data_hashes, query_hashes, alpha_m, beta_n):
    
    print("Start")
    
    offset = 0
    
    # No offset check
    full_amount = data_hashes.count()
    left_rdd = data_hashes.filter( lambda x: count(x[1], query_hashes, offset) < alpha_m)
    amount_of_candidates = 0
    
    while True:
        
        print("Offset", offset)
        print(amount_of_candidates)
        
        if amount_of_candidates < beta_n:
            offset += 1
        else:
            break
        
        left_rdd = left_rdd.filter( lambda x: count(x[1], query_hashes, offset) < alpha_m )
        amount_of_candidates = full_amount - left_rdd.count()

    print("End")
    candidate_rdd = data_hashes.subtractByKey(left_rdd)
    return candidate_rdd.keys()

def count(hashes_1, hashes_2, offset):
    counter = 0
    pos = 0
    while pos != len(hashes_1):
        if abs(hashes_1[pos] - hashes_2[pos]) <= offset:
            counter += 1
        pos += 1
    return counter