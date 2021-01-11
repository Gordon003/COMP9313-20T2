## import modules here
from time import time

########## Question 1 ##########
# do not change the heading of the function
def c2lsh(data_hashes, query_hashes, alpha_m, beta_n):
    
    # Calculate each data minimum offset to be candidate
    rdd_data = data_hashes.map(lambda x: count_collide( x, query_hashes, alpha_m ) )
    
    # Get max offset
    _, max_offset = rdd_data.max()
    
    # Generate map from their offset to max offset
    rdd = rdd_data.flatMap(lambda x: range(x[1], max_offset + 1))
    
    # Give each element a value of 1
    rdd = rdd.map(lambda x: (x, 1) )
    
    # Reduce each key by incrementing its value
    rdd = rdd.reduceByKey(lambda x, y: x + y)
    
    # Get keys that has a higher candidates than beta_n
    rdd = rdd.filter(lambda x: x[1] >= beta_n)
    
    # Get the minimum offset
    min_offset, _ = rdd.min()
    
    # Get all candidates within minimum offset
    rdd = rdd_data.filter(lambda x: x[1] <= min_offset).keys()
    
    return rdd

# Find the minimum offset to be counted as candidate
def count_collide(data_hashes, query_hashes, alpha_m):
    
    rID, hashes = data_hashes
    
    # Dictionary to keep track of all diffs
    offset_dict = {}
    
    # Calculate and add diff to dictionary
    for pos in range(len(hashes)):
        diff = int(abs(hashes[pos] - query_hashes[pos]))
        if diff in offset_dict:
            offset_dict[diff] += 1
        else:
            offset_dict[diff] = 1
    
    # Sort by keys
    offset_list = sorted(offset_dict.keys())
    
    # Find the minimum offset to be satisfied
    count = 0
    for x in offset_list:
        count += offset_dict[x]    
        if count >= alpha_m:
            return (rID, x)
    
    # Return rID withs its minimum offset
    return (rID, offset_dict[-1])