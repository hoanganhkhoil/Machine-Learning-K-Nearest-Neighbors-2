# Author: Khoi Hoang
# K-Neighbor-Not-Using-SKlearn
# Predict-Breast-Cancer
# Data-Taken-From-Wisconsin

import numpy as np
from math import sqrt
#import matplotlib.pyplot as plt
import warnings
#from matplotlib import style
from collections import Counter
import pandas as pd
import random
#style.use('fivethirtyeight')


def k_nearest_neighbors(dataset, to_be_predicted, k=3):
    if len(dataset) >= k:
       warnings.warn('K is smaller than the number of classes')

    distances = []
    for group in dataset:
        for case in dataset[group]:
            euclidean_distance = np.linalg.norm(np.array(case) - np.array(to_be_predicted))
            distances.append([euclidean_distance, group])

    # Get only the label of the least or nearest distance up to k 
    votes = [distance[1] for distance in sorted(distances)[:k]]
    
    # Choose the most common class that appears in the list votes
    vote_result = Counter(votes).most_common(1)[0][0]

    #print votes
    #print Counter(votes).most_common(1)   
    
    return vote_result

    
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

# Convert all data to float type
full_data = df.astype(float).values.tolist()
#print full_data[:5]

random.shuffle(full_data)
#print full_data[:5]

test_size = 0.2
data_set = {2:[],4:[]}  # label 2 and label 4
test_set = {2:[],4:[]}   # label 2 and label 4

data = full_data[:-int(test_size * len(full_data))] # 80% of data 
test_data = full_data[-int(test_size * len(full_data)):]  # 20% of data

# Populate the dataset
for case in data:
    data_set[case[-1]].append(case[:-1])  # append everything up to the label

# Populate the testset
for case in test_data:
    test_set[case[-1]].append(case[:-1])  # append everything up to the label


correct = 0
total = 0


for group in test_set:
    for case in test_set[group]:
        vote = k_nearest_neighbors(data_set, case, k=5)
        if group == vote:
            correct += 1
        total += 1

print ('accuracy:' , float(correct)/total)


#for i in dataset:
#    for feature in dataset[i]:
#        plt.scatter(feature[0],feature[1],s=100,color=i)


#plt.scatter(new_features[0],new_features[1],s=100,color='g')
#plt.show()
