import numpy as np


# create data to work on
def create_sim_data(train_size, test_size,dim):


    print("creating simulation data...")

    header = []
    for i in range(dim):
        s = "pos" + str(i)
        header.append(s)
    b1 = np.empty([train_size, 1])
    a1 = 2*np.random.rand(train_size, dim) - 1
    b2 = np.empty([test_size,1])
    a2 = 2*np.random.rand(test_size,dim) - 1
    for i in range(train_size):
        if (a1[i,0] + a1[i,1] - a1[i,2] >=0 )and(a1[i,1] > 0):
            b1[i] = 1
        else:
            b1[i] = 0
    for i in range(test_size):
        if (a2[i,0] + a2[i,1] - a2[i,2] >=0 )and(a2[i,1] > 0):
            b2[i] = 1
        else:
            b2[i] = 0

    training_data = np.concatenate((a1, b1), axis=1)
    testing_data = np.concatenate((a2,b2), axis=1)
    return training_data, testing_data, header
