"""
 A simple Decision Tree Classifier, with no pruning
"""
import  test as tst
import  time
import numpy as np
import matplotlib.pyplot as plt
import funs as m
import classes as c
import create_data as cd

header = ["posx", "posy", "posz"]
# Column labels.
# These are used only to print the tree.

def build_tree(rows):
    """Builds the tree.

    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.
    """

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = m.find_best_split(rows)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return c.Leaf(rows)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = m.partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.

    return c.Decision_Node(question, true_branch, false_branch)



if __name__ == '__main__':

    #create simulation data to be classified
    train_sizes = [200]
    test_size = 50
    dim = 3
    elapsed_time = np.empty([len(train_sizes),1])
    for i,train_size in enumerate(train_sizes):
        start_time = time.time()
        training_data, testing_data, header = cd.create_sim_data(train_size,test_size,dim)


        #welcome
        m.welcome()
        start_time = time.time()
        my_tree = build_tree(training_data)
        print("time elapsed for tree training = ",(start_time - time.time()))
        print("Classifier trained")
        print("Printing decision tree..")

        m.print_tree(my_tree)
        filehandler = tst.FileOperations()
        tst.pdf(filehandler,my_tree)
        print("Printing Boundary..")
        m.print_boundary(-1,1,my_tree)
        print("Printing Test data..")
        m.print_tests(testing_data,my_tree)
#        plt.show()
        elapsed_time[i] = time.time() - start_time
    plt.plot(train_sizes,elapsed_time)
    plt.show()
# Next steps
# - add support for missing (or unseen) attributes
# - prune the tree to prevent overfitting
# - add support for greater dimensions
