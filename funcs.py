import numpy as np
import classes as c
import matplotlib.pyplot as plt
import itertools
import time

#welcome promt
def welcome():
    print("Hello, this is an oblique decision tree classifier\n")


def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])


def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)



def simple_split(n_features,rows):
    """find the best split out of the simple ones
        and return its information gain & question object
    """
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = gini(rows)
    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            question = c.Question(col, val)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain >= best_gain:
                best_gain, best_question = gain, question
    return best_gain, best_question



def oblique_split(n_features,rows):
    # check all the possible oblique (+- 45 deg ) splits

    # get all the possible combinations of (-1,1) for the  dimension given
    coefs = list(itertools.product([-1,1], repeat= n_features))
    # keep only half of them, since the others are the same
    coefs = coefs[int(len(coefs)/2):]
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = gini(rows)
    for i in range(len(coefs)): #for each set of coefs
        for j, test_value in enumerate(np.arange(-2, 2, 0.005)): #for each of the 45 degree line
                question = c.Question_linear(test_value, coefs[i])
                true_rows, false_rows = partition(rows, question)
                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue
                gain = info_gain(true_rows, false_rows, current_uncertainty)
                if gain >= best_gain:
                    best_gain, best_question = gain, question

    return best_gain, best_question


def find_best_split(rows):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    n_features = len(rows[0]) - 1  # number of columns
    print("no features = ",n_features)
    start_time = time.time()
    best_simple_gain, best_simple_question = simple_split(n_features,rows)
    best_oblique_gain, best_oblique_question = oblique_split(n_features,rows)
    print("time for splits = ",start_time-time.time())
    if best_simple_gain >= best_oblique_gain :
        return best_simple_gain, best_simple_question
    else:
        return best_oblique_gain, best_oblique_question

def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, c.Leaf):
        return node.predictions
    if node.question.match(row,1):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

def gini(rows):
    """Calculate the Gini Impurity for a list of rows.
    """
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity

def info_gain(left, right, current_uncertainty):
    """Information Gain.

    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

def partition(rows, question):
    """Partitions a dataset.

    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row[:-1],0):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs

def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf

    if isinstance(node, c.Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


def print_boundary(upper,lower,my_tree):

    d = np.empty([50*50,1])
    f2 = np.empty([50,1])
    xx = np.linspace(lower,upper,50)
    yy = np.linspace(lower,upper,50)
    x,y = np.meshgrid(xx,yy)
    full = np.c_[x.ravel(), y.ravel()]

    for i,bb in enumerate(full):
        d[i] = list(classify(bb,my_tree).keys())[0]
    d = d.reshape(x.shape)
    plt.contour(x,y,d)

def print_tests(testing_data,my_tree):

    dot = np.empty([testing_data[:,0].size,1])
    for i,row in enumerate(testing_data):
        d = classify(row,my_tree)
        dot[i] = list(d.keys())[0]
    plt_data = np.concatenate((testing_data[:,0:2],dot),axis=1)
    for i in range(testing_data[:,0].size):
        c = "red" if plt_data[i,2] == 1 else "green"
        plt.scatter(plt_data[i,0],plt_data[i,1],c=c)
