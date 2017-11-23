import funs as m
import numpy as np
import dt as main

class Question_linear:
    #check if a linear combination with coefs of feautures >= test_value
    def __init__(self,test_value,coefs):
        self.test_value = test_value
        self.coefs = coefs
        self.type = "linear"
        self.dim = len(coefs)

    def match(self,example,flag):
        #this is a bug
       # if (example.size >= 3):
       #     example = example[:-1]
        val = np.dot(example,self.coefs)
        return val >= self.test_value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = ">="
        s = ""
        for i in range(self.dim):
            if i == self.dim - 1:
                s += '('+str(self.coefs[i])+')' + "*" + str(main.header[i]) + " " + condition + " " + str(self.test_value)
            else:
                s += '('+str(self.coefs[i])+')' + "*" + str(main.header[i]) + " +"
        return s

class Question:


    """A Question is used to partition a dataset.

    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. See the demo below.
    """

    def __init__(self, column, value):
        self.column = column
        self.value = value
        self.type = "ordinary"

    def match(self, example,flag):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if m.is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if m.is_numeric(self.value):
            condition = ">="
        return "%s %s %s" % (
            main.header[self.column], condition, str(self.value))

class Leaf:
    """A Leaf node classifies data.

    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows):
        self.predictions = m.class_counts(rows)


class Decision_Node:
    """A Decision Node asks a question.

    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

