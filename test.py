import classes as c
from collections import deque

class FileOperations:

    def __init__(self):
        self.file1 = open("tree", "w")
        self.file1.write("digraph Tree{\nnode[shape=box];\n")

    def write_to_file(self,num,num_in,node):
        if isinstance(node,c.Leaf):
            self.file1.write(str(num) + '[label = "' + str(node.predictions)+ '"];')
        else:
            self.file1.write(str(num) + '[label = "' + str(node.question) + '"];\n')
            s1 = str(num) + "->" + str(num_in+1) + '[labeldistance=2.5,labelangle=45,headlabel="True"];\n'
            s2 = str(num) + "->" + str(num_in + 2) + '[labeldistance=2.5,labelangle=45,headlabel="False"];\n'
            self.file1.write(s1)
            self.file1.write(s2)

def pdf(filehandler,node):
    q = deque([])
    q.append(node)
    num_out = 0
    num_in = 0
    while (len(list(q))>0):
        n = q.popleft()
        filehandler.write_to_file(num_out,num_in,n)
        num_out += 1
        if isinstance(n, c.Leaf):
            #if it has no children
            continue
        else:
            #enque the children
            q.append(n.true_branch)
            q.append(n.false_branch)
            num_in +=2
