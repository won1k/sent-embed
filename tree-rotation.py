import nltk
from random import random 
from StringIO import StringIO

DEPTH_LIMIT = 4
TREE_PROB = 0.6

prefix = open("data/prefix_test.txt", "r")
trees = []

# Load trees into list
for line in prefix:
	trees.append(nltk.Tree.fromstring(line.strip()))

prefix.close()


# Tree rotation (right)
def tree_rotation(tree):
	pivot = tree[0]
	tree[0] = pivot[1]
	pivot[1] = tree
	return pivot

rotated = [tree_rotation(t) for t in trees]

# Print rotated in prefix/infix
def print_infix(t, out, depth = 0):
    if type(t) == str:
        print >>out, t, 
    else:
        print >>out, "(",
        for i in range(len(t)):
            print_infix(t[i], out, depth +1),
            if i < len(t) - 1:
                print >>out, t.label(),
            # print_infix(t[1], out, depth+1),
        print >>out, ")",
    if depth == 0:
        print >>out, ""

def print_prefix(t, out, depth = 0):  
    if type(t) == str:
        print >>out, t, 
    else:
        print >>out, "(",
        print >>out, t.label(),
        for i in range(len(t)):
            print_prefix(t[i], out, depth +1),
            
        print >>out, ")",
    if depth == 0:
        print >>out, ""

infix_rot = open("data/infix_test_rot.txt","w")
prefix_rot = open("data/prefix_test_rot.txt","w")

all = []
seen = {}
for t in rotated:
    a = StringIO()
    b = StringIO()
    print_infix(t, a)
    print_prefix(t, b)
    all.append((b.getvalue().strip(), a.getvalue().strip()))
    
for a in all:
    print >>prefix_rot, a[0]
    print >>infix_rot, a[1]

infix_rot.close()
prefix_rot.close()


