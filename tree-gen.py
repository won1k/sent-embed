import nltk
from random import random 
from StringIO import StringIO

DEPTH_LIMIT = 4
TREE_PROB = 0.6

def make_expression(depth = 0):
    if depth > 1 and (depth >= DEPTH_LIMIT or random() < TREE_PROB) :
        return nltk.Tree(int(random() * 2), [])
        # return nltk.Tree(int(1), [])
    n = 2
    if random() < 0.5:
        return nltk.Tree("+", [make_expression(depth+1) for i in range(n)])
    else:
        return nltk.Tree("*", [make_expression(depth+1) for i in range(n)])


infix = open("data/infix_test.txt", "w")
prefix = open("data/prefix_test.txt", "w")

def print_infix(t, out, depth = 0):
    if len(t) == 0:
        print >>out, t.label(), 
    else:
        if depth != 0:
            print >>out, "(",
        for i in range(len(t)):
            print_infix(t[i], out, depth +1),
            if i < len(t) - 1:
                print >>out, t.label(),
            # print_infix(t[1], out, depth+1),
        if depth != 0:
            print >>out, ")",
    if depth == 0:
        print >>out, ""

def print_prefix(t, out, depth = 0):  
    if len(t) == 0:
        print >>out, t.label(), 
    else:
        print >>out, "(",
        print >>out, t.label(),
        for i in range(len(t)):
            print_prefix(t[i], out, depth +1),
            
        print >>out, ")",
    if depth == 0:
        print >>out, ""

all = []
seen = {}
for i in range(25000):
    m = make_expression()
    a = StringIO()
    b = StringIO()
    print_infix(m, a)
    print_prefix(m, b)
    all.append((b.getvalue().strip(), a.getvalue().strip()))
    
#all.sort(key= lambda a: (len(a[0]), a[0]) )
for a in all:
    print >>prefix, a[0]
    print >>infix, a[1]

infix.close()
prefix.close()
