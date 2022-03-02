#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
Output parser for the analyzer
"""

# ------------  output a parse tree ------------


def printTree(ptree, depth):

    node = ptree.begin()

    print("".rjust(depth * 2), end="")
    info = node.get_info()
    if info.is_head():
        print("+", end="")

    nch = node.num_children()
    if nch == 0:
        w = info.get_word()
        print("({0} {1} {2})".format(w.get_form(), w.get_lemma(), w.get_tag()), end="")

    else:
        print("{0}_[".format(info.get_label()))

        for i in range(nch):
            child = node.nth_child_ref(i)
            printTree(child, depth + 1)

        print("".rjust(depth * 2), end="")
        print("]", end="")

    print("")


# ------------  output a parse tree ------------
def printDepTree(dtree, depth):

    node = dtree.begin()

    print("".rjust(depth * 2), end="")

    info = node.get_info()
    link = info.get_link()
    linfo = link.get_info()
    print("{0}/{1}/".format(link.get_info().get_label(), info.get_label()), end="")

    w = node.get_info().get_word()
    print("({0} {1} {2})".format(w.get_form(), w.get_lemma(), w.get_tag()), end="")

    nch = node.num_children()
    if nch > 0:
        print(" [")

        for i in range(nch):
            d = node.nth_child_ref(i)
            if not d.begin().get_info().is_chunk():
                printDepTree(d, depth + 1)

        ch = {}
        for i in range(nch):
            d = node.nth_child_ref(i)
            if d.begin().get_info().is_chunk():
                ch[d.begin().get_info().get_chunk_ord()] = d

        for i in sorted(ch.keys()):
            printDepTree(ch[i], depth + 1)

        print("".rjust(depth * 2), end="")
        print("]", end="")

    print("")
