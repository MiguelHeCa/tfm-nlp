#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
Output parser for the analyzer
"""
import sys
import os
import io

import pyfreeling


class freeling_analyzer(object):
    def __init__(self, folder, lang):
        self.folder = folder
        self.lang = lang
        self.tk = None
        self.sp = None
        self.sid = None
        self.mf = None
        self.tg = None
        self.sen = None
        self.parser = None
        self.dep = None

    # ------------  output a parse tree ------------
    def printTree(self, ptree, depth):

        node = ptree.begin()

        print("".rjust(depth * 2), end="")
        info = node.get_info()
        if info.is_head():
            print("+", end="")

        nch = node.num_children()
        if nch == 0:
            w = info.get_word()
            print(
                "({0} {1} {2})".format(w.get_form(), w.get_lemma(), w.get_tag()), end=""
            )

        else:
            print("{0}_[".format(info.get_label()))

            for i in range(nch):
                child = node.nth_child_ref(i)
                self.printTree(child, depth + 1)

            print("".rjust(depth * 2), end="")
            print("]", end="")

        print("")

    # ------------  output a parse tree ------------
    def printDepTree(self, dtree, depth):

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
                    self.printDepTree(d, depth + 1)

            ch = {}
            for i in range(nch):
                d = node.nth_child_ref(i)
                if d.begin().get_info().is_chunk():
                    ch[d.begin().get_info().get_chunk_ord()] = d

            for i in sorted(ch.keys()):
                self.printDepTree(ch[i], depth + 1)

            print("".rjust(depth * 2), end="")
            print("]", end="")

        print("")

    def setup(self):
        # Check whether we know where to find FreeLing data files
        if "FREELINGDIR" not in os.environ:
            if sys.platform == "win32" or sys.platform == "win64":
                os.environ["FREELINGDIR"] = "C:\\Program Files"
            else:
                os.environ["FREELINGDIR"] = "/usr/local"
            print(
                "FREELINGDIR environment variable not defined, trying ",
                os.environ["FREELINGDIR"],
                file=sys.stderr,
            )

        if not os.path.exists(os.environ["FREELINGDIR"] + "/share/freeling"):
            print(
                "Folder",
                os.environ["FREELINGDIR"] + "/share/freeling",
                "not found.\n" +
                "Please set FREELINGDIR environment variable to FreeLing installation directory",
                file=sys.stderr,
            )
            sys.exit(1)

        # Location of FreeLing configuration files.
        DATA = os.environ["FREELINGDIR"] + "/share/freeling/"

        # Init locales
        pyfreeling.util_init_locale("default")

        # create language detector. Used just to show it. Results are printed
        # but ignored (after, it is assumed language is LANG)
        # la = pyfreeling.lang_ident(DATA + "common/lang_ident/ident-few.dat")

        # create options set for maco analyzer.
        # Default values are Ok, except for data files.
        LANG = self.lang
        op = pyfreeling.maco_options(LANG)
        op.set_data_files(
            "",
            DATA + "common/punct.dat",
            DATA + LANG + "/dicc.src",
            DATA + LANG + "/afixos.dat",
            "",
            DATA + LANG + "/locucions.dat",
            DATA + LANG + "/np.dat",
            DATA + LANG + "/quantities.dat",
            DATA + LANG + "/probabilitats.dat",
        )

        # create analyzers
        self.tk = pyfreeling.tokenizer(DATA + LANG + "/tokenizer.dat")
        self.sp = pyfreeling.splitter(DATA + LANG + "/splitter.dat")
        self.sid = self.sp.open_session()
        self.mf = pyfreeling.maco(op)

        # activate morpho modules to be used in next call
        self.mf.set_active_options(
            False,
            True,
            True,
            True,  # select which among created
            True,
            True,
            False,
            True,  # submodules are to be used.
            True,
            True,
            True,
            True,
        )
        # default: all created submodules are used

        # create tagger, sense anotator, and parsers
        self.tg = pyfreeling.hmm_tagger(DATA + LANG + "/tagger.dat", True, 2)
        self.sen = pyfreeling.senses(DATA + LANG + "/senses.dat")
        self.parser = pyfreeling.chart_parser(
            DATA + LANG + "/chunker/grammar-chunk.dat"
        )
        self.dep = pyfreeling.dep_txala(
            DATA + LANG + "/dep_txala/dependences.dat", self.parser.get_start_symbol()
        )

    def process(self, msg):
        self.setup()

        for lin in io.StringIO(msg):
            l = self.tk.tokenize(lin)
            ls = self.sp.split(self.sid, l, False)

            ls = self.mf.analyze(ls)
            ls = self.tg.analyze(ls)
            ls = self.sen.analyze(ls)
            ls = self.parser.analyze(ls)
            ls = self.dep.analyze(ls)

            # output results
            for s in ls:
                ws = s.get_words()
                for w in ws:
                    print(
                        w.get_form()
                        + " "
                        + w.get_lemma()
                        + " "
                        + w.get_tag()
                        + " "
                        + w.get_senses_string()
                    )
                print("")

                tr = s.get_parse_tree()
                self.printTree(tr, 0)

                dp = s.get_dep_tree()
                self.printDepTree(dp, 0)

        # clean up
        self.sp.close_session(self.sid)
