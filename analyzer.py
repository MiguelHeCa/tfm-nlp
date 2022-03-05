#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
Output parser for the analyzer
"""
import sys
import os
import io

import pyfreeling


class FreelingAnalyzer(object):
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
        self.setup()

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
        self.dep = pyfreeling.dep_lstm(
            DATA + LANG + "/dep_lstm/params-en.dat")

    def process(self, text, token=True, lemma=False, pos=False):
        features = {}
        if token:
            features['tokens'] = self.obtain_tokens(text)
        if lemma:
            features['lemmas'] = self.obtain_lemmas(text)
        if pos:
            features['PoS'] = self.obtain_pos(text)

        self.sp.close_session(self.sid)
        return features

    def obtain_tokens(self, text):
        results = {}
        for lin in io.StringIO(text.get_payload()):
            if lin.strip():
                lw = self.tk.tokenize(lin)
                ls = self.sp.split(self.sid, lw, False)
                if len(ls) > 0:
                    ws = ls[0].get_words()
                    for w in ws:
                        key = w.get_form()
                        add_to_dict(key, results)
        return results

    def obtain_lemmas(self, text):
        results = {}
        for lin in io.StringIO(text.get_payload()):
            if lin.strip():
                lw = self.tk.tokenize(lin)
                ls = self.sp.split(self.sid, lw, False)
                ls = self.mf.analyze(ls)
                if len(ls) > 0:
                    ws = ls[0].get_words()
                    for w in ws:
                        key = f'{w.get_form()}_{w.get_lemma()}'
                        add_to_dict(key, results)
        return results

    def obtain_pos(self, text):
        results = {}
        for lin in io.StringIO(text.get_payload()):
            if lin.strip():
                lw = self.tk.tokenize(lin)
                ls = self.sp.split(self.sid, lw, False)
                ls = self.mf.analyze(ls)
                ls = self.tg.analyze(ls)
                ls = self.sen.analyze(ls)
                if len(ls) > 0:
                    ws = ls[0].get_words()
                    for w in ws:
                        key = f'{w.get_form()}_{w.get_tag()}'
                        add_to_dict(key, results)
        return results


def add_to_dict(key, feature_dictionary):
    if key in feature_dictionary:
        feature_dictionary[key] += 1
    else:
        feature_dictionary[key] = 1
