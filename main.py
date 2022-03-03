#! /usr/bin/python3
# -*- coding: utf-8 -*-

"""
Main program for to parse mails
"""

import argparse
# import sys
import os

# import pyfreeling
import parser
import analyzer

# from pathlib import Path


def main():
    # Construct arguments parser
    ap = argparse.ArgumentParser()

    # Add arguments
    ap.add_argument(
        "-l", "--lang", default="en", help="language selection. Default English"
    )
    ap.add_argument("-r", "--rootdir", default="../maildir/", help="root directory")
    ap.add_argument("-s", "--sender", default="arora-h", help="person emails")

    args = vars(ap.parse_args())

    basedir = args["rootdir"] + "/" + args["sender"]

    # Parsing emails
    mail_dict = {}
    email_list = []
    for directory, subdirectory, filenames in os.walk(basedir):
        for filename in filenames:
            parser.raw_parse(os.path.join(directory, filename), email_list)
    email_list.sort(key=lambda x: x[0])
    pureThreads = parser.obtain_raw_threads(mail_dict, email_list)

    email_sample = email_list[0][1].get_payload()
    print(email_sample)

    # Analyze emails
    fla = analyzer.freeling_analyzer(basedir, args['lang'])
    fla.process(email_sample)


if __name__ == "__main__":
    main()
