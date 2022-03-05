#! /usr/bin/python3
# -*- coding: utf-8 -*-

"""
Main program for to parse mails
"""

import argparse
import os

import parser
import analyzer


def main():
    # Construct arguments parser
    ap = argparse.ArgumentParser()

    # Add arguments
    ap.add_argument(
        "-l", "--lang", default="en", help="language selection. Default English"
    )
    ap.add_argument("-r", "--rootdir", default="../maildir/", help="root directory")
    ap.add_argument("-s", "--sender", default="lay-k", help="person emails")

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

    anal = analyzer.FreelingAnalyzer(basedir, args["lang"])

    mailsWithFeatures = {}
    for mail in email_list[:1]:
        actualEmail = mail[1]
        mailsWithFeatures[mail] = parser.obtain_base_features(actualEmail)
        mailsWithFeatures[mail][actualEmail['message-id']].update(anal.process(actualEmail, lemma=True, pos=True))

    print(mailsWithFeatures[mail])


if __name__ == "__main__":
    main()
