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
        "-l", "--lang", default="en", help="Language selection. Default English"
    )
    ap.add_argument("-r", "--rootdir", default="../maildir/", help="Root directory")
    ap.add_argument("-s", "--sender", default="lay-k", help="Person emails folder")
    ap.add_argument("-t", "--token", action='store_false', help="Get token. Default True")
    ap.add_argument("-m", "--lemma", action='store_true', help="Get lemmas. Default False")
    ap.add_argument("-p", "--pos", action='store_true', help="Get PoS. Default False")

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

    print('Initializing Freeling...')
    anal = analyzer.FreelingAnalyzer(basedir, args["lang"])

    print('Getting features...')
    mailsWithFeatures = {}
    for mail in email_list[:2]:
        actualEmail = mail[1]
        mailsWithFeatures[mail] = parser.obtain_base_features(actualEmail)
        if args['token']:
            mailsWithFeatures[mail][actualEmail['message-id']]['tokens'] = anal.obtain_tokens(actualEmail)
        if args['lemma']:
            mailsWithFeatures[mail][actualEmail['message-id']]['lemmas'] = anal.obtain_lemmas(actualEmail)
        if args['pos']:
            mailsWithFeatures[mail][actualEmail['message-id']]['pos'] = anal.obtain_pos(actualEmail)

    print(mailsWithFeatures[mail])
    anal.close()


if __name__ == "__main__":
    main()
