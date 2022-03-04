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

    anal=analyzer.freeling_analyzer(basedir,"en")
    mailsWithFeatures={}
    for mail in email_list:
        actualEmail=mail[1]
        mailsWithFeatures[mail]=parser.obtain_base_features(actualEmail)
        ls=anal.obtain_tokens(actualEmail,mailsWithFeatures[mail])
        #Rellenar esta parte comentada
        #analyzer.obtain_lema(ls,mailsWithFeatures[actualEmail])
        #analyzer.obtain_PoS_features(ls,mailsWithFeatures[actualEmail])
    print(mailsWithFeatures)


    # email_sample = email_list[0][1].get_payload()
    # print(email_sample)
    # # Analyze emails
    # fla = analyzer.freeling_analyzer(basedir, args['lang'])
    # fla.setup()
    # fla.process(email_sample)


if __name__ == "__main__":
    main()
