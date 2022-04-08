#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 miguelHeCa <jose.miguel.heca@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Transforming data
"""

# Built-in modules
import email
import pprint

# Third-party modules

# Importing methods from modules
from pathlib import Path


def parse_email(input_file):
    """
    Obtain mail
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        message = f.read()

    return email


def get_email_id(input_file):
    """
    Set ID from emails parent directories
    """
    email_dir = '/'.join(str(input_file).split('/')[-3:])

    return email_dir


def main():
    emails_path = Path(Path.cwd().parent, 'maildir')
    emails_list = emails_path.rglob('*.')

    for email in emails_list:
        # print({email: parse_email(email)})
        pprint.pprint({get_email_id(email): parse_email(email)})


if __name__ == '__main__':
    main()
