#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 miguelHeCa <jose.miguel.heca@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Test for kwargs
"""


import pprint

from datetime import date


def get_dict(**kwargs):
    D = {}
    if 'To' in kwargs:
        D['To'] = kwargs.get('To')
    if 'From' in kwargs:
        D['From'] = kwargs.get('From')
    if 'Date' in kwargs:
        D['Date'] = kwargs.get('Date')

    try: d_list
    except NameError: d_list = None

    if d_list is None:
        print('Function works correctly')

    return D

if __name__ == '__main__':
    pp = pprint.PrettyPrinter(indent=4)

    eg = get_dict(To='sender', From='recipient', Date=date.today())

    pp.pprint(eg)

