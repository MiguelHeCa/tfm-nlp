#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 MiguelHeCa <josemiguel@heca.tech>
#
# Distributed under terms of the MIT license.

"""
Test for console interactions
"""

import argparse

from pathlib import Path

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument('-p', '--path', type=str, help='File path')

    args = ap.parse_args()

    file_name = Path(args.path)

    print(file_name)


main()
