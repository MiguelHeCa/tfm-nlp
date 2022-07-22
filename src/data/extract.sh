#! /usr/bin/env bash
#
# extract.sh
# Copyright (C) 2022 MiguelHeca <josemiguel@heca.tech>
#
# Distributed under terms of the MIT license.
#

DATA=raw
ENRON_DIR=maildir
ENRON_TAR=enron_mail_20150507.tar.gz

enron_download () {
    echo "Downloading emails..."
    wget https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz
    echo "Emails downloaded"
}

check_enron_dld () {
    if [ -f "$ENRON_TAR" ]
    then
        echo "Enron emails are downloaded"
    else
        echo "Enron emails are not downloaded"
        echo "Downloading"
        # enron_download
    fi
}

enron_extraction () {
    echo "Extracting Enron emails. This could take several minutes..."
    echo "Extracting emails"
    tar -xf $ENRON_TAR
    echo "Enron emails extracted."
}

check_data_dir () {
    if [ ! -d $DATA ]
    then
        echo "Directory $DATA does not exists."
        echo "Creating directory..."
        mkdir $DATA
        echo "Directory created"
    else
        echo "Directory $DATA exists."
    fi
}

check_data_dir
echo "Changin to directory"
cd raw

if [ -d $ENRON_DIR ]
then
    echo "Enron emails are already extracted."
else
    check_enron_dld
    enron_extraction
fi

echo "Enron emails are ready for analysis."
cd ..
