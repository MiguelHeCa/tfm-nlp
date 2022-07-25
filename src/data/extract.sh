#! /usr/bin/env bash
#
# extract.sh
# Copyright (C) 2022 MiguelHeca <josemiguel@heca.tech>
#
# Distributed under terms of the MIT license.
#
DATA_DIR=data/raw
ENRON_DIR=$DATA_DIR/maildir
ENRON_TAR=$DATA_DIR/enron_mail_20150507.tar.gz

enron_download () {
    echo "Downloading emails..."
    wget https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz -P $DATA_DIR
    echo "Emails downloaded"
}

check_enron_dld () {
    if [ -f "$ENRON_TAR" ]
    then
        echo "Enron emails are downloaded"
    else
        echo "Enron emails are not downloaded"
        echo "Downloading"
        enron_download
    fi
}

enron_extraction () {
    echo "Extracting Enron emails. This could take several minutes..."
    echo "Extracting emails"
    tar -xf $ENRON_TAR -C $DATA_DIR
    echo "Enron emails extracted."
}

check_data_dir () {
    if [ ! -d $DATA_DIR ]
    then
        echo "Directory $DATA_DIR does not exists."
        echo "Configuring project structure..."
        bash src/data/project-structure.sh
        echo "Directory ready"
    else
        echo "Directory $DATA_DIR exists."
    fi
}

check_data_dir

if [ -d $ENRON_DIR ]
then
    echo "Enron emails are already extracted."
else
    check_enron_dld
    enron_extraction
fi

echo "Enron emails are ready for analysis."
