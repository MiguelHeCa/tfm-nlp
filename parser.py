import os
from email.parser import Parser
import email.utils
import time


def messageIDtoSubject(mail_dict, messageID):
    return mail_dict[messageID]["subject"].replace(" ", "")


def raw_parse(inputfile, email_list):
    with open(inputfile, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read()
    parsedEmail = Parser().parsestr(data)
    timestamp = time.mktime(email.utils.parsedate(parsedEmail["date"]))
    email_list.append((timestamp, parsedEmail))


def obtain_raw_threads(mail_dict, email_list):
    subject_dict = {}
    rootMailSet = set()
    auxiliarRootStructure = {}

    for mail in email_list:
        actualEmail = mail[1]
        true_subject = actualEmail["subject"]
        true_subject = true_subject.replace(" ", "")
        mail_dict[actualEmail["message-id"]] = actualEmail
        if "Re:" != true_subject[0:3]:
            auxiliarRootStructure[true_subject] = actualEmail["message-id"]
            if true_subject not in subject_dict:
                subject_dict[true_subject] = []
        else:
            new_subject = true_subject.replace("Re:", "")
            if new_subject in subject_dict:
                subject_dict[new_subject].append(actualEmail["message-id"])
                rootMailSet.add(auxiliarRootStructure[new_subject])
            subject_dict[true_subject] = []
    threads = {}
    for mail in rootMailSet:
        subject = messageIDtoSubject(mail_dict, mail)
        childThread = subject_dict[subject]
        threads[mail] = childThread
    return threads
