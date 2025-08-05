from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
import os
import re
from bs4 import BeautifulSoup
from email import parser
from email import policy
from email.message import EmailMessage
from sklearn.model_selection import train_test_split
from collections import Counter

training_data_path='/home/victor/Documents/SpamClassifier/training_data'
training_spam_path=os.path.join(training_data_path,'spam')
training_ham_path=os.path.join(training_data_path,'ham')

def read_email(type_:str='spam',file_name:str=None)->EmailMessage:
    path=training_spam_path
    if type_=='ham':
        path=training_ham_path
    with open(os.path.join(path,file_name),'rb') as f:
        return parser.BytesParser(policy=policy.default).parse(f)

def html_to_string(mail):
    content = BeautifulSoup(mail.get_content(), 'html.parser')
    return content.get_text().strip()

class WordCounter:
    def __init__(self):
        self.counter=Counter()
    def clean_and_count(self,X):
        pass

def main():
  ham_names = [name for name in sorted(os.listdir(training_ham_path))]   #list of file names
  spam_names = [name for name in sorted(os.listdir(training_spam_path))]

  spam_emails=[read_email('spam',name) for name in spam_names]  #list of email instances
  ham_emails=[read_email('ham',name) for name in ham_names]

  X=np.array(ham_emails+spam_emails,dtype=object)
  y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

  X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

if __name__=='__main__':
    main()