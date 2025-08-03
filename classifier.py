from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
import os
from email import parser
from email import policy

training_data_path='/home/victor/Documents/SpamClassifier/training_data'
test_data_path='/home/victor/Documents/SpamClassifier/test_data'
training_spam_path=os.path.join(training_data_path,'spam')
training_ham_path=os.path.join(training_data_path,'ham')

def read_email(type_:str='spam',file_name:str=None):
    path=training_spam_path
    if type_=='ham':
        path=training_ham_path
    with open(os.path.join(path,file_name),'rb') as f:
        return parser.BytesParser(policy=policy.default).parse(f)

def main():

  ham_names = [name for name in sorted(os.listdir(training_ham_path))]   #list of file names
  spam_names = [name for name in sorted(os.listdir(training_spam_path))]

  spam_emails=[read_email('spam',name) for name in spam_names]  #list of email instances
  ham_emails=[read_email('ham',name) for name in ham_names]


if __name__=='__main__':
    main()