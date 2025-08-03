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

def get_names(type_:str='spam')->list[str]:
    path = os.path.join(training_spam_path, 'spam_names.txt')
    if type=='ham':
        path= os.path.join(training_ham_path,'ham_names.txt')
    with open(path, 'r') as f:
        text=f.read().split(' ')
    return text

class MailPipeline:
    def __init__(self):
        self.__spam_names=get_names('spam')
        self.__ham_names=get_names('ham')
        self.table=pd.DataFrame()
    def read_email(self):
        pars=parser.BytesParser(policy=policy.default)
        with open(os.path.join(training_spam_path,self.__spam_names[0]),'rb') as f:
            content=pars.parse(f)

def main():
   mp=MailPipeline()
   mp.read_email()
if __name__=='__main__':
    main()