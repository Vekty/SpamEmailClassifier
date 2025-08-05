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
from urlextract import URLExtract

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

class EmailToWordCounter:
    def __init__(self):
        self.counter=Counter()
        self.extractor=URLExtract()
        self.count=[]
    def clean_and_count(self,X):
        for x in X:
            content=x.get_content().strip()
            if(x.get_content_type()=='text/html'):
                content=html_to_string(x)
            if self.extractor.has_urls(content):    #any urls will be transformed to the string 'URL'
                urls = set(self.extractor.find_urls(content))
                for url in urls:
                    content=content.replace(url,'URL')
            content=re.sub(r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', 'NUMBER', content) #any numbers will be transformed to the string 'NUMBER'
            content=re.sub('[\W_]+', ' ', content, flags=re.M) #remove punctuation
            self.count+=Counter(content.split())
        return self.count
def main():
  ham_names = [name for name in sorted(os.listdir(training_ham_path))]   #list of file names
  spam_names = [name for name in sorted(os.listdir(training_spam_path))]

  spam_emails=[read_email('spam',name) for name in spam_names]  #list of email instances
  ham_emails=[read_email('ham',name) for name in ham_names]

  X=np.array(ham_emails+spam_emails,dtype=object)
  y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

  X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
  X_train_copy=X_train.copy()
  emwc=EmailToWordCounter()
  print(emwc.clean_and_count(X_train_copy))
if __name__=='__main__':
    main()