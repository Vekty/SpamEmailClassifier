import numpy as np
import os
import re
import copy
from bs4 import BeautifulSoup
from email import parser
from email import policy
from email.message import EmailMessage
from sklearn.model_selection import train_test_split
from collections import Counter
from urlextract import URLExtract
from sklearn.base import BaseEstimator,TransformerMixin
from scipy.sparse import csr_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

training_data_path='/home/victor/Documents/SpamClassifier/training_data'
training_spam_path=os.path.join(training_data_path,'spam')
training_ham_path=os.path.join(training_data_path,'ham')

def read_email(type_:str='spam',file_name:str=None)->EmailMessage:
    path=training_spam_path
    if type_=='ham':
        path=training_ham_path
    with open(os.path.join(path,file_name),'rb') as f:
        return parser.BytesParser(policy=policy.default).parse(f)

def html_to_plain_text(html):
    html=BeautifulSoup(html,'html.parser')
    return html.get_text()

def email_to_text(email):
    html = None
    for part in email.walk():
        ctype = part.get_content_type()
        if not ctype in ("text/plain", "text/html"):
            continue
        try:
            content = part.get_content()
        except:
            content = str(part.get_payload())
        if ctype == "text/plain":
            return content
        else:
            html = content
    if html:
        return html_to_plain_text(html)

class EmailToWordCounter(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.extractor=URLExtract()
        self.count=[]
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        for x in X:
                content=email_to_text(x)
                if content is not None:
                    if self.extractor.has_urls(content):    #any urls will be transformed to the string 'URL'
                        urls = set(self.extractor.find_urls(content))
                        for url in urls:
                            content=content.replace(url,'URL')
                    content=re.sub(r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', 'NUMBER', content) #any numbers will be transformed to the string 'NUMBER'
                    content=re.sub('[\W_]+', ' ', content, flags=re.M) #remove punctuation
                    self.count.append(Counter(content.split()))
        return np.array(self.count)

class AttributeAdder(BaseEstimator,TransformerMixin):
        def __init__(self):
            self.word_list=[]
            self.vocabulary_={}
        def fit(self,X,y=None):
            for x in X:
                self.word_list = self.word_list + list(x.keys())
            self.word_list = sorted(set(self.word_list))
            self.vocabulary_= {word: index + 1 for index, word in enumerate(self.word_list)}
            return self
        def transform(self,X):
            data = []
            rows = []
            cols = []
            for row, x in enumerate(X):
                for word, count in x.items():
                    rows.append(row)
                    data.append(count)
                    cols.append(self.vocabulary_.get(word))
            return csr_matrix((data, (rows, cols)), shape=(len(X), len(self.word_list) + 1))
def main():
  ham_names = [name for name in sorted(os.listdir(training_ham_path))]   #list of file names
  spam_names = [name for name in sorted(os.listdir(training_spam_path))]

  spam_emails=[read_email('spam',name) for name in spam_names]  #list of email instances
  ham_emails=[read_email('ham',name) for name in ham_names]

  X=np.array(ham_emails+spam_emails,dtype=object)
  y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

  X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
  X_train_copy=copy.deepcopy(X_train)

  mail_pipeline=Pipeline([('EmailTransformer',EmailToWordCounter()),('AttributeAdder',AttributeAdder())])
  X_train_copy=mail_pipeline.fit_transform(X_train_copy)

  model=LogisticRegression()
  model.fit(X_train_copy,y_train)
  predictions=model.predict(X_train_copy)

  print(predictions)
if __name__=='__main__':
    main()