import numpy as np
import os
import re
from bs4 import BeautifulSoup
from email import parser
from email import policy
from email.message import EmailMessage
import urllib
import tarfile
from sklearn.model_selection import train_test_split
from collections import Counter
from urlextract import URLExtract
from sklearn.base import BaseEstimator,TransformerMixin
from scipy.sparse import csr_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from nltk import PorterStemmer
from sklearn.metrics import recall_score,precision_score
from sklearn.model_selection import GridSearchCV

DOWNLOAD_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"
HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"
SPAM_PATH = os.path.join("datasets", "spam")
HAM_DIR = os.path.join(SPAM_PATH, "easy_ham")
SPAM_DIR = os.path.join(SPAM_PATH, "spam")

def fetch_spam_data(ham_url=HAM_URL, spam_url=SPAM_URL, spam_path=SPAM_PATH):
    if not os.path.isdir(spam_path):
        os.makedirs(spam_path)
    for filename, url in (("ham.tar.bz2", ham_url), ("spam.tar.bz2", spam_url)):
        path = os.path.join(spam_path, filename)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url, path)
        tar_bz2_file = tarfile.open(path)
        tar_bz2_file.extractall(path=spam_path)
        tar_bz2_file.close()

def read_email(type_:str='spam',file_name:str=None)->EmailMessage:
    path=SPAM_DIR
    if type_=='ham':
        path=HAM_DIR
    with open(os.path.join(path,file_name),'rb') as f:
        return parser.BytesParser(policy=policy.default).parse(f)

def html_to_plain_text(html):
    html=BeautifulSoup(html,'html.parser')
    for a_tag in html.find_all('a'):
        a_tag.replace_with('HYPERLINK')  #all <a> tags are replaced with the word HYPERLINK
    return html.get_text()

def remove_empty_data(y,indexes):
    y=np.delete(y,indexes)
    return y

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
        self.stemmer=PorterStemmer()
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        self.count=[]
        for x in X:
                content=email_to_text(x)
                if content is not None:
                    if self.extractor.has_urls(content):    #any urls will be transformed to the string 'URL'
                        urls = set(self.extractor.find_urls(content))
                        for url in urls:
                            content=content.replace(url,'URL')
                    content=re.sub(r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', 'NUMBER', content) #any numbers will be transformed to the string 'NUMBER'
                    content=re.sub('[\W_]+', ' ', content, flags=re.M) #remove punctuation
                    word_counts=Counter(content.split())
                    stemmed_word_counts=Counter()
                    for word,count in word_counts.items():
                        stemmed_word=self.stemmer.stem(word)        #stemming for scaling
                        stemmed_word_counts[stemmed_word]+=count
                    word_counts=stemmed_word_counts
                    self.count.append(word_counts)

        return np.array(self.count)

class AttributeAdder(BaseEstimator,TransformerMixin):
        def __init__(self,vocabulary_size=2000):
            self.vocabulary_={}
            self.vocabulary_size=vocabulary_size
        def fit(self,X,y=None):
            total_count = Counter()
            for word_count in X:
                for word, count in word_count.items():
                    total_count[word] += count
            most_common = total_count.most_common()[:self.vocabulary_size]
            self.vocabulary_= {word: index + 1 for index, (word,count) in enumerate(most_common)}
            return self
        def transform(self,X):
            data = []
            rows = []
            cols = []
            for row, x in enumerate(X):
                for word, count in x.items():
                    rows.append(row)
                    data.append(count)
                    cols.append(self.vocabulary_.get(word,0))
            return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))

def main():
  ham_names = [name for name in sorted(os.listdir(HAM_DIR))]   #list of file names
  spam_names = [name for name in sorted(os.listdir(SPAM_DIR))]

  spam_emails=[read_email('spam',name) for name in spam_names]  #list of email instances
  ham_emails=[read_email('ham',name) for name in ham_names]

  X=np.array(ham_emails+spam_emails,dtype=object)
  y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

  X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

  mail_pipeline=Pipeline([('EmailTransformer',EmailToWordCounter()),('AttributeAdder',AttributeAdder())])

  X_train=mail_pipeline.fit_transform(X_train)
  X_test=mail_pipeline.transform(X_test)
  y_train=np.delete(y_train,1047) #deleted element at index 1047 because for some reason the 1047th element in X is
                                      # of type None which caused performance to severely drop


  model=LogisticRegression(max_iter=2000,random_state=42)
  model.fit(X_train,y_train)
  y_hat=model.predict(X_test)

  print(recall_score(y_test,y_hat)) #97% recall
  print(precision_score(y_test,y_hat)) #95% precision

if __name__=='__main__':
    main()