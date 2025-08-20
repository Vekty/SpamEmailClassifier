# Email Spam Classifier  

A machine learning project to classify emails as **spam** or **ham (legitimate)** using **Logistic Regression** and custom **feature engineering**.  

The main goal was to achieve **high recall**, since it is safer to incorrectly mark a legitimate email as spam than to miss a spam email (users can always check the spam folder).  

---

## Dataset  

- **Source**: [SpamAssassin Public Corpus (Easy Dataset)](https://spamassassin.apache.org/old/publiccorpus/)  
- **Total**: 3,000 emails  
  - 2,500 ham  
  - 500 spam  
- **Split**: 80% training / 20% testing  

---

## Approach  

### Model  
- **Logistic Regression** (scikit-learn)  
- Chosen for **simplicity, interpretability, and strong performance**  

### Feature Engineering  

**Bag-of-Words Representation**  
- Counted word occurrences per email  
- Replaced all hyperlinks with `"HYPERLINK"`  
- Replaced all numbers with `"NUMBER"`  
- Removed punctuation  
- Applied **stemming** to reduce words to their root form  

**Custom Features**  
- Presence of **HTML** in the email  
- **Number of parts** (multi-part emails often spam)  
- **Count of uppercase characters** (spammers use CAPITALIZATION to grab attention)  

**Pipelines**  
- Implemented custom transformers (`EmailToWordCounter`, `AttributeAdder`)  
- Used scikit-learn’s `Pipeline` to automate preprocessing + modeling  

---

## Results  

**Initial Model (Bag-of-Words + Stemming)**  
- Recall: **97%**  
- Precision: **94%**  

<img width="1087" alt="Initial Results" src="https://github.com/user-attachments/assets/5f8cb08a-1922-4524-bef7-879433df60aa" />  

**Improved Model (With Extra Features)**  
- Recall: **98%**  
- Precision: **98%**  

<img width="1227" alt="Improved Results" src="https://github.com/user-attachments/assets/0515e997-8b64-41ef-a6bd-6f4ce63544b7" />  

---

## Conclusions  

- Logistic Regression + feature engineering achieved **near-perfect classification**.  
- Custom features (HTML presence, uppercase count, multi-part detection) gave a measurable boost.  

**Future improvements could include:**  
- Analyzing sender names (spammers often use odd formats)  
- Exploring **n-grams** for context  
- Comparing with **Naive Bayes, SVM, or deep learning models**  
