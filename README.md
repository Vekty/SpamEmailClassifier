The goal of this project was to build an e-mail spam classifier with a good enough recall for classification task practice and also to practice feature engineering when data isn't nicely available in a CSV format.
The dataset used for this project was taken from https://spamassassin.apache.org/old/publiccorpus/ (the easy dataset).
I used the Logistic Regression model ( scikit-learn implementation) and after I realised it produced excellent results I didn't bother testing any other model (with K-fold cross-validation) since it went beyond the scope of what I wanted to do (and also because at the moment of making this it was the only classification algorithm I fully understood).

The dataset contains 2500 ham emails and 500 spam emails and it was split 80-20 (80% training_set 20% test_set). 
<img width="1920" height="1080" alt="Screenshot_20250820_203617" src="https://github.com/user-attachments/assets/a559d63f-133b-431a-a88d-840693429713" />
