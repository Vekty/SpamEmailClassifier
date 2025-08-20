The goal of this project was to build an e-mail spam classifier with a good enough recall for classification task practice and also to practice feature engineering when data isn't nicely available in a CSV format (in this case when it's in a text format).
I thought it be better to aim for a high recall as I believe it's better to mark some ham emails as spam since when people don't find an email they are used to checking the spam folder anyway. Also, accuracy didn't seem like the best performance metric since I did have a lot more ham emails than spam emails.
The dataset used for this project was taken from https://spamassassin.apache.org/old/publiccorpus/ (the easy dataset).
I used the Logistic Regression model ( scikit-learn implementation) and after I realised it produced excellent results I didn't bother testing any other model (with K-fold cross-validation) since it went beyond the scope of what I wanted to do (and also because at the moment of making this it was the only classification algorithm I fully understood).
The dataset contains 2500 ham emails and 500 spam emails and it was split 80-20 (80% training set,20% test set). 
<img width="1327" height="549" alt="PHOTO1" src="https://github.com/user-attachments/assets/00e43b62-aa01-4a51-becf-ab943c2078d1" />
As for the features I started making a 'bag of words' (basically for each email i counted the number of appeareances of each word and made them into a vector) and also I thought it was a good idea for any hyperlink to be marked just as a single word, 'HYPERLINK' and same for each url and number since it didn't really matter what it was, it just mattered if they existed and how many they were, also I removed any punctuation.).
I used stemming to normalize the text features (by reducing words to a common root), which simplifies the feature space for logistic regression.
I also initially used scikit-learn's Pipeline class to automatize the process of transforming the emails(EmailToWordCounter) and making the actual feature vectors(AttributeAdder).
The initial results were very promising, 97% recall and 94% precision!:
<img width="1087" height="457" alt="Screenshot_20250820_205259" src="https://github.com/user-attachments/assets/5f8cb08a-1922-4524-bef7-879433df60aa" />
But still I felt like could improve it, so I added more features.
I noticed that emails that contained HTML tended to be more spam than ham so i thought that was an important feature to have along with the number of parts in case it was a multi-part email since again I noticed that was pretty important. Also, spam emails tend to have more a lot of uppercase characters to catch your attention, so I added a count of that as well.
<img width="1227" height="503" alt="Screenshot_20250820_203037" src="https://github.com/user-attachments/assets/0515e997-8b64-41ef-a6bd-6f4ce63544b7" />
The results were even better! I reached 98% recall and precision. 
Of course, there were still things I could improve, such as more carefully analysing the sender's names( they tend to be weird) but for now I feel that I was satisfied with the results.




  
  
