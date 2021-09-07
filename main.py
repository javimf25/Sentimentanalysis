import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB , GaussianNB , BernoulliNB , MultinomialNB
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

#Reding and preprocessin dataset
dataset=pd.read_csv("smile-annotations-final.csv" )
dataset.columns =['TweetId', 'Tweet', 'Sentiment']
print(dataset.head())
print(dataset.Sentiment.value_counts())
print(dataset.info())
check_for_nan = dataset.isnull().values.any()
print (check_for_nan)
#Count of words and tokenizer declaration
token = RegexpTokenizer(r'\w+')
ngramstested= [(1,1),(1,2),(2,2),(2,3),(3,3)]
print("Models tested with CountVectorizer")
print("\n")
for grams in ngramstested:
    print("Ngrams="+str(grams))
    CV=CountVectorizer(stop_words='english', ngram_range= grams, tokenizer=token.tokenize)
    counts= CV.fit_transform(dataset['Tweet'])
    #split between training and test set
    x_train, x_test, y_train, y_test= train_test_split(counts,dataset['Sentiment'],test_size=0.25,random_state=5)
    #Models tested declaration
    CMB=ComplementNB()
    BER=BernoulliNB()
    GAU=GaussianNB()
    MNB=MultinomialNB()
    CMB.fit(x_train,y_train)
    BER.fit(x_train,y_train)
    GAU.fit(x_train.todense(),y_train)
    MNB.fit(x_train,y_train)
    predictedCMB=CMB.predict(x_test)
    predictedBER=BER.predict(x_test)
    predictedGAU=GAU.predict(x_test.todense())
    predictedMNB=MNB.predict(x_test)
    acc_score_CMB=metrics.accuracy_score(predictedCMB,y_test)
    acc_score_BER=metrics.accuracy_score(predictedBER,y_test)
    acc_score_GAU=metrics.accuracy_score(predictedGAU,y_test)
    acc_score_NMB= metrics.accuracy_score(predictedMNB, y_test)
    print("Acurracy with ComplementNB model: "+str('{:04.2f}'.format(acc_score_CMB*100))+'%')
    print("Acurracy with BernoulliNB model: "+str('{:04.2f}'.format(acc_score_BER*100))+'%')
    print("Acurracy with GaussianNB model: " + str('{:04.2f}'.format(acc_score_GAU * 100)) + '%')
    print("Acurracy with MultinomialNB model: " + str('{:04.2f}'.format(acc_score_NMB * 100)) + '%')
print("\n")
print("Models tested with TF-IDF")
tfidf=TfidfVectorizer()
count2= tfidf.fit_transform(dataset['Tweet'])
#split between training and test set
x_train, x_test, y_train, y_test= train_test_split(count2,dataset['Sentiment'],test_size=0.3,random_state=5)
#Models tested declaration
CMB=ComplementNB()
BER=BernoulliNB()
GAU=GaussianNB()
MNB=MultinomialNB()
CMB.fit(x_train,y_train)
BER.fit(x_train,y_train)
GAU.fit(x_train.todense(),y_train)
MNB.fit(x_train,y_train)
#predictions for each model
predictedCMB=CMB.predict(x_test)
predictedBER=BER.predict(x_test)
predictedGAU=GAU.predict(x_test.todense())
predictedMNB=MNB.predict(x_test)
#accuracy score computation for each model
acc_score_CMB=metrics.accuracy_score(predictedCMB,y_test)
acc_score_BER=metrics.accuracy_score(predictedBER,y_test)
acc_score_GAU=metrics.accuracy_score(predictedGAU,y_test)
acc_score_NMB= metrics.accuracy_score(predictedMNB, y_test)
print("Acurracy with ComplementNB model: "+str('{:04.2f}'.format(acc_score_CMB*100))+'%')
print("Acurracy with BernoulliNB model: "+str('{:04.2f}'.format(acc_score_BER*100))+'%')
print("Acurracy with GaussianNB model: " + str('{:04.2f}'.format(acc_score_GAU * 100)) + '%')
print("Acurracy with MultinomialNB model: " + str('{:04.2f}'.format(acc_score_NMB * 100)) + '%')
print("\n")
print("Logistic Regression testing")
#Logistic regression testing
LogRgression=LogisticRegression()
LogRgression.fit(x_train,y_train)
predictedLG=LogRgression.predict(x_test)
acc_score_LG=metrics.accuracy_score(predictedLG, y_test)
print("Acurracy with LogisticRegression model: " + str('{:04.2f}'.format(acc_score_LG * 100)) + '%')
#knn model testing
k=[3,5,7,9,11,13,15]
print("\n")
print("Knn testing")
for ki in k:
    knnclassifier=KNeighborsClassifier(n_neighbors=ki,algorithm='brute')
    knnclassifier.fit(x_train,y_train)
    predictedKnn= knnclassifier.predict(x_test)
    acc_score_KNN= metrics.accuracy_score(predictedKnn, y_test)
    print("Acurracy with KNN model " + str(ki)+" neighbours:  " + str(acc_score_KNN * 100)+ '%')
