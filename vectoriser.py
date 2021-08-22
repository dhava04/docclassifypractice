import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def CountVector(txt):

    # build Bag of Words on extracted text
    cv = CountVectorizer(binary=False, min_df=5, max_df=1.0, ngram_range=(1,2))
    cv_fit = cv.fit_transform(createfit(path))
    print(cv_fit.shape)
    cv_features = cv.transform(txt)
    return(cv_features)

def TfidfVector(txt):

    # build term frequency–inverse document frequency features on extracted text
    tv = TfidfVectorizer(use_idf=True, min_df=5, max_df=1.0, ngram_range=(1,2),
                         sublinear_tf=True)
    tv_fit = tv.fit_transform(createfit(path))
    # transform extracted text into Features
    tv_features = tv.transform(txt)
    return(tv_features)