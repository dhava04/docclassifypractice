import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

from Pytess import extractText 
from TextPreprocessing import *

app = Flask(__name__)
print(app)
path = r"C:\Users\dhava\Desktop\Document Classification\DemoDoc"

# Deserialize Vectorization Pickle file for Transformation
cv = pickle.load(open("CountVector.pkl","rb"))
tv = pickle.load(open("TfidfVector.pkl","rb"))

# Deserialize Model Pickle file for prediction
cv_lgr_model  = pickle.load(open("cv_lgr_model.pkl" ,"rb"))
tv_lgr_model  = pickle.load(open("tv_lgr_model.pkl" ,"rb"))
cv_svc_model  = pickle.load(open("cv_svc_model.pkl" ,"rb"))
tv_svc_model  = pickle.load(open("tv_svc_model.pkl" ,"rb"))
cv_sgd_model  = pickle.load(open("cv_sgd_model.pkl" ,"rb"))
tv_sgd_model  = pickle.load(open("tv_sgd_model.pkl" ,"rb"))
cv_mnb_model  = pickle.load(open("cv_mnb_model.pkl" ,"rb"))
tv_mnb_model  = pickle.load(open("tv_mnb_model.pkl" ,"rb"))

@app.route('/')
def home():
    return render_template('index.html')
    
@app.route("/predict", methods = ['GET', 'POST'])
def predict():
 
    if request.method == 'POST':
        
        # check if there is a file in the request
        if 'file' not in request.files:
            return render_template('upload.html', msg='No file selected')
        file = request.files['file']
        
        # Check for file uploaded and initiate document processing
        if file:
            
            # call  OCR function to extract the text from document as string
            extractedtext = extractText(os.path.join(path, file.filename))
            
            #Load extracted text from document to a Pandas DataFrame
            df = pd.DataFrame([[extractedtext]],columns=['Text'])
            
            # PreProcess Extracted annd remove Speical characters, Symbols, Unicode character, Tokenize, remove stop words, Lemmatize
            processedText = pre_process_corpus(df)
            
            #Apply Count Vectorization Transformer on Clean Preprocessed text
            CountVectorFeatures = cv.transform(processedText)
            
            #Apply Term Frequenct and Inverse Document Frequency Transformer
            TfidfVectorFeatures = tv.transform(processedText)
            

            Prediction = cv_svc_model.predict(CountVectorFeatures)
            if Prediction[0]==0:
                Transformed_prediction = "Misc"
            else:
                Transformed_prediction = "TaxForm"
            
            #CV_LGR_Prediction = cv_lgr_model.predict(CountVectorFeatures)
            #TV_LGR_Prediction = tv_lgr_model.predict(TfidfVectorFeatures)
            #CV_SVC_Prediction = cv_svc_model.predict(CountVectorFeatures)
            #TV_SVC_Prediction = tv_svc_model.predict(TfidfVectorFeatures)
            #CV_SGD_Prediction = cv_sgd_model.predict(CountVectorFeatures)
            #TV_SGD_Prediction = tv_sgd_model.predict(TfidfVectorFeatures)
            #CV_MNB_Prediction = cv_mnb_model.predict(CountVectorFeatures)
            #TV_MNB_Prediction = tv_mnb_model.predict(TfidfVectorFeatures)

            return render_template('result.html',Transformed_prediction=Transformed_prediction)
        return render_template('upload.html')
    elif request.method == 'GET':
        return render_template('upload.html')
        
if __name__ == '__main__':
    app.run(port=1234)
