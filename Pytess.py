import os  
import sys 
import requests
import pandas as pd

#  libraries required for pdfplumber
import pdfplumber 

#  libraries required for Pytesseract
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extractText(filePath):
    finalString = ""
    # convert PDF file passed to images
    pages = convert_from_path(filePath,500,size=(1654,2340),poppler_path=r'C:\Users\dhava\Desktop\Learning\Document Classification\poppler-21.03.0\Library\bin')

    
    pageCounter = 0
    #save each image to disk
    for page in pages:
        pageCounter = pageCounter + 1
        saveFileName = "page_"+str(pageCounter)+".jpg"
        page.save(saveFileName, 'JPEG')
        imageToString = str(((pytesseract.image_to_string(Image.open(saveFileName),lang='eng'))))
        finalString = finalString.join(imageToString)
    
    return finalString