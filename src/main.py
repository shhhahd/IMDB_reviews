# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 18:40:28 2025

@author: lenovo
"""



from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware  # ✅ ضروري

app = FastAPI()

# ✅ تفعيل الـ CORS علشان HTML تقدر تبعت بيانات
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # مسموح من أي مصدر
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("../models/sentiment_model.pkl")

#data shape
class ReviewRequest(BaseModel):
    review: str
    
@app.post("/predict")  
def predict_sentiment(data: ReviewRequest):
    
    review_text = data.review
    
    prediction = model.predict([review_text])[0]
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    
    return {"sentiment": sentiment}