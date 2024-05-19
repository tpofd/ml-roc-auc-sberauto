import dill
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from dict import categorical_features, social_media_utms, organic_utms_medium

app = FastAPI()
model = joblib.load('models/catboost_model.pkl')
encoder = joblib.load('models/encoder.pkl')

class Form(BaseModel):
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str


class Prediction(BaseModel):
    predict: int


@app.get('/status')
def status():
    return "App is started"


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])

    df['from_social'] = df['utm_source'].isin(social_media_utms)
    df['from_social'] = df['from_social'].astype(int)

    df['is_organic'] = df['utm_medium'].isin(organic_utms_medium)
    df['is_organic'] = df['is_organic'].astype(int)

    df_encode = df[categorical_features].copy()

    for feature in categorical_features:
        feature_encode = []
        with open(f'encode/{feature}_encoder.pickle', 'rb') as f:
            d = pickle.load(f)
            feature_encode.append(d)

        df_encode[feature] = feature_encode[0].get(df[feature][0])

    y = model.predict(df_encode)
    return {'predict': int(y[0])}