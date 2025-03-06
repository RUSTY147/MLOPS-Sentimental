from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI()
analyzer = SentimentIntensityAnalyzer()

class TextInput(BaseModel):
    text: str

@app.post("/sentiment")
async def analyze_sentiment(data: TextInput):
    if not data.text:
        raise HTTPException(status_code=400, detail="No text provided")

    sentiment = analyzer.polarity_scores(data.text)
    return {"text": data.text, "sentiment": sentiment}
