from pydantic import BaseModel, Field

class PredictionRequest (BaseModel):
    text: str = Field (min_length=1, description="Financial news or sentence", example="Apple reported record quarterly earnings.")



class PredictionResponse (BaseModel):
    sentiment: str
    confidence: float
    probabilities: dict[str, float]