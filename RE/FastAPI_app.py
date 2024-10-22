from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Import the recommendation functions from previous steps
# from recommendation_system import hybrid_recommendation

app = FastAPI()

# Define request model
class RecommendationRequest(BaseModel):
    user_id: int
    property_id: int

# Define response model
class RecommendationResponse(BaseModel):
    property_id: int
    hybrid_score: float

# API route for getting property recommendations
@app.post("/recommend/", response_model=RecommendationResponse)
def get_recommendation(request: RecommendationRequest):
    # Use the hybrid recommendation function
    hybrid_score = hybrid_recommendation(request.user_id, request.property_id)
    
    return RecommendationResponse(property_id=request.property_id, hybrid_score=hybrid_score)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

