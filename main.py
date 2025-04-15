from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from recommender import MutualFundRecommender  # Import your MutualFundRecommender class

app = FastAPI(title="Mutual Fund Recommendation API")

# Set up CORS middleware to allow all origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the recommender once so it can be reused.
recommender = MutualFundRecommender("comprehensive_mutual_funds_data.csv")

@app.get("/health")
def read_health():
    """Health check endpoint."""
    return {"status": "ok"}

@app.get("/recommender")
def get_recommendations(
    risk_tolerance: int = Query(3, description="Numeric risk tolerance level"),
    investment_mode: str = Query("SIP", description="Investment mode: SIP or Lumpsum"),
    available_amount: float = Query(1500.0, description="Amount available for investment"),
    horizon: int = Query(3, description="Investment horizon in years: 1, 3, or 5"),
    use_ml: bool = Query(True, description="Enable ML-based recommendation enhancements")
):
    """
    Retrieve top mutual fund recommendations based on the provided query parameters.
    """
    result = recommender.recommend_funds(
        risk_tolerance, investment_mode, available_amount, horizon, use_ml
    )
    
    if result.empty:
        raise HTTPException(status_code=404, detail="No funds match the provided criteria.")

    # Return the top 10 recommendations as a list of dictionaries.
    return {"recommendations": result.head(10).to_dict(orient="records")}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
