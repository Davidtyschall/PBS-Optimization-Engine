"""
PBS Optimizer API
=================
Main FastAPI application entry point.
"""
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import preferences, sequences, packets

# Create app
app = FastAPI(
    title="PBS Optimizer API",
    description="API for pilot scheduling preference parsing and sequence scoring",
    version="1.0.0",
)

# CORS middleware (allow frontend to call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://bidline-two.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(preferences.router, prefix="/api")
app.include_router(sequences.router, prefix="/api")
app.include_router(packets.router, prefix="/api")


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


# Root
@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "PBS Optimizer API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "parse_preferences": "POST /api/preferences/parse",
            "list_packets": "GET /api/packets",
            "list_sequences": "GET /api/sequences",
            "score_sequences": "POST /api/sequences/score",
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)