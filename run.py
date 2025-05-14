import uvicorn
import logging

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Run the FastAPI application
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=8000, 
        log_level="info",
        reload=False
    )