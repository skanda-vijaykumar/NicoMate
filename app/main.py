from app.config import STATIC_DIR
from app.db.database import initialize_database, load_session_mapping
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Global variables
session_mapping = {}
app_ready = False

# Import routes after app creation to avoid circular imports
from app.api.routes import router as api_router  # noqa: E402

# Include API routes
app.include_router(api_router)


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    global session_mapping, app_ready

    try:
        # Set environment variable to indicate we're running in Docker
        os.environ["RUNNING_IN_DOCKER"] = "True"

        # Initialize database
        logging.info("Initializing database...")
        initialize_database()

        # Load session mapping
        session_mapping = load_session_mapping()
        logging.info(f"Loaded {len(session_mapping)} existing sessions")

        # Initialize data and models
        from app.api.dependencies import initialize_data_and_models

        await initialize_data_and_models()

        # Mark app as ready
        app_ready = True
        logging.info("Application startup complete - ready to handle requests")

    except Exception as e:
        logging.error(f"Error during startup: {str(e)}")
        # Don't re-raise the exception to allow the app to start even with errors
        logging.warning(
            "Application started with errors. Some functionality may be limited."
        )
