from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
import uuid
import logging
from datetime import datetime
import urllib.parse
import os

from app.main import session_mapping, app_ready
from app.db.models import get_session_history
from app.db.database import get_next_chat_id, get_db_connection
from app.api.dependencies import startup_complete, get_agent
from app.config import TEMPLATES_DIR
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

router = APIRouter()

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    from app.api.dependencies import (
        vector_index_markdown, 
        keyword_index_markdown, 
        vector_index_markdown_lab, 
        keyword_index_markdown_lab,
        agent_queue
    )
    
    return {
        "status": "ready" if app_ready else "initializing",
        "indices_loaded": all([
            vector_index_markdown is not None,
            keyword_index_markdown is not None,
            vector_index_markdown_lab is not None,
            keyword_index_markdown_lab is not None
        ]),
        "agents_available": agent_queue.qsize() if hasattr(agent_queue, 'qsize') else "unknown"
    }

@router.post("/chat")
async def chat(request: Request):
    """Chat endpoint for processing user messages."""
    # Wait for startup to complete before processing any requests
    if not app_ready:
        logging.info("Waiting for application startup to complete...")
        # Wait with timeout to avoid hanging indefinitely
        if not startup_complete.wait(timeout=60):
            raise HTTPException(
                status_code=503, 
                detail="Application is still initializing. Please try again in a few moments."
            )

    try:
        body = await request.json()
        session_id = body.get('sessionId')
        user_input = body['message']
        
        if session_id not in session_mapping:
            raise HTTPException(status_code=404, detail="Session not found")
            
        chat_id = session_mapping[session_id]['chat_id'] 
        
        session_history = get_session_history(session_id, chat_id)
        history_messages = session_history.get_messages()
        
        chat_history = []
        for msg in history_messages:
            if isinstance(msg, HumanMessage):
                chat_history.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                chat_history.append(f"AI: {msg.content}")
            elif isinstance(msg, SystemMessage):
                chat_history.append(f"System: {msg.content}")

        formatted_chat_history = "\n".join(chat_history)

        # Get tools and agent
        from app.api.dependencies import create_tools
        tools = create_tools()
        agent = await get_agent(tools)
        
        session_history.add_message(HumanMessage(content=user_input))

        # Determine routing (general vs selection)
        from app.services.routing import determine_route
        route = await determine_route(user_input, formatted_chat_history)
        
        if route == 'selection':
            # Handle connector selection route
            from app.services.connector_service import generate_connector_selection
            return StreamingResponse(
                generate_connector_selection(user_input, session_id, session_history),
                media_type="text/plain"
            )
        else:
            # Handle general route
            from app.services.chat_service import generate_response
            return StreamingResponse(
                generate_response(user_input, formatted_chat_history, agent, session_history),
                media_type="text/plain"
            )
            
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f'An error occurred while processing your request: {str(e)}'
        )

@router.post("/new_session")
async def new_session():
    """Create a new chat session."""
    if not app_ready:
        logging.info("Waiting for application startup to complete before creating new session...")
        if not startup_complete.wait(timeout=60):
            raise HTTPException(
                status_code=503, 
                detail="Application is still initializing. Please try again in a few moments."
            )
            
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT nextval('chat_id_seq')")
                chat_id = cur.fetchone()[0]
                session_id = str(uuid.uuid4())
                session_mapping[session_id] = {
                    'chat_id': chat_id, 
                    'timestamp': datetime.now().timestamp(),
                    'connector_selector': None
                }
                session_history = get_session_history(session_id, chat_id)
                session_history.add_message(AIMessage(
                    content="Hello! Welcome to Nicomatic customer support chat. How can I assist you today?"
                ))
                
                return {"sessionId": session_id, "chatId": chat_id}
    except Exception as e:
        logging.error(f"Error creating new session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_class=HTMLResponse)
async def index_page():
    """Serve the main HTML page."""
    try:
        index_path = os.path.join(TEMPLATES_DIR, "index.html")
        with open(index_path, encoding='utf-8') as f:
            content = f.read()
            return content
    except Exception as e:
        logging.error(f"Error reading index.html: {str(e)}")
        raise HTTPException(status_code=500, detail='Error reading index.html')

@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get chat session history."""
    try:
        session_info = session_mapping.get(session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="Session not found")
            
        chat_id = session_info['chat_id']
        session_history = get_session_history(session_id, chat_id)
        history_messages = session_history.get_messages()
        
        messages = []
        for message in history_messages:
            if isinstance(message, HumanMessage):
                messages.append({'sender': 'user', 'content': message.content})
            elif isinstance(message, AIMessage):
                messages.append({'sender': 'bot', 'content': message.content})
            elif isinstance(message, SystemMessage):
                messages.append({'sender': 'system', 'content': message.content})
                
        return JSONResponse(content={'messages': messages})
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in get_session: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail='An error occurred while retrieving the session.'
        )

@router.post("/suggestion")
async def suggestion(request: Request):
    """Handle system suggestions (for testing)."""
    try:
        body = await request.json()
        session_id = body.get('sessionId')
        user_suggestion = body.get('message')
        
        if not isinstance(user_suggestion, str):
            user_suggestion = str(user_suggestion)
            
        session_info = session_mapping.get(session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="Session not found")
            
        chat_id = session_info.get('chat_id')
        
        try:
            session_history = get_session_history(session_id, chat_id)
            session_history.add_message(SystemMessage(content=user_suggestion))
            logging.info(f"Suggestion stored for session {session_id}, chat {chat_id}")
            return JSONResponse(content={
                'detail': 'Suggestion submitted successfully.',
                'sessionId': session_id,
                'chatId': chat_id
            })
            
        except Exception as db_error:
            logging.error(f"Database error storing suggestion: {str(db_error)}")
            raise HTTPException(
                status_code=500,
                detail='Error storing suggestion in database'
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in suggestion endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail='An error occurred while submitting suggestion.'
        )

@router.get("/source_document/{file_path:path}")
async def get_source_document(file_path: str, page: int = 1):
    """Serve source documents."""
    try:
        # Decode the file path
        decoded_path = urllib.parse.unquote(file_path)

        # Check if the file exists
        if not os.path.exists(decoded_path):
            raise HTTPException(status_code=404, detail="Source document not found") 

        _, ext = os.path.splitext(decoded_path) 
        ext = ext.lower() 

        if ext == ".pdf":
            # Serve the PDF with a page hint
            response = FileResponse(decoded_path, media_type="application/pdf") 
            response.headers["Content-Disposition"] = (
                f"inline; filename={os.path.basename(decoded_path)}"
            )
            return response 
        else:
            try:
                with open(decoded_path, "r", encoding="utf-8") as file: 
                    content = file.read() 
            except UnicodeDecodeError:
                try:
                    with open(decoded_path, "r", encoding="latin-1") as file:
                        content = file.read()
                except Exception:
                    raise HTTPException(
                        status_code=500, 
                        detail="Error reading file content with standard encodings."
                    )

            filename = os.path.basename(decoded_path)

            # Serve the text content as HTML
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head> 
                <title>Source Document: {filename}</title> 
                <style> 
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 900px; margin: 0 auto; }} 
                    pre {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; }}
                    h1 {{ color: #333; border-bottom: 1px solid #eee; padding-bottom: 10px; }} 
                    .filepath {{ color: #666; font-size: 0.9em; margin-bottom: 20px; }} 
                </style> 
            </head> 
            <body> 
                <h1>Source Document: {filename}</h1> 
                <div class="filepath">Full path: {decoded_path}</div> 
                <pre>{content}</pre> 
            </body> 
            </html>
            """ 
            return HTMLResponse(content=html_content) 

    except FileNotFoundError: 
        raise HTTPException(status_code=404, detail="Source document not found")
    except HTTPException:
        raise
    except Exception as e: 
        logging.error(f"Error reading source document '{decoded_path}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reading source document: {str(e)}")