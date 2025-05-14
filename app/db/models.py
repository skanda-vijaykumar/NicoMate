from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

class CustomPostgresChatMessageHistory(BaseChatMessageHistory):
    """Custom implementation of chat message history using PostgreSQL."""
    
    def __init__(self, table_name: str, session_id: str, chat_id: int, sync_connection):
        """
        Initialize the chat message history.
        
        Args:
            table_name (str): The name of the table to store messages.
            session_id (str): The ID of the current session.
            chat_id (int): The ID of the current chat.
            sync_connection: The database connection.
        """
        self.table_name = table_name
        self.session_id = session_id
        self.chat_id = chat_id
        self.sync_connection = sync_connection
    
    def add_message(self, message):
        """
        Add a message to the history.
        
        Args:
            message: The message to add (HumanMessage, AIMessage, or SystemMessage).
        """
        with self.sync_connection.cursor() as cursor:
            message_type = (
                "human" if isinstance(message, HumanMessage) 
                else "ai" if isinstance(message, AIMessage) 
                else "system"
            )
            cursor.execute(
                f"""
                INSERT INTO {self.table_name} 
                (session_id, chat_id, message, type) 
                VALUES (%s, %s, %s, %s)
                """, 
                (self.session_id, self.chat_id, str(message.content), message_type)
            )
            self.sync_connection.commit()

    def get_messages(self):
        """
        Get all messages from the history.
        
        Returns:
            list: A list of messages.
        """
        messages = []
        with self.sync_connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT type, message FROM {self.table_name}
                WHERE session_id = %s 
                ORDER BY created_at
                """,
                (self.session_id,)
            )
            # Convert database records to message objects
            for msg_type, content in cursor.fetchall():
                if msg_type == 'human':
                    messages.append(HumanMessage(content=content))
                elif msg_type == 'ai':
                    messages.append(AIMessage(content=content))
                elif msg_type == 'system':
                    messages.append(SystemMessage(content=content))
        return messages
    
    def clear(self):
        """Clear all messages from the history."""
        with self.sync_connection.cursor() as cursor:
            cursor.execute(
                f"""
                DELETE FROM {self.table_name} 
                WHERE session_id = %s
                """, 
                (self.session_id,)
            )
            self.sync_connection.commit()

def get_session_history(session_id: str, chat_id: int) -> BaseChatMessageHistory:
    """
    Get the chat message history for a session.
    
    Args:
        session_id (str): The ID of the session.
        chat_id (int): The ID of the chat.
        
    Returns:
        BaseChatMessageHistory: The chat message history.
    """
    from app.db.database import get_db_connection
    sync_connection = get_db_connection()
    return CustomPostgresChatMessageHistory("chat_history", session_id, chat_id, sync_connection)