import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

Base = declarative_base()

class ChatSession(Base):
    """Stores individual chat messages within a session."""
    __tablename__ = 'chat_sessions'
    id = Column(Integer, primary_key=True)
    session_id = Column(String(50), nullable=False, index=True)
    message = Column(Text, nullable=False)
    role = Column(String(10), nullable=False)  # 'user' or 'assistant'
    timestamp = Column(DateTime, default=datetime.now)
    
    # Foreign key relationship to SessionInfo
    session_info_id = Column(Integer, ForeignKey('session_info.id'), nullable=True)
    session_info = relationship("SessionInfo", back_populates="messages")

class SessionInfo(Base):
    """Stores metadata about chat sessions."""
    __tablename__ = 'session_info'
    id = Column(Integer, primary_key=True)
    session_id = Column(String(50), nullable=False, unique=True, index=True)
    user_id = Column(String(50), nullable=False, default='anonymous', index=True)
    agent_id = Column(String(50), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.now)
    last_active = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    title = Column(String(255), nullable=True)  # Optional title for the conversation
    
    # Relationship to messages
    messages = relationship("ChatSession", back_populates="session_info", cascade="all, delete-orphan")

class AgentProfile(Base):
    """Stores agent personality profiles."""
    __tablename__ = 'agent_profiles'
    id = Column(Integer, primary_key=True)
    agent_id = Column(String(50), nullable=False, unique=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    traits = Column(JSON, nullable=False)  # Stored as JSON array
    avatar = Column(String(50), nullable=True)  # Emoji or path to avatar image
    behavior_params = Column(JSON, nullable=False)  # Stored as JSON object
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

# Database setup function
def init_db(db_url='sqlite:///chat_history.db'):
    """Initialize the database with all tables.
    
    Args:
        db_url: Database connection URL.
        
    Returns:
        SQLAlchemy session factory.
    """
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)