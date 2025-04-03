import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy.pool import QueuePool
from cachetools import TTLCache, LRUCache
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import asyncio
import statistics
from functools import wraps

def handle_exceptions(func):
    """Decorator for handling exceptions in session manager methods."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SQLAlchemyError as e:
            logging.error(f"Database error in {func.__name__}: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error in {func.__name__}: {str(e)}")
            return None
    return wrapper

@dataclass
class ChatMessage:
    """Represents a single message in a chat conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SessionState:
    """Represents the state of a chat session with enhanced metadata."""
    session_id: str
    agent_id: str
    user_id: str = "anonymous"
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    messages: List[ChatMessage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class SessionManager:
    """Manages chat sessions with enhanced features, caching, and analytics."""
    
    def __init__(self, db_session_factory, cache_ttl: int = 3600, max_cache_size: int = 1000):
        """Initialize session manager with configurable caching.
        
        Args:
            db_session_factory: Factory function for database sessions
            cache_ttl: Time-to-live for active sessions cache in seconds
            max_cache_size: Maximum number of items in caches
        """
        self.db_session_factory = db_session_factory
        self.active_sessions = TTLCache(maxsize=max_cache_size, ttl=cache_ttl)
        self.session_cache = LRUCache(maxsize=max_cache_size)
        self.session_metrics = defaultdict(lambda: {
            'message_count': 0,
            'response_times': [],
            'user_ratings': [],
            'last_active': None,
            'errors': []
        })
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        self.loop = asyncio.new_event_loop()
        logging.info("Session manager initialized with enhanced features")

    @handle_exceptions
    def track_metrics(self, session_id: str, response_time: Optional[float] = None,
                     user_rating: Optional[int] = None, error: Optional[str] = None) -> None:
        """Track comprehensive session metrics including errors."""
        metrics = self.session_metrics[session_id]
        metrics['message_count'] += 1
        metrics['last_active'] = datetime.now()
        
        if response_time is not None:
            metrics['response_times'].append(response_time)
        if user_rating is not None:
            metrics['user_ratings'].append(user_rating)
        if error is not None:
            metrics['errors'].append({'timestamp': datetime.now(), 'error': error})

    @handle_exceptions
    def get_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """Get detailed analytics for a specific session."""
        metrics = self.session_metrics[session_id]
        response_times = metrics['response_times']
        user_ratings = metrics['user_ratings']
        
        return {
            'message_count': metrics['message_count'],
            'avg_response_time': statistics.mean(response_times) if response_times else None,
            'avg_user_rating': statistics.mean(user_ratings) if user_ratings else None,
            'last_active': metrics['last_active'],
            'error_count': len(metrics['errors']),
            'last_error': metrics['errors'][-1] if metrics['errors'] else None
        }

    @handle_exceptions
    def get_global_analytics(self) -> Dict[str, Any]:
        """Get comprehensive global analytics."""
        total_messages = sum(m['message_count'] for m in self.session_metrics.values())
        all_ratings = [r for m in self.session_metrics.values() for r in m['user_ratings']]
        all_times = [t for m in self.session_metrics.values() for t in m['response_times']]
        total_errors = sum(len(m['errors']) for m in self.session_metrics.values())
        
        return {
            'total_sessions': len(self.session_metrics),
            'total_messages': total_messages,
            'avg_response_time': statistics.mean(all_times) if all_times else None,
            'avg_user_rating': statistics.mean(all_ratings) if all_ratings else None,
            'total_errors': total_errors,
            'active_sessions': len(self.active_sessions)
        }

    @handle_exceptions
    def create_session(self, session_id: str, agent_id: str, user_id: str = "anonymous",
                      metadata: Optional[Dict[str, Any]] = None) -> Optional[SessionState]:
        """Create a new session with metadata support."""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]

        session = SessionState(
            session_id=session_id,
            agent_id=agent_id,
            user_id=user_id,
            metadata=metadata or {}
        )

        with self.db_session_factory() as db:
            from models import SessionInfo
            session_info = SessionInfo(
                session_id=session_id,
                user_id=user_id,
                agent_id=agent_id,
                created_at=session.created_at,
                last_active=session.last_active,
                metadata=session.metadata
            )
            db.add(session_info)
            db.commit()

            self.active_sessions[session_id] = session
            self.session_cache[session_id] = session
            logging.info(f"Created new session {session_id} for agent {agent_id} and user {user_id}")
            return session

    @handle_exceptions
    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get a session with optimized caching."""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.last_active = datetime.now()
            return session

        if session_id in self.session_cache:
            session = self.session_cache[session_id]
            self.active_sessions[session_id] = session
            session.last_active = datetime.now()
            return session

        return self._load_session_from_db(session_id)

    @handle_exceptions
    def _load_session_from_db(self, session_id: str) -> Optional[SessionState]:
        """Load a session from database with optimized queries."""
        if not session_id:
            return None

        with self.db_session_factory() as db:
            from models import ChatSession, SessionInfo

            session_info = db.query(SessionInfo).filter_by(session_id=session_id).first()
            if not session_info:
                return None

            session_state = SessionState(
                session_id=session_id,
                agent_id=session_info.agent_id,
                user_id=session_info.user_id,
                created_at=session_info.created_at,
                last_active=session_info.last_active,
                metadata=session_info.metadata or {}
            )

            messages = db.query(ChatSession).filter_by(session_id=session_id).order_by(ChatSession.timestamp).all()
            session_state.messages.extend([
                ChatMessage(
                    role=msg.role,
                    content=msg.message,
                    timestamp=msg.timestamp,
                    metadata=msg.metadata or {}
                ) for msg in messages
            ])

            self.active_sessions[session_id] = session_state
            self.session_cache[session_id] = session_state
            return session_state

    @handle_exceptions
    def get_session_history(self, session_id: str, limit: int = 0,
                          start_time: Optional[datetime] = None) -> List[ChatMessage]:
        """Get session history with time-based filtering."""
        session = self.get_session(session_id)
        if not session:
            return []

        messages = session.messages
        if start_time:
            messages = [m for m in messages if m.timestamp >= start_time]
        if limit > 0:
            messages = messages[-limit:]

        return messages

    @handle_exceptions
    def get_user_sessions(self, user_id: str,
                         active_only: bool = False) -> List[SessionState]:
        """Get user sessions with filtering options."""
        sessions = [s for s in self.active_sessions.values() if s.user_id == user_id]
        
        if not active_only:
            with self.db_session_factory() as db:
                from models import SessionInfo
                db_sessions = db.query(SessionInfo).filter_by(user_id=user_id).all()
                for db_session in db_sessions:
                    if db_session.session_id not in self.active_sessions:
                        session = self._load_session_from_db(db_session.session_id)
                        if session:
                            sessions.append(session)

        return sessions

    @handle_exceptions
    def add_message(self, session_id: str, role: str, content: str,
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a message with metadata support."""
        session = self.get_session(session_id)
        if not session:
            return False

        message = ChatMessage(role=role, content=content, metadata=metadata or {})
        session.messages.append(message)
        session.last_active = datetime.now()

        with self.db_session_factory() as db:
            from models import ChatSession
            db_message = ChatSession(
                session_id=session_id,
                message=content,
                role=role,
                metadata=message.metadata
            )
            db.add(db_message)
            db.commit()
            return True

    @handle_exceptions
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its associated data."""
        self.active_sessions.pop(session_id, None)
        self.session_cache.pop(session_id, None)
        self.session_metrics.pop(session_id, None)

        with self.db_session_factory() as db:
            from models import ChatSession, SessionInfo
            db.query(ChatSession).filter_by(session_id=session_id).delete()
            result = db.query(SessionInfo).filter_by(session_id=session_id).delete()
            db.commit()
            return result > 0

    def cleanup_inactive_sessions(self, max_age: timedelta = timedelta(days=7)) -> int:
        """Clean up inactive sessions older than specified age."""
        cutoff_time = datetime.now() - max_age
        deleted_count = 0

        with self.db_session_factory() as db:
            from models import SessionInfo
            old_sessions = db.query(SessionInfo).filter(SessionInfo.last_active < cutoff_time).all()
            
            for session in old_sessions:
                if self.delete_session(session.session_id):
                    deleted_count += 1

        return deleted_count

_session_manager = None

def init_session_manager(db_session_factory) -> SessionManager:
    """Initialize the session manager singleton."""
    global _session_manager
    _session_manager = SessionManager(db_session_factory)
    return _session_manager

def get_session_manager() -> SessionManager:
    """Get the session manager singleton instance."""
    if _session_manager is None:
        raise RuntimeError("Session manager not initialized. Call init_session_manager first.")
    return _session_manager