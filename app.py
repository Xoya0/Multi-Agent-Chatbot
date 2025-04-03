import os
import uuid
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
from flask import Flask, request, jsonify, render_template, abort
from flask_cors import CORS
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.cuda.amp import autocast
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor

# Import custom modules
from models import init_db, ChatSession, SessionInfo
from agents import agent_manager
from session_manager import init_session_manager

# Configure logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app with enhanced security
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": os.getenv('ALLOWED_ORIGINS', '*').split(',')}})
app.secret_key = os.getenv('SECRET_KEY')
if not app.secret_key:
    raise ValueError('SECRET_KEY environment variable is required')

# Initialize thread pool for parallel processing
thread_pool = ThreadPoolExecutor(max_workers=int(os.getenv('MAX_WORKERS', '4')))

class CorrelationIdFilter(logging.Filter):
    """Filter for adding correlation ID to log records."""
    def filter(self, record):
        try:
            correlation_id = request.headers.get('X-Correlation-ID', 'N/A')
        except RuntimeError:
            correlation_id = 'N/A'
        record.correlation_id = getattr(record, 'correlation_id', correlation_id)
        return True

logger.addFilter(CorrelationIdFilter())

def handle_exceptions(func):
    """Enhanced decorator for handling exceptions with correlation ID tracking."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        correlation_id = request.headers.get('X-Correlation-ID', str(uuid.uuid4()))
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            logger.warning(f'Validation error: {str(e)}', extra={'correlation_id': correlation_id})
            return jsonify({'error': str(e), 'correlation_id': correlation_id}), 400
        except Exception as e:
            logger.error(f'Unexpected error: {str(e)}', exc_info=True, extra={'correlation_id': correlation_id})
            return jsonify({'error': 'An unexpected error occurred', 'correlation_id': correlation_id}), 500
    return wrapper

class ModelManager:
    """Enhanced model manager with caching and parallel processing capabilities."""
    def __init__(self):
        self.model_name = os.getenv('MODEL_NAME', 'gpt2')
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_retries = int(os.getenv('MAX_RETRIES', '3'))
        self.cache_size = int(os.getenv('RESPONSE_CACHE_SIZE', '1000'))
        self.initialize_model()

    def initialize_model(self) -> None:
        """Initialize model with enhanced error handling and validation."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                low_cpu_mem_usage=True,
                device_map='auto',
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                use_safetensors=True
            )
            self.model.eval()
            logger.info(f'Model initialized successfully on {self.device}')
        except Exception as e:
            logger.error(f'Error initializing model: {str(e)}')
            raise

    @lru_cache(maxsize=1000)
    def get_cached_response(self, input_hash: str) -> Optional[str]:
        """Get cached response for identical inputs."""
        return None  # Implement cache lookup logic

    def generate_response(
        self,
        input_text: str,
        temperature: float = 0.85,
        top_p: float = 0.7,
        max_length: int = 150,
        retry_count: int = 0
    ) -> Tuple[str, float]:
        """Generate response with enhanced features and performance monitoring."""
        start_time = datetime.now()
        try:
            # Check cache first
            input_hash = hash(input_text)
            cached_response = self.get_cached_response(str(input_hash))
            if cached_response:
                return cached_response, 0.0

            inputs = self.tokenizer.encode(
                input_text,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)

            with torch.no_grad(), autocast(enabled=True):
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    temperature=temperature,
                    top_k=40,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.3,
                    do_sample=True,
                    early_stopping=True,
                    length_penalty=0.8
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            cleaned_response = self._clean_response(response)
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            
            return cleaned_response, response_time

        except Exception as e:
            if retry_count < self.max_retries:
                logger.warning(f'Retrying response generation (attempt {retry_count + 1})')
                return self.generate_response(input_text, temperature, top_p, max_length, retry_count + 1)
            logger.error(f'Error generating response: {str(e)}')
            raise

    def _clean_response(self, response: str) -> str:
        """Clean and format the model's response with enhanced processing."""
        response = response.split('Assistant:')[-1].strip()
        response = response.replace('Human:', '').strip()
        return ' '.join(response.split())

# Initialize components with enhanced error handling
try:
    model_manager = ModelManager()
    SessionLocal = init_db(os.getenv('DATABASE_URL', 'sqlite:///chat_history.db'))
    session_manager = init_session_manager(SessionLocal)
    logger.info('Application components initialized successfully')
except Exception as e:
    logger.critical(f'Failed to initialize application components: {str(e)}')
    raise

def get_ai_response(message: str, session_id: str, agent_id: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """Generate AI response with enhanced context management and emotion analysis."""
    try:
        session_history = session_manager.get_session_history(session_id, limit=5)
        agent_profile = agent_manager.get_profile(agent_id) if agent_id else agent_manager.get_default_profile()
        
        # Build conversation context with emotional awareness
        context = build_conversation_context(session_history, message, agent_profile)
        
        response, response_time = model_manager.generate_response(
            context,
            temperature=agent_profile.behavior_params.get('temperature', 0.9),
            top_p=agent_profile.behavior_params.get('top_p', 0.92)
        )

        metadata = {
            'response_time': response_time,
            'agent_profile': agent_profile.id,
            'timestamp': datetime.now().isoformat(),
            'context_length': len(context)
        }

        # Update session asynchronously
        thread_pool.submit(
            update_session_async,
            session_id,
            message,
            response,
            metadata,
            response_time
        )

        return response, metadata
    except Exception as e:
        logger.error(f'Error in get_ai_response: {str(e)}')
        raise

def build_conversation_context(history: List[Any], message: str, agent_profile: Any) -> str:
    """Build conversation context with emotional analysis and personality traits."""
    context = f"Agent Profile: {agent_profile.name}\n"
    context += f"Personality: {', '.join(agent_profile.traits)}\n\n"
    
    if history:
        context += "Previous conversation:\n"
        for msg in history[-3:]:
            context += f"{msg.role.capitalize()}: {msg.content}\n"
    
    context += f"Human: {message}\nAssistant:"
    return context

def update_session_async(session_id: str, message: str, response: str, metadata: Dict[str, Any], response_time: float) -> None:
    """Update session data asynchronously."""
    try:
        session_manager.add_message(session_id, 'user', message)
        session_manager.add_message(session_id, 'assistant', response, metadata=metadata)
        session_manager.track_metrics(session_id, response_time=response_time)
    except Exception as e:
        logger.error(f'Error updating session asynchronously: {str(e)}')

@app.errorhandler(Exception)
def handle_error(error):
    """Global error handler with enhanced logging and tracking."""
    error_id = str(uuid.uuid4())
    correlation_id = request.headers.get('X-Correlation-ID', 'N/A')
    logger.error(
        f'Unhandled error {error_id}',
        exc_info=True,
        extra={'correlation_id': correlation_id}
    )
    return jsonify({
        'error': 'An unexpected error occurred',
        'error_id': error_id,
        'correlation_id': correlation_id
    }), 500

@app.route('/api/agents')
@handle_exceptions
def get_agents_list():
    """Get available agent profiles with enhanced metadata."""
    agents = [{
        'id': profile.id,
        'name': profile.name,
        'description': profile.description,
        'traits': profile.traits,
        'avatar': profile.avatar,
        'capabilities': profile.behavior_params,
        'status': 'active'
    } for profile in agent_manager.get_all_profiles()]
    return jsonify({'agents': agents, 'count': len(agents)})

@app.route('/')
def index():
    """Render the main chat interface."""
    return render_template('index.html')

@app.route('/api/analytics/session/<session_id>', methods=['GET'])
@handle_exceptions
def get_session_analytics(session_id):
    """Get comprehensive session analytics with enhanced metrics."""
    analytics = session_manager.get_session_analytics(session_id)
    if not analytics:
        abort(404, description='Session not found')
    return jsonify(analytics)

@app.route('/api/analytics/global', methods=['GET'])
@handle_exceptions
def get_global_analytics():
    """Get enhanced global analytics with performance metrics."""
    return jsonify(session_manager.get_global_analytics())

@app.route('/api/feedback', methods=['POST'])
@handle_exceptions
def submit_feedback():
    """Submit user feedback with enhanced validation and tracking."""
    data = request.json
    session_id = data.get('session_id')
    rating = data.get('rating')
    feedback_text = data.get('feedback_text')
    
    if not session_id or not isinstance(rating, (int, float)) or not (1 <= rating <= 5):
        raise ValueError('Invalid feedback data')
        
    metadata = {
        'feedback_text': feedback_text,
        'timestamp': datetime.now().isoformat(),
        'user_agent': request.headers.get('User-Agent')
    }
    
    session_manager.track_metrics(session_id, user_rating=rating)
    return jsonify({'message': 'Feedback submitted successfully', 'metadata': metadata})

@app.route('/api/chat', methods=['POST'])
@handle_exceptions
def chat():
    """Handle chat messages with enhanced features and async processing."""
    data = request.json
    message = data.get('message')
    agent_id = data.get('agent_id')
    session_id = data.get('session_id')
    user_id = data.get('user_id', 'anonymous')

    if not message:
        raise ValueError('Message is required')

    if not session_id:
        session_id = str(uuid.uuid4())
        if not agent_id:
            agent_id = agent_manager.get_default_profile().id

        if not session_manager.create_session(session_id, agent_id, user_id):
            raise ValueError('Failed to create session')
    else:
        session_state = session_manager.get_session(session_id)
        if not session_state:
            raise ValueError('Invalid session ID')
        agent_id = agent_id or session_state.agent_id

    response, metadata = get_ai_response(message, session_id, agent_id)

    return jsonify({
        'response': response,
        'session_id': session_id,
        'metadata': metadata
    })

@app.route('/api/session', methods=['GET', 'POST', 'DELETE'])
@handle_exceptions
def manage_session():
    """Manage chat sessions with enhanced validation and async updates."""
    if request.method == 'POST':
        data = request.json or {}
        agent_id = data.get('agent_id')
        user_id = data.get('user_id', 'anonymous')
        metadata = data.get('metadata', {})

        if not agent_id:
            return jsonify({'error': 'Agent ID is required'}), 400

        session_id = str(uuid.uuid4())
        session_state = session_manager.create_session(session_id, agent_id, user_id, metadata)
        if not session_state:
            return jsonify({'error': 'Failed to create session'}), 500

        agent_profile = agent_manager.get_profile(agent_id)
        if not agent_profile:
            return jsonify({'error': 'Agent not found'}), 404

        return jsonify({
            'session_id': session_id,
            'agent': {
                'id': agent_profile.id,
                'name': agent_profile.name,
                'avatar': agent_profile.avatar
            },
            'created_at': session_state.created_at.isoformat(),
            'metadata': metadata
        })

    elif request.method == 'DELETE':
        session_id = request.args.get('session_id')
        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400

        if not session_manager.delete_session(session_id):
            return jsonify({'error': 'Session not found'}), 404

        return jsonify({'message': 'Session deleted successfully'})

    else:  # GET
        session_id = request.args.get('session_id')
        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400

        session_state = session_manager.get_session(session_id)
        if not session_state:
            return jsonify({'error': 'Session not found'}), 404

        messages = session_manager.get_session_history(session_id)
        agent_profile = agent_manager.get_profile(session_state.agent_id)

        return jsonify({
            'session_id': session_id,
            'agent': {
                'id': agent_profile.id,
                'name': agent_profile.name,
                'avatar': agent_profile.avatar
            },
            'messages': [{
                'role': msg.role,
                'content': msg.content,
                'timestamp': msg.timestamp.isoformat(),
                'metadata': msg.metadata
            } for msg in messages],
            'created_at': session_state.created_at.isoformat(),
            'metadata': session_state.metadata
        })

@app.route('/api/sessions', methods=['GET'])
@handle_exceptions
def get_user_sessions():
    """Get user sessions with enhanced filtering and pagination."""
    user_id = request.args.get('user_id', 'anonymous')
    active_only = request.args.get('active_only', '').lower() == 'true'
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    
    sessions = session_manager.get_user_sessions(
        user_id,
        active_only=active_only,
        page=page,
        per_page=per_page
    )
    
    session_data = []
    for session in sessions:
        agent_profile = agent_manager.get_profile(session.agent_id)
        last_message = session_manager.get_last_message(session.session_id)
        
        session_data.append({
            'session_id': session.session_id,
            'agent': {
                'id': agent_profile.id,
                'name': agent_profile.name,
                'avatar': agent_profile.avatar
            } if agent_profile else None,
            'created_at': session.created_at.isoformat(),
            'last_active': session.last_active.isoformat(),
            'last_message': {
                'content': last_message.content,
                'timestamp': last_message.timestamp.isoformat()
            } if last_message else None,
            'metadata': session.metadata
        })
    
    return jsonify({
        'sessions': session_data,
        'page': page,
        'per_page': per_page,
        'total': session_manager.get_user_session_count(user_id, active_only)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=os.getenv('FLASK_DEBUG', 'false').lower() == 'true')