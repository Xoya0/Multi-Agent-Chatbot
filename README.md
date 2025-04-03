# Neural Network Chat (NNC)

A sophisticated chat application powered by neural networks, featuring dynamic agent personalities, advanced session management, and comprehensive analytics.

## Important Note

This repository contains large data files that are not included in the Git repository due to size limitations. To obtain the complete dataset:

1. Download the movie quotes dataset files 
2. Place the downloaded files in the `data/` directory:
   - `moviequotes.memorable_nonmemorable_pairs.txt`
   - `moviequotes.memorable_quotes.txt`
   - `moviequotes.scripts.txt`

## Features

- **Dynamic Agent Personalities**: Customizable chat agents with distinct personality traits and behavior parameters
- **Advanced Session Management**: Robust handling of chat sessions with persistence and caching
- **Real-time Analytics**: Comprehensive metrics tracking for response times, user ratings, and session statistics
- **Scalable Architecture**: Built with Flask and SQLAlchemy for reliable performance
- **Security Features**: Correlation ID tracking and enhanced error handling

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for optimal performance)
- SQLite or compatible database system

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables in `.env` file:
   ```env
   SECRET_KEY=your_secret_key
   MODEL_NAME=bigscience/bloom-560m
   PORT=5000
   FLASK_DEBUG=false
   DATABASE_URL=sqlite:///chat_history.db
   ```

## Usage

1. Start the server:
   ```bash
   python app.py
   ```
2. Access the chat interface at `http://localhost:5000`

### API Endpoints

- `POST /api/chat`: Send messages to chat agents
- `GET /api/agents`: List available agent profiles
- `GET /api/analytics/session/<session_id>`: Get session analytics
- `GET /api/analytics/global`: View global performance metrics

## Development

### Training Custom Models

```bash
# Train the model with custom data
python train.py
```

### Running Tests

```bash
python test_conversation.py
```

### Project Structure

- `app.py`: Main application entry point
- `agents.py`: Agent personality management
- `models.py`: Database models
- `session_manager.py`: Chat session handling
- `train.py`: Model training utilities
- `test_conversation.py`: Test suite

## Features in Detail

### Agent Personality System

Agents can be configured with:
- Personality traits
- Response parameters (temperature, top_p)
- Behavioral characteristics

### Session Management

- Persistent session storage
- Caching for active sessions
- Automatic cleanup of inactive sessions
- Comprehensive session analytics

### Analytics Capabilities

- Response time tracking
- User satisfaction metrics
- Error rate monitoring
- Session activity analysis

## Security Considerations

- Correlation ID tracking for request tracing
- Rate limiting capabilities
- Input validation and sanitization
- Secure session management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
