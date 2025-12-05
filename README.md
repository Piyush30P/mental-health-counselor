# Mental Health Counselor AI

A multimodal emotion recognition system that combines facial expression analysis and text sentiment analysis to provide therapeutic support and mental health counseling through an AI-powered interface.

## 🌟 Features

- **Multimodal Emotion Recognition**: Combines facial expression detection and text sentiment analysis
- **Real-time Analysis**: Live emotion detection through webcam and text input
- **Therapeutic AI**: GPT-4 powered conversational agent trained for therapeutic responses
- **Secure Authentication**: JWT-based user authentication and session management
- **Admin Dashboard**: Administrative interface for managing users and sessions
- **Data Visualization**: Real-time emotion tracking and analytics
- **Responsive UI**: Modern React frontend with Tailwind CSS

## 🏗️ Architecture

### Backend (FastAPI)

- **Framework**: FastAPI with Python 3.10+
- **Database**: MongoDB for user data and session storage
- **ML Models**:
  - CNN model for facial emotion recognition
  - LSTM model for text sentiment analysis
  - Fusion model combining both modalities
- **AI Integration**: OpenAI GPT-4 for therapeutic conversations
- **Authentication**: JWT tokens with bcrypt password hashing

### Frontend (React + Vite)

- **Framework**: React 18 with TypeScript
- **Styling**: Tailwind CSS for modern, responsive design
- **Build Tool**: Vite for fast development and building
- **State Management**: React Context API
- **Components**: Modular component architecture

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Node.js 16 or higher
- MongoDB (local or cloud instance)
- OpenAI API key

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/Piyush30P/mental-health-counselor.git
cd mental-health-counselor
```

2. **Set up the backend**

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

3. **Set up environment variables**
   Create a `.env` file in the backend directory:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Database Configuration
MONGODB_URL=mongodb://127.0.0.1:27017
DATABASE_NAME=mental_health_db

# JWT Configuration
JWT_SECRET=your_jwt_secret_key_here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=True

# CORS
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
```

4. **Set up the frontend**

```bash
cd ../frontend
npm install
```

### Running the Application

1. **Start MongoDB** (if running locally)

```bash
mongod
```

2. **Start the backend server**

```bash
cd backend
# Activate virtual environment if not already active
.\venv\Scripts\Activate.ps1
python main.py
```

Backend will be available at `http://localhost:8000`

3. **Start the frontend development server**

```bash
cd frontend
npm run dev
```

Frontend will be available at `http://localhost:3000`

## 📁 Project Structure

```
mental-health-counselor/
├── backend/
│   ├── auth/                 # Authentication modules
│   ├── database/            # Database connection and models
│   ├── models/              # ML models (facial, text, fusion, LLM)
│   ├── routers/             # API route handlers
│   ├── weights/             # Pre-trained model weights
│   ├── Model_training/      # Training scripts
│   ├── main.py             # FastAPI application entry point
│   ├── config.py           # Configuration management
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── contexts/       # React context providers
│   │   ├── pages/          # Page components
│   │   ├── services/       # API service functions
│   │   └── types/          # TypeScript type definitions
│   ├── public/             # Static assets
│   └── package.json        # Node.js dependencies
└── README.md
```

## 🔧 API Endpoints

### Authentication

- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `GET /api/auth/me` - Get current user

### Emotion Recognition

- `POST /api/emotion/text` - Text emotion analysis
- `POST /api/emotion/facial` - Facial emotion analysis
- `POST /api/emotion/multimodal` - Combined analysis
- `WebSocket /api/emotion/realtime` - Real-time emotion detection

### User Management

- `GET /api/user/profile` - Get user profile
- `PUT /api/user/profile` - Update user profile
- `GET /api/user/sessions` - Get user sessions

### Admin (Protected)

- `GET /api/admin/users` - Get all users
- `GET /api/admin/sessions` - Get all sessions
- `GET /api/admin/analytics` - Get system analytics

## 🤖 Machine Learning Models

### Facial Emotion Recognition

- **Architecture**: CNN with multiple convolutional layers
- **Input**: 48x48 grayscale images
- **Output**: 7 emotion classes (angry, disgust, fear, happy, neutral, sad, surprise)
- **Accuracy**: ~65% on FER2013 dataset

### Text Sentiment Analysis

- **Architecture**: LSTM with embedding layer
- **Input**: Tokenized text sequences (max length: 100)
- **Output**: 6 emotion classes (anger, fear, joy, love, sadness, surprise)
- **Preprocessing**: Cleaning, tokenization, lemmatization, stopword removal

### Multimodal Fusion

- **Method**: Weighted averaging of facial and text predictions
- **Weights**: Text 60%, Facial 40% (configurable)
- **Incongruence Detection**: Identifies mismatches between modalities

## 🧠 Therapeutic AI

The system uses OpenAI's GPT-4 model fine-tuned for therapeutic conversations:

- **Context-aware responses** based on detected emotions
- **Therapeutic techniques** including CBT and mindfulness
- **Crisis detection** with appropriate resource recommendations
- **Session continuity** maintaining conversation history

## 🔐 Security Features

- **JWT Authentication**: Secure token-based authentication
- **Password Hashing**: bcrypt for secure password storage
- **CORS Protection**: Configured for specific allowed origins
- **Input Validation**: Comprehensive input sanitization
- **Rate Limiting**: Protection against API abuse

## 📊 Monitoring & Analytics

- **Real-time Metrics**: Emotion trends and patterns
- **Session Analytics**: User engagement and progress tracking
- **System Health**: Model performance and API metrics
- **User Insights**: Personalized mental health insights

## 🚀 Deployment

### Docker Deployment (Optional)

```bash
# Build and run with Docker Compose
docker-compose up -d
```

### Cloud Deployment

The application can be deployed on:

- **Backend**: Heroku, AWS EC2, DigitalOcean
- **Frontend**: Netlify, Vercel, AWS S3 + CloudFront
- **Database**: MongoDB Atlas, AWS DocumentDB

## 🧪 Testing

### Backend Tests

```bash
cd backend
pytest tests/
```

### Frontend Tests

```bash
cd frontend
npm test
```

### Model Testing

```bash
cd backend/Model_training
python 3_test_model.py
```

## 📈 Performance Optimization

- **Model Caching**: Pre-loaded models for faster inference
- **Connection Pooling**: Optimized database connections
- **Async Processing**: Non-blocking API operations
- **Frontend Optimization**: Code splitting and lazy loading

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- FER2013 dataset for facial emotion recognition training
- OpenAI for GPT-4 API
- The open-source ML and web development communities

## 📞 Support

For support, email [pisepiyush631@gmail.com](mailto:pisepiyush631@gmail.com) or create an issue in the GitHub repository.

## 🔗 Links

- [Live Demo](https://your-demo-url.com)
- [API Documentation](https://your-api-docs-url.com)
- [Project Repository](https://github.com/Piyush30P/mental-health-counselor)

---

**Note**: This is a research prototype for educational purposes. For actual mental health concerns, please consult qualified mental health professionals.
