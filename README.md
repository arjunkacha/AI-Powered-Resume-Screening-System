# AI-Powered Resume Screening System

An intelligent system for automated resume screening and candidate ranking using NLP and machine learning techniques.

## 🎯 Features

- Resume parsing (PDF/DOCX support)
- Information extraction (name, skills, experience)
- Multiple embedding methods (TF-IDF, BERT, spaCy)
- Job description analysis
- Candidate ranking using similarity scoring
- Interactive web interface
- RESTful API endpoints
- Model training and evaluation
- Data analysis and visualization

## 🛠️ Technology Stack

- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Database**: MongoDB (optional)
- **NLP Libraries**: spaCy, scikit-learn, transformers
- **Document Processing**: pdfminer.six, python-docx
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn

## 📁 Project Structure

```
resume-screening-system/
├── backend/               # FastAPI backend code
│   ├── api/              # API endpoints
│   └── models/           # ML models and embeddings
├── frontend/             # Streamlit UI
├── data/                 # Resumes and job descriptions
│   ├── raw/             # Raw resume files
│   └── processed/       # Processed data
├── embeddings/           # Saved vector embeddings
├── utils/                # Utility functions
│   ├── data_collector.py # Data collection and preprocessing
│   └── model_trainer.py  # Model training and evaluation
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/yourusername/resume-screening-system.git
cd resume-screening-system
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required models:
```bash
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords
```

5. Run the development server:
```bash
# Start backend
cd backend
uvicorn main:app --reload

# Start frontend (in a new terminal)
cd frontend
streamlit run app.py
```

## 📊 Data Collection and Processing

1. Collect resumes:
```bash
python utils/data_collector.py
```

2. Process and tag resumes:
```bash
python utils/data_collector.py --process
```

3. Perform EDA:
```bash
python utils/data_collector.py --eda
```

## 🤖 Model Training

1. Train models:
```bash
python utils/model_trainer.py
```

2. Evaluate models:
```bash
python utils/model_trainer.py --evaluate
```

3. Compare models:
```bash
python utils/model_trainer.py --compare
```

## 📝 API Documentation

Once the server is running, visit:
- API docs: http://localhost:8000/docs
- Frontend: http://localhost:8501

### API Endpoints

- `POST /parse-resume`: Parse a resume and extract information
- `POST /rank-resumes`: Rank resumes based on job description
- `GET /health`: Health check endpoint

## 📈 Model Performance

The system supports multiple embedding methods and classifiers:

1. Embedding Methods:
   - TF-IDF
   - BERT
   - spaCy

2. Classifiers:
   - Logistic Regression
   - Support Vector Machine (SVM)

Performance metrics are saved in `models/saved/` directory.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- spaCy for NLP capabilities
- Hugging Face for transformer models
- FastAPI for the backend framework
- Streamlit for the frontend interface 