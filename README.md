# AI Internship Recommendation Engine

A complete internship recommendation prototype for the AI & Data Science internship task.

## Core requirements implemented
- Student Profile Management
- Internship Database
- Content-Based Filtering
- Collaborative Filtering
- Hybrid Recommendation
- Recommendation Evaluation Metrics

## Bonus features implemented
- Deep-learning-ready recommendation architecture (model layer can be extended)
- Real-time recommendation updates after feedback
- Visualization dashboard
- Web application UI
- AI testing/evaluation metric
- User feedback integration

## Tech stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Algorithm
1. Internship text is converted into TF-IDF vectors.
2. Student profile is converted into a TF-IDF vector.
3. Cosine similarity produces content-based relevance.
4. User-item ratings are used with nearest-neighbour collaborative filtering.
5. Final ranking combines content and collaborative signals.
6. Domain/location preferences provide lightweight personalization.
7. Feedback is added to the interaction table and the model is rebuilt.

## Project structure
```text
ai_internship_recommendation_engine/
├── app.py
├── recommendation_engine.py
├── requirements.txt
├── README.md
├── data/
│   ├── internships.csv
│   └── interactions.csv
└── tests/
    └── test_engine.py
```

## Important
This is a portfolio/learning implementation using a synthetic internship dataset. Replace the sample data with authorized real internship data before production use.
