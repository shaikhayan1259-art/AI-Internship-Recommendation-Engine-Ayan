from recommendation_engine import RecommendationEngine

def test_recommendations():
    engine = RecommendationEngine("data/internships.csv", "data/interactions.csv")
    profile = {
        "skills": "Python Pandas Machine Learning",
        "domain": "Data Science",
        "education": "B.E./B.Tech",
        "location": "Pune"
    }
    result = engine.recommend(profile, user_id="Ayan", top_k=5)
    assert len(result) == 5
    assert "match_score" in result.columns
    assert result["match_score"].between(0, 100).all()

def test_evaluation():
    engine = RecommendationEngine("data/internships.csv", "data/interactions.csv")
    metrics = engine.evaluate(k=5)
    assert "Precision@5" in metrics
