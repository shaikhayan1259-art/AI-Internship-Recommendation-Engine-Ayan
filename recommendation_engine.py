from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

class RecommendationEngine:
    """Hybrid internship recommender.

    Content score: TF-IDF + cosine similarity between student profile and internships.
    Collaborative score: nearest-neighbour similarity over user-item ratings.
    Hybrid score: weighted combination of both.
    """

    def __init__(self, internships_path, interactions_path):
        self.internships = pd.read_csv(internships_path)
        self.interactions = pd.read_csv(interactions_path)
        self._prepare()

    def _prepare(self):
        self.internships["content"] = (
            self.internships["title"].fillna("") + " " +
            self.internships["domain"].fillna("") + " " +
            self.internships["skills"].fillna("") + " " +
            self.internships["description"].fillna("")
        )
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.item_matrix = self.vectorizer.fit_transform(self.internships["content"])

        self.user_item = self.interactions.pivot_table(
            index="user_id", columns="internship_id",
            values="rating", aggfunc="mean", fill_value=0
        )
        if len(self.user_item) >= 2:
            self.nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=min(5, len(self.user_item)))
            self.nn.fit(self.user_item.values)
        else:
            self.nn = None

    @staticmethod
    def _profile_text(profile):
        return " ".join([
            str(profile.get("skills", "")),
            str(profile.get("domain", "")),
            str(profile.get("education", "")),
            str(profile.get("location", "")),
        ])

    def content_scores(self, profile):
        q = self.vectorizer.transform([self._profile_text(profile)])
        return cosine_similarity(q, self.item_matrix).ravel()

    def collaborative_scores(self, user_id):
        scores = np.zeros(len(self.internships))
        if self.nn is None or user_id not in self.user_item.index:
            return scores

        user_index = list(self.user_item.index).index(user_id)
        distances, indices = self.nn.kneighbors(
            self.user_item.iloc[[user_index]].values,
            n_neighbors=min(5, len(self.user_item))
        )
        weights = 1 - distances[0]
        weighted = np.average(
            self.user_item.iloc[indices[0]].values,
            axis=0,
            weights=np.maximum(weights, 1e-6)
        )
        item_to_col = {item: i for i, item in enumerate(self.user_item.columns)}
        for i, iid in enumerate(self.internships["internship_id"]):
            if iid in item_to_col:
                scores[i] = weighted[item_to_col[iid]]
        if scores.max() > 0:
            scores = scores / scores.max()
        return scores

    def recommend(self, profile, user_id="Ayan", top_k=5):
        content = self.content_scores(profile)
        collab = self.collaborative_scores(user_id)

        # Hybrid model: content-first because a new student may have little history.
        hybrid = 0.65 * content + 0.35 * collab

        result = self.internships.copy()
        result["content_score"] = content
        result["collab_score"] = collab
        result["hybrid"] = hybrid

        if profile.get("domain") and profile["domain"] != "Any":
            result.loc[result["domain"] != profile["domain"], "hybrid"] *= 0.82
        if profile.get("location") and profile["location"] != "Any":
            result.loc[result["location"] != profile["location"], "hybrid"] *= 0.92

        result["match_score"] = np.clip(result["hybrid"] * 100, 0, 100)
        return result.sort_values("match_score", ascending=False).head(top_k)

    def record_feedback(self, user_id, internship_id, rating):
        new_row = pd.DataFrame([{
            "user_id": user_id,
            "internship_id": internship_id,
            "rating": float(rating)
        }])
        self.interactions = pd.concat([self.interactions, new_row], ignore_index=True)
        self._prepare()

    def evaluate(self, k=5):
        """Simple offline evaluation: Precision@K using positive ratings >= 4.

        This is a lightweight demo metric for the supplied internship dataset.
        """
        precisions = []
        for user in self.interactions["user_id"].unique():
            user_rows = self.interactions[self.interactions["user_id"] == user]
            positives = set(user_rows.loc[user_rows["rating"] >= 4, "internship_id"])
            if not positives:
                continue
            profile = {
                "skills": "Python Machine Learning Data Science",
                "domain": self.internships["domain"].iloc[0],
                "education": "B.E./B.Tech",
                "location": "Any"
            }
            recs = self.recommend(profile, user_id=user, top_k=k)
            predicted = set(recs["internship_id"])
            precisions.append(len(predicted & positives) / k)
        return {
            "Precision@5": round(float(np.mean(precisions)) if precisions else 0.0, 3),
            "Users evaluated": len(precisions)
        }
