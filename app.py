import streamlit as st
import pandas as pd
from recommendation_engine import RecommendationEngine

st.set_page_config(page_title="AI Internship Recommendation Engine", page_icon="🎓", layout="wide")

@st.cache_resource
def load_engine():
    return RecommendationEngine(
        "data/internships.csv",
        "data/interactions.csv"
    )

engine = load_engine()

st.title("🎓 AI Internship Recommendation Engine")
st.caption("Hybrid recommendation system • Content-Based + Collaborative Filtering")

with st.sidebar:
    st.header("Student Profile")
    name = st.text_input("Name", "Ayan")
    skills = st.text_input(
        "Skills",
        "Python, Pandas, Machine Learning, SQL"
    )
    domain = st.selectbox(
        "Preferred Domain",
        sorted(engine.internships["domain"].unique().tolist()),
    )
    location = st.selectbox(
        "Preferred Location",
        ["Any"] + sorted(engine.internships["location"].unique().tolist()),
    )
    education = st.selectbox(
        "Education",
        ["B.E./B.Tech", "B.Sc", "M.E./M.Tech", "MCA", "Other"]
    )
    top_k = st.slider("Recommendations", 3, 10, 5)

    if st.button("🔄 Generate Recommendations", use_container_width=True):
        st.session_state["run"] = True

if "run" not in st.session_state:
    st.session_state["run"] = True

profile = {
    "name": name,
    "skills": skills,
    "domain": domain,
    "location": location,
    "education": education,
}

tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Recommendations", "📊 Dashboard", "💬 Feedback", "🗃️ Internship Database"
])

with tab1:
    st.subheader("Top Internship Recommendations")
    recs = engine.recommend(profile, user_id=name, top_k=top_k)
    if recs.empty:
        st.warning("No recommendations found.")
    else:
        for _, row in recs.iterrows():
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.markdown(f"### {row['title']}")
                    st.write(f"**Company:** {row['company']}  •  **Domain:** {row['domain']}")
                    st.write(f"**Skills:** {row['skills']}")
                    st.write(f"**Location:** {row['location']}  •  **Duration:** {row['duration']}")
                with c2:
                    st.metric("Match", f"{row['match_score']:.0f}%")
                    st.caption(f"Content {row['content_score']:.0f}%")
                    st.caption(f"Collaborative {row['collab_score']:.0f}%")
                if st.button("👍 Interested", key=f"like_{row['internship_id']}"):
                    engine.record_feedback(name, row["internship_id"], 1)
                    st.success("Feedback saved.")

with tab2:
    st.subheader("Recommendation & Dataset Dashboard")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Internships", len(engine.internships))
    m2.metric("Users", engine.interactions["user_id"].nunique())
    m3.metric("Interactions", len(engine.interactions))
    m4.metric("Avg. Feedback", f"{engine.interactions['rating'].mean():.2f}")

    st.markdown("#### Domain distribution")
    st.bar_chart(engine.internships["domain"].value_counts())

    st.markdown("#### Model evaluation")
    metrics = engine.evaluate(k=5)
    st.dataframe(pd.DataFrame([metrics]), use_container_width=True)

with tab3:
    st.subheader("User Feedback Integration")
    feedback = st.radio("How relevant were your recommendations?", ["Not relevant", "Okay", "Very relevant"])
    rating = {"Not relevant": 1, "Okay": 3, "Very relevant": 5}[feedback]
    if st.button("Submit Feedback"):
        if len(recs):
            engine.record_feedback(name, recs.iloc[0]["internship_id"], rating)
            st.success("Thanks — feedback has been recorded and will affect future recommendations.")

with tab4:
    st.subheader("Internship Database")
    st.dataframe(engine.internships, use_container_width=True)
