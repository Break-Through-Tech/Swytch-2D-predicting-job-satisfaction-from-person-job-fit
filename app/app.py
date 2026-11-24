import pandas as pd, streamlit as st
from swytch import scoring, matching
from swytch.features import PERSON

LABELS = {"IMP_ADV": "Opportunities for advancement", "IMP_BEN": "Benefits", "IMP_CHAL": "Intellectual challenge",
          "IMP_IND": "Independence", "IMP_LOC": "Location", "IMP_RESP": "Level of responsibility",
          "IMP_SAL": "Salary", "IMP_SEC": "Job security", "IMP_SOC": "Contribution to society"}

st.title("How well does a job fit your values?")
st.caption("Estimates are based on survey data and occupation-level profiles, not on any specific employer.")
values = {k: st.slider(LABELS[k], 1, 4, 3, help="1 = not important, 4 = very important") for k in PERSON}

tab1, tab2 = st.tabs(["Evaluate a listing", "Recommendations"])
with tab1:
    title = st.text_input("Job title"); desc = st.text_area("Listing text (optional)")
    if st.button("Evaluate") and title:
        cands = matching.top_matches(f"{title} {desc}")
        choice = st.selectbox("Matched occupation (change if wrong)", cands, format_func=lambda c: c[1])
        pid = choice[0]; bd = scoring.breakdown(values, pid)
        st.subheader(scoring.label(values, pid))
        st.bar_chart(pd.DataFrame(bd).set_index("dimension")[["you", "job"]])
        for m in scoring.explain(bd): st.write("• " + m)
with tab2:
    if st.button("Recommend"):
        for r in scoring.recommend(values, 10):
            with st.expander(f"{r['occupation']}  |  {r['label']}"):
                for m in r["why"]: st.write("• " + m)