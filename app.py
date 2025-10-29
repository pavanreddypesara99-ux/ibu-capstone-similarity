import streamlit as st
import pandas as pd

st.set_page_config(page_title="IBU Capstone Similarity (Fresh)", layout="centered")
st.title("🎓 IBU Capstone – Data Check")

st.sidebar.header("Google Sheet (Publish to web → CSV)")
csv_url = st.sidebar.text_input(
    "Paste your CSV link here",
    "https://docs.google.com/spreadsheets/d/e/XXXX/pub?output=csv"  # replace with your link
)

try:
    df = pd.read_csv(csv_url)
    df.columns = [c.strip() for c in df.columns]
    st.success(f"Loaded {len(df)} rows.")
    if "Project Title" not in df.columns:
        st.error("CSV must have a 'Project Title' column.")
    else:
        st.write("First 10 titles:")
        st.write(df["Project Title"].fillna("").astype(str).head(10).to_list())
except Exception as e:
    st.error(f"Could not load CSV: {e}")

st.caption("Fresh start: once this works online, we'll add hybrid TF-IDF + SBERT next.")
