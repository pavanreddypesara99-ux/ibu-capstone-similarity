# app.py
import streamlit as st
import pandas as pd, numpy as np, requests, re, unicodedata, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="IBU Capstone Similarity (Hybrid)", layout="centered")
st.title("🎓 IBU Capstone Similarity (Hybrid TF-IDF + SBERT)")

# -----------------------------
# Sidebar: links you control
# -----------------------------
st.sidebar.header("🔗 Data connections")
csv_url = st.sidebar.text_input(
    "Google Sheet (Publish to Web → CSV) URL",
    "https://docs.google.com/spreadsheets/d/e/2PACX-XXXX/pub?output=csv"
)
apps_script_url = st.sidebar.text_input(
    "Google Apps Script URL (Submissions sheet)",
    "https://script.google.com/macros/s/XXXX/exec"
)
st.sidebar.caption("Tip: If you see pubhtml or an edit link, we’ll auto-convert it to CSV.")

# -----------------------------
# Helpers
# -----------------------------
def to_csv_url(url: str) -> str:
    url = url.strip()
    if "pubhtml" in url:
        # convert publish-HTML → publish-CSV
        base = url.split("?")[0].replace("pubhtml", "pub")
        return f"{base}?output=csv"
    if "/edit#gid=" in url and "/d/" in url:
        # convert edit link → export CSV for that gid
        sheet_id = url.split("/d/")[1].split("/")[0]
        gid = url.split("gid=")[1].split("&")[0]
        return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    return url

def normalize_text(s: str) -> str:
    s = str(s or "")
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^a-z0-9\s\-&']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

@st.cache_data(show_spinner=False)
def load_data(url: str) -> pd.DataFrame:
    url = to_csv_url(url)
    df = pd.read_csv(url)
    df.columns = [c.strip() for c in df.columns]
    if "Project Title" not in df.columns:
        raise ValueError("CSV must contain a 'Project Title' column.")
    df["Project Title"] = df["Project Title"].fillna("").astype(str)
    return df

@st.cache_resource(show_spinner=True)
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def build_tfidf(titles_norm: pd.Series):
    # 1–3 word n-grams help short titles; min_df=1 keeps rare terms
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1,3), min_df=1, stop_words="english")
    X = vec.fit_transform(titles_norm)
    return vec, X

@st.cache_resource(show_spinner=True)
def embed_titles(model, titles_norm: pd.Series):
    # normalized + L2-normalized embeddings
    embs = model.encode(titles_norm.tolist(), convert_to_tensor=True, normalize_embeddings=True)
    return embs

def hybrid_search(query: str, vec, X, df, model, embs, prefilter_k=150, top_k=10):
    qn = normalize_text(query)
    if not qn:
        return pd.DataFrame(), 0.0
    # Stage 1: TF-IDF prefilter
    qv = vec.transform([qn])
    tfidf = cosine_similarity(qv, X).flatten()
    k = min(prefilter_k, len(tfidf))
    pre_idx = np.argsort(-tfidf)[:k]

    # Stage 2: SBERT re-rank
    q_emb = model.encode([qn], convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(q_emb, embs[pre_idx]).cpu().numpy().flatten()
    order = np.argsort(-sims)[:top_k]
    idxs = pre_idx[order]
    scores = sims[order]

    # Build table (gracefully handle missing columns)
    def col(c, i, default=""):
        return df[c].iloc[i] if c in df.columns else default

    out = pd.DataFrame({
        "Rank": np.arange(1, len(idxs)+1),
        "Similarity": np.round(scores, 4),
        "Existing Title": [df["Project Title"].iloc[i] for i in idxs],
        "Year": [col("Year", i, "") for i in idxs],
        "Program": [col("Program", i, "") for i in idxs],
        "Supervisor": [col("Supervisor", i, col("Faculty Advisor (last, first)", i, "")) for i in idxs],
        "Report URL": [col("Report URL", i, "") for i in idxs],
    })
    return out, (scores[0] if len(scores) else 0.0)

def bucket(score: float) -> str:
    if score >= 0.82: return "Very similar (likely overlap)"
    if score >= 0.74: return "Related (check scope)"
    if score >= 0.65: return "Loosely related"
    return "Not similar"

def register_title(script_url: str, title: str):
    # Minimal payload; extend later (student id, program, year…)
    payload = {"project_title": title, "timestamp": int(time.time())}
    r = requests.post(script_url, json=payload, timeout=12)
    return r.status_code

# -----------------------------
# Load data + models
# -----------------------------
try:
    df = load_data(csv_url)
    st.success(f"Loaded {len(df)} titles from sheet.")
except Exception as e:
    st.error(f"Could not load sheet: {e}")
    st.stop()

titles_norm = df["Project Title"].apply(normalize_text)
model = load_model()
vec, X = build_tfidf(titles_norm)
embs = embed_titles(model, titles_norm)

# -----------------------------
# UI – title-only workflow
# -----------------------------
st.subheader("🔍 Check your title")
user_title = st.text_input("Enter proposed capstone title")
colA, colB = st.columns(2)
with colA:
    threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.80, 0.01)
with colB:
    top_k = st.selectbox("Top results", [5, 10, 15, 20], index=1)

if st.button("Check Similarity", type="primary"):
    results, best = hybrid_search(user_title, vec, X, df, model, embs, prefilter_k=150, top_k=int(top_k))
    if results.empty:
        st.warning("No results. Check your input.")
    else:
        st.markdown(f"**Highest similarity:** {best:.3f} → _{bucket(best)}_")
        st.dataframe(results, use_container_width=True, hide_index=True)

        if best < threshold:
            st.success("✅ Below threshold — eligible to register.")
            if apps_script_url.strip():
                if st.button("Register this title"):
                    try:
                        code = register_title(apps_script_url.strip(), user_title.strip())
                        if code == 200:
                            st.success("Submitted to Submissions sheet.")
                        else:
                            st.error(f"Apps Script error (status {code}).")
                    except Exception as e:
                        st.error(f"Submit failed: {e}")
            else:
                st.info("Add your Apps Script URL in the sidebar to enable registration.")
        else:
            st.warning("⚠️ Above threshold — possible overlap. Refine your title.")

st.markdown("---")
st.caption("MVP: Title-only. Hybrid = TF-IDF prefilter → SBERT re-rank. IDs/metadata can be added later.")
