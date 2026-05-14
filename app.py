import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re

# Download nltk data
nltk.download('punkt')

# -----------------------------
# Function to extract text from PDF
# -----------------------------


def extract_text_from_pdf(uploaded_file):
    text = ""

    pdf_reader = PdfReader(uploaded_file)

    for page in pdf_reader.pages:
        extracted = page.extract_text()

        if extracted:
            text += extracted

    return text


# -----------------------------
# Function to clean text
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text


# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("📄 AI Resume Screener")
st.write("Upload a resume and compare it with a job description.")

# -----------------------------
# Resume Upload
# -----------------------------
uploaded_resume = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)

# -----------------------------
# Job Description Input
# -----------------------------
job_description = st.text_area(
    "Paste Job Description Here"
)

# -----------------------------
# Skills List
# -----------------------------
skills = [
    "python",
    "java",
    "html",
    "css",
    "javascript",
    "sql",
    "machine learning",
    "data analysis",
    "django",
    "flask",
    "react",
    "c++",
    "git",
    "mongodb"
]

# -----------------------------
# Analyze Button
# -----------------------------
if st.button("Analyze Resume"):

    if uploaded_resume is not None and job_description != "":

        # Extract resume text
        resume_text = extract_text_from_pdf(uploaded_resume)

        # Clean texts
        resume_clean = clean_text(resume_text)
        jd_clean = clean_text(job_description)

        # NLP Similarity
        documents = [resume_clean, jd_clean]

        cv = CountVectorizer()
        matrix = cv.fit_transform(documents)

        similarity_score = cosine_similarity(matrix)[0][1]

        match_percentage = round(similarity_score * 100, 2)

        # -----------------------------
        # Detect Skills
        # -----------------------------
        detected_skills = []

        for skill in skills:
            if skill.lower() in resume_clean:
                detected_skills.append(skill)

        # -----------------------------
        # Results Section
        # -----------------------------
        st.subheader("📊 Results")

        st.write(f"### Match Percentage: {match_percentage}%")

        # Match Category
        if match_percentage >= 70:
            st.success("✅ Strong Match for the Job")

        elif match_percentage >= 40:
            st.warning("⚠️ Moderate Match")

        else:
            st.error("❌ Low Match")

        # -----------------------------
        # Skills Section
        # -----------------------------
        st.subheader("🛠 Detected Skills")

        if detected_skills:

            for skill in detected_skills:
                st.write(f"✔ {skill}")

        else:
            st.write("No matching skills found.")

    else:
        st.warning("Please upload a resume and paste a job description.")
