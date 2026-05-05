import streamlit as st
from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI

from chains.extract_chain import get_extract_chain
from chains.match_chain import get_match_chain
from chains.score_chain import get_score_chain
from chains.explain_chain import get_explain_chain


# Load environment
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found in .env file")
    st.stop()


# LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    api_key=api_key
)

# Chains
extract_chain = get_extract_chain(llm)
match_chain = get_match_chain(llm)
score_chain = get_score_chain(llm)
explain_chain = get_explain_chain(llm)


# Load file helper
def load_file(uploaded_file):
    if uploaded_file is not None:
        return uploaded_file.read().decode("utf-8")
    return ""


# UI
st.set_page_config(page_title="Resume Screening AI", layout="centered")

st.title("📄 AI Resume Screening System")
st.write("Upload Resume + Job Description and get AI evaluation")

# Upload files
resume_file = st.file_uploader("Upload Resume (.txt)", type=["txt"])
job_file = st.file_uploader("Upload Job Description (.txt)", type=["txt"])

if st.button("Run Analysis"):

    if resume_file is None or job_file is None:
        st.warning("Please upload both Resume and Job Description")
        st.stop()

    resume = load_file(resume_file)
    job_description = load_file(job_file)

    with st.spinner("Analyzing resume..."):

        extract = extract_chain.invoke({
            "resume": resume,
            "job_description": job_description
        }).content

        match = match_chain.invoke({
            "resume": resume,
            "job_description": job_description
        }).content

        score = score_chain.invoke({
            "match_data": match
        }).content

        explain = explain_chain.invoke({
            "score": score,
            "match_data": match
        }).content


    # OUTPUT UI
    st.subheader("📊 Extracted Skills")
    st.write(extract)

    st.subheader("🔍 Match Analysis")
    st.write(match)

    st.subheader("⭐ Score")
    st.success(score)

    st.subheader("🧠 Explanation")
    st.write(explain)