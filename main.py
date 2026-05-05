from dotenv import load_dotenv
import os

load_dotenv()

from langchain_openai import ChatOpenAI

# Import chains (your current structure)
from chains.extract_chain import get_extract_chain
from chains.match_chain import get_match_chain
from chains.score_chain import get_score_chain
from chains.explain_chain import get_explain_chain


# Load API key (safety check)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")


# LLM (use modern model)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    api_key=api_key
)


# Build chains
extract_chain = get_extract_chain(llm)
match_chain = get_match_chain(llm)
score_chain = get_score_chain(llm)
explain_chain = get_explain_chain(llm)


# Load file helper
def load_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# Run full pipeline
def run_pipeline(resume, job_description):

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

    return extract, match, score, explain


# MAIN EXECUTION
if __name__ == "__main__":

    job_description = load_file("data/job_desc.txt")

    resumes = {
        "AVG": load_file("data/resume_avg.txt"),
        "STRONG": load_file("data/resume_strong.txt"),
        "WEAK": load_file("data/resume_weak.txt"),
    }

    print("\n===== RESUME SCREENING SYSTEM =====\n")

    for name, resume in resumes.items():

        print(f"\n========== {name} RESUME ==========\n")

        extract, match, score, explain = run_pipeline(resume, job_description)

        print("\n--- EXTRACT ---")
        print(extract)

        print("\n--- MATCH ---")
        print(match)

        print("\n--- SCORE ---")
        print(score)

        print("\n--- EXPLANATION ---")
        print(explain)

        print("\n---------------------------------\n")