from langchain_core.prompts import ChatPromptTemplate

match_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert resume-job matching system."),
    ("user", """
Compare the resume with the job description.

Job Description:
{job_description}

Resume:
{resume}

Identify:
- matched_skills
- missing_skills

Return output in JSON format:
{
    "matched_skills": [],
    "missing_skills": []
}
""")
])