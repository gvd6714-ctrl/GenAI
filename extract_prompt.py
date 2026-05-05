from langchain_core.prompts import ChatPromptTemplate

extract_prompt = ChatPromptTemplate.from_messages([
    ("system", "You extract key skills from resumes."),
    ("user", """
Resume:
{resume}

Job:
{job_description}

Extract skills, tools, technologies.
""")
])