from langchain_core.prompts import ChatPromptTemplate

score_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an ATS resume scoring system."),
    ("user", """
Based on the matching result below:

{match_data}

Assign a score between 0 and 100.

Rules:
- More matched skills = higher score
- More missing skills = lower score
- Be strict and realistic

Return ONLY the numeric score.
""")
])