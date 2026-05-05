from langchain_core.prompts import ChatPromptTemplate

explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert resume evaluator who explains decisions clearly."),
    ("user", """
Explain why this candidate received the score.

Score:
{score}

Matching Details:
{match_data}

Provide a short, clear, and professional explanation.
""")
])