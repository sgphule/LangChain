from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

text = """
The rise of autonomous AI agents is reshaping how we think about digital workflows. Instead of relying on a single model to handle every step, modern systems distribute tasks across specialized components that plan, reason, and execute in coordination. This shift mirrors how teams operate in the real world, where collaboration and role‑based expertise lead to better outcomes. As these agentic systems mature, they’re opening new possibilities for research, automation, and intelligent decision‑making.
Soccer has a way of pulling people in with its rhythm, emotion, and unpredictability. A single match can shift from calm possession to explosive counterattacks in seconds, keeping fans on edge throughout. What makes the sport so captivating isn’t just the goals but the stories unfolding behind every pass — the teamwork, the strategy, the individual brilliance, and the sheer determination players bring to the pitch. Whether it’s a local match on a dusty field or a packed stadium roaring during a championship final, soccer creates a shared energy that connects people across cultures and generations.
Outside the world of AI, industries are undergoing their own transformations driven by data and connectivity. Manufacturing, for instance, is moving toward highly adaptive production lines where machines adjust in real time based on sensor feedback. This evolution isn’t just about efficiency — it’s about creating environments that can learn, anticipate issues, and optimize themselves without constant human intervention. The result is a more resilient and responsive industrial ecosystem.
Meanwhile, on the human side of technology, the demand for intuitive tools continues to grow. People expect systems that understand context, adapt to their needs, and reduce cognitive load rather than add to it. Whether it’s a student researching complex topics or a professional managing multiple projects, the goal is the same: technology that feels like a partner rather than a barrier. As AI becomes more capable, the challenge is ensuring it remains accessible, trustworthy, and aligned with real human goals.
"""

docs = text_splitter.create_documents([text])
print(len(docs))
print(docs)
print("--------------------1st doc--------------------")
print(docs[0].page_content)
print("--------------------2nd doc--------------------")
print(docs[1].page_content)
print("--------------------3rd doc--------------------")
print(docs[2].page_content)
print("--------------------4th doc--------------------")
print(docs[3].page_content)

#Still experimental. Doesn't work smoothly at the moment.