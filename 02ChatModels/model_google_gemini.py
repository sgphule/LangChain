from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=1.9)
result = model.invoke("Could you help me writing 6 line poem on AI?")
print(result.content)
