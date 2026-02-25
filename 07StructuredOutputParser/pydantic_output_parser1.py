from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# 1. Define your structured output using Pydantic
class Facts(BaseModel):
    fact_1: str = Field(description="Fact 1 about the topic")
    fact_2: str = Field(description="Fact 2 about the topic")
    fact_3: str = Field(description="Fact 3 about the topic")

parser = PydanticOutputParser(pydantic_object=Facts)

# 2. LLM setup
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# 3. Prompt template with format instructions
template = PromptTemplate(
    template=(
        "Generate exactly 3 facts about {topic}.\n"
        "You MUST respond in valid JSON only.\n"
        "{format_instructions}\n"
        "Do not add explanations, commentary, or extra text."
    ),
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 4. LCEL chain
chain = template | model | parser

# 5. Invoke
result = chain.invoke({"topic": "Quantum computing"})
print(result.fact_1)
print(result.fact_2)
print(result.fact_3)

