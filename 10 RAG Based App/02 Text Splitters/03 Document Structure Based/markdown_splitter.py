from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """
# 🧠 Autonomous Multi‑Agent AI Framework

Welcome to the Autonomous Multi‑Agent AI Framework — a modular system for building intelligent agents that collaborate to solve complex tasks. This project demonstrates how planning, reasoning, memory, and execution can be distributed across multiple specialized agents, all coordinated through a central orchestrator. It’s designed for researchers, students, and developers exploring modern agentic AI workflows.

---

## 🚀 Key Features

- **Modular agent architecture** with clear roles  
- **Task planning and decomposition** using LLM‑powered reasoning  
- **Shared memory layer** for context persistence  
- **Orchestrator** that coordinates agent communication  
- **Example workflows** for research, automation, and knowledge retrieval  
- **Easily extensible** for custom agents and tools  

---

## ▶️ Getting Started

Install dependencies and run the demo:

```bash
pip install -r requirements.txt
python examples/multi_agent_demo.py
```
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size = 500,
    chunk_overlap = 0
)

chunks = splitter.split_text(text)
print(len(chunks))
print("--------------------1st Chunk--------------------")
print(chunks[0])
print("--------------------2nd Chunk--------------------")
print(chunks[1])
print("--------------------3rd Chunk--------------------")
print(chunks[2])
