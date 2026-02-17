from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

import streamlit as st

load_dotenv()
model = ChatOpenAI()

st.title("Research Tool")
st.header("AI Research Paper Summarizer")
st.subheader("Generate clear, structured summaries of academic papers")

papers = st.selectbox( "Select Research Paper Name",
                            ["Deep Learning Foundations",
                                    "ImageNet Classification with Deep CNNs (AlexNet)",
                                    "Deep Residual Learning (ResNet)",
                                    "Generative Adversarial Networks (GANs)",
                                    "Denoising Diffusion Probabilistic Models (DDPM)",
                                    "Attention Is All You Need (Transformer)",
                                    "BERT: Pre-training of Deep Bidirectional Transformers",
                                    "GPT‑3: Language Models Are Few-Shot Learners",
                                    "Sequence to Sequence Learning with Neural Networks",
                                    "Neural Machine Translation by Jointly Learning to Align and Translate",
                                    "Playing Atari with Deep Reinforcement Learning (DQN)",
                                    "AlphaGo / AlphaGo Zero",
                                    "YOLO: Real-Time Object Detection",
                                    "Vision Transformers (ViT)",
                                    "CLIP: Learning Transferable Visual Models from Natural Language Supervision",
                                    "DALL·E / Diffusion Models",
                                    "Scalable and Secure AI Inference in Healthcare",] )

styles = st.selectbox(
    "Select Explanation Style",
    [
        "Beginner‑Friendly — Simple and intuitive",
        "Technical — Precise and academically rigorous",
        "Code‑Oriented — Focused on implementation and examples",
        "Mathematical — Emphasis on formulas and theoretical depth"
    ]
)


length = st.selectbox(
    "Select Explanation Length",
    [
        "Short — 1 to 2 paragraphs",
        "Medium — 3 to 5 paragraphs",
        "Long — Detailed, in‑depth explanation"
    ]
)

template = PromptTemplate(template=""" 
Please summarize the research paper titled "{papers}" using the following guidelines: 
Explanation Style: {styles} Explanation Length: {length} 
Requirements: 1. Mathematical Details: 
- Include key equations or concepts when relevant. 
- Use intuitive analogies to simplify complex mathematical ideas. 
2. Accuracy: - Base the summary strictly on information contained in the paper. 
- If any required detail is missing from the paper, respond with "Insufficient Information" rather than inferring or guessing. 
3. Quality: - Ensure the summary is clear, coherent, and aligned with the requested style and length. 
- Maintain a professional and academically appropriate tone. Produce the final summary accordingly. """,
input_variables=["papers", "styles", "length"],
validate_template=True
)

prompt = template.invoke({
    "papers": papers,
    "styles": styles,
    "length": length
})

if st.button("Summarize"):
    result = model.invoke(prompt)
    st.write(result.content)
