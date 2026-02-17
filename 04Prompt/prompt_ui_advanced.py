from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt

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

template = load_prompt("template.json")

if st.button("Summarize"):
    chain = template | model
    result = chain.invoke({
        "papers": papers,
        "styles": styles,
        "length": length
    })
    st.write(result.content)
