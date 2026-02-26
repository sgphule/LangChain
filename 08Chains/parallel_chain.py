from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

text = """
YOLOv8 was released by Ultralytics on January 10, 2023, offering cutting-edge performance in terms of accuracy and 
speed. Building upon the advancements of previous YOLO versions, YOLOv8 introduced new features and optimizations that 
make it an ideal choice for various object detection tasks in a wide range of applications.
Key Features of YOLOv8
    Advanced Backbone and Neck Architectures: YOLOv8 employs state-of-the-art backbone and neck architectures, 
    resulting in improved feature extraction and object detection performance.
    Anchor-free Split Ultralytics Head: YOLOv8 adopts an anchor-free split Ultralytics head, which contributes to 
    better accuracy and a more efficient detection process compared to anchor-based approaches.
    Optimized Accuracy-Speed Tradeoff: With a focus on maintaining an optimal balance between accuracy and speed, 
    YOLOv8 is suitable for real-time object detection tasks in diverse application areas.
    Variety of Pretrained Models: YOLOv8 offers a range of pretrained models to cater to various tasks and performance 
    requirements, making it easier to find the right model for your specific use case.
Supported Tasks and Modes
The YOLOv8 series offers a diverse range of models, each specialized for specific tasks in computer vision. These 
models are designed to cater to various requirements, from object detection to more complex tasks like instance 
segmentation, pose/keypoints detection, oriented object detection, and classification.
Each variant of the YOLOv8 series is optimized for its respective task, ensuring high performance and accuracy. 
Additionally, these models are compatible with various operational modes including Inference, Validation, Training, 
and Export, facilitating their use in different stages of deployment and development.
"""
load_dotenv()

model0 = ChatOpenAI()
model1 = ChatAnthropic(model='claude-sonnet-4-5-20250929')

prompt1 = PromptTemplate(
    template = "Generate short and simple notes from following text \n {text}",
    input_variables = ["text"]
)

prompt2 = PromptTemplate(
    template = "Generate 5 questions from following text \n {text}",
    input_variables = ["text"]
)

prompt3 = PromptTemplate(
    template="Merge provided notes and quiz into a single document \n"
             "Do NOT remove, rewrite, shorten, or summarize ANY questions.\n" 
             "Preserve ALL content exactly as provided.\n\n" 
             "NOTES:\n{notes}\n\n" 
             "QUIZ:\n{quiz}",
    input_variables = ["notes", "quiz"]
)

parser = StrOutputParser()
parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | model0 | parser,
        "quiz": prompt2 | model1 | parser
    }
)

merge_chain = prompt3 | model1 | parser
final_chain = parallel_chain | merge_chain
result = final_chain.invoke({"text": text})
print(result)
final_chain.get_graph().print_ascii()