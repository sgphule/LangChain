from langchain_core.prompts import PromptTemplate

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

template.save("template.json")