import random


class FakeLLM:
    def __init__(self):
        print("LLM Created")


    def predict(self, prompt):
        response_list = [
            'Berlin is the Capital of Germany',
            'Ronaldo is a soccer player',
            'RAG stands for Retrieval Augmented Generation'
        ]
        return {"response": random.choice(response_list)}


class FakePromptTemplate:
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables= input_variables


    def format(self, input_dict):
        return self.template.format(**input_dict)


class FakeLLMChain:
    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

    def run(self, input_dict):
        final_prompt = self.prompt.format(input_dict)
        result = self.llm.predict(final_prompt)
        print("result2:", result)
        return result["response"]


llm = FakeLLM()
#result = llm.predict("Who is Ronaldo?")
#print(result)

template = FakePromptTemplate(template="Explain in a {length} about {topic}",
                              input_variables=["topic"])

prompt = template.format({"length": "short", "topic": "RAG"})
print("prompt:", prompt)
result = llm.predict(prompt)
print("result1", result)

chain = FakeLLMChain(llm, template)

chain_output = chain.run({"length" : "short", "topic" : "RAG"})

print("Chain result:", chain_output)
print("Here for FakeLLM we need call to PREDICT while for FakePromptTemplate need call to FORMAT")
print("Let's standardize this for creating flexible chains")
print("To make multiple call on fake_llm")

