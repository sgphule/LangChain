from abc import ABC, abstractmethod
import random

class Runnable(ABC):

    @abstractmethod
    def invoke(self, input_data):
        pass

class FakeLLM(Runnable):
    def __init__(self):
        print("LLM Created")


    def invoke(self, prompt):
        response_list = [
            'Berlin is the Capital of Germany',
            'Ronaldo is a soccer player',
            'RAG stands for Retrieval Augmented Generation'
        ]
        return {"response": random.choice(response_list)}


    def predict(self, prompt):
        response_list = [
            'Berlin is the Capital of Germany',
            'Ronaldo is a soccer player',
            'RAG stands for Retrieval Augmented Generation'
        ]
        print("This \"predict" "method is going to be deprecated")
        return {"response": random.choice(response_list)}


class FakePromptTemplate(Runnable):
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables= input_variables

    def invoke(self, input_dict):
        return self.template.format(**input_dict)

    def format(self, input_dict):
        print("This \"format" "method is going to be deprecated")
        return self.template.format(**input_dict)

class FakeStrOutputParser(Runnable):
    def __init__(self):
        pass


    def invoke(self, input_dict):
        return input_dict["response"]


class RunnableConnector(Runnable):
    def __init__(self, runnable_list):
        self.runnable_list = runnable_list

    def invoke(self, input_data):
        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)
        return input_data

template = FakePromptTemplate(template="Explain in a {length} about {topic}",
                              input_variables=["length", "topic"])

llm = FakeLLM()

parser = FakeStrOutputParser()
chain = RunnableConnector([template, llm, parser])

final_output = chain.invoke({"length":"long", "topic": "Germany"})
print("final_output:", final_output)

template1 = FakePromptTemplate(
    template="write a joke about a {topic}",
    input_variables=["topic"]
)

template2 = FakePromptTemplate(
    template="Explain the following joke {response}",
    input_variables=["response"]
)

chain1 = RunnableConnector([template1, llm])
chain2 = RunnableConnector([template2, llm, parser])
final_chain = RunnableConnector([chain1, chain2])
final_chain.invoke({"topic":"SOCCER"})