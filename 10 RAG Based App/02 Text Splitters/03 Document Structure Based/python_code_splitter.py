from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade

    def get_name(self):
        return self.name

    def display_details(self):
        print("Student Details:")
        print("Name:", self.name)
        print("Age:", self.age)
        print("Grade:", self.grade)


student1 = Student("Sudarshan", 25, "A+")

print("Getter Method Output:", student1.get_name())

student1.display_details()
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size = 200,
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
