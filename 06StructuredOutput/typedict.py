from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

new_person: Person= {'name': "Sudarshan", 'age': 99}

print(new_person)