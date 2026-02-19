from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = "Raja"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=7, description="A Decimal value represents the CGPA of student")


new_student1 = {"age": '144',"email":"dfgdf@gmail.com", "cgpa":9.0}

student1 = Student(**new_student1)
print(student1)
print(type(student1.age))

new_student2 = {"age": '194',"email":"aks@gmail.com"}

student2 = Student(**new_student2)
print(student2)
print(type(student2.age))
print("#####################")
student2_dict = dict(student2)
print(student2_dict)
student1_json = student1.model_dump_json()
print(student1_json)
