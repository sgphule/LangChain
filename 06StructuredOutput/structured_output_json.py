from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field
from pydantic_core.core_schema import json_schema

load_dotenv()

model1 = ChatOpenAI() #gpt‑3.5‑turbo
model2 = ChatOpenAI(model="gpt-4.1-mini")
model3 = ChatOpenAI(model="gpt-4o-mini")

json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": { "type": "string" },
      "description": "list all key themes mentioned in the review"
    },
    "summary": {
      "type": "string",
      "description": "Brief Summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["POS", "NEG", "MIX"],
      "description": "Sentiment of the review either positive, negative or mixed"
    },
    "pros": {
      "type": ["array", "null"],
      "items": { "type": "string" },
      "description": "List down all the pros mentioned"
    },
    "cons": {
      "type": ["array", "null"],
      "items": { "type": "string" },
      "description": "List down all the cons mentioned"
    },
    "author": {
      "type": ["string", "null"],
      "description": "Name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}


structured_model1 = model1.with_structured_output(json_schema, method="function_calling")
structured_model2 = model2.with_structured_output(json_schema)
structured_model3 = model3.with_structured_output(json_schema)


result1 = structured_model1.invoke("""A well‑balanced smartphone that delivers smooth performance, a sharp display, and 
                       reliable battery life. The camera system captures detailed photos in most lighting conditions, 
                       though low‑light shots could be stronger. Build quality feels premium, and the software 
                       experience is clean and responsive. While not the absolute fastest device in its class, it 
                       offers excellent value for everyday users.""")

print(result1)
print(type(result1))
print(result1["summary"])
print(result1["sentiment"])

result2 = structured_model2.invoke("""The smartphone feels underwhelming, with sluggish performance that becomes 
                    noticeable during everyday tasks. Its display lacks brightness and struggles outdoors, while the 
                    battery drains faster than expected. The camera delivers inconsistent results, especially in low 
                    light, and the build quality doesn’t inspire confidence. Overall, it falls short of what you’d 
                    expect at its price point.""")

print(result2)
print(type(result2))
print(result2["sentiment"])

result3 = structured_model3.invoke("""The smartphone struggles to stand out, offering an experience that feels 
                    inconsistent for its price. Performance dips appear during multitasking, and the camera system 
                    delivers mixed results, especially in challenging lighting. While the design is modern, the overall 
                    package leaves room for improvement.
                    Pros
                    Sleek and modern design
                    Comfortable in‑hand feel
                    Clean software interface
                    Decent display quality for everyday use
                    Cons
                    Noticeable performance slowdowns during multitasking
                    Battery life drains faster than expected
                    Camera quality is inconsistent, especially in low light
                    Build quality feels less premium than competitors""")

print(result3)
print(type(result3))
print(result3["sentiment"])





