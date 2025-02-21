
import os

from ollama import chat
import json
from typing import List
import requests
from pydantic import BaseModel, Field

def search_kb(question: str):
    """
    Load the whole knowledge base from the JSON file.
    (This is a mock function for demonstration purposes, we don't search)
    """
    with open("kb.json", "r") as f:
        return json.load(f)
    
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Get the answer to the user's question from the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                },
                "required": ["question"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

   
system_prompt = "You are a helpful assistant that answers questions from the knowledge base about our e-commerce store."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the return policy?"},
]

completion = chat(
    model="llama3.2:3b",
    messages=messages,
    tools=tools,
)

def call_function(name, args):
    if name == "search_kb":
        return search_kb(**args)
    
if "tool_calls" in completion.message:
    for tool_call in completion.message["tool_calls"]:
        name = tool_call["function"]["name"]
        args = tool_call["function"]["arguments"] 
        result = call_function(name, args)
        messages.append(
            {"role": "tool", "tool_call_id": tool_call.get("id", ""), "content": json.dumps(result)}
        )

class KBResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question.")
    source: int = Field(description="The record id of the answer.")

completion2 = chat(
    model="llama3.2:3b",
    messages=messages,
    tools=tools,
    format=KBResponse.model_json_schema(),
)

response_json = json.loads(completion2.message["content"])  # Ensure response is in JSON format
final_response = KBResponse(**response_json)
print(final_response.answer)