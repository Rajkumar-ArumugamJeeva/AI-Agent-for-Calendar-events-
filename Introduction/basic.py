
from pydantic import BaseModel, Field
from ollama import chat
import json
from typing import List
import requests
class CalendarEvent(BaseModel):
    event_name: str
    date: str
    participants: List[str]

def get_weather(latitude, longitude):
    """This is a publically available API that returns the weather for a given location."""
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    data = response.json()
    return data["current"]

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for provided coordinates in celsius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
                
            },
            
        },
    },
]

system_prompt = "You are a helpful weather assistant."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What's the weather like in Ottawa today?"},
]

completion =  chat(
    #
    # Required parameters
    #
    messages=messages,
    tools=tools,

    # The language model which will generate the completion.
    model="llama3.2:3b",
    # format=CalendarEvent.model_json_schema(),

)
# print(completion.model_dump())




def call_function(name, args):
    if name == "get_weather":
        return get_weather(**args)
# print(completion.message())

if "tool_calls" in completion.message:
    for tool_call in completion.message["tool_calls"]:
        name = tool_call["function"]["name"]
        args = tool_call["function"]["arguments"] 
        result = call_function(name, args)
        messages.append(
            {"role": "tool", "tool_call_id": tool_call.get("id", ""), "content": json.dumps(result)}
        )

class WeatherResponse(BaseModel):
    temperature: float = Field(
        description="The current temperature in celsius for the given location."
    )
    response: str = Field(
        description="A natural language response to the user's question."
    )

completion2 =  chat(
    #
    # Required parameters
    #
    messages=messages,
    tools=tools,

    # The language model which will generate the completion.
    model="llama3.2:3b",
    # response_format=WeatherResponse,
    format=WeatherResponse.model_json_schema(),

)


response_json = json.loads(completion2.message["content"])  # Ensure response is in JSON format
final_response = WeatherResponse(**response_json)
print(final_response.response)








