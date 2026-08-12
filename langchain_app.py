import os
import json
import requests

from fastapi import FastAPI
from pydantic import BaseModel, Field

from langserve import add_routes

from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from langgraph_app import run_langgraph


# ============================================================
# API KEY
# ============================================================

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not configured.")


# ============================================================
# LANGCHAIN TOOLS
# ============================================================

@tool
def search_movies(genre: str) -> str:
    """Search for Indian movies by genre."""

    movies = {
        "sci-fi": "Cargo, 2.0, Mr. India",
        "comedy": "3 Idiots, Hera Pheri, Munna Bhai M.B.B.S.",
        "action": "RRR, Vikram, Baahubali"
    }

    return movies.get(
        genre.lower(),
        "No movies found for that genre."
    )


@tool
def change_to_f(temp_c: float) -> float:
    """Convert Celsius to Fahrenheit."""

    return temp_c * 1.8 + 32


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""

    geo_url = "https://geocoding-api.open-meteo.com/v1/search"

    geo_response = requests.get(
        geo_url,
        params={
            "name": city,
            "count": 1,
            "language": "en",
            "format": "json"
        },
        timeout=10
    )

    geo_response.raise_for_status()

    data = geo_response.json()

    if "results" not in data:
        return f"Could not find city: {city}"

    location = data["results"][0]

    weather_response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "current": "temperature_2m,weather_code",
            "temperature_unit": "celsius"
        },
        timeout=10
    )

    weather_response.raise_for_status()

    current = weather_response.json()["current"]

    return json.dumps({
        "city": location["name"],
        "country": location.get("country"),
        "temperature_celsius": current["temperature_2m"],
        "weather_code": current["weather_code"]
    })


# ============================================================
# LANGCHAIN AGENT
# ============================================================

tools = [
    get_weather,
    search_movies,
    change_to_f
]

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key=API_KEY,
    temperature=0
)

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are restricted to Indian weather and Indian cinema. "
        "For anything outside these topics, respond exactly: "
        "'I am not authorized to answer questions outside of Indian "
        "weather and cinema.'"
    )
)


# ============================================================
# LANGSERVE
# ============================================================

class AgentInput(BaseModel):
    input: str = Field(
        description="Message for the agent"
    )


def format_input(x):

    return {
        "messages": [
            {
                "role": "user",
                "content": x["input"]
            }
        ]
    }


def extract_response(result):

    if isinstance(result, dict):

        messages = result.get("messages")

        if messages:

            last = messages[-1]

            content = getattr(
                last,
                "content",
                None
            )

            if content is not None:
                return str(content)

    return str(result)


agent_chain = (
    RunnableLambda(format_input)
    | agent
    | RunnableLambda(extract_response)
).with_types(
    input_type=AgentInput,
    output_type=str
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="LangChain + LangGraph AI",
    version="1.0"
)


@app.get("/")
def root():

    return {
        "status": "running",
        "langchain": "/agent/playground/",
        "langgraph": "/langgraph"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# LangServe
add_routes(
    app,
    agent_chain,
    path="/agent"
)


# ============================================================
# LANGGRAPH ENDPOINT
# ============================================================

class LangGraphRequest(BaseModel):
    task: str


@app.post("/langgraph")
def langgraph_endpoint(request: LangGraphRequest):

    result = run_langgraph(
        request.task
    )

    return {
        "result": result
    }


# ============================================================
# RENDER ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "langchain_app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )