import os
import json
import requests
 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langserve import add_routes

from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

# Import the LangGraph workflow from the other file
from langgraph_app import run_langgraph


# ============================================================
# 1. GEMINI API KEY
# ============================================================

GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not configured."
    )


# ============================================================
# 2. LANGCHAIN TOOLS
# ============================================================

@tool
def search_movies(genre: str) -> str:
    """Search for Indian movies by genre."""

    movies = {
        "sci-fi": "Cargo, 2.0, Mr. India",
        "science fiction": "Cargo, 2.0, Mr. India",
        "comedy": "3 Idiots, Hera Pheri, Munna Bhai M.B.B.S.",
        "action": "RRR, Vikram, Baahubali",
        "romance": "Jab We Met, Veer-Zaara, Dilwale Dulhania Le Jayenge",
        "thriller": "Andhadhun, Drishyam, Kahaani"
    }

    return movies.get(
        genre.lower().strip(),
        "No movies found for that genre."
    )


@tool
def change_to_f(temp_c: float) -> float:
    """Convert Celsius temperature to Fahrenheit."""

    return round(temp_c * 1.8 + 32, 2)


@tool
def get_weather(city: str) -> str:
    """Get current weather for an Indian city."""

    try:

        # ----------------------------------------------------
        # Geocoding
        # ----------------------------------------------------

        geo_url = (
            "https://geocoding-api.open-meteo.com/v1/search"
        )

        geo_params = {
            "name": city,
            "count": 1,
            "language": "en",
            "format": "json"
        }

        geo_response = requests.get(
            geo_url,
            params=geo_params,
            timeout=10
        )

        geo_response.raise_for_status()

        geo_data = geo_response.json()

        if (
            "results" not in geo_data
            or not geo_data["results"]
        ):
            return (
                f"Could not find weather data "
                f"for city: {city}"
            )

        location = geo_data["results"][0]

        latitude = location["latitude"]
        longitude = location["longitude"]

        # ----------------------------------------------------
        # Weather
        # ----------------------------------------------------

        weather_url = (
            "https://api.open-meteo.com/v1/forecast"
        )

        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,weather_code",
            "temperature_unit": "celsius"
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10
        )

        weather_response.raise_for_status()

        weather_data = weather_response.json()

        current = weather_data["current"]

        result = {
            "resolved_city": location["name"],
            "country": location.get("country"),
            "temperature_celsius": current["temperature_2m"],
            "weather_code": current["weather_code"]
        }

        return json.dumps(result)

    except requests.RequestException as e:

        return f"Weather API error: {str(e)}"

    except Exception as e:

        return f"Unexpected weather error: {str(e)}"


# ============================================================
# 3. TOOLS LIST
# ============================================================

tools = [
    get_weather,
    search_movies,
    change_to_f
]


# ============================================================
# 4. INITIALIZE GEMINI / LANGCHAIN AGENT
# ============================================================

llm_flash = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


agent = create_agent(
    model=llm_flash,
    tools=tools,
    system_prompt=(
        "You are a specialized AI agent restricted ONLY to "
        "Indian weather and Indian cinema.\n\n"

        "You are authorized to answer questions about:\n"
        "1. Weather in Indian cities.\n"
        "2. Indian movies.\n"
        "3. Indian cinema.\n"
        "4. Celsius to Fahrenheit conversion when it is "
        "related to weather.\n\n"

        "For any other topic, question, role, or general "
        "knowledge outside Indian weather and cinema, "
        "you must say exactly:\n\n"

        "I am not authorized to answer questions outside "
        "of Indian weather and cinema."
    )
)


# ============================================================
# 5. LANGSERVE INPUT MODEL
# ============================================================

class AgentInput(BaseModel):
    input: str = Field(
        description="Message to send to the movie and weather agent."
    )


# ============================================================
# 6. FORMAT INPUT FOR AGENT
# ============================================================

def format_for_agent(x):

    if isinstance(x, dict):
        user_input = x["input"]

    else:
        user_input = x.input

    return {
        "messages": [
            {
                "role": "user",
                "content": user_input
            }
        ]
    }


# ============================================================
# 7. EXTRACT FINAL AGENT RESPONSE
# ============================================================

def extract_text_response(agent_output):

    if not isinstance(agent_output, dict):
        return str(agent_output)

    messages = agent_output.get("messages")

    if messages:

        last_message = messages[-1]

        content = getattr(
            last_message,
            "content",
            None
        )

        if content is not None:

            # Gemini content can sometimes be a list
            if isinstance(content, list):

                text_parts = []

                for item in content:

                    if isinstance(item, dict):

                        text_parts.append(
                            item.get("text", "")
                        )

                    else:

                        text_parts.append(
                            str(item)
                        )

                return "\n".join(text_parts)

            return str(content)

    return str(agent_output)


# ============================================================
# 8. CREATE LANGCHAIN CHAIN
# ============================================================

formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(
    input_type=AgentInput,
    output_type=str
)


# ============================================================
# 9. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Indian Movie & Weather AI",
    version="1.0.0",
    description=(
        "LangChain + LangGraph application for "
        "Indian weather, Indian cinema and "
        "Python coding workflows."
    )
)


# ============================================================
# 10. ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "running",
        "application": "Indian Movie & Weather AI",
        "langchain": "/agent/playground/",
        "langgraph": "/langgraph",
        "documentation": "/docs"
    }


# ============================================================
# 11. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# 12. LANGSERVE ROUTE
# ============================================================

add_routes(
    app,
    formatted_agent_chain,
    path="/agent"
)


# ============================================================
# 13. LANGGRAPH REQUEST MODEL
# ============================================================

class LangGraphRequest(BaseModel):

    task: str = Field(
        description="Python coding task for the LangGraph workflow."
    )


# ============================================================
# 14. LANGGRAPH GET ENDPOINT
# ============================================================

@app.get("/langgraph")
def langgraph_info():

    return {
        "status": "available",
        "message": (
            "LangGraph endpoint is available. "
            "Use POST /langgraph to execute a task."
        ),
        "method": "POST",
        "example": {
            "task": (
                "Write a Python program to "
                "calculate factorial"
            )
        }
    }


# ============================================================
# 15. LANGGRAPH POST ENDPOINT
# ============================================================

@app.post("/langgraph")
def langgraph_endpoint(
    request: LangGraphRequest
):

    try:

        print(
            f"[LangGraph] Received task: {request.task}"
        )

        result = run_langgraph(
            request.task
        )

        print(
            "[LangGraph] Execution completed."
        )

        return {
            "success": True,
            "task": request.task,
            "result": result
        }

    except Exception as e:

        print(
            f"[LangGraph] ERROR: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# 16. LOCAL SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get("PORT", 8000)
    )

    uvicorn.run(
        "langchain_app:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
