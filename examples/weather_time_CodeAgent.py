import requests
import urllib.parse
from datetime import datetime
from zoneinfo import ZoneInfo
from smolagents import CodeAgent, LiteLLMModel, tool

@tool
def get_weather(location: str) -> str:
    """Get the real-time weather information for a specified city or location.

    Args:
        location: Clean city name or location without special characters. E.g., 'London', 'New York', 'Kathmandu'.
    """
    try:
        url = f"https://wttr.in/{location}?format=%C,+%t"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return f"{location}: {response.text.strip()}"
        return f"Error: Unable to fetch weather data for '{location}' (HTTP {response.status_code})."
    except Exception as e:
        return f"Error fetching weather data for '{location}': {str(e)}"

@tool
def get_current_datetime(location: str) -> str:
    """Get current local date and time for a location.

    Args:
        location: City name or IANA timezone string (e.g., 'Asia/Kathmandu', 'Europe/London', 'America/New_York'). If a city name is given, convert it to IANA timezone format before invoking (e.g., 'Kathmandu' -> 'Asia/Kathmandu').
    """
    try:
        try:
            tz = ZoneInfo(location)
            now = datetime.now(tz)
            return f"{location}: {now.strftime('%Y-%m-%d %H:%M:%S')}"
        except Exception:
            pass

        safe_loc = urllib.parse.quote(location)
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={safe_loc}&count=1&format=json"
        res = requests.get(geo_url, timeout=5)

        if res.status_code == 200 and res.json().get("results"):
            timezone_str = res.json()["results"][0]["timezone"]
            tz = ZoneInfo(timezone_str)
            now = datetime.now(tz)
            return f"{location} ({timezone_str}): {now.strftime('%Y-%m-%d %H:%M:%S')}"

        return f"Error: Unable to locate timezone for '{location}'."
    except Exception as e:
        return f"Error fetching date/time for '{location}': {str(e)}"


CUSTOM_INSTRUCTIONS = """
Additional rules:
- Convert city names to their corresponding IANA timezone string in 'Continent/City' format before calling get_current_datetime (e.g., 'Kathmandu' -> 'Asia/Kathmandu').
- Keep the final answer concise and combine both results into a single sentence.
"""

def run_code_agent(question: str, max_steps: int = 6):
    model = LiteLLMModel(
        model_id="ollama_chat/qwen2.5-coder:14b",
        api_base="http://127.0.0.1:11434",
        num_ctx=8192,
    )

    # Build a default agent first to obtain its built-in prompt_templates dict
    base_agent = CodeAgent(
        model=model,
        tools=[get_weather, get_current_datetime],
        add_base_tools=False,
    )
    prompt_templates = base_agent.prompt_templates
    prompt_templates["system_prompt"] += CUSTOM_INSTRUCTIONS

    agent = CodeAgent(
        model=model,
        tools=[get_weather, get_current_datetime],
        max_steps=max_steps,
        verbosity_level=1,
        planning_interval=None,
        name=None,
        description=None,
        add_base_tools=False,
        prompt_templates=prompt_templates,
    )

    result = agent.run(question)
    print(f"\n[AGENT FINISHED]\nFinal Answer: {result}")
    return result

if __name__ == "__main__":
    run_code_agent("What is the current local time, and what is the weather in Kathmandu?")
