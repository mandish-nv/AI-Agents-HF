import json
import re
import requests
from datetime import datetime
from smolagents import ChatMessage, LiteLLMModel
from datetime import datetime
import urllib.parse
from zoneinfo import ZoneInfo

# 1. Define System Prompt Template
SYSTEM_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

1. get_weather:
   Description: Get the real-time weather information for a specified city or location.
   Parameters: {"location": {"type": "string", "description": "Clean city name or location without special characters. E.g., 'London', 'New York', 'Kathmandu'"}}

2. get_current_datetime:
   Description: Get current local date and time. Convert city names to their corresponding IANA timezone string in 'Continent/City' format before invoking (e.g., convert 'Kathmandu' -> 'Asia/Kathmandu', 'New York' -> 'America/New_York').
   Parameters: {"location": {"type": "string", "description": "Standard IANA timezone string (e.g., 'Asia/Kathmandu', 'Europe/London', 'America/New_York')"}}

Instructions:
- Always call BOTH tools if the user asks for both weather and local time. Do not skip tool calls or assume one city's weather applies to another.

The way you use the tools is by specifying a JSON blob inside a markdown code block.
Specifically, this JSON must have an `action` key (the tool name) and an `action_input` key (a dictionary of tool arguments).

Example tool usage:

```json
{
  "action": "get_weather",
  "action_input": {"location": "New York"}
}
```

```json
{
  "action": "get_current_datetime",
  "action_input": {"location": "America/New_York"}
}
```

ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about one action to take. Only one action at a time in this format:
Action:
```json
$JSON_BLOB
```
Observation: the result of the action. This Observation is unique, complete, and the source of truth.
(this Thought/Action/Observation can repeat N times. You should take several steps when needed. Send only a SINGLE action at a time.)

You must always end your output with the following format:

Thought: I now know the final answer
Final Answer: the final answer to the original input question

Now begin! Reminder to ALWAYS use the exact characters `Final Answer:` when you provide a definitive answer."""


# 2. Define Tools
def get_weather(location: str) -> str:
    """Fetch real-time weather information for a given location using wttr.in weather service."""
    try:
        url = f"https://wttr.in/{location}?format=%C,+%t"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return f"{location}: {response.text.strip()}"
        return f"Error: Unable to fetch weather data for '{location}' (HTTP {response.status_code})."
    except Exception as e:
        return f"Error fetching weather data for '{location}': {str(e)}"


def get_current_datetime(location: str) -> str:
    """Get the current local date and time dynamically for any city or timezone."""
    try:
        # Step 1: Check if input is already a valid IANA timezone (e.g. 'Asia/Kathmandu')
        try:
            tz = ZoneInfo(location)
            now = datetime.now(tz)
            return f"{location}: {now.strftime('%Y-%m-%d %H:%M:%S')}"
        except Exception:
            pass

        # Step 2: Dynamically look up timezone via Open-Meteo Free Geocoding API
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


TOOL_REGISTRY = {
    "get_weather": get_weather,
    "get_current_datetime": get_current_datetime,
}


# 3. Helper to Extract JSON Action from Markdown
def parse_action_json(llm_output: str) -> dict | None:
    """Extract and parse the JSON block inside markdown backticks."""
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", llm_output, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            return None
    return None


# 4. ReAct Execution Loop
def run_react_agent(question: str, max_steps: int = 5):
    # Initialize LiteLLMModel
    model = LiteLLMModel(
        model_id="ollama_chat/qwen2.5-coder:7b",
        api_base="http://127.0.0.1:11434",
        num_ctx=8192,
    )

    prompt_history = f"{SYSTEM_PROMPT}\n\nQuestion: {question}\nThought:"
    print(f"=== User Question: {question} ===\n")

    for step in range(max_steps):
        # Format input using ChatMessage object
        message = ChatMessage(
            role="user",
            content=[{"type": "text", "text": prompt_history}],
        )

        # Call model with stop sequence on Observation:
        response = model([message], stop_sequences=["Observation:"])
        llm_response = response.content

        print(f"Thought:{llm_response}")

        # Check for Final Answer termination condition
        if "Final Answer:" in llm_response:
            final_answer = llm_response.split("Final Answer:")[-1].strip()
            print(f"\n[AGENT FINISHED]\nFinal Answer: {final_answer}")
            return final_answer

        # Parse Action JSON
        action_data = parse_action_json(llm_response)
        if not action_data or "action" not in action_data:
            print("\n[ERROR] Failed to extract valid JSON action block. Stopping.")
            break

        action_name = action_data.get("action")
        action_input = action_data.get("action_input", {})

        # Execute Tool Call
        if action_name in TOOL_REGISTRY:
            tool_func = TOOL_REGISTRY[action_name]
            observation = (
                tool_func(**action_input)
                if isinstance(action_input, dict)
                else tool_func(action_input)
            )
        else:
            observation = f"Error: Tool '{action_name}' is not registered."

        print(f"Observation: {observation}\n")

        # Update prompt history for next iteration
        prompt_history += f"{llm_response}\nObservation: {observation}\nThought:"

    print("[AGENT FAILED] Exceeded maximum step limit.")


if __name__ == "__main__":
    run_react_agent("What is the current local time, and what is the weather in Kathmandu?")