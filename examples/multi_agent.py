"""
Multi-agent system (smolagents + local Ollama):
Manager agent -> Web agent (search + travel-time tool) -> Plotly map.

Execution flow
1. The manager makes an initial plan.
2. It calls web_agent for filming locations.
3. The worker searches, visits pages, and returns a structured final_answer.
4. The manager repeats this for supercar factories.
5. It computes the travel times, builds a DataFrame, and plots the map.
6. write_image saves the PNG, check_plot validates it, and final_answer(fig) ends the run.

"""
import math
import os
from typing import Optional, Tuple

from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    LiteLLMModel,
    VisitWebpageTool,
    tool,
)

# ---------------------------------------------------------------- tool
@tool
def calculate_cargo_travel_time(
    origin_coords: Tuple[float, float],
    destination_coords: Tuple[float, float],
    cruising_speed_kmh: Optional[float] = 750.0,
) -> float:
    """
    Calculate the travel time for a cargo plane between two points on Earth using great-circle distance.

    Args:
        origin_coords: Tuple of (latitude, longitude) for the starting point
        destination_coords: Tuple of (latitude, longitude) for the destination
        cruising_speed_kmh: Optional cruising speed in km/h (defaults to 750 km/h for typical cargo planes)

    Returns:
        float: The estimated travel time in hours
    """
    lat1, lon1 = map(math.radians, origin_coords)
    lat2, lon2 = map(math.radians, destination_coords)

    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    distance = 6371.0 * 2 * math.asin(math.sqrt(a))  # haversine, km

    actual_distance = distance * 1.1  # +10% for non-direct routes / ATC
    return round(actual_distance / cruising_speed_kmh + 1.0, 2)  # +1h takeoff/landing


# -------------------------------------------------------------- models
API_BASE = "http://127.0.0.1:11434"


def make_model(model_id: str) -> LiteLLMModel:
    return LiteLLMModel(
        model_id=model_id,
        api_base=API_BASE,
        num_ctx=8192,
    )


worker_model = make_model("ollama_chat/qwen2.5-coder:7b")
# Optional stronger manager: swap in "ollama_chat/qwen2.5-coder:14b"
manager_model = make_model("ollama_chat/qwen2.5-coder:14b")

# -------------------------------------------------------------- agents
web_agent = CodeAgent(
    model=worker_model,
    tools=[
        DuckDuckGoSearchTool(max_results=5),
        VisitWebpageTool(max_output_length=3000),  # keep pages small for 8K ctx
        calculate_cargo_travel_time,
    ],
    name="web_agent",
    instructions=(
        "Use ONLY the provided tools (web_search, visit_webpage). Never import requests, "
        "bs4 or any other library. Never invent data: only report locations and coordinates "
        "found in search results or visited pages. Keep final_answer short."
    ),
    executor_kwargs={"timeout_seconds": 60},
    description=(
        "Browses the web to find information. Give it one focused question at a time "
        "(e.g. a list of locations with latitude/longitude). Returns concise text."
    ),
    verbosity_level=1,
    max_steps=10,
)


def check_plot(final_answer, agent_memory) -> bool:
    """Local final-answer check: the map image must exist and be non-empty."""
    path = "saved_map.png"
    assert os.path.exists(path) and os.path.getsize(path) > 0, (
        "Plot missing: save it with fig.write_image('saved_map.png') before final_answer."
    )
    return True


manager_agent = CodeAgent(
    model=manager_model,
    tools=[calculate_cargo_travel_time],
    managed_agents=[web_agent],
    additional_authorized_imports=[
        "geopandas",
        "plotly",
        "plotly.express",
        "shapely",
        "json",
        "pandas",
        "numpy",
    ],
    planning_interval=5,
    # Default is 30s; a managed agent call runs inside this executor and needs far longer.
    executor_kwargs={"timeout_seconds": 1800},
    verbosity_level=2,
    final_answer_checks=[check_plot],
    max_steps=15,
)

# ----------------------------------------------------------------- run
TASK = """
Find all Batman filming locations in the world, calculate the time to transfer via cargo plane to here (we're in Gotham, 40.7128° N, 74.0060° W).
Also give me some supercar factories with the same cargo plane transfer time. You need at least 6 points in total.
Represent this as spatial map of the world, with the locations represented as scatter points with a color that depends on the travel time, and save it to saved_map.png!

Here's an example of how to plot and return a map:
import plotly.express as px
df = px.data.carshare()
fig = px.scatter_map(df, lat="centroid_lat", lon="centroid_lon", text="name", color="peak_hour", size=100,
     color_continuous_scale=px.colors.sequential.Magma, size_max=15, zoom=1)
fig.show()
fig.write_image("saved_map.png")
final_answer(fig)

Never try to process strings using code: when you have a string to read, just print it and you'll see it.
"""

if __name__ == "__main__":
    manager_agent.visualize()
    result = manager_agent.run(TASK)
    print(result)
    # Inspect the figure from agent state:
    fig = manager_agent.python_executor.state.get("fig")
    if fig is not None:
        fig.write_html("saved_map.html")
        print("Saved saved_map.png and saved_map.html")