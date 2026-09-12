# Use Class-based (Tool(...)) when:
# Dynamic runtime creation: Generating tools programmatically on the fly (e.g., parsing OpenAPI specs, database schemas, or JSON configs).
# Wrapping third-party functions: Exposing library code or legacy functions as tools without modifying their original source code.
# Custom metadata overrides: Providing an LLM with a specialized name or description that differs from the actual underlying Python function name.
# Building tool factories: Writing generic adapter pipelines, plugin engines, or multi-tenant agent systems.

from typing import Callable, Any, List, Tuple


class Tool:
    """A class representing a reusable piece of code (Tool)."""

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        arguments: List[Tuple[str, str]],
        outputs: str,
    ):
        self.name = name
        self.description = description
        self.func = func
        self.arguments = arguments
        self.outputs = outputs

    def to_string(self) -> str:
        """Return a string representation of the tool for system prompts."""
        args_str = ", ".join(
            [f"{arg_name}: {arg_type}" for arg_name, arg_type in self.arguments]
        )
        return (
            f"Tool Name: {self.name} | "
            f"Description: {self.description} | "
            f"Arguments: ({args_str}) | "
            f"Outputs: {self.outputs}"
        )

    def __call__(self, *args, **kwargs) -> Any:
        """Invoke the underlying function."""
        return self.func(*args, **kwargs)


# 1. Define standard python function
def multiply_func(a: int, b: int) -> int:
    return a * b


# 2. Manually instantiate the Tool class
calculator_tool = Tool(
    name="calculator",
    description="Multiply two integers together.",
    func=multiply_func,
    arguments=[("a", "int"), ("b", "int")],
    outputs="int",
)

# 3. Format into System Instruction
SYSTEM_INSTRUCTION = f"""You are an AI assistant equipped with tools.

[AVAILABLE TOOLS]
{calculator_tool.to_string()}
"""

if __name__ == "__main__":
    print("=== SYSTEM INSTRUCTION ===")
    print(SYSTEM_INSTRUCTION)

    print("=== TOOL INVOCATION ===")
    print(f"Tool Schema: {calculator_tool.to_string()}")
    result = calculator_tool(a=6, b=7)
    print(f"Execution Result: {result}")