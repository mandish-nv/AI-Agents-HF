# Use Decorator (@tool) when:
# Hand-crafting native code: You are writing custom Python functions specifically designed for an LLM agent.
# Single source of truth: You want the function's docstring and type hints to automatically drive the schema without duplicate documentation.
# Prioritizing clean syntax: You want readable, Pythonic code with minimal boilerplate.

import inspect
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


def tool(func: Callable) -> Tool:
    """
    Decorator that inspects function signature and docstrings to build a Tool.
    """
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or "No description provided."

    arguments = []
    for param_name, param in sig.parameters.items():
        param_type = (
            param.annotation.__name__
            if param.annotation != inspect.Parameter.empty
            else "Any"
        )
        arguments.append((param_name, param_type))

    return_type = (
        sig.return_annotation.__name__
        if sig.return_annotation != inspect.Signature.empty
        else "Any"
    )

    return Tool(
        name=func.__name__,
        description=doc,
        func=func,
        arguments=arguments,
        outputs=return_type,
    )


# 1. Define tool using decorator
@tool
def calculator(a: int, b: int) -> int:
    """Multiply two integers together."""
    return a * b


# 2. Format into System Instruction
SYSTEM_INSTRUCTION = f"""You are an AI assistant equipped with tools.

[AVAILABLE TOOLS]
{calculator.to_string()}
"""

if __name__ == "__main__":
    print("=== SYSTEM INSTRUCTION ===")
    print(SYSTEM_INSTRUCTION)

    print("=== TOOL INVOCATION ===")
    print(f"Tool Schema: {calculator.to_string()}")
    result = calculator(a=6, b=7)
    print(f"Execution Result: {result}")