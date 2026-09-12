from smolagents import LiteLLMModel, ChatMessage
from transformers import AutoTokenizer

# 1. Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")

# 2. Define message history
messages = [
    {"role": "system", "content": "You are an AI assistant with access to various tools."},
    {"role": "user", "content": "Hi !"},
    {"role": "assistant", "content": "Hi human, what can help you with ?"},
    {"role": "user", "content": "Write a simple Python function to add two numbers."},
]

# 3. Render prompt using chat template
rendered_prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

print(f"Rendered Prompt: \n{rendered_prompt}")

# 4. Initialize the model
model = LiteLLMModel(
    model_id="ollama_chat/qwen2.5-coder:7b",
    api_base="http://127.0.0.1:11434",
    num_ctx=8192,
)

# 5. Pass rendered prompt into ChatMessage
message = ChatMessage(
    role="user", 
    content=[{"type": "text", "text": rendered_prompt}]
)

# 6. Execute model call
response = model([message])

print("Response from model:")
print(response.content)
