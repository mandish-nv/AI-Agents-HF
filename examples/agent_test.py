from smolagents import LiteLLMModel, ChatMessage

# Initialize the model
model = LiteLLMModel(
    model_id="ollama_chat/qwen2.5-coder:7b",
    api_base="http://127.0.0.1:11434",
    num_ctx=8192,
)

# Structure each content element with both "type" and "text"
message = ChatMessage(
    role="user", 
    content=[{"type": "text", "text": "Write a simple Python function to add two numbers."}]
)

# Pass the message list to the model
response = model([message])

print("Response from model:")
print(response.content)