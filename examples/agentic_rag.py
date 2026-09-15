from langchain_community.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from smolagents import Tool, CodeAgent, LiteLLMModel
from langchain_community.retrievers import BM25Retriever

class PartyPlanningRetrieverTool(Tool):
    name = "party_planning_retriever"
    description = "Searches and returns a list of relevant party planning ideas as structured dictionaries for Alfred's superhero-themed party."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be a query related to party planning or superhero themes.",
        }
    }
    # 1. Update output type to allow structured objects/arrays
    output_type = "object"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(
            docs, k=3  # Retrieve the top 3 documents
        )

    def forward(self, query: str) -> list:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.retriever.invoke(query)
        
        # 2. Return a structured list of dictionaries natively
        return [
            {
                "idea": doc.page_content,
                "source": doc.metadata.get("source", "Unknown")
            }
            for doc in docs
        ]

# Simulate a knowledge base about party planning
party_ideas = [
    {"text": "A superhero-themed masquerade ball with luxury decor, including gold accents and velvet curtains.", "source": "Party Ideas 1"},
    {"text": "Hire a professional DJ who can play themed music for superheroes like Batman and Wonder Woman.", "source": "Entertainment Ideas"},
    {"text": "For catering, serve dishes named after superheroes, like 'The Hulk's Green Smoothie' and 'Iron Man's Power Steak.'", "source": "Catering Ideas"},
    {"text": "Decorate with iconic superhero logos and projections of Gotham and other superhero cities around the venue.", "source": "Decoration Ideas"},
    {"text": "Interactive experiences with VR where guests can engage in superhero simulations or compete in themed games.", "source": "Entertainment Ideas"}
]

source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"]})
    for doc in party_ideas
]

# Split the documents into smaller chunks for more efficient search
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)
docs_processed = text_splitter.split_documents(source_docs)

# Create the retriever tool
party_planning_retriever = PartyPlanningRetrieverTool(docs_processed)

# Initialize the Ollama model via LiteLLMModel
model = LiteLLMModel(
    model_id="ollama_chat/qwen2.5-coder:7b",
    api_base="http://127.0.0.1:11434",
    num_ctx=8192,
)

# Initialize the agent with the local Ollama model
agent = CodeAgent(tools=[party_planning_retriever], model=model)

# Example usage
response = agent.run(
    "Find ideas for a luxury superhero-themed party, including entertainment, catering, and decoration options. Categorize them into a dictionary."
)

print(response)