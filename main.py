from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from dotenv import load_dotenv

load_dotenv()

# Chosen LLM end embedding model
llm = Ollama(model="deepseek-r1:1.5b", request_timeout=60.0)
embedding_model = resolve_embed_model("local:BAAI/bge-m3")

# Parser for PDF files
parser = LlamaParse(result_type="markdown")
file_extractor = {".pdf": parser}

# Storing Data in Vector Store
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embedding_model)

# Running a query on the vector index
query_engine = vector_index.as_query_engine(llm=llm)

# Example query to test the query engine
# response = query_engine.query("What is a topic of the document?")
# print(response)

# Creating a tool for the query engine
tools = [
    QueryEngineTool(query_engine=query_engine,metadata=ToolMetadata(name="Cybersecurity Regulations", description="Use this to questions about cybersecurity regulations"))
]

# code_llm = Ollama(model="codellama")

agent = ReActAgent.from_tools(
    tools=tools,
    llm=llm,
    verbose=True,
    context="The purpose of this agent is to answer questions about cybersecurity regulations using the provided documents.",
)
while (prompt := input("Enter your question (or 'q' to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)