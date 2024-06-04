# Import necessary modules
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes

######################START OF MY CODE######################
import os  # For accessing environment variables
from langchain_nvidia_ai_endpoints import ChatNVIDIA  # NVIDIA AI endpoints for chat models
from langchain_core.output_parsers import StrOutputParser  # String output parser
from langchain_core.prompts import ChatPromptTemplate  # Chat prompt template
import requests  # For making HTTP requests
from langchain.vectorstores.chroma import Chroma  # Chroma vector store
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings  # NVIDIA AI endpoints for embeddings
from langchain.prompts import ChatPromptTemplate  # Chat prompt template
from langchain_core.runnables import RunnableLambda  # Runnables for chaining operations

# Constants
CHROMA_PATH = "../../data/chroma_small"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
######################END OF MY CODE######################

# Initialize FastAPI app
app = FastAPI()

# Redirect root URL to docs
@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

######################START OF MY CODE######################

# Function to craft the prompt
def craft_prompt_function(inputs):
    # Extract the most recent query text from inputs
    query_text = str(inputs['undefined'][len(inputs['undefined']) - 1]['content'])
    
    # Prepare the DB with embeddings
    embedding_function = NVIDIAEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    # Search the DB for relevant documents
    results = db.similarity_search_with_relevance_scores(query_text, k=10)

    # Check if results are sufficient
    if len(results) == 0 or results[0][1] < 0.5:
        print(f"Unable to find matching results.")
    
    # Prepare passages for reranking
    passages = [{'text': doc.page_content} for doc, _ in results]

    # Invoke NVIDIA rerank-qa-mistral-4b model for reranking the passages
    invoke_url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"

    headers = {
        "Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}",
        "Accept": "application/json",
    }

    payload = {
        "model": "nv-rerank-qa-mistral-4b:1",
        "query": {
            "text": query_text
        },
        "passages": passages
    }

    # Make the HTTP POST request
    session = requests.Session()
    response = session.post(invoke_url, headers=headers, json=payload)
    response.raise_for_status()
    response_body = response.json()
    print("response_body: ", response_body)

    # Process the response to extract rankings
    if 'rankings' not in response_body or not response_body['rankings']:
        print("No rankings found in response body.")
    else:
        rankings = response_body['rankings']
        sorted_rankings = sorted(rankings, key=lambda x: x['logit'])

        if len(rankings) < 3:
            print("Not enough rankings found in response body.")
        else:
            top_3_indices = [ranking['index'] for ranking in sorted_rankings[:3]]
            top_3_results = [results[index][0] for index in top_3_indices]
            results = top_3_results
            print("Top 3 results: ", results)

    # Create context text from the top results
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])

    # Update inputs with context and question
    inputs['context'] = [{
        'type': 'system',
        'content': context_text
    }]
    inputs['question'] = [{
        'type': 'human',
        'content': query_text
    }]
    inputs['undefined'] = []

    return inputs

# Initialize the components
embedding_function = NVIDIAEmbeddings()
chroma_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

chat_model = ChatNVIDIA(model="meta/llama2-70b")
output_parser = StrOutputParser()

# Define the prompt and model chain
chain = (
    RunnableLambda(craft_prompt_function) |  # First step: craft the prompt
    _prompt |  # Apply the prompt template
    chat_model |  # Pass through the NVIDIA chat model
    output_parser  # Parse the output
) 

# Add routes to FastAPI app
add_routes(
    app, 
    chain,
    path="/docs-agent",
    playground_type="chat",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True
)

######################END OF MY CODE######################
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
