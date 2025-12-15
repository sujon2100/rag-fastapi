The Architecture

#Architecture Description: 
-FastAPI: handles incoming HTTP requests
-VectorStore module: connects to Pinecone
-LLM interfence module: calls OpenAI or Local LLM
-Docker container: reproducible dev environment
-Architecture Diagram: User-->FastAPI-->VectorStore-->LLM-->Response

Think of your system as a conversation between four specialists, each doing only one job.
A user sends a question to your FastAPI endpoint.
FastAPI itself does not think. It only coordinates. Its job is to move data between the right components in the right order.
The first thing FastAPI does is not call Llama.
Instead, it sends the user’s question to a sentence-transformer embedding model (all-MiniLM-L6-v2).
This model converts the question into a 384-dimensional vector.
At this point, the question has no words anymore — only meaning.
That vector is sent to Pinecone.
Pinecone is your system’s long-term semantic memory.
It stores embeddings of documents you previously uploaded (FastAPI notes, Pinecone explanations, RAG concepts, etc.).
It compares the query vector against stored vectors and returns the most semantically similar text chunks.
If Pinecone finds matches, it returns the raw text associated with those vectors.
If it finds nothing, the system stops early and refuses to hallucinate.
Only after relevant context is found does Llama get involved.
Llama is running locally on your machine via Ollama.
It is not queried directly by the user.
It is given a carefully constructed prompt that includes:
the retrieved context from Pinecone
the user’s original question
an instruction to answer only using the provided context
Llama’s role is not memory.
Llama’s role is reasoning and explanation.
The final answer is then returned to the user through FastAPI.

Flow:
Client (Postman / UI)
        |
        v
FastAPI (/rag/query)
        |
        v
Query Embedding (SentenceTransformers)
        |
        v
Vector Search (Pinecone)
        |
        v
Context Injection
        |
        v
LLM Generation (Ollama - Llama 3)
        |
        v
Final Answer (JSON)

                      ┌──────────────────────┐
                      │      Frontend /      │
                      │   Query Client UI     │
                      └──────────▲───────────┘
                                 │ HTTP POST
                                 ▼
                     ┌──────────────────────────┐
                     │       FastAPI Backend     │
                     │ (app/main, routes, etc.) │
                     └──────────▲───────────────┘
                                 │ Calls
 ┌───────────────────────────────┴────────────────────────────┐
 │                     RAG Pipeline Components                │
 │                                                            │
 │   ┌─────────────────┐      ┌──────────────────────────┐    │
 │   │   Vector Store   │      │    Local LLM (Ollama)     │  │
 │   │   Pinecone DB    │◀────▶│   Llama3 / Model API     │   │
 │   │  (384 dim index) │      │  (localhost network)      │  │
 │   └─────────────────┘      └──────────────────────────┘     │
 │             ▲                             ▲                 │
 │  embed query│                             │ generate answer│
 │             │                             │                 │
 └────────────────────────────────────────────────────────────┘

