#Architecture Description: 
#FastAPI: handles incoming HTTP requests
#VectorStore module: connects to Pinecone
#LLM interfence module: calls OpenAI or Local LLM
#Docker container: reproducible dev environment
#Architecture Diagram: User-->FastAPI-->VectorStore-->LLM-->Response