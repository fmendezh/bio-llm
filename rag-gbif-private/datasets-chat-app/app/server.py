from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from rag_chroma_private import chain as rag_chroma_private_chain
from rag_chroma_private import agent_executor as rag_chroma_private_agent_executor
from rag_chroma_private import Input,Output

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(app, rag_chroma_private_chain, path="/rag-chroma-private")

add_routes(app, rag_chroma_private_agent_executor.with_types(input_type=Input, output_type=Output), path="/agent")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
