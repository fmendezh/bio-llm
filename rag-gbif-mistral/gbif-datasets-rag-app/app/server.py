from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from .gbif_datasets_rag.chain import chain as gbif_datasets_rag_chain

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


add_routes(app, gbif_datasets_rag_chain, path="/gbif-datasets")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
