# Load
from langchain.chat_models import ChatOllama
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

embeddings_open = OllamaEmbeddings(model="mistral")

mistral = ChatOllama(model="mistral")

vectordb = Chroma(persist_directory="./embeddings_chroma_db",
                  embedding_function=embeddings_open,
                  collection_name="rag-gbif-datasets"
                  )
retriever = vectordb.as_retriever()

# RAG prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# RAG chain
chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        | prompt
        | mistral
        | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)
