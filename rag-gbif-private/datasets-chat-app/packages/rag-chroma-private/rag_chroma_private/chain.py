# Load
from langchain.chat_models import ChatOllama
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain.document_loaders import JSONLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.tools import StructuredTool
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.tools.render import format_tool_to_openai_function
from langchain.schema import AIMessage
from pygbif import registry
from .models import DatasetSearchResults, Result, Facet, Count
import json



def dataset_search(country: str = None, type: str = None, subtype: str = None, license: str = None, keyword: str = None, decade: int = None, publishing_country: str = None, project_id: str = None, hosting_country: str = None, continent: str = None):
    '''use this tool when you need to query GBIF datasets using the parameters: country, type, subtype, license, keyword, decade, publishing country, projectId, hosting country, continent'''
    results = registry.dataset_search(country=country, type=type, subtype=subtype, license=license, keyword=keyword, decade=decade, publishingCountry=publishing_country, projectId=project_id, hostingCountry=hosting_country, continent=continent, limit=10)
    return AIMessage(content=json.dumps(results))

loader = JSONLoader('datasmall/datasets.jsonl', jq_schema='.', json_lines=True, text_content=False)
data = loader.load()

# Split

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=all_splits,
    collection_name="rag-gbif-datasets",
    embedding=GPT4AllEmbeddings(),
)
retriever = vectorstore.as_retriever()

# Prompt
# Optionally, pull from the Hub
# from langchain import hub
# prompt = hub.pull("rlm/rag-prompt")
# Or, define your own:
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM
# Select the LLM that you downloaded
ollama_llm = "mistral"
model = ChatOllama(model=ollama_llm)

model_f = OllamaFunctions(model="mistral")

tools = [StructuredTool.from_function(dataset_search)]

llm_with_tools = model_f.bind(functions=[format_tool_to_openai_function(t) for t in tools])


search_template = '''
  Using the GBIF 'dataset_search' tool. Only call this tool to answer the query: {input}

  Format instructions:
  {format_instructions}
'''

parser = PydanticOutputParser(pydantic_object=DatasetSearchResults)
assistant_system_message = """You are a helpful assistant. \
Use tools (only if necessary) to best answer the users questions."""
prompt_agent = ChatPromptTemplate.from_messages(
    [
        ("system", assistant_system_message),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt_agent
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()
)


agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)

# initialize conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

# initialize agent with tools
agent = initialize_agent(
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    tools=tools,
    llm=model_f,
    verbose=False,
    max_iterations=3,
    return_direct=True,
    early_stopping_method='force',
    memory=conversational_memory
)



# RAG chain
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    __root__: str

class Input(BaseModel):
    input: str


class Output(BaseModel):
    output: str

chain = chain.with_types(input_type=Question)
