# Load
from langchain.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import StructuredTool
from langchain.agents import initialize_agent, AgentType
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.schema.output_parser import StrOutputParser
from langchain.tools.render import format_tool_to_openai_function
from pygbif import registry, species, occurrences
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.schema import AIMessage, HumanMessage
from typing import List, Tuple
import json


def dataset_search(type: str = None, subtype: str = None, license: str = None, keyword: str = None,
                   decade: int = None, publishing_country: str = None, project_id: str = None,
                   hosting_country: str = None, continent: str = None):
    '''use this tool when you need to query GBIF datasets using the parameters: type, subtype, license, keyword, decade, publishing country, projectId, hosting country, continent'''
    results = registry.dataset_search(type=type, subtype=subtype, license=license, keyword=keyword,
                                      decade=decade, publishingCountry=publishing_country, projectId=project_id,
                                      hostingCountry=hosting_country, continent=continent, limit=10)
    return json.dumps(results)


def species_search(q: str = None, rank: str = None, higherTaxonKey: int = None, status: str = None,
                   isExtinct: bool = None,
                   habitat: str = None, nameType: str = None, datasetKey: str = None, nomenclaturalStatus: str = None,
                   type: str = None):
    '''use this tool when you need to query GBIF species using the parameters: q or query, rank, higherTaxonKey, status, isExtinct, habitat, nameType, datasetKey, nomenclaturalStatus, type'''
    results = species.name_lookup(q=q, rank=rank, higherTaxonKey=higherTaxonKey, status=status, isExtinct=isExtinct,
                                  habitat=habitat, nameType=nameType, datasetKey=datasetKey,
                                  nomenclaturalStatus=nomenclaturalStatus, type=type, limit=10, offset=0)
    return json.dumps(results)

def _format_chat_history(chat_history: List[Tuple[str, str]]):
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

model_f = ChatOllama(model="mistral")

tools = [StructuredTool.from_function(dataset_search), StructuredTool.from_function(species_search)]

llm_with_tools = model_f.bind(functions=[format_tool_to_openai_function(t) for t in tools])

search_template = '''
  Use the GBIF 'dataset_search' tool or the 'species_search' tool to best answer user questions.
'''
prompt_agent = ChatPromptTemplate.from_messages(
    [
        ("system", search_template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = (
        {
            "input": lambda x: x["input"],
            "chat_history": lambda x: _format_chat_history(x["chat_history"]),
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt_agent
        | llm_with_tools
        | StrOutputParser()
)

class AgentInput(BaseModel):
    input: str
    chat_history: List[Tuple[str, str]] = Field(
        ..., extra={"widget": {"type": "chat", "input": "input", "output": "output"}}
    )

# initialize agent with tools
agent_executor = initialize_agent(
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    tools=tools,
    llm=model_f,
    verbose=False,
    max_iterations=5,
    return_direct=True,
    handle_parsing_errors=True,
    early_stopping_method='force',
    max_execution_time=600000,
).with_types(input_type=AgentInput)

