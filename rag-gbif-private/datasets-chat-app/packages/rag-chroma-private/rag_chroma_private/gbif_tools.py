from langchain.tools import BaseTool
from pygbif import registry


class GbifApiTool(BaseTool):
    name = "GBIF API tool"
    description = "use this tool when you need to query GBIF datasets using the parameters: country, type, subtype, license, keyword, decade, publishing country, projectId, hosting country, continent"

    def _run(self, country: str, type: str, subtype: str, license: str, keyword: str, decad: int, publishing_country: str, project_id: str, hosting_country: str, continent: str):
        return registry.dataset_search(country=country, type=type, subtype=subtype, license=license, keyword=keyword, decade=decade, publishingCountry=publishing_country, projectId=project_id, hostingCountry=hosting_country, continent=continent, limit=10)

    def _arun(self, radius: int):
        raise NotImplementedError("This tool does not support async")