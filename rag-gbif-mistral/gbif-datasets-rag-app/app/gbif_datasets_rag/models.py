from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class Result(BaseModel):
    key: str
    title: Optional[str]
    doi: Optional[str]
    description: Optional[str]
    type: Optional[str]
    hostingOrganizationKey: Optional[str]
    hostingOrganizationTitle: Optional[str]
    hostingCountry: Optional[str] = None
    publishingCountry: Optional[str] = None
    publishingOrganizationKey: Optional[str]
    publishingOrganizationTitle: Optional[str]
    endorsingNodeKey: Optional[str]
    license: Optional[str]
    decades: Optional[List[int]] = None
    keywords: Optional[List[str]] = None
    recordCount: Optional[int]
    nameUsagesCount: Optional[int]
    subtype: Optional[str] = None
    networkKeys: Optional[List[str]] = None


class Count(BaseModel):
    name: str
    count: int


class Facet(BaseModel):
    field: str
    counts: List[Count]


class DatasetSearchResults(BaseModel):
    offset: int
    limit: int
    endOfRecords: bool
    count: int
    results: Optional[List[Result]]
    facets: Optional[List[Facet]]
