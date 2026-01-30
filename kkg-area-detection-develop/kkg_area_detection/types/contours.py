# coding: utf-8
from typing import List

from typing_extensions import TypedDict, NotRequired


class ContourVertex(TypedDict):
    x: int
    y: int


class ContourInfo(TypedDict):
    id: int
    vertices: List[ContourVertex]
    holes: NotRequired[List[List[ContourVertex]]]
