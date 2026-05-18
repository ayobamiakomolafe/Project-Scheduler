from pydantic import BaseModel, Field
from typing import List

class TaskModel(BaseModel):
    id: str
    name: str
    duration: int = Field(gt=0)
    resources: List[str]
    cost: float = Field(ge=0)
    dependencies: List[str] = []

class ProjectModel(BaseModel):
    tasks: List[TaskModel]

   