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


# 1. Define strict schemas to enforce structural integrity across tasks
class TaskItem(BaseModel):
    id: str = Field(..., description="Unique identifier like T1, T2")
    name: str = Field(..., description="Name of the project task")
    duration: int = Field(default=0, description="Estimated duration in days")
    resources: List[str] = Field(default_factory=list, description="Roles needed, e.g., ['Senior Dev']")
    cost: float = Field(default=0.0, description="Calculated cost based on resource salary")
    dependencies: List[str] = Field(default_factory=list, description="IDs of prerequisite tasks")

class ProjectWBS(BaseModel):
    tasks: List[TaskItem]