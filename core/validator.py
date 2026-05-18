from core.models import ProjectModel

def validate_project(data):
    return ProjectModel(**data)