from core.models import ProjectModel

def validate_project(data, auto_fix_min_duration=None):
    """Validate project data before constructing ProjectModel.

    If `auto_fix_min_duration` is provided (an int), tasks with non-positive
    durations will be set to that minimum. Returns:
      - If `auto_fix_min_duration` is None: returns `ProjectModel` or raises `ValueError`.
      - If `auto_fix_min_duration` is set: returns a tuple `(ProjectModel, fixes)` where
        `fixes` is a list of `(index, id, previous_duration)` for each fixed task.
    """
    tasks = data.get("tasks") if isinstance(data, dict) else None
    if tasks is None:
        if auto_fix_min_duration is not None:
            return ProjectModel(**data), []
        return ProjectModel(**data)

    invalid = []
    for idx, t in enumerate(tasks):
        dur = t.get("duration") if isinstance(t, dict) else None
        if not isinstance(dur, (int, float)) or dur <= 0:
            tid = t.get("id") if isinstance(t, dict) else None
            invalid.append((idx, tid, dur))

    if invalid:
        if auto_fix_min_duration is not None:
            fixes = []
            for idx, tid, prev in invalid:
                if isinstance(tasks[idx], dict):
                    tasks[idx]["duration"] = int(auto_fix_min_duration)
                    fixes.append((idx, tid, prev))
            data["tasks"] = tasks
            model = ProjectModel(**data)
            return model, fixes
        else:
            parts = [f"{i} (id={tid})" for i, tid, _ in invalid]
            raise ValueError("Invalid task durations (must be > 0) at indices: " + ", ".join(parts))

    model = ProjectModel(**data)
    if auto_fix_min_duration is not None:
        return model, []
    return model
