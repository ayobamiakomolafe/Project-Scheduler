from datetime import datetime, timedelta

def schedule_tasks(tasks):
    task_map = {t.id: t for t in tasks}
    start_dates = {}

    def get_start(task):
        if task.id in start_dates:
            return start_dates[task.id]

        if not task.dependencies:
            start_dates[task.id] = datetime.today()
        else:
            dep_ends = []
            for dep in task.dependencies:
                dep_task = task_map[dep]
                dep_start = get_start(dep_task)
                dep_end = dep_start + timedelta(days=dep_task.duration)
                dep_ends.append(dep_end)

            start_dates[task.id] = max(dep_ends)

        return start_dates[task.id]

    schedule = []

    for t in tasks:
        start = get_start(t)
        end = start + timedelta(days=t.duration)

        schedule.append({
            "task": t.name,
            "start": start,
            "end": end,
            "duration": t.duration,
            "resources": t.resources,
            "cost": t.cost,
            "dependencies": t.dependencies
        })

    return schedule