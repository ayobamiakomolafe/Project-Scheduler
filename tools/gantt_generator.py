import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


def generate_gantt(schedule, output_file="gantt.png"):
    """
    Generate a responsive Gantt chart.

    Parameters:
        schedule (list): List of task dictionaries:
            {
                "task": str,
                "start": datetime,
                "end": datetime
            }

        output_file (str): Output image filename
    """

    if not schedule:
        raise ValueError("Schedule cannot be empty")

    # ---------------------------------------------------
    # Dynamic figure sizing
    # ---------------------------------------------------

    total_days = (
        max(item["end"] for item in schedule)
        - min(item["start"] for item in schedule)
    ).days

    fig_width = max(10, total_days * 0.3)
    fig_height = max(4, len(schedule) * 0.6)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # ---------------------------------------------------
    # Plot tasks
    # ---------------------------------------------------

    for item in schedule:
        start_num = mdates.date2num(item["start"])
        duration = (item["end"] - item["start"]).days

        ax.barh(
            y=item["task"],
            width=duration,
            left=start_num,
            height=0.5,
            align="center"
        )

    # ---------------------------------------------------
    # Format x-axis as dates
    # ---------------------------------------------------

    ax.xaxis_date()

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m-%d")
    )

    # Auto-adjust date spacing
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)

    plt.xticks(rotation=45)

    # ---------------------------------------------------
    # Labels and styling
    # ---------------------------------------------------

    ax.set_xlabel("Date")
    ax.set_ylabel("Tasks")
    ax.set_title("Project Gantt Chart")

    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    # Put first task at top
    ax.invert_yaxis()

    # ---------------------------------------------------
    # Auto-fit everything
    # ---------------------------------------------------

    plt.tight_layout()

    # ---------------------------------------------------
    # Save image
    # ---------------------------------------------------

    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


