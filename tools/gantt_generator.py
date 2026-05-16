import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

def generate_gantt(schedule):
    plt.figure()

    for item in schedule:
        duration = (item["end"] - item["start"]).days
        plt.barh(item["task"], duration)

    plt.xlabel("Days")
    plt.title("Project Gantt Chart")

    plt.savefig("gantt.png", bbox_inches="tight")