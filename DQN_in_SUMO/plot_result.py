import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

config = {
    "figure.figsize": (8, 4),
    "font.size": 18,
    "mathtext.fontset": "stix",
    "font.sans-serif": "Times New Roman",
    "axes.unicode_minus": False,
    "axes.labelsize": 18,
    "savefig.dpi": 300,
    "savefig.format": "svg"
}
plt.rcParams.update(config)


def rolling_average(sequence, steps):
    new_sequence = []
    for i in range(len(sequence) - steps + 1):
        steps_sum = 0
        current_step = 0
        while current_step < steps:
            steps_sum += sequence[i + current_step]
            current_step += 1
        steps_average = steps_sum / steps
        new_sequence.append(steps_average)
    return new_sequence


def smoothing_average(sequence, last_weight):
    new_sequence = []
    for i in range(len(sequence)):
        if i == 0:
            smoothed = sequence[i]
        else:
            smoothed = (1 - last_weight) * sequence[i] + \
                       last_weight * new_sequence[i - 1]
        new_sequence.append(smoothed)
    return new_sequence


training_date = "2023.05.06-17.52.23"


result_path = os.path.join("output_model", training_date, "reward.csv")
result = pd.read_csv(result_path, index_col="Episode")
average_reward = result.Average_Reward.to_list()

fig = plt.figure(figsize=(12, 6))

ax1 = plt.plot(np.arange(1, 201), average_reward, c="#214c41", alpha=0.3)
ax2 = plt.plot(np.arange(1, 201), smoothing_average(average_reward, 0.6), c="#214c41")
plt.xlim([0, 200])
plt.xlabel("Episodes")
plt.ylabel("Average reward")

fig.tight_layout()

plt.show()