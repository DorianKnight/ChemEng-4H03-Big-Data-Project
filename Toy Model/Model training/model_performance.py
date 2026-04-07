# Author: Dorian Knight
# Created: April 5th 2026
# Updated: April 5th 2026
# Description: Model performance over time - model was run in separate code and accuracies were written down and included here

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
index = [1,2,3,4,5,6,7,8]
accuracies = [75,62.5,50,62.5, 50, 50, 50, 75]
ax.plot(index, accuracies)
ax.set_title("Model Accuracy vs Times Trained")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim([0,100])
ax.set_xlabel("Times trained from scratch")
fig.savefig("Model Accuracy vs Train Cycles.png")
plt.show()