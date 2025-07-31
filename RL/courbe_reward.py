import numpy as np
import matplotlib.pyplot as plt

distance_to_goal = np.linspace(0, 50, 1000)
reward =  40*np.exp(-distance_to_goal / 20)

plt.plot(distance_to_goal, reward, label="Reward vs Distance",linestyle=':', color='blue')
plt.xlabel("Distance to goal")
plt.ylabel("Reward")
plt.title("Courbe : reward = 40 * exp(-distance_to_goal / 20)")
plt.grid(True)
plt.legend()
plt.show()