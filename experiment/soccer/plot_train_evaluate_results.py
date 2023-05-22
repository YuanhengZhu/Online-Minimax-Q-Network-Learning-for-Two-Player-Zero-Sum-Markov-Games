import numpy as np
import matplotlib.pyplot as plt
import json


f = open(".\\save\\train_m2qn_soccer_box_300\\train_evaluate_results")
train_evaluate_results = json.load(f)

plt.figure('soccer_evaluate_results')

win_traj, lose_traj, draw_traj = [], [], []
for i in range(len(train_evaluate_results['timesteps'])):
    win_traj.append(train_evaluate_results['evaluate_results'][i]['win'])
    lose_traj.append(train_evaluate_results['evaluate_results'][i]['lose'])
    draw_traj.append(train_evaluate_results['evaluate_results'][i]['draw'])

plt.xlabel('num_timesteps')
plt.ylabel('rate')
plt.plot(train_evaluate_results['timesteps'],
         win_traj, 'ro', label='win')
plt.plot(train_evaluate_results['timesteps'],
         lose_traj, 'g+', label='lose')
plt.plot(train_evaluate_results['timesteps'],
         draw_traj, 'b.', label='draw')

plt.legend(loc='best')
plt.grid()
plt.show(block=False)
plt.pause(1)

while True:
    pass