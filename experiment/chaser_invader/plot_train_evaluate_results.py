import numpy as np
import matplotlib.pyplot as plt
import json


f = open(".\\save\\train_m2qn_with_dp_opp_chaser_invader_300\\train_evaluate_results")
f = open(".\\save\\train_m2qn_chaser_invader_100\\train_evaluate_results")
f = open(".\\save\\train_m2qn_with_random_opp_chaser_invader_300\\train_evaluate_results")
train_evaluate_results = json.load(f)

plt.figure('chaser_invader_evaluate_results')

win_traj, lose_traj, draw_traj, reward_traj = [], [], [], []
for i in range(len(train_evaluate_results['timesteps'])):
    win_traj.append(train_evaluate_results['evaluate_results'][i]['win'])
    lose_traj.append(train_evaluate_results['evaluate_results'][i]['lose'])
    draw_traj.append(train_evaluate_results['evaluate_results'][i]['draw'])
    reward_traj.append(train_evaluate_results['evaluate_results'][i]['reward'])

plt.subplot(2,1,1)
plt.ylabel('rate')
plt.plot(train_evaluate_results['timesteps'],
         win_traj, label='win')
plt.plot(train_evaluate_results['timesteps'],
         lose_traj, label='lose')
plt.plot(train_evaluate_results['timesteps'],
         draw_traj, label='draw')

plt.legend(loc='best')
plt.grid()
plt.show(block=False)

plt.subplot(2,1,2)
plt.xlabel('num_timesteps')
plt.plot(train_evaluate_results['timesteps'],
         reward_traj)
plt.grid()
plt.show(block=False)

plt.pause(1000)
