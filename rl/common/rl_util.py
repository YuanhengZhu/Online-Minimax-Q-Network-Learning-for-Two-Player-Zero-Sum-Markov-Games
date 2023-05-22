import warnings
from scipy.optimize import linprog
import numpy as np
import matplotlib.pyplot as plt

def minimax(Q):
    c = np.zeros(np.shape(Q)[0] + 1)
    c[0] = -1
    A_ub = np.ones((np.shape(Q)[1], np.shape(Q)[0] + 1))
    A_ub[:, 1:] = -Q.T
    b_ub = np.zeros(np.shape(Q)[1])
    A_eq = np.ones((1, np.shape(Q)[0] + 1))
    A_eq[0, 0] = 0
    b_eq = [1]
    bounds = ((None, None),) + ((0, 1),) * np.shape(Q)[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

    if res.success:
        z = res.x[1:]
        z[z < 0.01] = 0.0
        return z / np.sum(z)
    else:
        print("Alert: linear programming fails in solution, so just minimax ")
        prob = np.zeros(np.shape(Q)[0])
        act_best_worst = np.argmax(np.amin(Q, axis=1), axis=0)
        prob[act_best_worst] = 1
        return prob


def get_linear_epsilon_value(step, schedule_timesteps, initial_eps=1.0, final_eps=0.1):
    fraction = min(float(step) / schedule_timesteps, 1.0)
    return initial_eps + fraction * (final_eps - initial_eps)


def evaluate_soccer(env, act_func, bct_func, total_runs=100):
    run_results = {'win': 0, 'lose': 0, 'draw': 0}
    for _ in range(total_runs):
        obs = env.reset()
        done = False
        while not done:
            act = act_func(obs)
            bct = bct_func(obs)
            new_obs, rew, done, info = env.step(act, bct)

            if done:
                if rew > 0:
                    run_results['win'] += 1
                elif rew < 0:
                    run_results['lose'] += 1
                else:
                    run_results['draw'] += 1
                break
            else:
                obs = new_obs

    for key in run_results.keys():
        run_results[key] /= total_runs

    return run_results


def plot_soccer_evaluate_results(timesteps, results):
    if len(results) == 0:
        return

    fig = plt.figure('soccer_evaluate_results')

    if len(results) == 1:
        plt.clf()
        plt.xlabel('num_timesteps')
        plt.ylabel('rate')
        plt.plot(timesteps[-1], results[-1]['win'], 'ro', label='win')
        plt.plot(timesteps[-1], results[-1]['lose'], 'g+', label='lose')
        plt.plot(timesteps[-1], results[-1]['draw'], 'b.', label='draw')
    else:
        plt.plot(timesteps[-1], results[-1]['win'], 'ro')
        plt.plot(timesteps[-1], results[-1]['lose'], 'g+')
        plt.plot(timesteps[-1], results[-1]['draw'], 'b.')
    plt.xlim(0, timesteps[-1])
    plt.ylim(0, 1)
    plt.legend(loc='best')
    plt.grid()
    plt.show(block=False)
    plt.pause(1)


def evaluate_chaser_invader(env, act_func, bct_func, total_runs=100):
    run_results = {'win': 0, 'lose': 0, 'draw': 0, 'reward': 0}
    for _ in range(total_runs):
        obs = env.reset()
        done = False
        while not done:
            act = act_func(obs)
            bct = bct_func(obs)
            new_obs, rew, done, info = env.step(act, bct)

            if done:
                if rew > 0:
                    run_results['win'] += 1
                elif rew < 0:
                    run_results['lose'] += 1
                else:
                    run_results['draw'] += 1
                run_results['reward'] += rew
                break
            else:
                obs = new_obs

    for key in run_results.keys():
        run_results[key] /= total_runs

    return run_results

def plot_chaser_invader_evaluate_results(timesteps, results):
    if len(results) == 0:
        return

    plt.figure('chaser_invader_evaluate_results')

    if len(results) == 1:
        plt.clf()
        plt.subplot(2,1,1)
        plt.ylabel('rate')
        plt.plot(timesteps[-1], results[-1]['win'], 'ro', label='win')
        plt.plot(timesteps[-1], results[-1]['lose'], 'g+', label='lose')
        plt.plot(timesteps[-1], results[-1]['draw'], 'b.', label='draw')
    else:
        plt.subplot(2, 1, 1)
        plt.plot(timesteps[-1], results[-1]['win'], 'ro')
        plt.plot(timesteps[-1], results[-1]['lose'], 'g+')
        plt.plot(timesteps[-1], results[-1]['draw'], 'b.')
    plt.xlim(0, timesteps[-1])
    plt.ylim(0, 1)
    plt.legend(loc='best')
    plt.grid()
    plt.show(block=False)

    plt.subplot(2, 1, 2)
    plt.xlabel('num_timesteps')
    plt.ylabel('reward')
    plt.plot(timesteps[-1], results[-1]['reward'], 'ko')
    plt.xlim(0, timesteps[-1])
    plt.ylim(-10, 10)

    plt.grid()
    plt.show(block=False)
    plt.pause(1)