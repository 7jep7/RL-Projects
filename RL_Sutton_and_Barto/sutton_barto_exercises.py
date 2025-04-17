import numpy as np
import matplotlib.pyplot as plt

def run_bandit_experiment(steps=10000, epsilon=0.1, k=10, alpha=0.1, non_stationary=True, initial_Q=0.0, use_sample_avg_method=True, use_epsilon_greedy=True, c=2):
    if non_stationary:
        q_star = np.zeros(k)
    else:
        q_star = np.random.normal(0, 1, k)

    Q = np.ones(k) * initial_Q
    N = np.zeros(k)

    perc_of_optimal_action = np.zeros(steps)
    avg_reward = np.zeros(steps)

    for run in range(steps):
        if use_epsilon_greedy:
            max_index = np.random.choice(np.flatnonzero(Q == Q.max()))
            if np.random.rand() < epsilon:
                i = np.random.randint(k)
            else:
                i = max_index
        else:
            i = np.argmax(Q + c * np.sqrt(np.log(run + 1) / (N + 1e-5)))

        reward = np.random.normal(q_star[i], 1)
        N[i] += 1

        if use_sample_avg_method:
            Q[i] += (1 / N[i]) * (reward - Q[i])
        else:
            Q[i] += alpha * (reward - Q[i])

        optimal_action = np.argmax(q_star)
        perc_of_optimal_action[run] = 1 / (run + 1) * (perc_of_optimal_action[max(0, run - 1)] * run + (i == optimal_action))
        avg_reward[run] = 1 / (run + 1) * (avg_reward[max(0, run - 1)] * run + reward)

        if non_stationary:
            q_star += np.random.normal(loc=0.0, scale=0.01, size=k)

    return avg_reward, perc_of_optimal_action

def plot_results(avg_rewards, perc_optimal_actions, labels):
    fig, ax1 = plt.subplots()
    x_axis_steps = np.arange(1, len(avg_rewards[0]) + 1)

    for i in range(len(avg_rewards)):
        ax1.plot(x_axis_steps, avg_rewards[i], label=f"Avg Reward ({labels[i]})")

    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Average Reward", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    for i in range(len(perc_optimal_actions)):
        ax2.plot(x_axis_steps, perc_optimal_actions[i], linestyle="--", label=f"% Optimal ({labels[i]})")

    ax2.set_ylabel("% Optimal Action", color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(loc="lower right")

    plt.title("Comparison of Different Experiment Settings")
    fig.tight_layout()
    plt.show()

def run_jack_car_rental_experiment(num_cars=20, num_days=30, num_runs=10, rental_rate=10, return_rate=3, gamma=0.9):
    V = np.zeros((num_cars + 1, num_cars + 1))

    def rental_request(location):
        return np.random.poisson(3 if location == 1 else 4)

    def car_return(location):
        return np.random.poisson(3 if location == 1 else 2)

    def reward(cars_loc1, cars_loc2, cars_moved):
        return rental_rate * (cars_loc1 + cars_loc2) - (abs(cars_moved) * 2)

    policy = np.zeros((num_cars + 1, num_cars + 1), dtype=int)
    for run in range(num_runs):
        for day in range(num_days):
            for s1 in range(num_cars + 1):
                for s2 in range(num_cars + 1):
                    action_values = []
                    for a in range(-min(s2, num_cars - s1), min(s1, num_cars - s2) + 1):
                        cars_rented_loc1 = min(s1 - a, rental_request(1))
                        cars_rented_loc2 = min(s2 + a, rental_request(2))

                        cars_returned_loc1 = min(num_cars - (s1 - a - cars_rented_loc1), car_return(1))
                        cars_returned_loc2 = min(num_cars - (s2 + a - cars_rented_loc2), car_return(2))

                        next_s1 = max(0, min(num_cars, s1 - a - cars_rented_loc1 + cars_returned_loc1))
                        next_s2 = max(0, min(num_cars, s2 + a - cars_rented_loc2 + cars_returned_loc2))

                        action_values.append(reward(cars_rented_loc1, cars_rented_loc2, a) + gamma * V[next_s1][next_s2])

                    V[s1][s2] = max(action_values)
                    policy[s1][s2] = int(np.argmax(action_values) - min(s1, num_cars - s2))

    return policy

def plot_policy(policy, num_cars):
    """
    Plots the policy for Jack's car rental problem as a discrete grid.

    Args:
        policy (np.ndarray): The policy array with shape (num_cars + 1, num_cars + 1).
        num_cars (int): The maximum number of cars at each location.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(policy.T, origin="lower", cmap="coolwarm", extent=[0, num_cars, 0, num_cars])
    plt.colorbar(label="Number of Cars Moved")
    plt.title("Optimal Policy for Jack's Car Rental Problem")
    plt.xlabel("Cars at Location 1 (s1)")
    plt.ylabel("Cars at Location 2 (s2)")
    plt.xticks(np.arange(0, num_cars + 1, step=5))
    plt.yticks(np.arange(0, num_cars + 1, step=5))
    plt.grid(visible=True, color="gray", linestyle="--", linewidth=0.5)
    plt.show()

def main():
    # # Run bandit experiments
    # avg_reward_1, perc_optimal_1 = run_bandit_experiment(initial_Q=0.0, non_stationary=False, steps=1000, use_sample_avg_method=True, use_epsilon_greedy=True)
    # avg_reward_2, perc_optimal_2 = run_bandit_experiment(initial_Q=0.0, non_stationary=False, steps=1000, use_sample_avg_method=True, use_epsilon_greedy=False)

    # plot_results(
    #     avg_rewards=[avg_reward_1, avg_reward_2],
    #     perc_optimal_actions=[perc_optimal_1, perc_optimal_2],
    #     labels=["epsilon greedy", "UCB action selection"]
    # )

    # Run Jack's car rental experiment
    policy = run_jack_car_rental_experiment(num_cars=20, num_days=30, num_runs=10)
    plot_policy(policy, num_cars=20)

if __name__ == "__main__":
    main()