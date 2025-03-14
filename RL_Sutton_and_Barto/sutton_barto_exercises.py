import numpy as np
import matplotlib.pyplot as plt

#2.5 - experiment to demonstrate the difficulties of the sample-average methods for nonstationary problems, 
# i. e. problems where reward probabilities change with time
def run_bandit_experiment(steps=10000, epsilon=0.1, k=10, alpha=0.1, non_stationary=True, initial_Q=0.0, use_sample_avg_method = True, use_epsilon_greedy = True, c=2):

    if non_stationary:
        q_star = np.zeros(k) #np.random.normal(0, 1, k)# #true value (expected reward) of action a; 
                            # for stationary case, select from normal distribution rather than starting with all zeros
    else:
        q_star = np.random.normal(0, 1, k)

    Q = np.ones(k) * initial_Q                  #stores best estimate of q_star for each action (=slot machine) based on all previous selections
    N = np.zeros(k)                             #counts how often each action (=slot machine) was selected

    perc_of_optimal_action = np.zeros(steps)    #pre-allocate space for storing results
    avg_reward = np.zeros(steps)                #pre-allocate space for storing results

    print("Initial q_star values:", q_star)  # BEFORE

    #perform the experiment
    for run in range(steps):

        if use_epsilon_greedy:
            # Find all indices where Q is at its maximum, Randomly select one of the max indices
            max_index = np.random.choice(np.flatnonzero(Q == Q.max()))

            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                i = np.random.randint(k)  # Pick any action randomly
            else:
                i = max_index  # Pick best action
        else: #use UCB (upper confidence bound) action selection
                i = np.argmax(Q + c * np.sqrt(np.log(run)/N))  # Index of best action; where best = best balance between exploitation Q and exploration (high uncertainty -> higher chance that action might be of high value)

        reward = np.random.normal(q_star[i], 1) #the reward in a specific iteration fluctuates around the expected q_star value
        N[i] += 1

        if use_sample_avg_method:
            Q[i] += (1 / N[i]) * (reward - Q[i])  # Sample-average instead of constant α

        else:
            alpha = 0.1
            Q[i] += alpha * (reward - Q[i]) #constant stepsize alpha means less significance on rewards that are further in the past 
                                    # (desirable for nonstationary problems)

        
        optimal_action = np.argmax(q_star)  # Index of best action
        perc_of_optimal_action[run] = 1/(run+1) * (perc_of_optimal_action[max(0, run-1)]*run + (i == optimal_action))
        avg_reward[run]             = 1/(run+1) * (avg_reward[max(0, run-1)]*run + reward)

        if non_stationary:
            #nonstationary problem, i. e. q_star values change (random walk)
            q_star += np.random.normal(loc=0.0, scale=0.01, size=10)

    print("Final q_star values:", q_star)  # AFTER (for nonstationary case)
    return avg_reward, perc_of_optimal_action

def plot_results(avg_rewards, perc_optimal_actions, labels):
    fig, ax1 = plt.subplots()
    x_axis_steps = np.arange(1, len(avg_rewards[0]) + 1)  # Steps, for plotting

    # Plot average rewards
    for i in range(len(avg_rewards)):
        ax1.plot(x_axis_steps, avg_rewards[i], label=f"Avg Reward ({labels[i]})")
    
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Average Reward", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.legend(loc="upper left")

    # Secondary y-axis for % optimal action
    ax2 = ax1.twinx()
    for i in range(len(perc_optimal_actions)):
        ax2.plot(x_axis_steps, perc_optimal_actions[i], linestyle="--", label=f"% Optimal ({labels[i]})")

    ax2.set_ylabel("% Optimal Action", color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(loc="lower right")

    plt.title("Comparison of Different Experiment Settings")
    fig.tight_layout()
    plt.show()


# Run two experiments (example: different Q initialization)
avg_reward_1, perc_optimal_1 = run_bandit_experiment(initial_Q=0.0, non_stationary=False, steps=1000, use_sample_avg_method = True, use_epsilon_greedy=True)  # Normal Q-init
avg_reward_2, perc_optimal_2 = run_bandit_experiment(initial_Q=0.0, non_stationary=False, steps=1000, use_sample_avg_method = True, use_epsilon_greedy=False)  # Optimistic Q-init

# Call the updated plot_results function
plot_results(
    avg_rewards=[avg_reward_1, avg_reward_2],
    perc_optimal_actions=[perc_optimal_1, perc_optimal_2],
    labels=["epsilon greedy", "UCB action selection"]
)