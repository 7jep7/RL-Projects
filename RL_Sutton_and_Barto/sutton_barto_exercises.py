import numpy as np
import matplotlib.pyplot as plt

#2.5 - experiment to demonstrate the difficulties of the sample-average methods for nonstationary problems, 
# i. e. problems where reward probabilities change with time
def run_bandit_experiment(steps=10000, epsilon=0.1, k=10, alpha=0.1):

    q_star = np.zeros(k) #true value (expected reward) of action a
    Q = np.zeros(k) #stores best estimate of q_star for each action (=slot machine) based on all previous selections
    N = np.zeros(k) #counts how often each action (=slot machine) was selected

    perc_of_optimal_action = np.zeros(steps) #pre-allocate space for storing results
    avg_reward = np.zeros(steps) #pre-allocate space for storing results

    #perform the experiment
    for run in range(steps):

        # Find all indices where Q is at its maximum, Randomly select one of the max indices
        max_index = np.random.choice(np.flatnonzero(Q == Q.max()))

        if np.random.rand() < epsilon:
            i = np.random.randint(k)  # Pick any action randomly
        else:
            i = max_index  # Pick best action

        reward = np.random.normal(q_star[i], 1) #the reward in a specific iteration fluctuates around the expected q_star value
        N[i] += 1

        alpha = 0.1
        Q[i] += alpha * (reward - Q[i]) #constant stepsize alpha means less significance on rewards that are further in the past 
                                    # (desirable for nonstationary problems)
        
        optimal_action = np.argmax(q_star)  # Index of best action
        perc_of_optimal_action[run] = 1/(run+1) * (perc_of_optimal_action[max(0, run-1)]*run + (i == optimal_action))
        avg_reward[run]             = 1/(run+1) * (avg_reward[max(0, run-1)]*run + reward)

        #nonstationary problem, i. e. q_star values change (random walk)
        q_star += np.random.normal(loc=0.0, scale=0.01, size=10)
    
    return avg_reward, perc_of_optimal_action

def plot_results(avg_reward, perc_of_optimal_action):
    fig, ax1 = plt.subplots()
    x_axis_steps = np.arange(1, len(avg_reward) + 1)

    ax1.plot(x_axis_steps, avg_reward, 'b-', label="Average Reward")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Average Reward", color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(x_axis_steps, perc_of_optimal_action, 'r-', label="% Optimal Action")
    ax2.set_ylabel("% Optimal Action", color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title("Average Reward and % Optimal Action vs Steps")
    fig.tight_layout()
    plt.show()


# Run the experiment
avg_reward, perc_of_optimal_action = run_bandit_experiment()
plot_results(avg_reward, perc_of_optimal_action)