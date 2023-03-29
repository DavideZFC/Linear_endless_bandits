import numpy as np

def test_algorithm(policy, env, T, seeds):

    regret_matrix = np.zeros((seeds, T))

    opt = env.get_optimum()
    for seed in range(seeds):

        policy.reset()
        for t in range(1,T):
            arm = policy.pull_arm()
            reward, expected_regret = env.pull_arm(arm)
            regret_matrix[seed, t] = regret_matrix[seed, t-1] + expected_regret
            policy.update(arm, reward)
    
    return regret_matrix

