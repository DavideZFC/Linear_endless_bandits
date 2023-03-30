import numpy as np
import time

def evaluate_learner(env, policy, hor=5000, epi=5):
    '''
    evaluates a policy on a given environment
        env = environment
        policy = policy to evaluate
        hor = time horizon, number of steps to compute
        epi = number of episodes to average the cumulative regret

    returns
        policy = the trained policy
        the mean regret across the episodes
        its standard deviation
        the time elapsed
    '''
    start_time = time.time()
    opt = env.get_optimum()
    global_regret = np.zeros((epi, hor))
    for ep in range(epi):
        policy.reset()
        regrets = np.zeros(hor)
        for t in range(hor):
            arm = policy.choose_arm()
            reward = env.pull_arm(arm)
            policy.update(arm, reward)

            regrets[t] = opt - env.y[arm]
        
        cumuregret = np.cumsum(regrets)
        global_regret[ep,:] = cumuregret
    
    mean_global_regret = np.mean(global_regret, axis=0)
    if (epi>1):
        std_global_regret = np.std(global_regret, axis=0)
    else:
        std_global_regret = np.zeros_like(mean_global_regret)
    end_time = time.time()
    return policy, mean_global_regret, std_global_regret, end_time-start_time
    
            
