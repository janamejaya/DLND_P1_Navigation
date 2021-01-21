import numpy as np
import random
import time
from collections import deque
import torch

class Discrete_Action_Experiments():
    """Interacts with and learns from the environment using discrete actions."""
    
    def __init__(self, nruns, nepisodes, maxt, start_eps, end_eps, decay_eps, current_agent, 
                 current_env, target_score, num_episodes_score_avg):
        """Initialize a Discrete Experiment object.
        Params
        ======
            nruns (int): Number of times the experiment will be run
            nepisodes (int): Number of episodes in each run
            maxt (int): Maximum number of steps per episode
            start_eps (float): starting value of epsilon, for epsilon-greedy action selection
            end_eps (float): minimum value of epsilon
            decay_eps (float): multiplicative factor (per episode) for decreasing epsilon
            current_agent (Class): selected agent
            current_env (Class): selected_environment
            target_score (float): target score to be achieved for successful run
            num_episodes_score_avg (int): number of scores over which a running average will be monitored
        """
        self.nruns = nruns
        self.nepisodes = nepisodes
        self.maxt = maxt
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.decay_eps = decay_eps
        self.agent = current_agent
        self.env = current_env,
        self.target_score = target_score
        self.num_episodes_score_avg = num_episodes_score_avg
        self.brain_name = self.env[0].brain_names[0]

    def execute_one_episode(self, current_eps):
        env_info = self.env[0].reset(train_mode=True)[self.brain_name]
        state = env_info.vector_observations[0]   # get the next state
        score = 0
        
        # For each time-step during the episode
        for t in range(self.maxt):
            # Select an action
            action = self.agent.act(state, current_eps).astype(int)
            
            # Get the reward, next state, and episode termination info
            env_info = self.env[0].step(action)[self.brain_name]
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            
            # Update the agent over the transition from state to next_state
            self.agent.step(state, action, reward, next_state, done)
            
            # Update state information
            state = next_state
            
            # Add reward to the score
            score += reward
            if done:
                break
        return score
    
    def execute_one_run(self, runid):
        """
        Execute one run with several episodes until the average score over the last num_episodes_score_avg
        exceeds the target value of target_score
        """
        # Initialize scores
        scores = []
        scores_window = deque(maxlen=self.num_episodes_score_avg)
        
        # Initialize eps
        eps = self.start_eps
        
        # Start the timer
        start_time=time.time()
        print('\n')
        for episode_num in range(self.nepisodes):
            # Reset the environment
            env_info = self.env[0].reset(train_mode=True)[self.brain_name]
            
            # Get the initial state
            state = env_info.vector_observations[0]
            
            # Run one episode and return the score for that episode
            # score corresponds to the total reward
            current_score = self.execute_one_episode(eps)
            
            # Append current_score to the scores list
            scores.append(current_score)
            scores_window.append(current_score)
            
            #decrease epsilon
            eps = max(self.end_eps, eps*self.decay_eps)
            
            # Get the average score from scores_window
            avg_score = np.mean(scores_window)
            
            # Show the average score
            if episode_num%self.num_episodes_score_avg==0:
                print('Run {:d} \tEpisode_Num {:d} \tAverage Score: {:.2f}'.format(runid, episode_num, avg_score))

            if avg_score>=self.target_score:
                end_time = time.time()
                
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f} \tTotal time/seconds: {:.2f}'.format(episode_num, avg_score, end_time-start_time))
                torch.save(self.agent.qnetwork_local.state_dict(), 'checkpoint_'+str(runid)+'.pth')
                
                break

        end_time=time.time()
        
        # Return scores for this run, the number of episodes required and the total time taken
        return scores, avg_score, episode_num, end_time-start_time
    
    def run_experiment(self):
        # Initial list to store list of scores from each run
        all_scores = []
        all_num_episodes = []
        all_total_times = []
        all_avg_scores = []
        
        # For each run
        for current_runid in range(self.nruns):
            # Perform the experiment for one run and return the scores, number of episodes and total time required
            scores, avg_score, num_episodes, total_time = self.execute_one_run(current_runid)
            
            # store the scores, num_episodes and total_time
            all_scores.append(scores)
            all_num_episodes.append(num_episodes)
            all_total_times.append(total_time)
            all_avg_scores.append(avg_score)
        
        # Find the average number of episodes required to reach the target score
        avg_number_of_episodes = np.mean(all_num_episodes)
        std_number_of_episodes = np.std(all_num_episodes)
        print('\nAverage number of episodes required to reach target score : {:2f} +/- {:2f}'.format(avg_number_of_episodes, std_number_of_episodes))
        
        # Find the average number time required to reach the target score
        #print('list of time taken = ', all_total_times)
        avg_time = np.mean(all_total_times)
        std_time = np.std(all_total_times)
        print('Average time/seconds per run required to reach target score : {:2f} +/- {:2f}'.format(avg_time, std_time))
        
        # Return all scores
        return all_scores, all_avg_scores
    

