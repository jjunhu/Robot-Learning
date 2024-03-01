import bc
import torch

def interact(env, learner, expert, observations, actions, checkpoint_path, seed, num_epochs=100):
  """Interact with the environment and update the learner policy using DAgger.
   
    This function interacts with the given Gym environment and aggregates to
    the BC dataset by querying the expert.
    
    Parameters:
        env (Env)
            The gym environment (in this case, the Hopper gym environment)
        learner (Learner)
            A Learner object (policy)
        expert (ExpertActor)
            An ExpertActor object (expert policy)
        observations (list of numpy.ndarray)
            An initially empty list of numpy arrays 
        actions (list of numpy.ndarray)
            An initially empty list of numpy arrays 
        checkpoint_path (str)
            The path to save the best performing model checkpoint
        seed (int)
            The seed to use for the environment
        num_epochs (int)
            Number of epochs to run the train function for
    """
  # Interact with the environment and aggregate your BC Dataset by querying the expert
  NUM_INTERACTIONS = 50
  best_reward = float("-inf")
  best_state_dict = None
  for episode in range(NUM_INTERACTIONS):
      total_learner_reward = 0
      done = False
      obs = env.reset(seed=seed)
      while not done:
        # TODO: Implement Hopper environment interaction and dataset aggregation here
        with torch.no_grad():
            learner_action = learner.get_action(obs)
            expert_action = expert.get_expert_action(obs)
            observations.append(obs)
            
            # aggregate new expert action
            actions.append(expert_action)
            obs, reward, done, info = env.step(learner_action)
            total_learner_reward += reward
        if done:
          break
      print(f"After interaction {episode}, reward = {total_learner_reward}")
      if total_learner_reward > best_reward:
        best_reward = total_learner_reward
        best_state_dict = learner.state_dict()        
      bc.train(learner, observations, actions, "dagger_last_epoch", num_epochs)
          
  torch.save(best_state_dict, checkpoint_path)
  learner.load_state_dict(best_state_dict)

  return learner
      