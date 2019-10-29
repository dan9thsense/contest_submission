import importlib.util

from animalai.envs.gym.environment import AnimalAIEnv
from animalai.envs.arena_config import ArenaConfig

MULTIPLE_CONFIGS = True
NUM_EPISODES = 2

def main():
    # Load the agent from the submission
    # this is the standard way that python 3.5+ versions import a file
    # and a class within that file, from a specific path location
    # agent.py was copied into the docker image in the folder /aaio
    # and agent.py has a class, Agent, that we use to interact with
    # https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path

    print('Loading your agent')
    try:
        spec = importlib.util.spec_from_file_location('agent_module', '/aaio/agent.py')
        agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_module)
        submitted_agent = agent_module.Agent()
    except Exception as e:
        print('Your agent could not be loaded, make sure all the paths are absolute, error thrown:')
        raise e
    print('Agent successfully loaded')

    arena_config_in = ArenaConfig('/aaio/configs/1-Food.yaml')

    print('Resetting your agent')
    try:
        submitted_agent.reset(t=arena_config_in.arenas[0].t)
    except Exception as e:
        print('Your agent could not be reset:')
        raise e

    env = AnimalAIEnv(
        environment_filename='/aaio/test/env/AnimalAI',
        seed=0,
        retro=False,
        n_arenas=1,
        worker_id=1,
        docker_training=True,
    )

    print('Running ', NUM_EPISODES, ' episodes')
    configs = []
    configs.append('/aaio/configs/1-Food.yaml')
    if MULTIPLE_CONFIGS:
        configs.append('/aaio/configs/2-Preferences.yaml')
        configs.append('/aaio/configs/3-Obstacles.yaml')
        configs.append('/aaio/configs/4-Avoidance.yaml')
        configs.append('/aaio/configs/5-SpatialReasoning.yaml')
        configs.append('/aaio/configs/6-Generalization.yaml')
        configs.append('/aaio/configs/7-InternalMemory.yaml')
        configs.append('/aaio/configs/temporary_blackout.yaml')
        configs.append('/aaio/configs/permanent_blackout.yaml')
        configs.append('/aaio/configs/permanent_blackout_with_wall_and_bad_goal.yaml')
        configs.append('/aaio/configs/hot_zone.yaml')
        configs.append('/aaio/configs/movingFood.yaml')
        configs.append('/aaio/configs/forcedChoice.yaml')
        configs.append('/aaio/configs/objectManipulation.yaml')
        configs.append('/aaio/configs/allObjectsRandom.yaml')
        
    
    config_results = []
    for config in configs:
        print('starting ', config)
        average_num_actions = 0
        average_reward = 0
        num_time_outs = 0
        arena_config_in = ArenaConfig(config)
        for k in range(NUM_EPISODES):
            time_out = False
            num_actions = 0
            episode_reward = 0
            episode_results = []
            env.reset(arenas_configurations=arena_config_in)
            print('Episode {} starting'.format(k))
            try:
                obs, reward, done, info = env.step([0, 0])
                for i in range(arena_config_in.arenas[0].t):
                    action = submitted_agent.step(obs, reward, done, info)
                    num_actions += 1
                    obs, reward, done, info = env.step(action)
                    episode_reward += reward
                    if done:
                        if i == arena_config_in.arenas[0].t - 1:
                            time_out = True
                        submitted_agent.reset(arena_config_in.arenas[0].t)
                        break
            except Exception as e:
                print('Episode {} failed'.format(k))
                raise e
            print('Episode {0} completed, num actions {1}, reward {1}'.format(k, num_actions, episode_reward, time_out))
            #episode_results.append([config, k, num_actions, episode_reward, time_out])
            average_reward += episode_reward
            average_num_actions += num_actions
            if time_out:
                num_time_outs += 1
        config_results.append([config, average_num_actions / NUM_EPISODES, average_reward / NUM_EPISODES, num_time_outs])

    print("config results: config, avg number of actions, average reward, number of timeouts")
    for result in config_results:
        print(result)


if __name__ == '__main__':
    main()
