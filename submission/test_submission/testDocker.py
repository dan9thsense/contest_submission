import importlib.util

from animalai.envs.gym.environment import AnimalAIEnv
from animalai.envs.arena_config import ArenaConfig


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

    arena_config_in = ArenaConfig('/aaio/configs/7-InternalMemory.yaml')

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



    print('Running 5 episodes')

    for k in range(5):
        env.reset(arenas_configurations=arena_config_in)
        cumulated_reward = 0
        print('Episode {} starting'.format(k))
        try:
            obs, reward, done, info = env.step([0, 0])
            for i in range(arena_config_in.arenas[0].t):
                
                action = submitted_agent.step(obs, reward, done, info)
                obs, reward, done, info = env.step(action)
                cumulated_reward += reward
                if done:
                    submitted_agent.reset(arena_config_in.arenas[0].t)
                    break
        except Exception as e:
            print('Episode {} failed'.format(k))
            raise e
        print('Episode {0} completed, reward {1}'.format(k, cumulated_reward))

    print('SUCCESS')


if __name__ == '__main__':
    main()
