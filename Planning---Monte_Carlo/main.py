from Simulator import Simulator
import Agents
import sample_agent
from copy import deepcopy
import time

CONSTRUCTOR_TIMEOUT = 60
ACTION_TIMEOUT = 5
DIMENSIONS = (10, 10)
PENALTY = 10000


class Game:
    def __init__(self, an_input):
        self.initial_state = deepcopy(an_input)
        self.simulator = Simulator(self.initial_state)
        self.Name = []
        self.agents = []
        self.score = [0, 0]

    def initiate_agent(self, module, player_number):
        start = time.time()
        agent = module.Agent(self.initial_state, player_number)
        if time.time() - start > CONSTRUCTOR_TIMEOUT:
            raise ValueError(f'agent timed out on constructor!')
        return agent

    def get_action(self, agent, player):
        start = time.time()
        action = agent.act(self.simulator.get_state())
        finish = time.time()
        if finish - start > ACTION_TIMEOUT:
            self.score[player] -= PENALTY
            raise ValueError(f'{self.Name[player]} timed out on action!')
        return action

    def play_episode(self, swapped=False):
        length_of_episode = self.initial_state["turns to go"]
        for i in range(length_of_episode):
            for number, agent in enumerate(self.agents):
                try:
                    action = self.get_action(agent, number)
                except (AssertionError, ValueError)as e:
                    print(e)
                    self.score[number] -= PENALTY
                    return
                try:
                    self.simulator.act(action, number + 1)
                except (AssertionError, ValueError):
                    print(f'{agent.Name} chose illegal action!')
                    self.score[number] -= PENALTY
                    return
                print(f"{agent.Name} chose {action}")
            print(f"-----")
        if not swapped:
            self.score[0] += self.simulator.get_score()['player 1']
            self.score[1] += self.simulator.get_score()['player 2']
        else:
            self.score[0] += self.simulator.get_score()['player 2']
            self.score[1] += self.simulator.get_score()['player 1']
            print(f'***********  end of round!  ************ \n \n')

    def play_game(self):
        print(f'***********  starting a first round!  ************ \n \n')
        self.agents = [self.initiate_agent(Agents, 1),
                       self.initiate_agent(sample_agent, 2)]
        self.Name = ['Your agent', 'Rival agent']
        self.play_episode()

        print(f'***********  starting a second round!  ************ \n \n')
        self.simulator = Simulator(self.initial_state)

        self.agents = [self.initiate_agent(sample_agent, 1),
                       self.initiate_agent(Agents, 2)]
        self.Name = ['Rival agent', 'Your agent']
        self.play_episode(swapped=True)
        print(f'end of game!')
        return self.score


def main():
    an_input = {
        'map': [['P', 'P', 'P', 'P', 'P', 'I', 'P', 'P', 'P', 'P'],
                ['P', 'P', 'P', 'P', 'P', 'I', 'P', 'P', 'I', 'P'],
                ['P', 'P', 'I', 'P', 'P', 'I', 'P', 'I', 'P', 'I'],
                ['P', 'P', 'P', 'P', 'P', 'I', 'P', 'P', 'P', 'P'],
                ['P', 'I', 'P', 'P', 'P', 'I', 'P', 'P', 'P', 'P'],
                ['P', 'P', 'P', 'I', 'P', 'P', 'P', 'I', 'P', 'P'],
                ['P', 'P', 'I', 'P', 'P', 'P', 'P', 'P', 'I', 'I'],
                ['P', 'P', 'P', 'P', 'I', 'P', 'P', 'P', 'P', 'P'],
                ['P', 'P', 'P', 'P', 'P', 'P', 'I', 'P', 'P', 'P'],
                ['P', 'P', 'P', 'P', 'I', 'P', 'P', 'P', 'P', 'I']],
        'taxis': {'taxi 1': {'location': (4, 0), 'capacity': 2, 'player': 1},
                  'taxi 2': {'location': (3, 9), 'capacity': 1, "player": 1},
                  'taxi 3': {'location': (8, 3), 'capacity': 2, "player": 2},
                  'taxi 4': {'location': (4, 4), 'capacity': 1, "player": 2},
                  'taxi 5': {'location': (0, 0), 'capacity': 1, "player": 2},
                  },
        'passengers': {'Omer': {'location': (4, 8), 'destination': (1, 9), 'reward': 4},
                       'Gal': {'location': (2, 3), 'destination': (8, 1), 'reward': 6},
                       'Jana': {'location': (8, 6), 'destination': (5, 4), 'reward': 1},
                       'Reema': {'location': (7, 0), 'destination': (5, 0), 'reward': 7},
                       'Dana': {'location': (4, 7), 'destination': (2, 5), 'reward': 4},
                       'Kobi': {'location': (0, 6), 'destination': (6, 2), 'reward': 3},
                       },
        'turns to go': 200
    }
    # an_input = {
    #     'map': [['P', 'P', 'P', 'P', 'P', 'I', 'P', 'P', 'P', 'P'],
    #             ['P', 'P', 'P', 'P', 'P', 'I', 'P', 'P', 'I', 'P'],
    #             ['P', 'P', 'I', 'P', 'P', 'I', 'P', 'I', 'P', 'I'],
    #             ['P', 'P', 'P', 'P', 'P', 'I', 'P', 'P', 'P', 'P'],
    #             ['P', 'I', 'P', 'P', 'P', 'I', 'P', 'P', 'P', 'P'],
    #             ['P', 'P', 'P', 'I', 'P', 'P', 'P', 'I', 'P', 'P'],
    #             ['P', 'I', 'I', 'P', 'P', 'P', 'P', 'P', 'I', 'I'],
    #             ['P', 'P', 'P', 'I', 'I', 'P', 'P', 'P', 'P', 'P'],
    #             ['P', 'I', 'I', 'P', 'P', 'P', 'I', 'P', 'P', 'P'],
    #             ['P', 'P', 'P', 'P', 'I', 'P', 'P', 'P', 'P', 'I']],
    #     'taxis': {'taxi 1': {'location': (1, 0), 'capacity': 1, 'player': 1},
    #               'taxi 2': {'location': (5, 5), 'capacity': 0, 'player': 1},
    #               'taxi 3': {'location': (7, 2), 'capacity': 2, 'player': 2},
    #               'taxi 4': {'location': (7, 1), 'capacity': 1, 'player': 2}
    #               },
    #     'passengers': {'Omer': {'location': 'taxi 2', 'destination': (1, 9), 'reward': 4},
    #                    'Jana': {'location': (9, 6), 'destination': (5, 4), 'reward': 1},
    #                    'Dana': {'location': (4, 7), 'destination': (2, 5), 'reward': 4},
    #                    'Kobi': {'location': (0, 6), 'destination': (6, 2), 'reward': 3},
    #                    'Dave': {'location': (2, 6), 'destination': (3, 0), 'reward': 8},
    #                    'Oliver': {'location': (5, 1), 'destination': (7, 8), 'reward': 5},
    #                    'Tamar': {'location': (0, 3), 'destination': (4, 2), 'reward': 7},
    #                    'Efrat': {'location': 'taxi 1', 'destination': (5, 2), 'reward': 6},
    #                    'Avi': {'location': (7, 2), 'destination': (6, 6), 'reward': 9},
    #                    'Ali': {'location': (0, 3), 'destination': (8, 4), 'reward': 4}
    #                    },
    #     'turns to go': 20
    # }
    game = Game(an_input)
    results = game.play_game()
    print(f'Score for {Agents.Name} is {results[0]}, score for {sample_agent.Name} is {results[1]}')


if __name__ == '__main__':
    main()