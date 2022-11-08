from cv import ComputerVision
import json
from sys import exit
from pygame.locals import *
import matplotlib.pyplot as plt
import numpy as np


def main():
    qlearn = AQLearning()
    vision = ComputerVision()
    vision.setup()

    while True:
        quit = not vision.action(qlearn.act(vision.getX(), vision.getY(), vision.getV(), vision.getY1()))
        if quit:
            qlearn.save_qvalues()
            qlearn.save_training_states()
            break

        death, score =  vision.nextFrame()
        if death:
            qlearn.update_qvalues(score)
            #print(f"Episode: {qlearn.episode}, alpha: {qlearn.alpha}, score: {score}, max_score: {qlearn.max_score}")
            qlearn.showPerformance()


        
class AQLearning:
    def __init__(self):
        # Important values
        self.alpha = 0.7
        self.reward = {0: 0, 1: -1000}
        self.discount_factor = 0.95  # q-learning discount factor
        # Stabilize and converge to optimal policy
        self.alpha_decay = 0.00003  # 20,000 episodes to fully decay

        # Episodes
        self.episode = 0
        self.scores = []
        self.max_score = 0
        self.max_scores = []
        self.moves = []

        # Store values
        self.q_values = {}
        self.prev_state = "0_0_0_0"     # x = 0, y = 0, v= 0, y1 = 0
        self.prev_action = 0
        self.load_qvalues()
        self.load_training_states()

    def load_qvalues(self):
        """Load q values and from json file."""
        print("Loading Q-table states from json file...")
        try:
            with open("data/q_values_resume.json", "r") as f:
                self.q_values = json.load(f)
        except IOError:
            self.init_qvalues(self.prev_state)

    def init_qvalues(self, state):
        """
        Initialise q values if state not yet seen.
        :param state: current state
        """
        if self.q_values.get(state) is None:
            # q_values[state][0] (action = 0)
            # q_values[state][1] (action = 1)
            # q_values[state][2] (number of times)
            self.q_values[state] = [0, 0, 0]     

    def load_training_states(self):
        """Load current training state from json file."""
        #if self.train:
        print("Loading training states from json file...")
        try:
            with open("data/training_values_resume.json", "r") as f:
                training_state = json.load(f)
                self.episode = training_state['episodes'][-1]
                self.scores = training_state['scores']
                self.max_scores = training_state['max_scores']
                self.alpha = max(self.alpha - self.alpha * self.episode, 0.1)
                # self.epsilon = max(self.epsilon - self.epsilon_decay * self.episode, 0)
                if self.scores:
                    self.max_score = self.max_scores[-1]
                else:
                    self.max_score = 0
        except IOError:
            pass  

    def get_state(self, x, y, vel, y1):
        state = str(x) + "_" + str(y) + "_" + str(vel) + "_" + str(y1)
        self.init_qvalues(state)
        return state


    def act(self, x, y, v, y1):
        state = self.get_state(x, y, v, y1)
        self.moves.append((self.prev_state, self.prev_action, state))
        self.reduce_moves()
        # store the transition from previous state to current state
        self.prev_state = state
        
        # Best action with respect to current state, default is 0 (do nothing), 1 is flap
        self.prev_action = 0 if self.q_values[state][0] >= self.q_values[state][1] else 1

        return self.prev_action



    def update_qvalues(self, score):
        # rewards
        self.episode += 1
        self.scores.append(score)
        self.max_score = max(score, self.max_score)
        self.max_scores.append(self.max_score)

        history = list(reversed(self.moves))
        # Flag if the bird died in the top pipe, don't flap if this is the case
        high_death_flag = True if int(history[0][2].split("_")[1]) > 120 else False
        t, last_flap = 0, True
        for move in history:
            t += 1
            state, action, new_state = move
            self.q_values[state][2] += 1  # number of times this state has been seen
            curr_reward = self.reward[0]
            # Select reward
            if t <= 2:
                # Penalise last 2 states before dying
                curr_reward = self.reward[1]
                if action:
                    last_flap = False
            elif (last_flap or high_death_flag) and action:
                # Penalise flapping
                curr_reward = self.reward[1]
                last_flap = False
                high_death_flag = False

            self.q_values[state][action] = (1 - self.alpha) * (self.q_values[state][action]) + \
                                            self.alpha * (curr_reward + self.discount_factor *
                                                            max(self.q_values[new_state][0:2]))

        # Decay values for convergence
        if self.alpha > 0.1:
            self.alpha = max(self.alpha_decay - self.alpha_decay, 0.1)
        # if self.epsilon > 0:
        #     self.epsilon = max(self.epsilon - self.epsilon_decay, 0)

        # Don't need to reset previous action or state since this doesn't matter for all the beginning states
        # Although wikipedia mentions a reset of initial conditions tends to predict human behaviour more accurately
        self.moves = []  # clear history after updating strategies

        if score > max(self.scores, default=0):
            print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print("$$$$$$$$ NEW RECORD: %d $$$$$$$$" % score)
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")

    def reduce_moves(self, reduce_len=1000000):
        """
        Reduce length of moves if greater than reduce_len.
        :param reduce_len: reduce moves in memory if greater than this length, default 1 million
        """
        if len(self.moves) > reduce_len:
            history = list(reversed(self.moves[:reduce_len]))
            for move in history:
                state, action, new_state = move
                # Save q_values with default of 0 reward (bird not yet died)
                self.q_values[state][action] = (1 - self.alpha) * (self.q_values[state][action]) + \
                                               self.alpha * (self.reward[0] + self.discount_factor *
                                                             max(self.q_values[new_state][0:2]))
            self.moves = self.moves[reduce_len:]


    def save_qvalues(self):
        """Save q values to json file."""
        #if self.train:
        print(f"Saving Q-table with {len(self.q_values.keys())} states to file...")
        with open("data/q_values_resume.json", "w") as f:
            json.dump(self.q_values, f)

    def save_training_states(self):
        #if self.train:
        """Save current training state to json file."""
        print(f"Saving training states with {self.episode} episodes to file...")
        with open("data/training_values_resume.json", "w") as f:
            json.dump({'episodes': [i+1 for i in range(self.episode)],
                        'scores': self.scores}, f)

    
    def showPerformance(self):
        average = []
        num = 0
        sum_s = 0

        for s in self.scores:
            num += 1
            sum_s += s
            average.append(sum_s/num)
        
        if len(average) == 0:
            average.append(0)

        print("\nEpisode: {}, highest score: {}, average: {}".format(self.episode, max(self.scores, default=0), average[-1]))
        plt.figure(1)
        #plt.gca().get_xaxis().set_major_formatter(MaxNLocator(integer=True))
        plt.scatter(range(1, num+1), self.scores, c="green", s=3)
        plt.plot(range(1, num+1), average, 'b')
        plt.plot(range(1, num+1), self.max_scores, label='max_score', color='g')
        plt.xlim((1,num))
        plt.ylim((0,int(max(self.scores)*1.1)))

        plt.title("Score distribution")
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.show(block=False)
        plt.pause(0.001)

    


if __name__ == '__main__':
    main()
