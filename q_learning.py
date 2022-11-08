import argparse
from cv import ComputerVision
import json
import flappy_mod as Flappy
from sys import exit
import pygame
from pygame.locals import *
import matplotlib.pyplot as plt
import numpy as np
import random


def main():
    qlearn = AQLearning()
    vision = ComputerVision()
    vision.setup()
    while True:
        score = vision.getScore()
        death =  not vision.nextFrame()
        quit = not vision.action(qlearn.act(vision.getX(), vision.getY(), vision.getV(), vision.getY1()))
        if quit:
            qlearn.update_qvalues(score, death)
            qlearn.save_qvalues()
            qlearn.save_training_states()
            break
        if death:
            qlearn.update_qvalues(score, death)
            qlearn.save_qvalues()
            qlearn.save_training_states()
            #print(f"Episode: {qlearn.episode}, alpha: {qlearn.alpha}, score: {score}, max_score: {qlearn.max_score}")
            qlearn.showPerformance()

        
class AQLearning:
    def __init__(self):
        self.alpha = 0.7
        self.q_values = {}
        self.prev_state = "0_0_0_0"     # x = 0, y = 0, v= 0, y1 = 0
        self.curr_state = "0_0_0_0"
        self.prev_action = 0
        self.curr_action = 0
        self.score = 0

        # Episodes
        self.episode = 0
        self.scores = []
        self.max_score = 0

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
            self.q_values[state] = [0, 0]     

    def load_training_states(self):
        """Load current training state from json file."""
        #if self.train:
        print("Loading training states from json file...")
        try:
            with open("data/training_values_resume.json", "r") as f:
                training_state = json.load(f)
                self.episode = training_state['episodes'][-1]
                self.scores = training_state['scores']
                self.alpha = max(self.alpha - self.alpha * self.episode, 0.1)
                # self.epsilon = max(self.epsilon - self.epsilon_decay * self.episode, 0)
                if self.scores:
                    self.max_score = max(self.scores)
                else:
                    self.max_score = 0
        except IOError:
            pass  

    def get_state(self, x, y, vel, y1):
        state = str(x) + "_" + str(y) + "_" + str(vel) + "_" + str(y1)
        self.init_qvalues(state)
        return state

    def update_qvalues(self, score, death):
        # rewards
        self.episode += 1
        if score == self.score:
            # reward = 0
            reward = 0
        if death:
            reward = -1000
            self.episode += 1
            self.scores.append(score)

        self.score = score
        # Formula?
        self.q_values[self.prev_state][self.prev_action] = reward + self.alpha * max(self.q_values[self.curr_state][0], self.q_values[self.curr_state][1])
        self.previous_state = self.curr_state  # update the last_state with the current state

        if score > max(self.scores, default=0):
            print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print("$$$$$$$$ NEW RECORD: %d $$$$$$$$" % score)
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")

    def act(self, x, y, v, y1):
        # store the transition from previous state to current state
        self.prev_action = self.curr_action
        self.curr_state = self.get_state(x, y, v, y1)
        # Best action with respect to current state, default is 0 (do nothing), 1 is flap
        self.curr_action = 0 if self.q_values[self.curr_state][0] >= self.q_values[self.curr_state][1] else 1

        return self.curr_action


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

        print("\nEpisode: {}, highest score: {}, average: {}".format(num, max(self.scores, default=0), average[-1]))
        plt.figure(1)
        #plt.gca().get_xaxis().set_major_formatter(MaxNLocator(integer=True))
        plt.scatter(range(1, num+1), self.scores, c="green", s=3)
        plt.plot(range(1, num+1), average, 'b')
        plt.xlim((1,num))
        plt.ylim((0,int(max(self.scores)*1.1)))

        plt.title("Score distribution")
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.show(block=False)
        plt.pause(0.001)

    


if __name__ == '__main__':
    main()
