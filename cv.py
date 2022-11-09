import cv2 as cv
import numpy as np
from windowcapture import WindowCapture
import flappy_mod as Flappy
from time import time

""" WindowCapture.list_window_names()
exit() """

class ComputerVision:

    def __init__(self) -> None:
        self.movementInfo = None
        self.gameInfo = None
        self.wincap = None
        self.X = 0          # x-distance to next pipe
        self.Y = 0          # y-distance to next pipe
        self.lastY = 236    # last y-coordinate of the bird
        self.V = 0          # velocity of bird
        self.Y1 = 0         # y-distance between two pipes

    def getX(self):
        return self.X

    def getY(self):
        return self.Y
    
    def getV(self):
        return self.V

    def getY1(self):
        return self.Y1

    def setup(self):
        Flappy.launch()
        self.movementInfo = Flappy.initialSetup()
        self.gameInfo = Flappy.mainGameSetup(self.movementInfo)

        # initialize the WindowCapture class
        self.wincap = WindowCapture('Flappy Bird')

    def action(self, action):
        return Flappy.action(self.gameInfo, action)
    
    def keyInput(self):
        return Flappy.keyInput(self.gameInfo)

    def getScore(self):
        return self.gameInfo['score']

    def nextFrame(self):
        # Go to next frame
        #Flappy.action(self.gameInfo)
        #loop_time = time()
        self.gameInfo, crash = Flappy.mainGame(self.gameInfo)
        if (crash):
            score = self.gameInfo['score']
            self.setup()
            self.X = 0      
            self.Y = 0          
            self.lastY = 236    
            self.V = 0          
            self.Y1 = 0  
            return True, score


        # get a frane of the game
        frame = self.wincap.get_screenshot()
        #frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

        # Game lags really badly with more than 3 objects to detect
        # Will sometimes not detect because score will cover pipe
        # Hide score?

        # Up pipes
        pipe1_img = cv.imread('refer/pipe_1.png', cv.IMREAD_ANYCOLOR)
        upPipes = self.detect(pipe1_img, frame, 0.8, (0, 255, 0))

        # Down pipes - not really needed
        # Comment out to reduce lag
        #pipe3_img = cv.imread('refer/pipe_3.png', cv.IMREAD_ANYCOLOR)
        #downPipes = self.detect(pipe3_img, frame, 0.8, (255, 0, 0))

        # Bird
        # Doesn't work consistently
        # Using features is also inconsistent and slow
        # Removed rotation and blue bird
        bird1_img = cv.imread('refer/cropped_bird.png', cv.IMREAD_ANYCOLOR)
        bird2_img = cv.imread('refer/bird_1.png', cv.IMREAD_ANYCOLOR)
        birdLoc = self.detectBird(bird1_img, frame, 0.55, (0, 0, 255), bird2_img)

        if birdLoc:
            self.V = -(birdLoc[0][1] - self.lastY)
            self.lastY = birdLoc[0][1]
            self.X = 0
            self.Y = 0
        else:
            # Bird out of screen
            self.V = 0
            self.lastY = 0 
            # Impossible values   
            self.X = -1
            self.Y = 9999

        if birdLoc and upPipes:
            if upPipes[0][0] < birdLoc[0][0]:
                self.X = upPipes[0][1] - birdLoc[0][0]
                self.Y = birdLoc[0][0] - upPipes[0][1]
            else:
                self.X = upPipes[0][0] - birdLoc[0][0]
                self.Y = birdLoc[0][0] - upPipes[0][0]

        if upPipes and len(upPipes) > 1:
            self.Y1 = upPipes[0][1] - upPipes[1][1]

        cv.imshow('result.jpg', frame)

        # debug the loop rate
        #print('FPS {}'.format(1 / (time() - loop_time)))
        #loop_time = time()

        return False, self.gameInfo['score']


    def detect(self, image, frame, threshold, colour):
        result = cv.matchTemplate(frame, image, cv.TM_CCOEFF_NORMED)

        locations = np.where(result >= threshold)
        # Convert to (x,y) positions
        locations = list(zip(*locations[::-1]))
        #print(locations)

        # Create list of [x, y, w, h] rectangles
        width = image.shape[1]
        height = image.shape[0]

        rectangles = []
        for loc in locations:
            rect = [int(loc[0]), int(loc[1]), width, height]
            rectangles.append(rect)
            rectangles.append(rect)

        rectangles, weights = cv.groupRectangles(rectangles, 1, 0.5)
        rectangles = sorted(rectangles, key=lambda x: x[0])
        #print(rectangles)

        if len(rectangles):
            line_type = cv.LINE_4

            for (x, y, w, h) in rectangles:
                top_left = (x, y)
                bottom_right = (x + w, y + h)

                cv.rectangle(frame, top_left, bottom_right, colour, line_type)
        
        return rectangles

    def detectBird(self, image, frame, threshold, colour, reference):
        result = cv.matchTemplate(frame, image, cv.TM_CCOEFF_NORMED)

        locations = np.where(result >= threshold)
        # Convert to (x,y) positions
        locations = list(zip(*locations[::-1]))
        #print(locations)

        # Create list of [x, y, w, h] rectangles
        width = reference.shape[1]
        height = reference.shape[0]

        rectangles = []
        for loc in locations:
            rect = [int(loc[0]) - 10, int(loc[1]) - 2, width, height]
            rectangles.append(rect)
            rectangles.append(rect)

        rectangles, weights = cv.groupRectangles(rectangles, 1, 0.5)
        rectangles = sorted(rectangles, key=lambda x: x[0])
        #print(rectangles)

        if len(rectangles):
            line_type = cv.LINE_4

            for (x, y, w, h) in rectangles:
                top_left = (x, y)
                bottom_right = (x + w, y + h)

                cv.rectangle(frame, top_left, bottom_right, colour, line_type)
        
        return rectangles


def main():
    vision = ComputerVision()
    vision.setup()

    while True:
        vision.keyInput()
        # Go to next frame
        vision.nextFrame()
    
if __name__ == '__main__':
    main()

