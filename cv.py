import cv2 as cv
import numpy as np
from windowcapture import WindowCapture
import flappy_mod as Flappy
from time import time
#import flappy

""" WindowCapture.list_window_names()
exit() """

def main():
    Flappy.launch()
    movementInfo = Flappy.initialSetup()
    gameInfo = Flappy.mainGameSetup(movementInfo)

    # initialize the WindowCapture class
    wincap = WindowCapture('Flappy Bird')
    loop_time = time()

    while True:
        # Go to next frame
        crash = Flappy.mainGame(gameInfo)
        if (crash):
            Flappy.launch()
            movementInfo = Flappy.initialSetup()
            gameInfo = Flappy.mainGameSetup(movementInfo)


        # get an updated image of the game
        screenshot = wincap.get_screenshot()
        #frame = cv.cvtColor(screenshot, cv.COLOR_RGB2GRAY)

        # Game lags with more than 2 objects to detect
        # Will sometimes not detect because score will cover pipe
        # Hide score?

        # Up pipes
        pipe1_img = cv.imread('refer/pipe_1.png', cv.IMREAD_ANYCOLOR)
        upPipes = detect(pipe1_img, screenshot, 0.8, (0, 255, 0))

        # Down pipes
        pipe3_img = cv.imread('refer/pipe_3.png', cv.IMREAD_ANYCOLOR)
        downPipes = detect(pipe3_img, screenshot, 0.8, (255, 0, 0))

        # Bird
        bird_img = cv.imread('refer\yellowbird-midflap.png', cv.IMREAD_ANYCOLOR)
        birdLoc = detect(bird_img, screenshot,  0.6, (0, 0, 255))


        cv.imshow('result.jpg', screenshot)

        # debug the loop rate
        #print('FPS {}'.format(1 / (time() - loop_time)))
        #loop_time = time()


        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()
            break


def detect(image, screenshot, threshold, colour):
    result = cv.matchTemplate(screenshot, image, cv.TM_CCOEFF_NORMED)

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
    #print(rectangles)

    if len(rectangles):
        line_type = cv.LINE_4

        for (x, y, w, h) in rectangles:
            top_left = (x, y)
            bottom_right = (x + w, y + h)

            cv.rectangle(screenshot, top_left, bottom_right, colour, line_type)
    
    return rectangles


if __name__ == '__main__':
    main()

