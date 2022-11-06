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

        #pipe1
        pipe1_img = cv.imread('pipe1_1.png', cv.IMREAD_ANYCOLOR)
        detect(pipe1_img, screenshot, 0.7)

        """ pipe_w = pipe1_img.shape[1]
        pipe_h = pipe1_img.shape[0]

        #redbird1_img = cv.imread('assets/sprites/redbird-downflap.png')

        result = cv.matchTemplate(screenshot, pipe1_img, cv.TM_CCOEFF_NORMED)

        #Show matching heatmap
        #cv.imshow('Result', result)
        #cv.waitKey()

        #print(result)

        threshold = 0.7
        locations = np.where(result >= threshold)
        # Convert to (x,y) positions
        locations = list(zip(*locations[::-1]))
        #print(locations)

        # Create list of [x, y, w, h] rectangles
        rectangles = []
        for loc in locations:
            rect = [int(loc[0]), int(loc[1]), pipe_w, pipe_h]
            rectangles.append(rect)
            rectangles.append(rect)

        rectangles, weights = cv.groupRectangles(rectangles, 1, 0.5)
        print(rectangles)

        if len(rectangles):
            print('Found!')

            pipe_w = pipe1_img.shape[1]
            pipe_h = pipe1_img.shape[0]
            line_colour = (0, 255, 0)
            line_type = cv.LINE_4

            for (x, y, w, h) in rectangles:
                top_left = (x, y)
                bottom_right = (x + w, y + h)

                cv.rectangle(screenshot, top_left, bottom_right, line_colour, line_type)
                #cv.imwrite('result.jpg', screenshot)

            cv.imshow('result.jpg', screenshot)

        else:
            print('Not found.') """

        cv.imshow('result.jpg', screenshot)

        # debug the loop rate
        print('FPS {}'.format(1 / (time() - loop_time)))
        loop_time = time()


        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()
            break


def detect(image, screenshot, threshold):
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
    print(rectangles)

    if len(rectangles):
        print('Found!')
        line_colour = (0, 255, 0)
        line_type = cv.LINE_4

        for (x, y, w, h) in rectangles:
            top_left = (x, y)
            bottom_right = (x + w, y + h)

            cv.rectangle(screenshot, top_left, bottom_right, line_colour, line_type)


if __name__ == '__main__':
    main()

