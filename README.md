Flappy Bird Computer Vision Model
===============

Train a neural network to play Flappy Bird with computer vision.

Flappy Bird made in python from [here](https://github.com/sourabhv/FlapPyBird) 

Setup 
---------------------------

1. Install Python 3.x (recommended) from [here](https://www.python.org/download/releases/) (Or use your preffered package manager)

2. _Optional_: Setup a virtual environment from [here](https://pypi.org/project/virtualenv/)

3. Clone the repository: 
    ```bash
   $ git clone https://github.com/waddupasd123/flappybird-cv.git
   ```
4. Install dependencies:

   ```bash
   $ pip install -r requirements.txt
   ```

5. To run training network (q-learning):

   ```bash
   $ python q_learning.py
   ```
   To start training from the beginning, delete data/q_values_resume.json and data/training_values_resume.json.
   (Make sure to back it up because it takes a long time to train.)

   If you want to play the game with object detection, then run this command by itself:

   ```bash
   $ python cv.py
   ```
   Uncomment line 75 and 76 in cv.py to also detect the top pipes but it may affect fps performance.

Description
---------------------------
Uses Open CV (computer vision) to detect positions of bird and pipes. It will then use a q-learning model to train network.

![Bottom pipes](result.jpg)


References
---------------------------
1. Flappy Bird made in python from [here](https://github.com/sourabhv/FlapPyBird) 

2. OpenCV tutorials and code used from [here](https://learncodebygaming.com/blog/tutorial/opencv-object-detection-in-games)

3. Q-learning model code for Flappy Bird extracted from [here](https://github.com/anthonyli358/FlapPyBird-Reinforcement-Learning)
