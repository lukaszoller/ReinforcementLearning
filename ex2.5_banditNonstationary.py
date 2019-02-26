import numpy as np
import matplotlib.pyplot as plt

############################################################################
# Solution of exercise 2.5: k-armed bandit, nonstationary
#
# Barto, Sutton (2018): Reinforcement Learning. An Introduction. Second Edition.
# Cambridge. p.33.

# this is a modified version of the base bandit problem
# other than the base case, the q*(a) start out equal and then take independent
# random walks (by adding a normally distributed increment with mean zero and
# standard deviation 0.01 to all the q*(a) on each step.

############################################################################

# number of runs
runs = 2000
# maximum steps
maxSteps = 10000

# prepare arrays for performance evaluation
averageRewardArray = np.zeros(maxSteps)
optimalActionArray = np.zeros(maxSteps)

for runs in range(0,runs):

    ### Initialize parameters
    # number of arms
    numberOfArms = 10
    # epsilon
    epsilon = 0.1

    # Stepcount
    stepCount = 1

    # initialize action estimates
    actionEstimates = np.zeros(numberOfArms)
    # initialize true action values --> All q*(a) start equal
    changingActionValues = np.zeros(numberOfArms)

    # Reward function
    def reward(action):
        """Takes an action and returns a reward. The reward is chosen randomly from
        a normal distribution with mean q*(a) (trueActionValue) and variance 1"""
        return np.random.normal(changingActionValues[action], 1, 1)


    # Algorithm
    while stepCount < maxSteps:
    # Choose action
        # if randomNumber < epsilon explore, otherwise exploit
        randomNumber = np.random.uniform(0, 1, 1)
        if randomNumber < epsilon:
            action = np.random.choice(np.arange(1, numberOfArms))
        else:
            action = np.argmax(actionEstimates)

    # Evaluation : Check if optimal action is chosen and store information in optimalActionArray
        if action == np.argmax(changingActionValues):
            optimalActionArray[stepCount] = optimalActionArray[stepCount]+1

    ################ Difference to banditBaseCase ######################################################################
    # Change q*(a): each step a random increment is added to the true values
    # mean = 0; standard deviation = 0.01
        changingActionValues[action] = changingActionValues[action] + np.random.normal(0, 0.01, 1)

    ################ Difference to banditBaseCase ######################################################################


    # get Reward
        r = reward(action)
    # change Action Values (nonstationary Problem)
        changingActionValues = changingActionValues + np.random.normal(0, 0.01, 1)
    # Evaluation: Get Reward per Step
        averageRewardArray[stepCount] = averageRewardArray[stepCount]+r
    # update action Estimates
        actionEstimates[action] = actionEstimates[action] + 1/stepCount * (r - actionEstimates[action])

    # increase stepCount
        stepCount += 1


plt.plot(averageRewardArray/runs)
plt.show()

plt.plot(optimalActionArray/runs)
plt.show()