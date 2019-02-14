import numpy as np
import matplotlib.pyplot as plt

############################################################################
# k-armed bandit

# this is a replica of the 10-armed Testbed (Chapter 2.3)

############################################################################

# number of runs
runs = 2000
# maximum steps
maxSteps = 1000

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
    # initialize true action values
    trueActionValues = np.random.normal(0, 1, numberOfArms)

    # Reward function
    def reward(action):
        """Takes an action and returns a reward. The reward is chosen randomly from
        a normal distribution with mean q*(a) (trueActionValue) and variance 1"""
        return np.random.normal(trueActionValues[action], 1, 1)

    # for i in range(1,100):
    #     print(reward(1))

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
        if action == np.argmax(trueActionValues):
            optimalActionArray[stepCount] = optimalActionArray[stepCount]+1

    # get Reward
        r = reward(action)
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