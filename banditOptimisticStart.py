import numpy as np
import matplotlib.pyplot as plt

############################################################################
# Replica of figure 2.3: Optimistic initial values
#
# Barto, Sutton (2018): Reinforcement Learning. An Introduction. Second Edition.
# Cambridge. p.34.

# this is a modified version of the base bandit problem. In this case the
# initial action estimates q(a) are selected not 0 like in the base case
# but 5.

# Action value evaluation method: sample average


############################################################################

# number of runs
from numpy.core.multiarray import ndarray

runs: int = 2000
# maximum steps
maxSteps: int = 10000

# prepare arrays for performance evaluation
optimalActionArrayBase: ndarray = np.zeros(maxSteps)
optimalActionArrayOptimal: ndarray = np.zeros(maxSteps)

for runs in range(0, runs):

    ### Initialize parameters
    # number of arms
    numberOfArms = 10
    # epsilon
    epsilonBase = 0.1
    epsilonOptimal = 0

    # Stepcount
    stepCount = 1

    # initialize action estimates
    actionEstimatesBase = np.zeros(numberOfArms)
    actionEstimatesOptimal = np.zeros(numberOfArms) + 5
    # initialize true action values
    trueActionValuesBase = np.random.normal(0, 1, numberOfArms)
    trueActionValuesOptimal = np.random.normal(0, 1, numberOfArms)

    # Reward function
    def reward(action, trueActionValuesArray):
        """Takes an action and returns a reward. The reward is chosen randomly from
        a normal distribution with mean q*(a) (trueActionValue) and variance 1"""
        return np.random.normal(trueActionValuesArray[action], 1, 1)

    # Algorithm
    while stepCount < maxSteps:
        # Choose action

        # Base case: if randomNumber < epsilon explore, otherwise exploit
        randomNumberBase = np.random.uniform(0, 1, 1)
        if randomNumberBase < epsilonBase:
            actionBase = np.random.choice(np.arange(1, numberOfArms))
        else:
            actionBase = np.argmax(actionEstimatesBase)

        # Optimal: if randomNumber < epsilon explore, otherwise exploit
        randomNumberOptimal = np.random.uniform(0, 1, 1)
        if randomNumberOptimal < epsilonOptimal:
            actionOptimal = np.random.choice(np.arange(1, numberOfArms))
        else:
            actionOptimal = np.argmax(actionEstimatesOptimal)

    # Evaluation : Check if optimal action is chosen and store information in optimalActionArray
        if actionBase == np.argmax(trueActionValuesBase):
            optimalActionArrayBase[stepCount] = optimalActionArrayBase[stepCount]+1
        if actionOptimal == np.argmax(trueActionValuesOptimal):
            optimalActionArrayOptimal[stepCount] = optimalActionArrayOptimal[stepCount]+1

    # get Reward
        rBase = reward(actionBase, trueActionValuesBase)
        rOptimal = reward(actionOptimal, trueActionValuesOptimal)

        # update action Estimates
        actionEstimatesBase[actionBase] = actionEstimatesBase[actionBase] + 1 / stepCount * (rBase - actionEstimatesBase[actionBase])
        actionEstimatesOptimal[actionOptimal] = actionEstimatesOptimal[actionOptimal] + 1 / stepCount * (rOptimal - actionEstimatesOptimal[actionOptimal])

    # increase stepCount
        stepCount += 1


plt.plot(optimalActionArrayBase/runs, label="Base case")
plt.plot(optimalActionArrayOptimal/runs, label="Optimistic initial values")
plt.legend()
plt.show()

