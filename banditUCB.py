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

############################################################################
# Bandit UCB: 2.7 Upper-Confidence-Bound Action Selection
#
# Barto, Sutton (2018): Reinforcement Learning. An Introduction. Second Edition.
# Cambridge. p.35.

# This is a replica of the Figure 2.4 (page 36) with a comparision of UCB and
# epsilon-greedy.
# Other than epsilon-greedy, UCB selects actions in exploration not randomly but
# taking into account both how close their estimates are to being maximal and the
# uncertainties in those estimates.

############################################################################

# number of runs
runs = 2000
# maximum steps
maxSteps = 1000

# prepare arrays for performance evaluation
averageRewardArrayGreedy = np.zeros(maxSteps)
averageRewardArrayUCB = np.zeros(maxSteps)

for runs in range(0, runs):

    # Initialize parameters
    # number of arms
    numberOfArms = 10
    # epsilon
    epsilon = 0.1
    # Constant for UCB
    c = 2

    # Stepcount
    stepCount = 1

    # initialize action estimates
    actionEstimatesGreedy = np.zeros(numberOfArms)
    actionEstimatesUCB = np.zeros(numberOfArms)

    # initialize true action values
    trueActionValuesGreedy = np.random.normal(0, 1, numberOfArms)
    trueActionValuesUCB = np.random.normal(0, 1, numberOfArms)

    # Array for counting usage of action
    actionCountArray = np.zeros(numberOfArms)+1     # +1 --> avoid division by 0

    # Reward function
    def reward(action):
        """Takes an action and returns a reward. The reward is chosen randomly from
        a normal distribution with mean q*(a) (trueActionValue) and variance 1"""
        return np.random.normal(trueActionValuesGreedy[action], 1, 1)

    # Algorithm
    while stepCount < maxSteps:
        # Choose action Greedy
        # if randomNumber < epsilon explore, otherwise exploit
        randomNumberGreedy = np.random.uniform(0, 1, 1)
        if randomNumberGreedy < epsilon:
            actionGreedy = np.random.choice(np.arange(1, numberOfArms))
        else:
            actionGreedy = np.argmax(actionEstimatesGreedy)

        # Choose action UCB
        # if randomNumber < epsilon explore, otherwise exploit
        randomNumberUCB = np.random.uniform(0, 1, 1)
        if randomNumberUCB < epsilon:
            actionUCB = np.random.choice(np.arange(1, numberOfArms))
        else:
            actionUCB = np.argmax(actionEstimatesUCB + c*np.sqrt(np.log(stepCount)/actionCountArray))
            # actionUCB = np.argmax(actionEstimatesUCB)

        # Count action
        actionCountArray[actionUCB] = actionCountArray[actionUCB]+1

        # get Reward
        rGreedy = reward(actionGreedy)
        rUCB = reward(actionUCB)
        # Evaluation: Get Reward per Step
        averageRewardArrayGreedy[stepCount] = averageRewardArrayGreedy[stepCount] + rGreedy
        averageRewardArrayUCB[stepCount] = averageRewardArrayUCB[stepCount] + rUCB
        # update action Estimates
        actionEstimatesGreedy[actionGreedy] = actionEstimatesGreedy[actionGreedy] + 1 / stepCount * (rGreedy - actionEstimatesGreedy[actionGreedy])
        actionEstimatesUCB[actionUCB] = actionEstimatesUCB[actionUCB] + 1 / stepCount * (rUCB - actionEstimatesUCB[actionUCB])

        # increase stepCount
        stepCount += 1


plt.plot(averageRewardArrayGreedy / runs, label="Base case")
plt.plot(averageRewardArrayUCB / runs, label="UCB")
plt.legend()
plt.show()
