import numpy as np
import matplotlib.pyplot as plt

############################################################################
# Solution of exercise 2.11: Parameter plot of k-armed bandit, nonstationary
#
# Barto, Sutton (2018): Reinforcement Learning. An Introduction. Second Edition.
# Cambridge. p.44.

# this is a modified version of the base bandit problem
# other than the base case, the q*(a) start out equal and then take independent
# random walks (by adding a normally distributed increment with mean zero and
# standard deviation 0.01 to all the q*(a) on each step.

############################################################################

# number of runs
runs = 1000
# maximum steps
maxSteps = 20000

# prepare parameter array
exponentArray = range(-7, 3)
parameterArray = np.zeros(10)+2
parameterArray = np.power(parameterArray, exponentArray)

# prepare arrays for performance evaluation
avgRwrdPerParameterArray = np.zeros(len(parameterArray))

# first loop: over all parameters
for i in range(0, len(parameterArray)):

    # initialize average Reward
    avgRewardPerParameter = 0

    runCounter = 1

    # second loop over all runs
    for runs in range(0,runs):

        ### Initialize parameters
        # average reward per run
        avgRewardPerRun = 0
        # number of arms
        numberOfArms = 10
        # epsilon
        epsilon = parameterArray[i]

        # Stepcount
        stepCount = 1
        # Average denominator (get average over the last ... steps)
        avgDenominator = 10000

        # initialize action estimates
        actionEstimates = np.zeros(numberOfArms)
        # initialize true action values --> All q*(a) start equal
        changingActionValues = np.zeros(numberOfArms)

        # Reward function
        def reward(action):
            """Takes an action and returns a reward. The reward is chosen randomly from
            a normal distribution with mean q*(a) (trueActionValue) and variance 1"""
            return np.random.normal(changingActionValues[action], 1, 1)


        # 3rd loop: over all steps per run
        while stepCount < maxSteps:
            # Choose action
            # if randomNumber < epsilon explore, otherwise exploit
            randomNumber = np.random.uniform(0, 1, 1)
            if randomNumber < epsilon:
                action = np.random.choice(np.arange(1, numberOfArms))
            else:
                action = np.argmax(actionEstimates)

            ################ Difference to banditBaseCase ######################################################################
            # Change q*(a): each step a random increment is added to the true values
            # mean = 0; standard deviation = 0.01
            changingActionValues[action] = changingActionValues[action] + np.random.normal(0, 0.01, 1)

            ################ Difference to banditBaseCase ######################################################################


            # get Reward
            r = reward(action)
            # change Action Values (nonstationary Problem)
            changingActionValues = changingActionValues + np.random.normal(0, 0.01, 1)

            # Update action Estimates (average of last [averageDenominator] steps)
            if stepCount < avgDenominator:          # this is the common average with stepsize = 1/n
                actionEstimates[action] = actionEstimates[action] + 1/stepCount * (r - actionEstimates[action])
            else:
                actionEstimates[action] = (avgDenominator-1)*actionEstimates[action]/avgDenominator + 1/avgDenominator * (r - actionEstimates[action])

            # Evaluation: update average reward per run (over all steps)
            avgRewardPerRun = avgRewardPerRun + (r - avgRewardPerRun)*1/stepCount

            # increase stepCount
            stepCount += 1


        # update average reward per parameter (over all runs)
        avgRewardPerParameter = avgRewardPerParameter + (avgRewardPerRun - avgRewardPerParameter)*1/runCounter

    avgRwrdPerParameterArray[i] = avgRewardPerParameter

plt.plot(parameterArray, avgRwrdPerParameterArray)
plt.semilogx(basex=2)
plt.show()
