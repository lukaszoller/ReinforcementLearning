###########################################################################
# 1. makeMaze
###########################################################################
#
# Source: https://www.mathworks.com/examples/matlab/community/36132-maze-solving-using-q-learning-algorithm
#

import numpy as np

## Create Maze
# Parameters
# Side of maze
n = 12;
# Wallpenalty
wallPenalty = -50
# Path value
pathValue = 1
# Goal value
goalValue = 10

# Make maze with just walls
maze <- matrix(wallPenalty,n,n)

# add paths
for i in range(1, (n-3)*n):

for (i in 1:((n-3)*n)) {
maze[sample(1:n, 1), sample(1:n, 1)]= pathValue
}

# starting node
maze[1,1] = 1

# Goal node
maze[n,n] = goalValue

# display maze in matrix form
maze

###########################################################################
# 2. visualizeMaze
###########################################################################
#
# Source: https://www.mathworks.com/examples/matlab/community/36132-maze-solving-using-q-learning-algorithm
#

## Visualize Maze
# Check for at least one path between Start maze(1,1) and Goal maze(n,n)

ggplot

figure
imagesc(maze)               # print maze nicely
colormap(winter)            # specify color style
for i=1:n                   # for all location in maze set X for a wall
for j=1:n
if maze(i,j)==min(maze)
    text(j,i,'X','HorizontalAlignment','center')
end
end
end
# Place Start- and Goallabel
text(1,1,'START','HorizontalAlignment','center')
text(n,n,'GOAL','HorizontalAlignment','center')
axis off

###########################################################################
# 3. makeRewardMatrix
###########################################################################
#
# Source: https://www.mathworks.com/examples/matlab/community/36132-maze-solving-using-q-learning-algorithm
#

## Possible actions are:
# List of possible Actions
# Up    :  (i-n)
# Down  :  (i+n)
# Left  :  (i-1)
# Right :  (i+1)
# Diagonally SE  :  (i+n+1)
# Diagonally SW  :  (i+n-1)
# Diagonally NE  :  (i-n+1)
# Diagonally NW  :  (i-n-1)

##
# Make Reward matrix
Goal = 144;

# For each state (n*n) create zerosmatrix with all possible states (n*n)
# until now all states are considered possible.
reward=zeros(n*n);

# Fill this (n*n, n*n) matrix with the rewards from the maze
for i=1:Goal
reward(i,:)=reshape(maze',1,Goal);
end

# for all impossible States (not reacheble in 1 Step) set infinity.
for i=1:Goal
for j=1:Goal
if j~=i-n  && j~=i+n  && j~=i-1 && j~=i+1 && j~=i+n+1 && j~=i+n-1 && j~=i-n+1 && j~=i-n-1
reward(i,j)=-Inf;
end
end
end
for i=1:n:Goal
for j=1:i+n
if j==i+n-1 || j==i-1 || j==i-n-1
reward(i,j)=-Inf;
reward(j,i)=-Inf;
end
end
end


###########################################################################
# 4. fillQtable
###########################################################################
#
# Source: https://www.mathworks.com/examples/matlab/community/36132-maze-solving-using-q-learning-algorithm
#
# This q-Learning algorithm uses a 100# greedy policy.




## Q-Learning algorithm
# initialize q matrix

q = randn(size(reward));            #
gamma = 0.9;
alpha = 0.2;
maxItr = 50;

# Repeat until Convergence OR Maximum Iterations
for i=1:maxItr

# Starting from start position
cs=1;

# Repeat until Goal state is reached
while(1)

# possible actions for the chosen state
n_actions = find(reward(cs,:)>0);
# choose an action at random and set it as the next state
ns = n_actions(randi([1 length(n_actions)],1,1));

# find all the possible actions for the selected state
n_actions = find(reward(ns,:)>=0);

# find the maximum q-value i.e, next state with best action
max_q = 0;
for j=1:length(n_actions)
max_q = max(max_q,q(ns,n_actions(j)));
end

# Update q-values as per Bellman's equation
q(cs,ns)=reward(cs,ns)+gamma*max_q;

# Check whether the episode has completed i.e Goal has been reached
if(cs == Goal)
    break;
end

# Set current state as next state
cs=ns;
end
end


###########################################################################
# 5. solveMaze
###########################################################################
#
# Source: https://www.mathworks.com/examples/matlab/community/36132-maze-solving-using-q-learning-algorithm
#



## Solve the maze
# Starting from the first position
start = 1;
move = 0;
path = start;

# Iterating until Goal-State is reached
while(move~=Goal)
    [~,move]=max(q(start,:));

    # Deleting chances of getting stuck in small loops  (upto order of 4)
    if ismember(move,path)
        [~,x]=sort(q(start,:),'descend');
    move=x(2);
    if ismember(move,path)
        [~,x]=sort(q(start,:),'descend');
    move=x(3);
    if ismember(move,path)
        [~,x]=sort(q(start,:),'descend');
    move=x(4);
    if ismember(move,path)
        [~,x]=sort(q(start,:),'descend');
    move=x(5);
    end
    end
    end
    end

    # Appending next action/move to the path
    path=[path,move];
    start=move;
    end



    ###########################################################################
    # 6. displaySolution
    ###########################################################################
    #
    # Source: https://www.mathworks.com/examples/matlab/community/36132-maze-solving-using-q-learning-algorithm
    #


    ## Optimal Path between Start and Goal

    fprintf('Final path: #s',num2str(path))

    fprintf('Total steps: #d',length(path))

    # reproducing path to matrix path
    pmat=zeros(n,n);
    [q, r]=quorem(sym(path),sym(n));
    q=double(q);r=double(r);
    q(r~=0)=q(r~=0)+1;r(r==0)=n;
    for i=1:length(q)
    pmat(q(i),r(i))=50;
    end

    ## Plot the maze


    figure
    imagesc(pmat)
    colormap(white)
    for i=1:n
    for j=1:n
    if maze(i,j)==min(maze)
        text(j,i,'X','HorizontalAlignment','center')
    end
    if pmat(i,j)==50
        text(j,i,'\bullet','Color','red','FontSize',28)
    end
    end
    end
    text(1,1,'START','HorizontalAlignment','right')
    text(n,n,'GOAL','HorizontalAlignment','right')
    hold on
    imagesc(maze,'AlphaData',0.2)
    colormap(winter)
    hold off
    axis off


