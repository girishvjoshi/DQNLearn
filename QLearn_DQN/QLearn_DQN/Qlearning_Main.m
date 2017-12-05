%{
Auhtor: Girish Joshi
Date: 11/20/2017
This Script solves the Cart pole balance task using DQN

Input : Qlearning( No of Epochs, No of Itteration in each Epoch, Dicount Factor,
epsilon, simulation Choice )
%}
% Training Initialization
CartPoleQlearn = DQNLearn(100,300,1,1,false);

% Training Start
CartPoleQlearn.QLearningTrain();

%% Testing the DQN Network
test_simSteps = 4000;
CartPoleQlearn.DQNTest(test_simSteps);

