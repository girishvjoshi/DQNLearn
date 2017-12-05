%{ 
Auhtor: Girish Joshi
Date: 11/20/2017
This class caries out Action Value iteration using DQN
%}

classdef DQNLearn < CartPole  
    
    properties (Access = 'private')
        
        maxEpoch = [];
        maxIttr = [];
        gamma = [];
        batch = [];
        Q = [];
        newQ = [];
        targetQ = [];
        maxQ = [];
        epsilon = [];
        epsilonDecay = [];
        annealing = [];
        totalReward = [];
        testtotalReward = [];
        UpdateIndex = [];
        TDerror = [];
        net = [];
        net_prev = [];
        replayBuffer = [];
        bufferSize = [];
        sampleSize = [];
        netWeights = [];
        simChoice;    
    end
   
    properties (Access = 'public')        
        epochAvgReward = [];
        epochLength = [];        
    end
    
    methods (Access = 'public')
        
        function obj = DQNLearn(epoch,ittr,gamma,epsilon,simStatus)
                        
            obj = obj@CartPole([0 0 0 0],simStatus);
            if (nargin == 0)                 
                obj.maxEpoch = 100;
                obj.maxIttr = 1000;
                obj.gamma = 0.99;
                obj.epsilon = 0.8;   
            else                
                obj.maxEpoch = epoch;
                obj.maxIttr = ittr;
                obj.gamma = gamma;
                obj.epsilon = epsilon;                               
            end
            
            obj.batch = 1;
            obj.epsilonDecay = 0.999;
            obj.annealing = 0.8;
            obj.totalReward = 0;
            obj.goal = false;
            obj.resetCode = false;
            
            %Initialize the Q-Network
            layer1Size = 10;
            layer2Size = 20;            
            obj.net = fitnet([layer1Size layer2Size],'trainlm');
            obj.net_prev = obj.net;
            obj.net.trainParam.lr = 0.1;
            obj.net.trainParam.epochs = 10;
            obj.net.trainParam.showWindow = false;
            obj.net.trainParam.lr_dec = 0.8;
            obj.net = train(obj.net,rand(length(obj.state),25),rand(length(obj.actions),25));
            obj.netWeights = getwb(obj.net);
            
            % Initialize Data Recording
            obj.epochLength = zeros(1,obj.maxEpoch);
            obj.epochAvgReward = zeros(1,obj.maxEpoch);
            
            % Initialize the replay Buffer
            [state,Action,reward,next_state,done] = obj.doAction(1);
            obj.bufferSize = 10000;
            obj.replayBuffer = [state,Action,reward,next_state,done];
            obj.sampleSize = 100;
            
        end
        % Iniate DQN-Learning
        function QLearningTrain(obj)
            
            for epochs = 1:obj.maxEpoch                
                %Reset the parameters
                obj.totalReward = 0;
                obj.bonus = 0;
                % Epsilon Decay                
                obj.epsilon = obj.epsilon*obj.epsilonDecay;                
                              
                obj.randomInitState();
                
                for itr_no = 1:obj.maxIttr
                    
                    cartForce = obj.selectAction();                   
                    [state,Action,reward,next_state,done] = obj.doAction(cartForce); % Propogate the Plant                   
                    obj.addtoReplaybuffer(state,Action,reward,next_state,done);                  
                    obj.state = next_state;
                    %Aggregating the Total Reward for every Epoch
                    obj.totalReward = obj.totalReward + reward;
                    
                    if obj.resetCode
                        break;
                    end
                    
                end
                
                % Store the Average Reward and Epoch Length for plotting
                % the network Performance
                obj.epochAvgReward(epochs) = obj.totalReward/itr_no;
                obj.epochLength(epochs) = itr_no;
                
                if itr_no == obj.maxIttr                    
                    disp(['Episode',num2str(epochs),':Successful Balance Achieved!- Average Reward:', num2str(obj.epochAvgReward(epochs))]);                    
                elseif obj.resetCode == true                    
                    disp(['Episode',num2str(epochs),': Reset Condition reached!!!- Average Reward:', num2str(obj.epochAvgReward(epochs))]);
                    obj.resetCode = false;                    
                end                
              
                obj.trainOnBuffer()               
            end
            
        end
        
        function addtoReplaybuffer(obj,state,Action,reward,next_state,done)            
            if length(obj.replayBuffer) < obj.bufferSize                
                obj.replayBuffer = [obj.replayBuffer;[state,Action,reward,next_state,done]];
            else
                obj.replayBuffer(1,:) = [];
                obj.replayBuffer = [obj.replayBuffer;[state,Action,reward,next_state,done]];
            end
        end
        
        function DQNTest(obj,simLength)
            
            obj.simOnOff = 1;
            obj.testtotalReward = 0;
            obj.epsilon = -Inf;
            obj.initSim();
            obj.randomInitState();            
            
            for testIter  = 1:simLength                
                cartForce = obj.selectAction();
                [~,~,reward,next_state,done] = obj.doAction(cartForce);
                obj.state = next_state;
                obj.testtotalReward = obj.testtotalReward + reward;
                if done
                    break; % If Reset Condition is Reached; Break
                end
            end
        end
    end
    
    methods (Access = 'private')
        
        function Qval = genQvalue(obj,state)            
            Qval = obj.net(state');            
        end
        
        function trainOnBuffer(obj)
            
            sampledrawfromBuffer = datasample(obj.replayBuffer,min(obj.sampleSize,length(obj.replayBuffer)));
            stateBatch = sampledrawfromBuffer(:,[1:4]);
            actionBatch = sampledrawfromBuffer(:,5);
            rewardBatch = sampledrawfromBuffer(:,6);
            nextstateBatch = sampledrawfromBuffer(:,[7:10]);
            doneBatch = sampledrawfromBuffer(:,11);
            valueBatch = zeros(length(obj.actions),1);
            
            for count = 1:length(sampledrawfromBuffer)
                
                value = obj.genQvalue(stateBatch(count,:));
                aIdx = find(~(obj.actions-actionBatch(count)));
                if doneBatch(count)
                    value(aIdx) = rewardBatch(count);
                else
                    value(aIdx) = rewardBatch(count) + obj.gamma*max(obj.genQvalue(nextstateBatch(count,:)));
                end
                valueBatch(:,count) = value;
            end
            
            obj.net = setwb(obj.net,obj.netWeights);
            obj.net = train(obj.net, stateBatch',valueBatch);
            obj.netWeights = getwb(obj.net);
            
        end
        
        function selectedAction = selectAction(obj)
            
            if rand <= obj.epsilon                
                actionIndex = randi(obj.actionCardinality,1);                
            else                
                obj.Q = obj.genQvalue(obj.state);                
                [~,actionIndex] = max(obj.Q);                
            end
            
            selectedAction = obj.actions(actionIndex);
        end
        
    end
end
