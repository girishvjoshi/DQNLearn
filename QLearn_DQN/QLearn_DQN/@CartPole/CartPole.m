%{ 
Auhtor: Girish Joshi
Date: 11/20/2017
This Class file simulates the Cart-Pole Dynamics
%}

classdef CartPole < handle
    
    properties (Access = 'protected')
        
        state = [];
        timeStep = [];
        massCart = [];
        massPole = [];
        poleLength = [];
        gravity = [];
        substeps = [];
        actionCardinality = [];
        actions = [];
        pendLimitAngle = [];
        cartLimitRange = [];
        reward = [];
        bonus = [];
        goal = [];
        resetCode = [];
        simOnOff;
        panel;
        cart;
        pole
        dot;
        arrow;
    end
    
    methods (Access = 'protected')
        
        function obj = CartPole(startPoint,simChoice)
            
            if nargin == 0                
                obj.state = [0 0 0 0];
                obj.simOnOff = 0;                
            else                
                obj.state = startPoint;
                obj.simOnOff = simChoice;                
            end
            obj.timeStep = 0.05;
            obj.massCart = 5;
            obj.massPole = 0.5;
            obj.gravity = 9.81;
            obj.poleLength = 1;
            obj.substeps = 1;
            obj.reward = 0;
            obj.bonus = 0;
            obj.actions = [-1 1];
            obj.actionCardinality = length(obj.actions);
            obj.pendLimitAngle = deg2rad(10);
            obj.cartLimitRange = 10;
            
            if obj.simOnOff == 1
                obj.initSim
            end
            
        end
        
        function [state,Action,reward,next_state,done] = doAction(obj,Action)
            
            state = obj.state;            
            next_state = obj.RK4(Action);   
            obj.checkIfGoalReached();            
            reward = obj.reward;            
            done = obj.resetCode;            
            if obj.simOnOff == 1
                obj.simCartpole(Action);
            end
            
        end
        
        function randomInitState(obj)
            obj.state = [-10*(1-2*rand) -1.5*(1-2*rand) 0.17*rand 0.1*rand];
        end
        
        function checkIfGoalReached(obj)
            
            obj.generateReward();
            
            if norm([obj.state(3) obj.state(4)]) < 0.01                
                obj.bonus = 10;
                obj.goal = true;
                obj.resetCode = false;
                
            elseif abs(obj.state(3)) >  obj.pendLimitAngle                
                obj.bonus = -10;     %punishement for falling down
                obj.resetCode = true;
                
            elseif abs(obj.state(1)) > obj.cartLimitRange                
                obj.bonus = -10;     %punishement for moving too far
                obj.resetCode = true;
                
            else
                obj.bonus = 0;
                obj.resetCode = false;
            end
            
            obj.reward = obj.reward + obj.bonus;
            obj.goal = false;
            obj.bonus = 0;
        end
        
        function initSim(obj)
            
            obj.panel = figure;
            obj.panel.Position = [100 100 1200 600];
            obj.panel.Color = [1 1 1];
            hold on
            obj.cart = plot(0,0,'m','Linewidth',50);
            obj.pole = plot(0,0,'b','LineWidth',10); % Pendulum stick
            axPend = obj.pole.Parent;
            axPend.XTick = [];
            axPend.YTick = [];
            axPend.Visible = 'off';
            axPend.Position = [0.35 0.4 0.3 0.3];
            axPend.Clipping = 'off';
            axis equal
            axis([-10 10 -5 5]);
            obj.dot = plot(0,0,'.k','MarkerSize',50);
            obj.arrow = quiver(0,0,-3,0,'linewidth',7,'color','r','MaxHeadSize',15);
            hold off         
            
        end
    end
    
    methods(Access = 'private')
        
        function Xdot = dynamicsCP(obj,state,Action)
            
            theta = state(3);            
            theta_dot = state(4);            
            A = [cos(theta) obj.poleLength; obj.massCart+obj.massPole  obj.massPole*obj.poleLength*cos(theta)];            
            B = [-obj.gravity*sin(theta); Action+obj.massPole*obj.poleLength*theta_dot^2*sin(theta)];            
            dynamic = 1\A*B;            
            Xdot = [state(2) dynamic(1)  state(4) dynamic(2)];
            
        end
        
        function Xstep = RK4(obj,Action)
            
            for i = 1:obj.substeps
                k1 = obj.dynamicsCP(obj.state,Action);
                k2 = obj.dynamicsCP(obj.state+obj.timeStep/2*k1,Action);
                k3 = obj.dynamicsCP(obj.state+obj.timeStep/2*k2,Action);
                k4 = obj.dynamicsCP(obj.state+obj.timeStep*k3,Action);                
                Xstep = obj.state + obj.timeStep/6*(k1 + 2*k2 + 2*k3 + k4);        
                % Map pendulum Angle from 0 to 360 to -180 to 180
                if Xstep(3)>pi
                    Xstep(3) = -pi + (Xstep(3)-pi);
                elseif Xstep(3)<-pi
                    Xstep(3) = pi - (-pi - Xstep(3));
                end                
            end            
        end
        
        function generateReward(obj)            
            obj.reward = 0;
            % obj.reward = (-obj.state(3).^2 -0.25*obj.state(4).^2-0.05*obj.state(1).^2);  %Uncomment for shaped Reward          
        end
        
        function simCartpole(obj,Action)
            
            set(obj.pole,'XData',[obj.state(1) obj.state(1)-10*sin(obj.state(3))]);
            set(obj.pole,'YData',[0 10*cos(obj.state(3))]);
            set(obj.cart,'XData',[obj.state(1)-2.5 obj.state(1)+2.5]);
            set(obj.cart,'YData',[0 0]);
            set(obj.dot,'XData',obj.state(1));
            set(obj.dot,'YData',0);
            set(obj.arrow,'XData',obj.state(1)+sign(Action)*3);
            set(obj.arrow,'YData',0);
            set(obj.arrow,'UData',sign(Action)*3);
            drawnow;
            
        end        
    end    
end