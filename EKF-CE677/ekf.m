
control_points = [-10	0;
0	-10;
10	0;
0	10 ];

observations = [ 9.72	9.72	22.08	22.08;
8.77	8.77	20.74	20.74;
7.97	7.97	19.42	19.42;
7.34	7.34	18.1	18.1;
6.93	6.93	16.81	16.81;
6.79	6.79	15.53	15.53;
6.93	6.93	14.28	14.28;
7.34	7.34	13.06	13.06;
7.97	7.97	11.88	11.88;
8.77	8.77	10.76	10.76;
9.72	9.72	9.72	9.72;
10.76	10.76	8.77	8.77;
11.88	11.88	7.97	7.97;
13.06	13.06	7.34	7.34;
14.28	14.28	6.93	6.93;
15.53	15.53	6.79	6.79;
16.81	16.81	6.93	6.93;
18.1	18.1	7.34	7.34;
19.42	19.42	7.97	7.97;
20.74	20.74	8.77	8.77;
22.08	22.08	9.72	9.72 ];

%state vector will consists of X,Y,V_x,V_y
%{
syms xc yc vx vy

%Define the intial estimate value(X_0 and P_0)
eqn1 = sqrt((control_points(1,1) - xc)^2 + (control_points(1,2) - yc)^2) == 9.72;
eqn2 = sqrt((control_points(2,1) - xc)^2 + (control_points(2,2) - yc)^2) == 9.72;

sol = solve([eqn1, eqn2 ], [xc, yc]);


eqn3 = sqrt((control_points(1,1) - xc)^2 + (control_points(1,2) - yc)^2) == 8.77;
eqn4 = sqrt((control_points(2,1) - xc)^2 + (control_points(2,2) - yc)^2) == 8.77;

sol1 = solve([eqn3, eqn4 ], [xc, yc]);
%}

%initial assumed values of state and covariance by solving above equations.

initial_state = [-9.715 ;1.0475; -9.715 ; 1.0475];

%if we are assuming that there is no correlation between the confidence
%on the state is overestimated so we assume a correlation.
intital_covariance  = [0.8 , 0 , 0 ,0 ; 
                       0 , 0.8 , 0 ,0 ; 
                       0 , 0 , 0.8 ,0 ;
                       0 , 0 , 0 ,0.8 ];

%state propagation model
del_t = 1;

F_k  = [1,del_t,0,0;
        0,1,0,0;
        0,0,1,del_t;
        0,0,0,1];

%Different values of Q and R taken to study the difference.

%Q and R are normally distributed
Q = [0.1 , 0 , 0 ,0 ; 
     0 , 0.1 , 0 ,0 ; 
     0 , 0 , 0.1 ,0 ;
     0 , 0 , 0 ,0.1 ];


 
R = [0.1 , 0 , 0 ,0 ; 
     0 , 0.1 , 0 ,0 ; 
     0 , 0 , 0.1 ,0 ;
     0 , 0 , 0 ,0.1 ];

%Different values of Q and R taken%
%{
Q1 = [0.5 , 0 , 0, 0 ; 
     0 , 0.5 , 0 ,0 ; 
     0 , 0 , 0.5 ,0 ;
     0 , 0 , 0 ,0.5 ]; 
 
R1 = [0.5 , 0 , 0, 0 ; 
     0 , 0.5 , 0 ,0 ; 
     0 , 0 , 0.5 ,0 ;
     0, 0 , 0 ,0.5 ];

 
Q2 = [0.2 , 0 , 0, 0 ; 
     0 , 0.4 , 0 ,0 ; 
     0 , 0 , 0.6 ,0 ;
     0 , 0 , 0 ,0.8 ]; 
 
R2 = [0.1 , 0 , 0, 0 ; 
     0 , 0.3 , 0 ,0 ; 
     0 , 0 , 0.5 ,0;
     0 , 0 , 0 ,0.7 ];

%}

%for first iteration we take the input values as intial state and
%covariance matrix.

input_state_vector_for_prediction = initial_state;
input_covariance_matrix_for_prediction = intital_covariance;


trace_P = [];
x_values = [];
y_values = [];
vel_X = [];
vel_Y = [];

for i=1:length(observations)
    [xhat_minus, P_minus] = predict_state(F_k,input_state_vector_for_prediction,input_covariance_matrix_for_prediction,Q);

    [xhat_plus,P_plus] = correct_states(i,xhat_minus,P_minus,R,control_points,xc,vx,yc,vy,observations);
    
    %disp(P_plus)
    %we get the trace of P_plus which is the posterior estimation error.
    trace_P(i) = trace(P_plus);
    
    input_state_vector_for_prediction = xhat_plus;
    input_covariance_matrix_for_prediction = P_plus;
    
    
    fprintf('\n\n Iteration no : %d\n', i)
    disp("Updated Coordinate Values")
    disp("X = ")
    disp(xhat_plus(1))
    disp("Y = ")
    disp(xhat_plus(3))
    
    x_values(i) = xhat_plus(1);
    vel_X(i) = xhat_plus(2);
    y_values(i) = xhat_plus(3);
    vel_Y(i) = xhat_plus(4);
    
end
 

function[xhat_minus, P_minus] = predict_state(F_k,X,P,Q)
    xhat_minus = F_k * X;
    P_minus = F_k * P * transpose(F_k) + Q;
end


function [xhat_plus,P_plus] = correct_states(iter,xhat_minus,P_minus,R,control_points,xc,vx,yc,vy,observations)

    %now the measurements are avaialble
    % Defining the measurement model.
    %as the measurement model is not linear we will linearize it.
    % we will linearize it about the nominal point xhat_minus.

    measurement_model = [ sqrt((control_points(1,1) - xc)^2 + (control_points(1,2) - yc)^2);
                      sqrt((control_points(2,1) - xc)^2 + (control_points(2,2) - yc)^2);
                      sqrt((control_points(3,1) - xc)^2 + (control_points(3,2) - yc)^2);
                      sqrt((control_points(4,1) - xc)^2 + (control_points(4,2) - yc)^2)];

    h_at_xhat_minus = subs(measurement_model , [xc,yc], [vpa(xhat_minus(1)),vpa(xhat_minus(3))]);

    H_k = jacobian(measurement_model,[xc,vx,yc,vy]);

    %evaluate H_k at xhat_minus
    H_k_at_xhat_minus = eval(subs(H_k,[xc,yc], [vpa(xhat_minus(1)),vpa(xhat_minus(3))]));
    
    
    innovation = [observations(iter,1) - h_at_xhat_minus(1);
                  observations(iter,2) - h_at_xhat_minus(2);
                  observations(iter,3) - h_at_xhat_minus(3);
                  observations(iter,4) - h_at_xhat_minus(4)];
    
   
    tmp = inv( H_k_at_xhat_minus * vpa(P_minus) * transpose(H_k_at_xhat_minus) + R);         
    K_k = P_minus * transpose(H_k_at_xhat_minus) * tmp;

    
    xhat_plus = xhat_minus + K_k * innovation;
    
    
    I = eye(4);
    P_plus = (I - K_k * H_k_at_xhat_minus)*P_minus*transpose(I - K_k * H_k_at_xhat_minus) ...
        + K_k * R * transpose(K_k);
end
