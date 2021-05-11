dtmc 
 
const double r_0;
const double r_1;


module multi_param_bee_agents_3
       // ai - state of agent i: 3:init 1:success -j: failure when j amount of pheromone present 
       a0 : [-2..3] init 3; 
       a1 : [-2..3] init 3; 
       b : [0..1] init 0; 

       //  initial transition
       []   a0 = 3 & a1 = 3 & b = 0 -> 1.0*r_0: (a0'=0) & (a1'=0) + (1-r_0): (a0'=1) & (a1'=-1);

       []   a0 = 1 & a1 = -1 & b = 0 -> 1.0*r_1: (a0'=1) & (a1'=0) + (1-r_1): (a0'=1) & (a1'=1); 


       //  final transitions
       [zero]   a0 = 0 & a1 = 0 & b = 0 -> (a0'=0) & (a1'=0) & (b'=1);
       [one]    a0 = 1 & a1 = 0 & b = 0 -> (a0'=1) & (a1'=0) & (b'=1);
       [two]    a0 = 1 & a1 = 1 & b = 0 -> (a0'=1) & (a1'=1) & (b'=1);
endmodule 

label "bscc_1" = a0 = 0 & a1 = 0 & b = 1;
label "bscc_2" = a0 = 1 & a1 = 0 & b = 1;
label "bscc_3" = a0 = 1 & a1 = 1 & b = 1;


rewards "mean"
       [zero] true: 0;
       [one] true: 1;
       [two] true: 2;
endrewards

rewards "mean_squared"
       [zero] true: 0;
       [one] true: 2;
       [two] true: 4;
endrewards

rewards "mean2" 
       a0 = 0 & a1 = 0: 0;
       a0 = 1 & a1 = 0: 1;
       a0 = 1 & a1 = 1: 2;
endrewards 
rewards "mean_squared2" 
       a0 = 0 & a1 = 0: 0;
       a0 = 1 & a1 = 0: 2;
       a0 = 1 & a1 = 1: 4;
endrewards 
