dtmc 
 
const double p;
const double q1;
const double q2;

module multi_param_agents_3
       // ai - state of agent i:  -1:init, 0:total_failure, 1:success, 2:failure_after_first_attempt
       // where success denotes decision to sting, failure the opposite
       // b = 1: 'final'/leaf/BSCC state flag
       a0 : [-1..2] init -1; 
       a1 : [-1..2] init -1; 
       a2 : [-1..2] init -1; 
       b : [0..1] init 0; 

       //  initial transition
       []   a0 = -1 & a1 = -1  & a2 = -1 -> 1.0*p*p*p: (a0'=1) & (a1'=1) & (a2'=1) + 3.0*p*p*(1-p): (a0'=1) & (a1'=1) & (a2'=2) + 3.0*p*(1-p)*(1-p): (a0'=1) & (a1'=2) & (a2'=2) + 1.0*(1-p)*(1-p)*(1-p): (a0'=2) & (a1'=2) & (a2'=2);

       // some ones, some zeros transitions
       []   a0 = 0 & a1 = 0 & a2 = 0 -> (a0'= 0) & (a1'= 0) & (a2'= 0) & (b'=1);
       []   a0 = 1 & a1 = 0 & a2 = 0 -> (a0'= 1) & (a1'= 0) & (a2'= 0) & (b'=1);
       []   a0 = 1 & a1 = 1 & a2 = 0 -> (a0'= 1) & (a1'= 1) & (a2'= 0) & (b'=1);
       []   a0 = 1 & a1 = 1 & a2 = 1 -> (a0'= 1) & (a1'= 1) & (a2'= 1) & (b'=1);

       // some ones, some twos transitions
       []   a0 = 1 & a1 = 2 & a2 = 2 -> q1:(a0'= 1) & (a1'= 1) & (a2'= 2) + 1-q1:(a0'= 1) & (a1'= 2) & (a2'= 0);
       []   a0 = 1 & a1 = 1 & a2 = 2 -> q2:(a0'= 1) & (a1'= 1) & (a2'= 1) + 1-q2:(a0'= 1) & (a1'= 1) & (a2'= 0);

       // some ones, some twos, some zeros transitions
       []   a0 = 1 & a1 = 2 & a2 = 0 -> q1: (a0' = 1) & (a1' = 1) & (a2' = 0) + 1-q1: (a0' = 1) & (a1' = 0) & (a2' = 0);

       // all twos transition
       []   a0 = 2 & a1 = 2  & a2 = 2 -> (a0'= 0) & (a1'= 0) & (a2'= 0);
endmodule

label "bscc_0" = (a0=0) & (a1=0) & (a2=0) & (b=1);
label "bscc_1" = (a0=1) & (a1=0) & (a2=0) & (b=1);
label "bscc_2" = (a0=1) & (a1=1) & (a2=0) & (b=1);
label "bscc_3" = (a0=1) & (a1=1) & (a2=1) & (b=1);

label "survived" = (a0=1) & (a1=1) & (a2=1) & (b=1);
