dtmc 
 
const double p;
const double q1;
const double q2;
const double q3;
const double q4;
const double q5;
const double q6;
const double q7;
const double q8;
const double q9;

module multi_param_agents_10
       // ai - state of agent i:  -1:init, 0:total_failure, 1:success, 2:failure_after_first_attempt
       // where success denotes decision to sting, failure the opposite
       // b = 1: 'final'/leaf/BSCC state flag
       a0 : [-1..2] init -1; 
       a1 : [-1..2] init -1; 
       a2 : [-1..2] init -1; 
       a3 : [-1..2] init -1; 
       a4 : [-1..2] init -1; 
       a5 : [-1..2] init -1; 
       a6 : [-1..2] init -1; 
       a7 : [-1..2] init -1; 
       a8 : [-1..2] init -1; 
       a9 : [-1..2] init -1; 
       b : [0..1] init 0; 

       //  initial transition
       []   a0 = -1 & a1 = -1  & a2 = -1  & a3 = -1  & a4 = -1  & a5 = -1  & a6 = -1  & a7 = -1  & a8 = -1  & a9 = -1 -> 1.0*p*p*p*p*p*p*p*p*p*p: (a0'=1) & (a1'=1) & (a2'=1) & (a3'=1) & (a4'=1) & (a5'=1) & (a6'=1) & (a7'=1) & (a8'=1) & (a9'=1) + 10.0*p*p*p*p*p*p*p*p*p*(1-p): (a0'=1) & (a1'=1) & (a2'=1) & (a3'=1) & (a4'=1) & (a5'=1) & (a6'=1) & (a7'=1) & (a8'=1) & (a9'=2) + 45.0*p*p*p*p*p*p*p*p*(1-p)*(1-p): (a0'=1) & (a1'=1) & (a2'=1) & (a3'=1) & (a4'=1) & (a5'=1) & (a6'=1) & (a7'=1) & (a8'=2) & (a9'=2) + 120.0*p*p*p*p*p*p*p*(1-p)*(1-p)*(1-p): (a0'=1) & (a1'=1) & (a2'=1) & (a3'=1) & (a4'=1) & (a5'=1) & (a6'=1) & (a7'=2) & (a8'=2) & (a9'=2) + 210.0*p*p*p*p*p*p*(1-p)*(1-p)*(1-p)*(1-p): (a0'=1) & (a1'=1) & (a2'=1) & (a3'=1) & (a4'=1) & (a5'=1) & (a6'=2) & (a7'=2) & (a8'=2) & (a9'=2) + 252.0*p*p*p*p*p*(1-p)*(1-p)*(1-p)*(1-p)*(1-p): (a0'=1) & (a1'=1) & (a2'=1) & (a3'=1) & (a4'=1) & (a5'=2) & (a6'=2) & (a7'=2) & (a8'=2) & (a9'=2) + 210.0*p*p*p*p*(1-p)*(1-p)*(1-p)*(1-p)*(1-p)*(1-p): (a0'=1) & (a1'=1) & (a2'=1) & (a3'=1) & (a4'=2) & (a5'=2) & (a6'=2) & (a7'=2) & (a8'=2) & (a9'=2) + 120.0*p*p*p*(1-p)*(1-p)*(1-p)*(1-p)*(1-p)*(1-p)*(1-p): (a0'=1) & (a1'=1) & (a2'=1) & (a3'=2) & (a4'=2) & (a5'=2) & (a6'=2) & (a7'=2) & (a8'=2) & (a9'=2) + 45.0*p*p*(1-p)*(1-p)*(1-p)*(1-p)*(1-p)*(1-p)*(1-p)*(1-p): (a0'=1) & (a1'=1) & (a2'=2) & (a3'=2) & (a4'=2) & (a5'=2) & (a6'=2) & (a7'=2) & (a8'=2) & (a9'=2) + 10.0*p*(1-p)*(1-p)*(1-p)*(1-p)*(1-p)*(1-p)*(1-p)*(1-p)*(1-p): (a0'=1) & (a1'=2) & (a2'=2) & (a3'=2) & (a4'=2) & (a5'=2) & (a6'=2) & (a7'=2) & (a8'=2) & (a9'=2) + 1.0*(1-p)*(1-p)*(1-p)*(1-p)*(1-p)*(1-p)*(1-p)*(1-p)*(1-p)*(1-p): (a0'=2) & (a1'=2) & (a2'=2) & (a3'=2) & (a4'=2) & (a5'=2) & (a6'=2) & (a7'=2) & (a8'=2) & (a9'=2);

       // some ones, some zeros transitions
       []   a0 = 0 & a1 = 0 & a2 = 0 & a3 = 0 & a4 = 0 & a5 = 0 & a6 = 0 & a7 = 0 & a8 = 0 & a9 = 0 -> (a0'= 0) & (a1'= 0) & (a2'= 0) & (a3'= 0) & (a4'= 0) & (a5'= 0) & (a6'= 0) & (a7'= 0) & (a8'= 0) & (a9'= 0) & (b'=1);
       []   a0 = 1 & a1 = 0 & a2 = 0 & a3 = 0 & a4 = 0 & a5 = 0 & a6 = 0 & a7 = 0 & a8 = 0 & a9 = 0 -> (a0'= 1) & (a1'= 0) & (a2'= 0) & (a3'= 0) & (a4'= 0) & (a5'= 0) & (a6'= 0) & (a7'= 0) & (a8'= 0) & (a9'= 0) & (b'=1);
       []   a0 = 1 & a1 = 1 & a2 = 0 & a3 = 0 & a4 = 0 & a5 = 0 & a6 = 0 & a7 = 0 & a8 = 0 & a9 = 0 -> (a0'= 1) & (a1'= 1) & (a2'= 0) & (a3'= 0) & (a4'= 0) & (a5'= 0) & (a6'= 0) & (a7'= 0) & (a8'= 0) & (a9'= 0) & (b'=1);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 0 & a4 = 0 & a5 = 0 & a6 = 0 & a7 = 0 & a8 = 0 & a9 = 0 -> (a0'= 1) & (a1'= 1) & (a2'= 1) & (a3'= 0) & (a4'= 0) & (a5'= 0) & (a6'= 0) & (a7'= 0) & (a8'= 0) & (a9'= 0) & (b'=1);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 1 & a4 = 0 & a5 = 0 & a6 = 0 & a7 = 0 & a8 = 0 & a9 = 0 -> (a0'= 1) & (a1'= 1) & (a2'= 1) & (a3'= 1) & (a4'= 0) & (a5'= 0) & (a6'= 0) & (a7'= 0) & (a8'= 0) & (a9'= 0) & (b'=1);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 1 & a4 = 1 & a5 = 0 & a6 = 0 & a7 = 0 & a8 = 0 & a9 = 0 -> (a0'= 1) & (a1'= 1) & (a2'= 1) & (a3'= 1) & (a4'= 1) & (a5'= 0) & (a6'= 0) & (a7'= 0) & (a8'= 0) & (a9'= 0) & (b'=1);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 1 & a4 = 1 & a5 = 1 & a6 = 0 & a7 = 0 & a8 = 0 & a9 = 0 -> (a0'= 1) & (a1'= 1) & (a2'= 1) & (a3'= 1) & (a4'= 1) & (a5'= 1) & (a6'= 0) & (a7'= 0) & (a8'= 0) & (a9'= 0) & (b'=1);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 1 & a4 = 1 & a5 = 1 & a6 = 1 & a7 = 0 & a8 = 0 & a9 = 0 -> (a0'= 1) & (a1'= 1) & (a2'= 1) & (a3'= 1) & (a4'= 1) & (a5'= 1) & (a6'= 1) & (a7'= 0) & (a8'= 0) & (a9'= 0) & (b'=1);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 1 & a4 = 1 & a5 = 1 & a6 = 1 & a7 = 1 & a8 = 0 & a9 = 0 -> (a0'= 1) & (a1'= 1) & (a2'= 1) & (a3'= 1) & (a4'= 1) & (a5'= 1) & (a6'= 1) & (a7'= 1) & (a8'= 0) & (a9'= 0) & (b'=1);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 1 & a4 = 1 & a5 = 1 & a6 = 1 & a7 = 1 & a8 = 1 & a9 = 0 -> (a0'= 1) & (a1'= 1) & (a2'= 1) & (a3'= 1) & (a4'= 1) & (a5'= 1) & (a6'= 1) & (a7'= 1) & (a8'= 1) & (a9'= 0) & (b'=1);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 1 & a4 = 1 & a5 = 1 & a6 = 1 & a7 = 1 & a8 = 1 & a9 = 1 -> (a0'= 1) & (a1'= 1) & (a2'= 1) & (a3'= 1) & (a4'= 1) & (a5'= 1) & (a6'= 1) & (a7'= 1) & (a8'= 1) & (a9'= 1) & (b'=1);

       // some ones, some twos transitions
       []   a0 = 1 & a1 = 2 & a2 = 2 & a3 = 2 & a4 = 2 & a5 = 2 & a6 = 2 & a7 = 2 & a8 = 2 & a9 = 2 -> q1:(a0'= 1) & (a1'= 1) & (a2'= 2) & (a3'= 2) & (a4'= 2) & (a5'= 2) & (a6'= 2) & (a7'= 2) & (a8'= 2) & (a9'= 2) + 1-q1:(a0'= 1) & (a1'= 2) & (a2'= 2) & (a3'= 2) & (a4'= 2) & (a5'= 2) & (a6'= 2) & (a7'= 2) & (a8'= 2) & (a9'= 0);
       []   a0 = 1 & a1 = 1 & a2 = 2 & a3 = 2 & a4 = 2 & a5 = 2 & a6 = 2 & a7 = 2 & a8 = 2 & a9 = 2 -> q2:(a0'= 1) & (a1'= 1) & (a2'= 1) & (a3'= 2) & (a4'= 2) & (a5'= 2) & (a6'= 2) & (a7'= 2) & (a8'= 2) & (a9'= 2) + 1-q2:(a0'= 1) & (a1'= 1) & (a2'= 2) & (a3'= 2) & (a4'= 2) & (a5'= 2) & (a6'= 2) & (a7'= 2) & (a8'= 2) & (a9'= 0);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 2 & a4 = 2 & a5 = 2 & a6 = 2 & a7 = 2 & a8 = 2 & a9 = 2 -> q3:(a0'= 1) & (a1'= 1) & (a2'= 1) & (a3'= 1) & (a4'= 2) & (a5'= 2) & (a6'= 2) & (a7'= 2) & (a8'= 2) & (a9'= 2) + 1-q3:(a0'= 1) & (a1'= 1) & (a2'= 1) & (a3'= 2) & (a4'= 2) & (a5'= 2) & (a6'= 2) & (a7'= 2) & (a8'= 2) & (a9'= 0);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 1 & a4 = 2 & a5 = 2 & a6 = 2 & a7 = 2 & a8 = 2 & a9 = 2 -> q4:(a0'= 1) & (a1'= 1) & (a2'= 1) & (a3'= 1) & (a4'= 1) & (a5'= 2) & (a6'= 2) & (a7'= 2) & (a8'= 2) & (a9'= 2) + 1-q4:(a0'= 1) & (a1'= 1) & (a2'= 1) & (a3'= 1) & (a4'= 2) & (a5'= 2) & (a6'= 2) & (a7'= 2) & (a8'= 2) & (a9'= 0);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 1 & a4 = 1 & a5 = 2 & a6 = 2 & a7 = 2 & a8 = 2 & a9 = 2 -> q5:(a0'= 1) & (a1'= 1) & (a2'= 1) & (a3'= 1) & (a4'= 1) & (a5'= 1) & (a6'= 2) & (a7'= 2) & (a8'= 2) & (a9'= 2) + 1-q5:(a0'= 1) & (a1'= 1) & (a2'= 1) & (a3'= 1) & (a4'= 1) & (a5'= 2) & (a6'= 2) & (a7'= 2) & (a8'= 2) & (a9'= 0);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 1 & a4 = 1 & a5 = 1 & a6 = 2 & a7 = 2 & a8 = 2 & a9 = 2 -> q6:(a0'= 1) & (a1'= 1) & (a2'= 1) & (a3'= 1) & (a4'= 1) & (a5'= 1) & (a6'= 1) & (a7'= 2) & (a8'= 2) & (a9'= 2) + 1-q6:(a0'= 1) & (a1'= 1) & (a2'= 1) & (a3'= 1) & (a4'= 1) & (a5'= 1) & (a6'= 2) & (a7'= 2) & (a8'= 2) & (a9'= 0);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 1 & a4 = 1 & a5 = 1 & a6 = 1 & a7 = 2 & a8 = 2 & a9 = 2 -> q7:(a0'= 1) & (a1'= 1) & (a2'= 1) & (a3'= 1) & (a4'= 1) & (a5'= 1) & (a6'= 1) & (a7'= 1) & (a8'= 2) & (a9'= 2) + 1-q7:(a0'= 1) & (a1'= 1) & (a2'= 1) & (a3'= 1) & (a4'= 1) & (a5'= 1) & (a6'= 1) & (a7'= 2) & (a8'= 2) & (a9'= 0);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 1 & a4 = 1 & a5 = 1 & a6 = 1 & a7 = 1 & a8 = 2 & a9 = 2 -> q8:(a0'= 1) & (a1'= 1) & (a2'= 1) & (a3'= 1) & (a4'= 1) & (a5'= 1) & (a6'= 1) & (a7'= 1) & (a8'= 1) & (a9'= 2) + 1-q8:(a0'= 1) & (a1'= 1) & (a2'= 1) & (a3'= 1) & (a4'= 1) & (a5'= 1) & (a6'= 1) & (a7'= 1) & (a8'= 2) & (a9'= 0);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 1 & a4 = 1 & a5 = 1 & a6 = 1 & a7 = 1 & a8 = 1 & a9 = 2 -> q9:(a0'= 1) & (a1'= 1) & (a2'= 1) & (a3'= 1) & (a4'= 1) & (a5'= 1) & (a6'= 1) & (a7'= 1) & (a8'= 1) & (a9'= 1) + 1-q9:(a0'= 1) & (a1'= 1) & (a2'= 1) & (a3'= 1) & (a4'= 1) & (a5'= 1) & (a6'= 1) & (a7'= 1) & (a8'= 1) & (a9'= 0);

       // some ones, some twos, some zeros transitions
       []   a0 = 1 & a1 = 2 & a2 = 0 & a3 = 0 & a4 = 0 & a5 = 0 & a6 = 0 & a7 = 0 & a8 = 0 & a9 = 0 -> q1: (a0' = 1) & (a1' = 1) & (a2' = 0) & (a3' = 0) & (a4' = 0) & (a5' = 0) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0) + 1-q1: (a0' = 1) & (a1' = 0) & (a2' = 0) & (a3' = 0) & (a4' = 0) & (a5' = 0) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 2 & a2 = 2 & a3 = 0 & a4 = 0 & a5 = 0 & a6 = 0 & a7 = 0 & a8 = 0 & a9 = 0 -> q1: (a0' = 1) & (a1' = 1) & (a2' = 2) & (a3' = 0) & (a4' = 0) & (a5' = 0) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0) + 1-q1: (a0' = 1) & (a1' = 2) & (a2' = 0) & (a3' = 0) & (a4' = 0) & (a5' = 0) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 2 & a2 = 2 & a3 = 2 & a4 = 0 & a5 = 0 & a6 = 0 & a7 = 0 & a8 = 0 & a9 = 0 -> q1: (a0' = 1) & (a1' = 1) & (a2' = 2) & (a3' = 2) & (a4' = 0) & (a5' = 0) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0) + 1-q1: (a0' = 1) & (a1' = 2) & (a2' = 2) & (a3' = 0) & (a4' = 0) & (a5' = 0) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 2 & a2 = 2 & a3 = 2 & a4 = 2 & a5 = 0 & a6 = 0 & a7 = 0 & a8 = 0 & a9 = 0 -> q1: (a0' = 1) & (a1' = 1) & (a2' = 2) & (a3' = 2) & (a4' = 2) & (a5' = 0) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0) + 1-q1: (a0' = 1) & (a1' = 2) & (a2' = 2) & (a3' = 2) & (a4' = 0) & (a5' = 0) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 2 & a2 = 2 & a3 = 2 & a4 = 2 & a5 = 2 & a6 = 0 & a7 = 0 & a8 = 0 & a9 = 0 -> q1: (a0' = 1) & (a1' = 1) & (a2' = 2) & (a3' = 2) & (a4' = 2) & (a5' = 2) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0) + 1-q1: (a0' = 1) & (a1' = 2) & (a2' = 2) & (a3' = 2) & (a4' = 2) & (a5' = 0) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 2 & a2 = 2 & a3 = 2 & a4 = 2 & a5 = 2 & a6 = 2 & a7 = 0 & a8 = 0 & a9 = 0 -> q1: (a0' = 1) & (a1' = 1) & (a2' = 2) & (a3' = 2) & (a4' = 2) & (a5' = 2) & (a6' = 2) & (a7' = 0) & (a8' = 0) & (a9' = 0) + 1-q1: (a0' = 1) & (a1' = 2) & (a2' = 2) & (a3' = 2) & (a4' = 2) & (a5' = 2) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 2 & a2 = 2 & a3 = 2 & a4 = 2 & a5 = 2 & a6 = 2 & a7 = 2 & a8 = 0 & a9 = 0 -> q1: (a0' = 1) & (a1' = 1) & (a2' = 2) & (a3' = 2) & (a4' = 2) & (a5' = 2) & (a6' = 2) & (a7' = 2) & (a8' = 0) & (a9' = 0) + 1-q1: (a0' = 1) & (a1' = 2) & (a2' = 2) & (a3' = 2) & (a4' = 2) & (a5' = 2) & (a6' = 2) & (a7' = 0) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 2 & a2 = 2 & a3 = 2 & a4 = 2 & a5 = 2 & a6 = 2 & a7 = 2 & a8 = 2 & a9 = 0 -> q1: (a0' = 1) & (a1' = 1) & (a2' = 2) & (a3' = 2) & (a4' = 2) & (a5' = 2) & (a6' = 2) & (a7' = 2) & (a8' = 2) & (a9' = 0) + 1-q1: (a0' = 1) & (a1' = 2) & (a2' = 2) & (a3' = 2) & (a4' = 2) & (a5' = 2) & (a6' = 2) & (a7' = 2) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 1 & a2 = 2 & a3 = 0 & a4 = 0 & a5 = 0 & a6 = 0 & a7 = 0 & a8 = 0 & a9 = 0 -> q2: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 0) & (a4' = 0) & (a5' = 0) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0) + 1-q2: (a0' = 1) & (a1' = 1) & (a2' = 0) & (a3' = 0) & (a4' = 0) & (a5' = 0) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 1 & a2 = 2 & a3 = 2 & a4 = 0 & a5 = 0 & a6 = 0 & a7 = 0 & a8 = 0 & a9 = 0 -> q2: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 2) & (a4' = 0) & (a5' = 0) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0) + 1-q2: (a0' = 1) & (a1' = 1) & (a2' = 2) & (a3' = 0) & (a4' = 0) & (a5' = 0) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 1 & a2 = 2 & a3 = 2 & a4 = 2 & a5 = 0 & a6 = 0 & a7 = 0 & a8 = 0 & a9 = 0 -> q2: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 2) & (a4' = 2) & (a5' = 0) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0) + 1-q2: (a0' = 1) & (a1' = 1) & (a2' = 2) & (a3' = 2) & (a4' = 0) & (a5' = 0) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 1 & a2 = 2 & a3 = 2 & a4 = 2 & a5 = 2 & a6 = 0 & a7 = 0 & a8 = 0 & a9 = 0 -> q2: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 2) & (a4' = 2) & (a5' = 2) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0) + 1-q2: (a0' = 1) & (a1' = 1) & (a2' = 2) & (a3' = 2) & (a4' = 2) & (a5' = 0) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 1 & a2 = 2 & a3 = 2 & a4 = 2 & a5 = 2 & a6 = 2 & a7 = 0 & a8 = 0 & a9 = 0 -> q2: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 2) & (a4' = 2) & (a5' = 2) & (a6' = 2) & (a7' = 0) & (a8' = 0) & (a9' = 0) + 1-q2: (a0' = 1) & (a1' = 1) & (a2' = 2) & (a3' = 2) & (a4' = 2) & (a5' = 2) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 1 & a2 = 2 & a3 = 2 & a4 = 2 & a5 = 2 & a6 = 2 & a7 = 2 & a8 = 0 & a9 = 0 -> q2: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 2) & (a4' = 2) & (a5' = 2) & (a6' = 2) & (a7' = 2) & (a8' = 0) & (a9' = 0) + 1-q2: (a0' = 1) & (a1' = 1) & (a2' = 2) & (a3' = 2) & (a4' = 2) & (a5' = 2) & (a6' = 2) & (a7' = 0) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 1 & a2 = 2 & a3 = 2 & a4 = 2 & a5 = 2 & a6 = 2 & a7 = 2 & a8 = 2 & a9 = 0 -> q2: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 2) & (a4' = 2) & (a5' = 2) & (a6' = 2) & (a7' = 2) & (a8' = 2) & (a9' = 0) + 1-q2: (a0' = 1) & (a1' = 1) & (a2' = 2) & (a3' = 2) & (a4' = 2) & (a5' = 2) & (a6' = 2) & (a7' = 2) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 2 & a4 = 0 & a5 = 0 & a6 = 0 & a7 = 0 & a8 = 0 & a9 = 0 -> q3: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 0) & (a5' = 0) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0) + 1-q3: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 0) & (a4' = 0) & (a5' = 0) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 2 & a4 = 2 & a5 = 0 & a6 = 0 & a7 = 0 & a8 = 0 & a9 = 0 -> q3: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 2) & (a5' = 0) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0) + 1-q3: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 2) & (a4' = 0) & (a5' = 0) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 2 & a4 = 2 & a5 = 2 & a6 = 0 & a7 = 0 & a8 = 0 & a9 = 0 -> q3: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 2) & (a5' = 2) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0) + 1-q3: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 2) & (a4' = 2) & (a5' = 0) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 2 & a4 = 2 & a5 = 2 & a6 = 2 & a7 = 0 & a8 = 0 & a9 = 0 -> q3: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 2) & (a5' = 2) & (a6' = 2) & (a7' = 0) & (a8' = 0) & (a9' = 0) + 1-q3: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 2) & (a4' = 2) & (a5' = 2) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 2 & a4 = 2 & a5 = 2 & a6 = 2 & a7 = 2 & a8 = 0 & a9 = 0 -> q3: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 2) & (a5' = 2) & (a6' = 2) & (a7' = 2) & (a8' = 0) & (a9' = 0) + 1-q3: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 2) & (a4' = 2) & (a5' = 2) & (a6' = 2) & (a7' = 0) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 2 & a4 = 2 & a5 = 2 & a6 = 2 & a7 = 2 & a8 = 2 & a9 = 0 -> q3: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 2) & (a5' = 2) & (a6' = 2) & (a7' = 2) & (a8' = 2) & (a9' = 0) + 1-q3: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 2) & (a4' = 2) & (a5' = 2) & (a6' = 2) & (a7' = 2) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 1 & a4 = 2 & a5 = 0 & a6 = 0 & a7 = 0 & a8 = 0 & a9 = 0 -> q4: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 1) & (a5' = 0) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0) + 1-q4: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 0) & (a5' = 0) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 1 & a4 = 2 & a5 = 2 & a6 = 0 & a7 = 0 & a8 = 0 & a9 = 0 -> q4: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 1) & (a5' = 2) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0) + 1-q4: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 2) & (a5' = 0) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 1 & a4 = 2 & a5 = 2 & a6 = 2 & a7 = 0 & a8 = 0 & a9 = 0 -> q4: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 1) & (a5' = 2) & (a6' = 2) & (a7' = 0) & (a8' = 0) & (a9' = 0) + 1-q4: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 2) & (a5' = 2) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 1 & a4 = 2 & a5 = 2 & a6 = 2 & a7 = 2 & a8 = 0 & a9 = 0 -> q4: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 1) & (a5' = 2) & (a6' = 2) & (a7' = 2) & (a8' = 0) & (a9' = 0) + 1-q4: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 2) & (a5' = 2) & (a6' = 2) & (a7' = 0) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 1 & a4 = 2 & a5 = 2 & a6 = 2 & a7 = 2 & a8 = 2 & a9 = 0 -> q4: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 1) & (a5' = 2) & (a6' = 2) & (a7' = 2) & (a8' = 2) & (a9' = 0) + 1-q4: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 2) & (a5' = 2) & (a6' = 2) & (a7' = 2) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 1 & a4 = 1 & a5 = 2 & a6 = 0 & a7 = 0 & a8 = 0 & a9 = 0 -> q5: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 1) & (a5' = 1) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0) + 1-q5: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 1) & (a5' = 0) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 1 & a4 = 1 & a5 = 2 & a6 = 2 & a7 = 0 & a8 = 0 & a9 = 0 -> q5: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 1) & (a5' = 1) & (a6' = 2) & (a7' = 0) & (a8' = 0) & (a9' = 0) + 1-q5: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 1) & (a5' = 2) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 1 & a4 = 1 & a5 = 2 & a6 = 2 & a7 = 2 & a8 = 0 & a9 = 0 -> q5: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 1) & (a5' = 1) & (a6' = 2) & (a7' = 2) & (a8' = 0) & (a9' = 0) + 1-q5: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 1) & (a5' = 2) & (a6' = 2) & (a7' = 0) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 1 & a4 = 1 & a5 = 2 & a6 = 2 & a7 = 2 & a8 = 2 & a9 = 0 -> q5: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 1) & (a5' = 1) & (a6' = 2) & (a7' = 2) & (a8' = 2) & (a9' = 0) + 1-q5: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 1) & (a5' = 2) & (a6' = 2) & (a7' = 2) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 1 & a4 = 1 & a5 = 1 & a6 = 2 & a7 = 0 & a8 = 0 & a9 = 0 -> q6: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 1) & (a5' = 1) & (a6' = 1) & (a7' = 0) & (a8' = 0) & (a9' = 0) + 1-q6: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 1) & (a5' = 1) & (a6' = 0) & (a7' = 0) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 1 & a4 = 1 & a5 = 1 & a6 = 2 & a7 = 2 & a8 = 0 & a9 = 0 -> q6: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 1) & (a5' = 1) & (a6' = 1) & (a7' = 2) & (a8' = 0) & (a9' = 0) + 1-q6: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 1) & (a5' = 1) & (a6' = 2) & (a7' = 0) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 1 & a4 = 1 & a5 = 1 & a6 = 2 & a7 = 2 & a8 = 2 & a9 = 0 -> q6: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 1) & (a5' = 1) & (a6' = 1) & (a7' = 2) & (a8' = 2) & (a9' = 0) + 1-q6: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 1) & (a5' = 1) & (a6' = 2) & (a7' = 2) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 1 & a4 = 1 & a5 = 1 & a6 = 1 & a7 = 2 & a8 = 0 & a9 = 0 -> q7: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 1) & (a5' = 1) & (a6' = 1) & (a7' = 1) & (a8' = 0) & (a9' = 0) + 1-q7: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 1) & (a5' = 1) & (a6' = 1) & (a7' = 0) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 1 & a4 = 1 & a5 = 1 & a6 = 1 & a7 = 2 & a8 = 2 & a9 = 0 -> q7: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 1) & (a5' = 1) & (a6' = 1) & (a7' = 1) & (a8' = 2) & (a9' = 0) + 1-q7: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 1) & (a5' = 1) & (a6' = 1) & (a7' = 2) & (a8' = 0) & (a9' = 0);
       []   a0 = 1 & a1 = 1 & a2 = 1 & a3 = 1 & a4 = 1 & a5 = 1 & a6 = 1 & a7 = 1 & a8 = 2 & a9 = 0 -> q8: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 1) & (a5' = 1) & (a6' = 1) & (a7' = 1) & (a8' = 1) & (a9' = 0) + 1-q8: (a0' = 1) & (a1' = 1) & (a2' = 1) & (a3' = 1) & (a4' = 1) & (a5' = 1) & (a6' = 1) & (a7' = 1) & (a8' = 0) & (a9' = 0);

       // all twos transition
       []   a0 = 2 & a1 = 2  & a2 = 2  & a3 = 2  & a4 = 2  & a5 = 2  & a6 = 2  & a7 = 2  & a8 = 2  & a9 = 2 -> (a0'= 0) & (a1'= 0) & (a2'= 0) & (a3'= 0) & (a4'= 0) & (a5'= 0) & (a6'= 0) & (a7'= 0) & (a8'= 0) & (a9'= 0);
endmodule 


 label "bscc_0"  = (a0=0) & (a1=0) & (a2=0) & (a3=0) & (a4=0) & (a5=0) & (a6=0) & (a7=0) & (a8=0) & (a9=0) & (b=1);
 label "bscc_1"  = (a0=1) & (a1=0) & (a2=0) & (a3=0) & (a4=0) & (a5=0) & (a6=0) & (a7=0) & (a8=0) & (a9=0) & (b=1);
 label "bscc_2"  = (a0=1) & (a1=1) & (a2=0) & (a3=0) & (a4=0) & (a5=0) & (a6=0) & (a7=0) & (a8=0) & (a9=0) & (b=1); 
 label "bscc_3"  = (a0=1) & (a1=1) & (a2=1) & (a3=0) & (a4=0) & (a5=0) & (a6=0) & (a7=0) & (a8=0) & (a9=0) & (b=1);
 label "bscc_4"  = (a0=1) & (a1=1) & (a2=1) & (a3=1) & (a4=0) & (a5=0) & (a6=0) & (a7=0) & (a8=0) & (a9=0) & (b=1);
 label "bscc_5"  = (a0=1) & (a1=1) & (a2=1) & (a3=1) & (a4=1) & (a5=0) & (a6=0) & (a7=0) & (a8=0) & (a9=0) & (b=1);
 label "bscc_6"  = (a0=1) & (a1=1) & (a2=1) & (a3=1) & (a4=1) & (a5=1) & (a6=0) & (a7=0) & (a8=0) & (a9=0) & (b=1);
 label "bscc_7"  = (a0=1) & (a1=1) & (a2=1) & (a3=1) & (a4=1) & (a5=1) & (a6=1) & (a7=0) & (a8=0) & (a9=0) & (b=1);
 label "bscc_8"  = (a0=1) & (a1=1) & (a2=1) & (a3=1) & (a4=1) & (a5=1) & (a6=1) & (a7=1) & (a8=0) & (a9=0) & (b=1);
 label "bscc_9"  = (a0=1) & (a1=1) & (a2=1) & (a3=1) & (a4=1) & (a5=1) & (a6=1) & (a7=1) & (a8=1) & (a9=0) & (b=1);
 label "bscc_10" = (a0=1) & (a1=1) & (a2=1) & (a3=1) & (a4=1) & (a5=1) & (a6=1) & (a7=1) & (a8=1) & (a9=1) & (b=1);
 
 label "survived" = (a0=1) & (a1=1) & (a2=1) & (a3=1) & (a4=1) & (a5=1) & (a6=0) & (a7=0) & (a8=0) & (a9=0) & (b=1) | (a0=1) & (a1=1) & (a2=1) & (a3=1) & (a4=1) & (a5=1) & (a6=1) & (a7=0) & (a8=0) & (a9=0) & (b=1) | (a0=1) & (a1=1) & (a2=1) & (a3=1) & (a4=1) & (a5=1) & (a6=1) & (a7=1) & (a8=0) & (a9=0) & (b=1) | (a0=1) & (a1=1) & (a2=1) & (a3=1) & (a4=1) & (a5=1) & (a6=1) & (a7=1) & (a8=1) & (a9=0) & (b=1) | (a0=1) & (a1=1) & (a2=1) & (a3=1) & (a4=1) & (a5=1) & (a6=1) & (a7=1) & (a8=1) & (a9=1) & (b=1);

