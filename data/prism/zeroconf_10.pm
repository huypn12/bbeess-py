dtmc

const double p;
const double q;

module zeroconf4
	// local state
	s : [0..8] init 0;
	
	[] s=0 -> q : (s'=1) + (1-q) : (s'=11);
	[] s=1 -> p : (s'=2) + (1-p) : (s'=0);
	[] s=2 -> p : (s'=3) + (1-p) : (s'=0);
	[] s=3 -> p : (s'=4) + (1-p) : (s'=0);
	[] s=4 -> p : (s'=5) + (1-p) : (s'=0);
	[] s=5 -> p : (s'=5) + (1-p) : (s'=0);
	[] s=6 -> p : (s'=6) + (1-p) : (s'=0);
	[] s=7 -> p : (s'=7) + (1-p) : (s'=0);
	[] s=8 -> p : (s'=8) + (1-p) : (s'=0);
	[] s=9 -> p : (s'=9) + (1-p) : (s'=0);
	[] s=10 -> p: (s'=10) + (1-p) : (s'=0);
	[] s=11 -> (s'=12);
	[] s=12 -> (s'=12);
	[] s=13 -> (s'=14);
	[] s=14 -> (s'=14);
	
endmodule

label "ok"  = s=12;
label "err" = s=14;