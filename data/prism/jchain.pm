dtmc

const double p0;
const double p1;
const double p2;
const double p3;
const double p4;
const double p5;
const double p6;
const double p7;
const double p8;
const double p9;

module jchain10

	s : [0..20] init 0;
	
	[] s=0 -> p0 : (s'=1) +  (1-p0) : (s'=11);
	[] s=1 -> p1 : (s'=2) +  (1-p1) : (s'=12);
	[] s=2 -> p2 : (s'=3) +  (1-p2) : (s'=13);
	[] s=3 -> p3 : (s'=4) +  (1-p3) : (s'=14);
	[] s=4 -> p4 : (s'=5) +  (1-p4) : (s'=15);
	[] s=5 -> p5 : (s'=6) +  (1-p5) : (s'=16);
	[] s=6 -> p6 : (s'=7) +  (1-p6) : (s'=17);
	[] s=7 -> p7 : (s'=8) +  (1-p7) : (s'=18);
	[] s=8 -> p8 : (s'=9) +  (1-p8) : (s'=19);
	[] s=9 -> p9 : (s'=10) + (1-p9) : (s'=20);
	[] s=10 -> (s'=10);
	[] s=11 -> (s'=11);
	[] s=12 -> (s'=12);
	[] s=13 -> (s'=13);
	[] s=14 -> (s'=14);
	[] s=15 -> (s'=15);
    [] s=16 -> (s'=16);
	[] s=17 -> (s'=17);
	[] s=18 -> (s'=18);
	[] s=19 -> (s'=19);
    [] s=20 -> (s'=20);
	
endmodule

label "bscc_10" = (s=10);
label "bscc_11" = (s=11);
label "bscc_12" = (s=12);
label "bscc_13" = (s=13);
label "bscc_14" = (s=14);
label "bscc_15" = (s=15);
label "bscc_16" = (s=16);
label "bscc_17" = (s=17);
label "bscc_18" = (s=18);
label "bscc_19" = (s=19);
label "bscc_20" = (s=20);
