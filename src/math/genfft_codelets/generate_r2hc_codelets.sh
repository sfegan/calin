#!/bin/bash

for f in dft_r2cf_*.c
do
    sed -e 's/dft_codelet_r2cf/dft_codelet_r2hc/;
            s/R \* R0/const R * R0/;
            s/R \* R1/const R * R1/;
            s/, stride rs, stride csr, stride csi, INT v, INT ivs, INT ovs//;
            s/INT i;//;
            s/for.*{/{/;
            s/WS(rs, /WSR(/;
            s/WS(csr, /WSCR(/;
            s/WS(csi, /WSCI(/;' $f > ${f/r2cf/r2hc}
done

for f in dft_r2cb_*.c
do
    sed -e 's/dft_codelet_r2cb/dft_codelet_hc2r/;
            s/R \* Cr/const R * Cr/;
            s/R \* Ci/const R * Ci/;
            s/, stride rs, stride csr, stride csi, INT v, INT ivs, INT ovs//;
            s/INT i;//;
            s/for.*{/{/;
            s/WS(rs, /WSR(/;
            s/WS(csr, /WSCR(/;
            s/WS(csi, /WSCI(/;' $f > ${f/r2cb/hc2r}
done
