/* Generated by: /Users/sfegan/GitHub/fftw3/genfft/gen_r2cf.native -n 30 -standalone -fma -generic-arith -compact -name dft_codelet_r2cf_30 */

/*
 * This function contains 158 FP additions, 70 FP multiplications,
 * (or, 102 additions, 14 multiplications, 56 fused multiply/add),
 * 63 stack variables, 8 constants, and 60 memory accesses
 */
void dft_codelet_r2cf_30(R * R0, R * R1, R * Cr, R * Ci, stride rs, stride csr, stride csi, INT v, INT ivs, INT ovs)
{
  DK(KP823639103, +0.823639103546331925877420039278190003029660514);
  DK(KP910592997, +0.910592997310029334643087372129977886038870291);
  DK(KP951056516, +0.951056516295153572116439333379382143405698634);
  DK(KP500000000, +0.500000000000000000000000000000000000000000000);
  DK(KP866025403, +0.866025403784438646763723170752936183471402627);
  DK(KP559016994, +0.559016994374947424102293417182819058860154590);
  DK(KP250000000, +0.250000000000000000000000000000000000000000000);
  DK(KP618033988, +0.618033988749894848204586834365638117720309180);
  {
    INT i;
    for (i = v; i > 0; i = i - 1, R0 = R0 + ivs, R1 = R1 + ivs, Cr = Cr + ovs, Ci = Ci + ovs, MAKE_VOLATILE_STRIDE(120, rs), MAKE_VOLATILE_STRIDE(120, csr), MAKE_VOLATILE_STRIDE(120, csi)) {
      E T1a, TC, T1f, TQ, T1d, T1e, T13, TJ, T18, TP, T16, T17, T2l,
       T2m, Tf;
      E TL, T1v, T23, T1o, T22, TU, T10, T2o, T2p, Tu, TM, T1K, T26,
       T1D, T25;
      E TX, T11;
      {
        E TB, T1c, Ty, T1b;
        T1a = R1[WS(rs, 7)];
        {
          E Tz, TA, Tw, Tx;
          Tz = R1[WS(rs, 13)];
          TA = R1[WS(rs, 1)];
          TB = SUB(Tz, TA);
          T1c = ADD(Tz, TA);
          Tw = R1[WS(rs, 10)];
          Tx = R1[WS(rs, 4)];
          Ty = SUB(Tw, Tx);
          T1b = ADD(Tw, Tx);
        }
        TC = FMA(KP618033988, TB, Ty);
        T1f = SUB(T1b, T1c);
        TQ = FNMS(KP618033988, Ty, TB);
        T1d = ADD(T1b, T1c);
        T1e = FNMS(KP250000000, T1d, T1a);
      }
      {
        E TI, T15, TF, T14;
        T13 = R0[0];
        {
          E TG, TH, TD, TE;
          TG = R0[WS(rs, 6)];
          TH = R0[WS(rs, 9)];
          TI = SUB(TG, TH);
          T15 = ADD(TG, TH);
          TD = R0[WS(rs, 3)];
          TE = R0[WS(rs, 12)];
          TF = SUB(TD, TE);
          T14 = ADD(TD, TE);
        }
        TJ = FMA(KP618033988, TI, TF);
        T18 = SUB(T14, T15);
        TP = FNMS(KP618033988, TF, TI);
        T16 = ADD(T14, T15);
        T17 = FNMS(KP250000000, T16, T13);
      }
      {
        E T1i, T1p, T1j, T1k, T3, T1l, T6, T1q, T1r, Ta, T1s, Td;
        T1i = R0[WS(rs, 5)];
        T1p = R1[WS(rs, 12)];
        {
          E T1, T2, T4, T5;
          T1 = R0[WS(rs, 8)];
          T2 = R0[WS(rs, 2)];
          T1j = ADD(T1, T2);
          T4 = R0[WS(rs, 11)];
          T5 = R0[WS(rs, 14)];
          T1k = ADD(T4, T5);
          T3 = SUB(T1, T2);
          T1l = ADD(T1j, T1k);
          T6 = SUB(T4, T5);
        }
        {
          E T8, T9, Tb, Tc;
          T8 = R1[WS(rs, 9)];
          T9 = R1[0];
          T1q = ADD(T9, T8);
          Tb = R1[WS(rs, 3)];
          Tc = R1[WS(rs, 6)];
          T1r = ADD(Tb, Tc);
          Ta = SUB(T8, T9);
          T1s = ADD(T1q, T1r);
          Td = SUB(Tb, Tc);
        }
        {
          E T7, Te, T1t, T1u;
          T2l = ADD(T1i, T1l);
          T2m = ADD(T1p, T1s);
          T7 = FMA(KP618033988, T6, T3);
          Te = FNMS(KP618033988, Td, Ta);
          Tf = ADD(T7, Te);
          TL = SUB(Te, T7);
          T1t = FNMS(KP250000000, T1s, T1p);
          T1u = SUB(T1q, T1r);
          T1v = FMA(KP559016994, T1u, T1t);
          T23 = FNMS(KP559016994, T1u, T1t);
          {
            E T1m, T1n, TS, TT;
            T1m = FNMS(KP250000000, T1l, T1i);
            T1n = SUB(T1j, T1k);
            T1o = FMA(KP559016994, T1n, T1m);
            T22 = FNMS(KP559016994, T1n, T1m);
            TS = FNMS(KP618033988, T3, T6);
            TT = FMA(KP618033988, Ta, Td);
            TU = ADD(TS, TT);
            T10 = SUB(TT, TS);
          }
        }
      }
      {
        E T1x, T1E, T1y, T1z, Ti, T1A, Tl, T1F, T1G, Tp, T1H, Ts;
        T1x = R0[WS(rs, 10)];
        T1E = R1[WS(rs, 2)];
        {
          E Tg, Th, Tj, Tk;
          Tg = R0[WS(rs, 13)];
          Th = R0[WS(rs, 7)];
          T1y = ADD(Tg, Th);
          Tj = R0[WS(rs, 1)];
          Tk = R0[WS(rs, 4)];
          T1z = ADD(Tj, Tk);
          Ti = SUB(Tg, Th);
          T1A = ADD(T1y, T1z);
          Tl = SUB(Tj, Tk);
        }
        {
          E Tn, To, Tq, Tr;
          Tn = R1[WS(rs, 14)];
          To = R1[WS(rs, 5)];
          T1F = ADD(To, Tn);
          Tq = R1[WS(rs, 8)];
          Tr = R1[WS(rs, 11)];
          T1G = ADD(Tq, Tr);
          Tp = SUB(Tn, To);
          T1H = ADD(T1F, T1G);
          Ts = SUB(Tq, Tr);
        }
        {
          E Tm, Tt, T1I, T1J;
          T2o = ADD(T1x, T1A);
          T2p = ADD(T1E, T1H);
          Tm = FMA(KP618033988, Tl, Ti);
          Tt = FNMS(KP618033988, Ts, Tp);
          Tu = ADD(Tm, Tt);
          TM = SUB(Tt, Tm);
          T1I = FNMS(KP250000000, T1H, T1E);
          T1J = SUB(T1F, T1G);
          T1K = FMA(KP559016994, T1J, T1I);
          T26 = FNMS(KP559016994, T1J, T1I);
          {
            E T1B, T1C, TV, TW;
            T1B = FNMS(KP250000000, T1A, T1x);
            T1C = SUB(T1y, T1z);
            T1D = FMA(KP559016994, T1C, T1B);
            T25 = FNMS(KP559016994, T1C, T1B);
            TV = FNMS(KP618033988, Ti, Tl);
            TW = FMA(KP618033988, Tp, Ts);
            TX = ADD(TV, TW);
            T11 = SUB(TW, TV);
          }
        }
      }
      {
        E T2n, T2q, T2u, T2w, T2x, T2y, T2t, T2v, T2r, T2s;
        T2n = SUB(T2l, T2m);
        T2q = SUB(T2o, T2p);
        T2u = ADD(T2n, T2q);
        T2w = ADD(T2l, T2m);
        T2x = ADD(T2o, T2p);
        T2y = ADD(T2w, T2x);
        T2r = ADD(T13, T16);
        T2s = ADD(T1a, T1d);
        T2t = SUB(T2r, T2s);
        T2v = ADD(T2r, T2s);
        Ci[WS(csi, 5)] = MUL(KP866025403, SUB(T2n, T2q));
        Cr[WS(csr, 5)] = FNMS(KP500000000, T2u, T2t);
        Cr[WS(csr, 15)] = ADD(T2t, T2u);
        Ci[WS(csi, 10)] = MUL(KP866025403, SUB(T2x, T2w));
        Cr[WS(csr, 10)] = FNMS(KP500000000, T2y, T2v);
        Cr[0] = ADD(T2v, T2y);
      }
      {
        E T2a, T2k, TR, TY, T2d, TZ, T12, T2b, T28, T2c, T21, T2h, T2g,
         T2i, T24;
        E T27, T29, T2j;
        T2a = SUB(T11, T10);
        T2k = SUB(TU, TX);
        TR = ADD(TP, TQ);
        TY = ADD(TU, TX);
        T2d = FNMS(KP500000000, TY, TR);
        TZ = SUB(TQ, TP);
        T12 = ADD(T10, T11);
        T2b = FNMS(KP500000000, T12, TZ);
        T24 = SUB(T22, T23);
        T27 = SUB(T25, T26);
        T28 = ADD(T24, T27);
        T2c = SUB(T27, T24);
        {
          E T1Z, T20, T2e, T2f;
          T1Z = FNMS(KP559016994, T18, T17);
          T20 = FNMS(KP559016994, T1f, T1e);
          T21 = SUB(T1Z, T20);
          T2h = ADD(T1Z, T20);
          T2e = ADD(T22, T23);
          T2f = ADD(T25, T26);
          T2g = SUB(T2e, T2f);
          T2i = ADD(T2e, T2f);
        }
        Ci[WS(csi, 12)] = MUL(KP951056516, ADD(TR, TY));
        Cr[WS(csr, 12)] = ADD(T2h, T2i);
        Ci[WS(csi, 3)] = MUL(KP951056516, ADD(TZ, T12));
        Cr[WS(csr, 3)] = ADD(T21, T28);
        Ci[WS(csi, 7)] = NEG(MUL(KP951056516, FNMS(KP910592997, T2c, T2b)));
        Ci[WS(csi, 8)] = NEG(MUL(KP951056516, FNMS(KP910592997, T2g, T2d)));
        Ci[WS(csi, 2)] = MUL(KP951056516, FMA(KP910592997, T2g, T2d));
        Ci[WS(csi, 13)] = MUL(KP951056516, FMA(KP910592997, T2c, T2b));
        T29 = FNMS(KP500000000, T28, T21);
        Cr[WS(csr, 7)] = FMA(KP823639103, T2a, T29);
        Cr[WS(csr, 13)] = FNMS(KP823639103, T2a, T29);
        T2j = FNMS(KP500000000, T2i, T2h);
        Cr[WS(csr, 2)] = FNMS(KP823639103, T2k, T2j);
        Cr[WS(csr, 8)] = FMA(KP823639103, T2k, T2j);
      }
      {
        E T1O, T1W, Tv, TK, T1P, TN, TO, T1X, T1M, T1Q, T1h, T1R, T1U,
         T1Y, T1w;
        E T1L, T1N, T1V;
        T1O = SUB(Tu, Tf);
        T1W = SUB(TM, TL);
        Tv = ADD(Tf, Tu);
        TK = SUB(TC, TJ);
        T1P = FMA(KP500000000, Tv, TK);
        TN = ADD(TL, TM);
        TO = ADD(TJ, TC);
        T1X = FMA(KP500000000, TN, TO);
        T1w = SUB(T1o, T1v);
        T1L = SUB(T1D, T1K);
        T1M = ADD(T1w, T1L);
        T1Q = SUB(T1L, T1w);
        {
          E T19, T1g, T1S, T1T;
          T19 = FMA(KP559016994, T18, T17);
          T1g = FMA(KP559016994, T1f, T1e);
          T1h = SUB(T19, T1g);
          T1R = ADD(T19, T1g);
          T1S = ADD(T1o, T1v);
          T1T = ADD(T1D, T1K);
          T1U = ADD(T1S, T1T);
          T1Y = SUB(T1T, T1S);
        }
        Ci[WS(csi, 9)] = MUL(KP951056516, SUB(Tv, TK));
        Cr[WS(csr, 9)] = ADD(T1h, T1M);
        Ci[WS(csi, 6)] = MUL(KP951056516, SUB(TN, TO));
        Cr[WS(csr, 6)] = ADD(T1R, T1U);
        Ci[WS(csi, 11)] = MUL(KP951056516, FNMS(KP910592997, T1Q, T1P));
        Ci[WS(csi, 4)] = MUL(KP951056516, FMA(KP910592997, T1Y, T1X));
        Ci[WS(csi, 14)] = MUL(KP951056516, FNMS(KP910592997, T1Y, T1X));
        Ci[WS(csi, 1)] = MUL(KP951056516, FMA(KP910592997, T1Q, T1P));
        T1N = FNMS(KP500000000, T1M, T1h);
        Cr[WS(csr, 1)] = FMA(KP823639103, T1O, T1N);
        Cr[WS(csr, 11)] = FNMS(KP823639103, T1O, T1N);
        T1V = FNMS(KP500000000, T1U, T1R);
        Cr[WS(csr, 4)] = FMA(KP823639103, T1W, T1V);
        Cr[WS(csr, 14)] = FNMS(KP823639103, T1W, T1V);
      }
    }
  }
}
