/* Generated by: /Users/sfegan/GitHub/fftw3/genfft/gen_r2cf.native -n 48 -standalone -fma -generic-arith -compact -name dft_codelet_r2cf_48 */

/*
 * This function contains 266 FP additions, 106 FP multiplications,
 * (or, 162 additions, 2 multiplications, 104 fused multiply/add),
 * 82 stack variables, 5 constants, and 96 memory accesses
 */
void dft_codelet_r2cf_48(R * R0, R * R1, R * Cr, R * Ci, stride rs, stride csr, stride csi, INT v, INT ivs, INT ovs)
{
  DK(KP866025403, +0.866025403784438646763723170752936183471402627);
  DK(KP923879532, +0.923879532511286756128183189396788286822416626);
  DK(KP414213562, +0.414213562373095048801688724209698078569671875);
  DK(KP707106781, +0.707106781186547524400844362104849039284835938);
  DK(KP500000000, +0.500000000000000000000000000000000000000000000);
  {
    INT i;
    for (i = v; i > 0; i = i - 1, R0 = R0 + ivs, R1 = R1 + ivs, Cr = Cr + ovs, Ci = Ci + ovs, MAKE_VOLATILE_STRIDE(192, rs), MAKE_VOLATILE_STRIDE(192, csr), MAKE_VOLATILE_STRIDE(192, csi)) {
      E T1y, T2K, T5, Ta, Tb, T1N, T3h, T1B, T3i, Tg, Tl, Tm, T1Q, T2J,
       T1F;
      E T2N, Ty, T22, T1U, T2M, T1I, T2Q, TJ, T23, T1X, T2P, T2X, T32,
       T3T, T3U;
      E T3V, T2b, T2w, T1t, T2U, T1i, T2Z, T28, T2x, T38, T3d, T3Y,
       T3Z, T40, T2i;
      E T2z, T16, T35, TV, T3a, T2f, T2A;
      {
        E T1, T6, T4, T1w, T9, T1x, T1L, T1M;
        T1 = R0[0];
        T6 = R0[WS(rs, 12)];
        {
          E T2, T3, T7, T8;
          T2 = R0[WS(rs, 8)];
          T3 = R0[WS(rs, 16)];
          T4 = ADD(T2, T3);
          T1w = SUB(T3, T2);
          T7 = R0[WS(rs, 20)];
          T8 = R0[WS(rs, 4)];
          T9 = ADD(T7, T8);
          T1x = SUB(T8, T7);
        }
        T1y = ADD(T1w, T1x);
        T2K = SUB(T1w, T1x);
        T5 = ADD(T1, T4);
        Ta = ADD(T6, T9);
        Tb = ADD(T5, Ta);
        T1L = FNMS(KP500000000, T4, T1);
        T1M = FNMS(KP500000000, T9, T6);
        T1N = ADD(T1L, T1M);
        T3h = SUB(T1L, T1M);
      }
      {
        E Tc, Th, Tf, T1z, Tk, T1A, T1O, T1P;
        Tc = R0[WS(rs, 6)];
        Th = R0[WS(rs, 18)];
        {
          E Td, Te, Ti, Tj;
          Td = R0[WS(rs, 14)];
          Te = R0[WS(rs, 22)];
          Tf = ADD(Td, Te);
          T1z = SUB(Te, Td);
          Ti = R0[WS(rs, 2)];
          Tj = R0[WS(rs, 10)];
          Tk = ADD(Ti, Tj);
          T1A = SUB(Tj, Ti);
        }
        T1B = ADD(T1z, T1A);
        T3i = SUB(T1z, T1A);
        Tg = ADD(Tc, Tf);
        Tl = ADD(Th, Tk);
        Tm = ADD(Tg, Tl);
        T1O = FNMS(KP500000000, Tf, Tc);
        T1P = FNMS(KP500000000, Tk, Th);
        T1Q = ADD(T1O, T1P);
        T2J = SUB(T1O, T1P);
      }
      {
        E To, Tt, Tr, T1D, Tw, T1E;
        To = R0[WS(rs, 3)];
        Tt = R0[WS(rs, 15)];
        {
          E Tp, Tq, Tu, Tv;
          Tp = R0[WS(rs, 11)];
          Tq = R0[WS(rs, 19)];
          Tr = ADD(Tp, Tq);
          T1D = SUB(Tq, Tp);
          Tu = R0[WS(rs, 23)];
          Tv = R0[WS(rs, 7)];
          Tw = ADD(Tu, Tv);
          T1E = SUB(Tv, Tu);
        }
        T1F = ADD(T1D, T1E);
        T2N = SUB(T1D, T1E);
        {
          E Ts, Tx, T1S, T1T;
          Ts = ADD(To, Tr);
          Tx = ADD(Tt, Tw);
          Ty = ADD(Ts, Tx);
          T22 = SUB(Ts, Tx);
          T1S = FNMS(KP500000000, Tr, To);
          T1T = FNMS(KP500000000, Tw, Tt);
          T1U = ADD(T1S, T1T);
          T2M = SUB(T1S, T1T);
        }
      }
      {
        E Tz, TE, TC, T1G, TH, T1H;
        Tz = R0[WS(rs, 21)];
        TE = R0[WS(rs, 9)];
        {
          E TA, TB, TF, TG;
          TA = R0[WS(rs, 5)];
          TB = R0[WS(rs, 13)];
          TC = ADD(TA, TB);
          T1G = SUB(TB, TA);
          TF = R0[WS(rs, 17)];
          TG = R0[WS(rs, 1)];
          TH = ADD(TF, TG);
          T1H = SUB(TG, TF);
        }
        T1I = ADD(T1G, T1H);
        T2Q = SUB(T1G, T1H);
        {
          E TD, TI, T1V, T1W;
          TD = ADD(Tz, TC);
          TI = ADD(TE, TH);
          TJ = ADD(TD, TI);
          T23 = SUB(TD, TI);
          T1V = FNMS(KP500000000, TC, Tz);
          T1W = FNMS(KP500000000, TH, TE);
          T1X = ADD(T1V, T1W);
          T2P = SUB(T1V, T1W);
        }
      }
      {
        E T18, T1j, T1o, T1d, T1b, T2V, T1m, T30, T1r, T31, T1g, T2W;
        T18 = R1[WS(rs, 1)];
        T1j = R1[WS(rs, 7)];
        T1o = R1[WS(rs, 19)];
        T1d = R1[WS(rs, 13)];
        {
          E T19, T1a, T1k, T1l;
          T19 = R1[WS(rs, 9)];
          T1a = R1[WS(rs, 17)];
          T1b = ADD(T19, T1a);
          T2V = SUB(T1a, T19);
          T1k = R1[WS(rs, 15)];
          T1l = R1[WS(rs, 23)];
          T1m = ADD(T1k, T1l);
          T30 = SUB(T1l, T1k);
        }
        {
          E T1p, T1q, T1e, T1f;
          T1p = R1[WS(rs, 3)];
          T1q = R1[WS(rs, 11)];
          T1r = ADD(T1p, T1q);
          T31 = SUB(T1q, T1p);
          T1e = R1[WS(rs, 21)];
          T1f = R1[WS(rs, 5)];
          T1g = ADD(T1e, T1f);
          T2W = SUB(T1f, T1e);
        }
        T2X = SUB(T2V, T2W);
        T32 = SUB(T30, T31);
        T3T = ADD(T30, T31);
        T3U = ADD(T2V, T2W);
        T3V = SUB(T3T, T3U);
        {
          E T29, T2a, T1n, T1s;
          T29 = ADD(T18, T1b);
          T2a = ADD(T1d, T1g);
          T2b = SUB(T29, T2a);
          T2w = ADD(T29, T2a);
          T1n = FNMS(KP500000000, T1m, T1j);
          T1s = FNMS(KP500000000, T1r, T1o);
          T1t = ADD(T1n, T1s);
          T2U = SUB(T1s, T1n);
        }
        {
          E T1c, T1h, T26, T27;
          T1c = FNMS(KP500000000, T1b, T18);
          T1h = FNMS(KP500000000, T1g, T1d);
          T1i = ADD(T1c, T1h);
          T2Z = SUB(T1c, T1h);
          T26 = ADD(T1j, T1m);
          T27 = ADD(T1o, T1r);
          T28 = SUB(T26, T27);
          T2x = ADD(T26, T27);
        }
      }
      {
        E TL, TW, T11, TQ, TO, T36, TZ, T3b, T14, T3c, TT, T37;
        TL = R1[WS(rs, 22)];
        TW = R1[WS(rs, 4)];
        T11 = R1[WS(rs, 16)];
        TQ = R1[WS(rs, 10)];
        {
          E TM, TN, TX, TY;
          TM = R1[WS(rs, 6)];
          TN = R1[WS(rs, 14)];
          TO = ADD(TM, TN);
          T36 = SUB(TN, TM);
          TX = R1[WS(rs, 12)];
          TY = R1[WS(rs, 20)];
          TZ = ADD(TX, TY);
          T3b = SUB(TY, TX);
        }
        {
          E T12, T13, TR, TS;
          T12 = R1[0];
          T13 = R1[WS(rs, 8)];
          T14 = ADD(T12, T13);
          T3c = SUB(T13, T12);
          TR = R1[WS(rs, 18)];
          TS = R1[WS(rs, 2)];
          TT = ADD(TR, TS);
          T37 = SUB(TS, TR);
        }
        T38 = SUB(T36, T37);
        T3d = SUB(T3b, T3c);
        T3Y = ADD(T3b, T3c);
        T3Z = ADD(T36, T37);
        T40 = SUB(T3Y, T3Z);
        {
          E T2g, T2h, T10, T15;
          T2g = ADD(TL, TO);
          T2h = ADD(TQ, TT);
          T2i = SUB(T2g, T2h);
          T2z = ADD(T2g, T2h);
          T10 = FNMS(KP500000000, TZ, TW);
          T15 = FNMS(KP500000000, T14, T11);
          T16 = ADD(T10, T15);
          T35 = SUB(T10, T15);
        }
        {
          E TP, TU, T2d, T2e;
          TP = FNMS(KP500000000, TO, TL);
          TU = FNMS(KP500000000, TT, TQ);
          TV = ADD(TP, TU);
          T3a = SUB(TP, TU);
          T2d = ADD(TW, TZ);
          T2e = ADD(T11, T14);
          T2f = SUB(T2d, T2e);
          T2A = ADD(T2d, T2e);
        }
      }
      {
        E Tn, TK, T2H, T2F, T2G, T2I;
        Tn = ADD(Tb, Tm);
        TK = ADD(Ty, TJ);
        T2H = ADD(Tn, TK);
        T2F = ADD(T2w, T2x);
        T2G = ADD(T2z, T2A);
        T2I = ADD(T2F, T2G);
        Cr[WS(csr, 12)] = SUB(Tn, TK);
        Ci[WS(csi, 12)] = SUB(T2F, T2G);
        Cr[WS(csr, 24)] = SUB(T2H, T2I);
        Cr[0] = ADD(T2H, T2I);
      }
      {
        E T2v, T2D, T2C, T2E, T2y, T2B;
        T2v = SUB(Tb, Tm);
        T2D = SUB(Ty, TJ);
        T2y = SUB(T2w, T2x);
        T2B = SUB(T2z, T2A);
        T2C = ADD(T2y, T2B);
        T2E = SUB(T2B, T2y);
        Cr[WS(csr, 6)] = FNMS(KP707106781, T2C, T2v);
        Ci[WS(csi, 6)] = FMA(KP707106781, T2E, T2D);
        Cr[WS(csr, 18)] = FMA(KP707106781, T2C, T2v);
        Ci[WS(csi, 18)] = NEG(FNMS(KP707106781, T2E, T2D));
      }
      {
        E T25, T2t, T2s, T2u, T2k, T2o, T2n, T2p;
        {
          E T21, T24, T2q, T2r;
          T21 = SUB(T5, Ta);
          T24 = ADD(T22, T23);
          T25 = FNMS(KP707106781, T24, T21);
          T2t = FMA(KP707106781, T24, T21);
          T2q = FMA(KP414213562, T2f, T2i);
          T2r = FNMS(KP414213562, T28, T2b);
          T2s = SUB(T2q, T2r);
          T2u = ADD(T2r, T2q);
        }
        {
          E T2c, T2j, T2l, T2m;
          T2c = FMA(KP414213562, T2b, T28);
          T2j = FNMS(KP414213562, T2i, T2f);
          T2k = SUB(T2c, T2j);
          T2o = ADD(T2c, T2j);
          T2l = SUB(Tg, Tl);
          T2m = SUB(T23, T22);
          T2n = FNMS(KP707106781, T2m, T2l);
          T2p = FMA(KP707106781, T2m, T2l);
        }
        Cr[WS(csr, 21)] = FNMS(KP923879532, T2k, T25);
        Ci[WS(csi, 21)] = NEG(FNMS(KP923879532, T2s, T2p));
        Cr[WS(csr, 3)] = FMA(KP923879532, T2k, T25);
        Ci[WS(csi, 3)] = FMA(KP923879532, T2s, T2p);
        Ci[WS(csi, 9)] = NEG(FNMS(KP923879532, T2o, T2n));
        Cr[WS(csr, 9)] = FNMS(KP923879532, T2u, T2t);
        Ci[WS(csi, 15)] = FMA(KP923879532, T2o, T2n);
        Cr[WS(csr, 15)] = FMA(KP923879532, T2u, T2t);
      }
      {
        E T3W, T49, T41, T48, T3R, T47, T45, T4b, T3S, T3X;
        T3S = SUB(T1i, T1t);
        T3W = FMA(KP866025403, T3V, T3S);
        T49 = FNMS(KP866025403, T3V, T3S);
        T3X = SUB(TV, T16);
        T41 = FNMS(KP866025403, T40, T3X);
        T48 = FMA(KP866025403, T40, T3X);
        {
          E T3P, T3Q, T43, T44;
          T3P = SUB(T1X, T1U);
          T3Q = SUB(T1B, T1y);
          T3R = FNMS(KP866025403, T3Q, T3P);
          T47 = FMA(KP866025403, T3Q, T3P);
          T43 = SUB(T1N, T1Q);
          T44 = SUB(T1I, T1F);
          T45 = FMA(KP866025403, T44, T43);
          T4b = FNMS(KP866025403, T44, T43);
        }
        {
          E T42, T4c, T46, T4a;
          T42 = SUB(T3W, T41);
          Ci[WS(csi, 10)] = FMA(KP707106781, T42, T3R);
          Ci[WS(csi, 14)] = NEG(FNMS(KP707106781, T42, T3R));
          T4c = ADD(T49, T48);
          Cr[WS(csr, 10)] = FNMS(KP707106781, T4c, T4b);
          Cr[WS(csr, 14)] = FMA(KP707106781, T4c, T4b);
          T46 = ADD(T3W, T41);
          Cr[WS(csr, 22)] = FNMS(KP707106781, T46, T45);
          Cr[WS(csr, 2)] = FMA(KP707106781, T46, T45);
          T4a = SUB(T48, T49);
          Ci[WS(csi, 2)] = FMA(KP707106781, T4a, T47);
          Ci[WS(csi, 22)] = NEG(FNMS(KP707106781, T4a, T47));
        }
      }
      {
        E T1v, T20, T4g, T4h, T1K, T4i, T1Z, T4d;
        {
          E T17, T1u, T4e, T4f;
          T17 = ADD(TV, T16);
          T1u = ADD(T1i, T1t);
          T1v = SUB(T17, T1u);
          T20 = ADD(T1u, T17);
          T4e = ADD(T3U, T3T);
          T4f = ADD(T3Z, T3Y);
          T4g = SUB(T4e, T4f);
          T4h = ADD(T4e, T4f);
        }
        {
          E T1C, T1J, T1R, T1Y;
          T1C = ADD(T1y, T1B);
          T1J = ADD(T1F, T1I);
          T1K = SUB(T1C, T1J);
          T4i = ADD(T1C, T1J);
          T1R = ADD(T1N, T1Q);
          T1Y = ADD(T1U, T1X);
          T1Z = ADD(T1R, T1Y);
          T4d = SUB(T1R, T1Y);
        }
        Ci[WS(csi, 4)] = FMA(KP866025403, T1K, T1v);
        Cr[WS(csr, 4)] = FMA(KP866025403, T4g, T4d);
        Ci[WS(csi, 20)] = FNMS(KP866025403, T1K, T1v);
        Cr[WS(csr, 20)] = FNMS(KP866025403, T4g, T4d);
        Cr[WS(csr, 8)] = SUB(T1Z, T20);
        Ci[WS(csi, 8)] = MUL(KP866025403, SUB(T4h, T4i));
        Cr[WS(csr, 16)] = ADD(T1Z, T20);
        Ci[WS(csi, 16)] = MUL(KP866025403, ADD(T4i, T4h));
      }
      {
        E T2L, T3v, T3F, T3j, T2S, T3G, T3D, T3L, T3m, T3w, T34, T3r,
         T3A, T3K, T3f;
        E T3q;
        {
          E T2O, T2R, T2Y, T33;
          T2L = FMA(KP866025403, T2K, T2J);
          T3v = FNMS(KP866025403, T2K, T2J);
          T3F = FMA(KP866025403, T3i, T3h);
          T3j = FNMS(KP866025403, T3i, T3h);
          T2O = FMA(KP866025403, T2N, T2M);
          T2R = FNMS(KP866025403, T2Q, T2P);
          T2S = SUB(T2O, T2R);
          T3G = ADD(T2O, T2R);
          {
            E T3B, T3C, T3k, T3l;
            T3B = FNMS(KP866025403, T38, T35);
            T3C = FMA(KP866025403, T3d, T3a);
            T3D = FNMS(KP414213562, T3C, T3B);
            T3L = FMA(KP414213562, T3B, T3C);
            T3k = FNMS(KP866025403, T2N, T2M);
            T3l = FMA(KP866025403, T2Q, T2P);
            T3m = ADD(T3k, T3l);
            T3w = SUB(T3l, T3k);
          }
          T2Y = FNMS(KP866025403, T2X, T2U);
          T33 = FNMS(KP866025403, T32, T2Z);
          T34 = FNMS(KP414213562, T33, T2Y);
          T3r = FMA(KP414213562, T2Y, T33);
          {
            E T3y, T3z, T39, T3e;
            T3y = FMA(KP866025403, T2X, T2U);
            T3z = FMA(KP866025403, T32, T2Z);
            T3A = FNMS(KP414213562, T3z, T3y);
            T3K = FMA(KP414213562, T3y, T3z);
            T39 = FMA(KP866025403, T38, T35);
            T3e = FNMS(KP866025403, T3d, T3a);
            T3f = FNMS(KP414213562, T3e, T39);
            T3q = FMA(KP414213562, T39, T3e);
          }
        }
        {
          E T2T, T3g, T3t, T3u;
          T2T = FMA(KP707106781, T2S, T2L);
          T3g = SUB(T34, T3f);
          Ci[WS(csi, 7)] = FMA(KP923879532, T3g, T2T);
          Ci[WS(csi, 17)] = NEG(FNMS(KP923879532, T3g, T2T));
          T3t = FMA(KP707106781, T3m, T3j);
          T3u = ADD(T3r, T3q);
          Cr[WS(csr, 7)] = FNMS(KP923879532, T3u, T3t);
          Cr[WS(csr, 17)] = FMA(KP923879532, T3u, T3t);
        }
        {
          E T3n, T3o, T3p, T3s;
          T3n = FNMS(KP707106781, T3m, T3j);
          T3o = ADD(T34, T3f);
          Cr[WS(csr, 19)] = FNMS(KP923879532, T3o, T3n);
          Cr[WS(csr, 5)] = FMA(KP923879532, T3o, T3n);
          T3p = FNMS(KP707106781, T2S, T2L);
          T3s = SUB(T3q, T3r);
          Ci[WS(csi, 5)] = NEG(FNMS(KP923879532, T3s, T3p));
          Ci[WS(csi, 19)] = FMA(KP923879532, T3s, T3p);
        }
        {
          E T3x, T3E, T3N, T3O;
          T3x = FNMS(KP707106781, T3w, T3v);
          T3E = SUB(T3A, T3D);
          Ci[WS(csi, 1)] = NEG(FNMS(KP923879532, T3E, T3x));
          Ci[WS(csi, 23)] = FMA(KP923879532, T3E, T3x);
          T3N = FMA(KP707106781, T3G, T3F);
          T3O = ADD(T3K, T3L);
          Cr[WS(csr, 23)] = FNMS(KP923879532, T3O, T3N);
          Cr[WS(csr, 1)] = FMA(KP923879532, T3O, T3N);
        }
        {
          E T3H, T3I, T3J, T3M;
          T3H = FNMS(KP707106781, T3G, T3F);
          T3I = ADD(T3A, T3D);
          Cr[WS(csr, 13)] = FNMS(KP923879532, T3I, T3H);
          Cr[WS(csr, 11)] = FMA(KP923879532, T3I, T3H);
          T3J = FMA(KP707106781, T3w, T3v);
          T3M = SUB(T3K, T3L);
          Ci[WS(csi, 11)] = FMA(KP923879532, T3M, T3J);
          Ci[WS(csi, 13)] = NEG(FNMS(KP923879532, T3M, T3J));
        }
      }
    }
  }
}
