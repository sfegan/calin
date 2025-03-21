/* Generated by: /Users/sfegan/GitHub/fftw3/genfft/gen_r2cf.native -n 56 -standalone -fma -generic-arith -compact -name dft_codelet_r2cf_56 */

/*
 * This function contains 368 FP additions, 190 FP multiplications,
 * (or, 184 additions, 6 multiplications, 184 fused multiply/add),
 * 96 stack variables, 7 constants, and 112 memory accesses
 */
void dft_codelet_r2cf_56(R * R0, R * R1, R * Cr, R * Ci, stride rs, stride csr, stride csi, INT v, INT ivs, INT ovs)
{
  DK(KP974927912, +0.974927912181823607018131682993931217232785801);
  DK(KP707106781, +0.707106781186547524400844362104849039284835938);
  DK(KP900968867, +0.900968867902419126236102319507445051165919162);
  DK(KP801937735, +0.801937735804838252472204639014890102331838324);
  DK(KP554958132, +0.554958132087371191422194871006410481067288862);
  DK(KP692021471, +0.692021471630095869627814897002069140197260599);
  DK(KP356895867, +0.356895867892209443894399510021300583399127187);
  {
    INT i;
    for (i = v; i > 0; i = i - 1, R0 = R0 + ivs, R1 = R1 + ivs, Cr = Cr + ovs, Ci = Ci + ovs, MAKE_VOLATILE_STRIDE(224, rs), MAKE_VOLATILE_STRIDE(224, csr), MAKE_VOLATILE_STRIDE(224, csi)) {
      E T1, Tb, TM, T3o, T3i, T2x, T1Y, T2I, To, Ty, TT, T3v, T3d, T2C,
       T29;
      E T2P, Tz, TJ, TW, T3y, T3f, T2E, T2e, T2S, Tc, Tm, TP, T3r, T3k,
       T2z;
      E T23, T2L, T1c, T2q, T5l, T4D, T3E, T2Z, T3Z, T1D, T2j, T5e,
       T4I, T3H, T36;
      E T4c, T1Q, T2m, T5g, T4K, T3I, T39, T4h, T1p, T2t, T5n, T4F,
       T3F, T32, T44;
      {
        E Ta, T1U, T4, T1W, T7, T1V, T8, T9, TL, T3n;
        T1 = R0[0];
        T8 = R0[WS(rs, 12)];
        T9 = R0[WS(rs, 16)];
        Ta = ADD(T8, T9);
        T1U = SUB(T9, T8);
        {
          E T2, T3, T5, T6;
          T2 = R0[WS(rs, 4)];
          T3 = R0[WS(rs, 24)];
          T4 = ADD(T2, T3);
          T1W = SUB(T3, T2);
          T5 = R0[WS(rs, 8)];
          T6 = R0[WS(rs, 20)];
          T7 = ADD(T5, T6);
          T1V = SUB(T6, T5);
        }
        Tb = ADD(T1, ADD(T4, ADD(T7, Ta)));
        TL = FNMS(KP356895867, Ta, T7);
        TM = FNMS(KP692021471, TL, T4);
        T3n = FNMS(KP356895867, T7, T4);
        T3o = FNMS(KP692021471, T3n, Ta);
        {
          E T3h, T2w, T1X, T2H;
          T3h = FMA(KP554958132, T1U, T1W);
          T3i = FMA(KP801937735, T3h, T1V);
          T2w = FMA(KP554958132, T1V, T1U);
          T2x = FNMS(KP801937735, T2w, T1W);
          T1X = FNMS(KP554958132, T1W, T1V);
          T1Y = FNMS(KP801937735, T1X, T1U);
          T2H = FNMS(KP356895867, T4, Ta);
          T2I = FNMS(KP692021471, T2H, T7);
        }
      }
      {
        E Tx, T25, Tr, T27, Tu, T26, Tv, Tw, TS, T3u;
        To = R0[WS(rs, 7)];
        Tv = R0[WS(rs, 19)];
        Tw = R0[WS(rs, 23)];
        Tx = ADD(Tv, Tw);
        T25 = SUB(Tw, Tv);
        {
          E Tp, Tq, Ts, Tt;
          Tp = R0[WS(rs, 11)];
          Tq = R0[WS(rs, 3)];
          Tr = ADD(Tp, Tq);
          T27 = SUB(Tq, Tp);
          Ts = R0[WS(rs, 15)];
          Tt = R0[WS(rs, 27)];
          Tu = ADD(Ts, Tt);
          T26 = SUB(Tt, Ts);
        }
        Ty = ADD(To, ADD(Tr, ADD(Tu, Tx)));
        TS = FNMS(KP356895867, Tx, Tu);
        TT = FNMS(KP692021471, TS, Tr);
        T3u = FNMS(KP356895867, Tu, Tr);
        T3v = FNMS(KP692021471, T3u, Tx);
        {
          E T3c, T2B, T28, T2O;
          T3c = FMA(KP554958132, T25, T27);
          T3d = FMA(KP801937735, T3c, T26);
          T2B = FMA(KP554958132, T26, T25);
          T2C = FNMS(KP801937735, T2B, T27);
          T28 = FNMS(KP554958132, T27, T26);
          T29 = FNMS(KP801937735, T28, T25);
          T2O = FNMS(KP356895867, Tr, Tx);
          T2P = FNMS(KP692021471, T2O, Tu);
        }
      }
      {
        E TI, T2a, TC, T2c, TF, T2b, TG, TH, TV, T3x;
        Tz = R0[WS(rs, 21)];
        TG = R0[WS(rs, 5)];
        TH = R0[WS(rs, 9)];
        TI = ADD(TG, TH);
        T2a = SUB(TH, TG);
        {
          E TA, TB, TD, TE;
          TA = R0[WS(rs, 25)];
          TB = R0[WS(rs, 17)];
          TC = ADD(TA, TB);
          T2c = SUB(TB, TA);
          TD = R0[WS(rs, 1)];
          TE = R0[WS(rs, 13)];
          TF = ADD(TD, TE);
          T2b = SUB(TE, TD);
        }
        TJ = ADD(Tz, ADD(TC, ADD(TF, TI)));
        TV = FNMS(KP356895867, TI, TF);
        TW = FNMS(KP692021471, TV, TC);
        T3x = FNMS(KP356895867, TF, TC);
        T3y = FNMS(KP692021471, T3x, TI);
        {
          E T3e, T2D, T2d, T2R;
          T3e = FMA(KP554958132, T2a, T2c);
          T3f = FMA(KP801937735, T3e, T2b);
          T2D = FMA(KP554958132, T2b, T2a);
          T2E = FNMS(KP801937735, T2D, T2c);
          T2d = FNMS(KP554958132, T2c, T2b);
          T2e = FNMS(KP801937735, T2d, T2a);
          T2R = FNMS(KP356895867, TC, TI);
          T2S = FNMS(KP692021471, T2R, TF);
        }
      }
      {
        E Tl, T1Z, Tf, T21, Ti, T20, Tj, Tk, TO, T3q;
        Tc = R0[WS(rs, 14)];
        Tj = R0[WS(rs, 26)];
        Tk = R0[WS(rs, 2)];
        Tl = ADD(Tj, Tk);
        T1Z = SUB(Tk, Tj);
        {
          E Td, Te, Tg, Th;
          Td = R0[WS(rs, 18)];
          Te = R0[WS(rs, 10)];
          Tf = ADD(Td, Te);
          T21 = SUB(Te, Td);
          Tg = R0[WS(rs, 22)];
          Th = R0[WS(rs, 6)];
          Ti = ADD(Tg, Th);
          T20 = SUB(Th, Tg);
        }
        Tm = ADD(Tc, ADD(Tf, ADD(Ti, Tl)));
        TO = FNMS(KP356895867, Tl, Ti);
        TP = FNMS(KP692021471, TO, Tf);
        T3q = FNMS(KP356895867, Ti, Tf);
        T3r = FNMS(KP692021471, T3q, Tl);
        {
          E T3j, T2y, T22, T2K;
          T3j = FMA(KP554958132, T1Z, T21);
          T3k = FMA(KP801937735, T3j, T20);
          T2y = FMA(KP554958132, T20, T1Z);
          T2z = FNMS(KP801937735, T2y, T21);
          T22 = FNMS(KP554958132, T21, T20);
          T23 = FNMS(KP801937735, T22, T1Z);
          T2K = FNMS(KP356895867, Tf, Tl);
          T2L = FNMS(KP692021471, T2K, Ti);
        }
      }
      {
        E T10, T13, T19, T16, T1a, T2o, T3X, T3W, T3V, T2X;
        T10 = R1[WS(rs, 3)];
        {
          E T11, T12, T17, T18, T14, T15;
          T11 = R1[WS(rs, 7)];
          T12 = R1[WS(rs, 27)];
          T13 = ADD(T11, T12);
          T17 = R1[WS(rs, 15)];
          T18 = R1[WS(rs, 19)];
          T19 = ADD(T17, T18);
          T14 = R1[WS(rs, 11)];
          T15 = R1[WS(rs, 23)];
          T16 = ADD(T14, T15);
          T1a = FNMS(KP356895867, T19, T16);
          T2o = FNMS(KP356895867, T13, T19);
          T3X = SUB(T18, T17);
          T3W = SUB(T12, T11);
          T3V = SUB(T15, T14);
          T2X = FNMS(KP356895867, T16, T13);
        }
        {
          E T1b, T2p, T5k, T4C, T2Y, T3Y;
          T1b = FNMS(KP692021471, T1a, T13);
          T1c = FNMS(KP900968867, T1b, T10);
          T2p = FNMS(KP692021471, T2o, T16);
          T2q = FNMS(KP900968867, T2p, T10);
          T5k = FNMS(KP554958132, T3W, T3V);
          T5l = FNMS(KP801937735, T5k, T3X);
          T4C = FMA(KP554958132, T3V, T3X);
          T4D = FNMS(KP801937735, T4C, T3W);
          T3E = ADD(T10, ADD(T13, ADD(T16, T19)));
          T2Y = FNMS(KP692021471, T2X, T19);
          T2Z = FNMS(KP900968867, T2Y, T10);
          T3Y = FMA(KP554958132, T3X, T3W);
          T3Z = FMA(KP801937735, T3Y, T3V);
        }
      }
      {
        E T1r, T1u, T1A, T1x, T1B, T2h, T4a, T49, T48, T34;
        T1r = R1[WS(rs, 24)];
        {
          E T1s, T1t, T1y, T1z, T1v, T1w;
          T1s = R1[0];
          T1t = R1[WS(rs, 20)];
          T1u = ADD(T1s, T1t);
          T1y = R1[WS(rs, 8)];
          T1z = R1[WS(rs, 12)];
          T1A = ADD(T1y, T1z);
          T1v = R1[WS(rs, 4)];
          T1w = R1[WS(rs, 16)];
          T1x = ADD(T1v, T1w);
          T1B = FNMS(KP356895867, T1A, T1x);
          T2h = FNMS(KP356895867, T1u, T1A);
          T4a = SUB(T1z, T1y);
          T49 = SUB(T1t, T1s);
          T48 = SUB(T1w, T1v);
          T34 = FNMS(KP356895867, T1x, T1u);
        }
        {
          E T1C, T2i, T5d, T4H, T35, T4b;
          T1C = FNMS(KP692021471, T1B, T1u);
          T1D = FNMS(KP900968867, T1C, T1r);
          T2i = FNMS(KP692021471, T2h, T1x);
          T2j = FNMS(KP900968867, T2i, T1r);
          T5d = FNMS(KP554958132, T49, T48);
          T5e = FNMS(KP801937735, T5d, T4a);
          T4H = FMA(KP554958132, T48, T4a);
          T4I = FNMS(KP801937735, T4H, T49);
          T3H = ADD(T1r, ADD(T1u, ADD(T1x, T1A)));
          T35 = FNMS(KP692021471, T34, T1A);
          T36 = FNMS(KP900968867, T35, T1r);
          T4b = FMA(KP554958132, T4a, T49);
          T4c = FMA(KP801937735, T4b, T48);
        }
      }
      {
        E T1E, T1H, T1N, T1K, T1O, T2k, T4f, T4e, T4d, T37;
        T1E = R1[WS(rs, 10)];
        {
          E T1F, T1G, T1L, T1M, T1I, T1J;
          T1F = R1[WS(rs, 14)];
          T1G = R1[WS(rs, 6)];
          T1H = ADD(T1F, T1G);
          T1L = R1[WS(rs, 22)];
          T1M = R1[WS(rs, 26)];
          T1N = ADD(T1L, T1M);
          T1I = R1[WS(rs, 18)];
          T1J = R1[WS(rs, 2)];
          T1K = ADD(T1I, T1J);
          T1O = FNMS(KP356895867, T1N, T1K);
          T2k = FNMS(KP356895867, T1H, T1N);
          T4f = SUB(T1M, T1L);
          T4e = SUB(T1G, T1F);
          T4d = SUB(T1J, T1I);
          T37 = FNMS(KP356895867, T1K, T1H);
        }
        {
          E T1P, T2l, T5f, T4J, T38, T4g;
          T1P = FNMS(KP692021471, T1O, T1H);
          T1Q = FNMS(KP900968867, T1P, T1E);
          T2l = FNMS(KP692021471, T2k, T1K);
          T2m = FNMS(KP900968867, T2l, T1E);
          T5f = FNMS(KP554958132, T4e, T4d);
          T5g = FNMS(KP801937735, T5f, T4f);
          T4J = FMA(KP554958132, T4d, T4f);
          T4K = FNMS(KP801937735, T4J, T4e);
          T3I = ADD(T1E, ADD(T1H, ADD(T1K, T1N)));
          T38 = FNMS(KP692021471, T37, T1N);
          T39 = FNMS(KP900968867, T38, T1E);
          T4g = FMA(KP554958132, T4f, T4e);
          T4h = FMA(KP801937735, T4g, T4d);
        }
      }
      {
        E T1d, T1g, T1m, T1j, T1n, T2r, T42, T41, T40, T30;
        T1d = R1[WS(rs, 17)];
        {
          E T1e, T1f, T1k, T1l, T1h, T1i;
          T1e = R1[WS(rs, 21)];
          T1f = R1[WS(rs, 13)];
          T1g = ADD(T1e, T1f);
          T1k = R1[WS(rs, 1)];
          T1l = R1[WS(rs, 5)];
          T1m = ADD(T1k, T1l);
          T1h = R1[WS(rs, 25)];
          T1i = R1[WS(rs, 9)];
          T1j = ADD(T1h, T1i);
          T1n = FNMS(KP356895867, T1m, T1j);
          T2r = FNMS(KP356895867, T1g, T1m);
          T42 = SUB(T1l, T1k);
          T41 = SUB(T1f, T1e);
          T40 = SUB(T1i, T1h);
          T30 = FNMS(KP356895867, T1j, T1g);
        }
        {
          E T1o, T2s, T5m, T4E, T31, T43;
          T1o = FNMS(KP692021471, T1n, T1g);
          T1p = FNMS(KP900968867, T1o, T1d);
          T2s = FNMS(KP692021471, T2r, T1j);
          T2t = FNMS(KP900968867, T2s, T1d);
          T5m = FNMS(KP554958132, T41, T40);
          T5n = FNMS(KP801937735, T5m, T42);
          T4E = FMA(KP554958132, T40, T42);
          T4F = FNMS(KP801937735, T4E, T41);
          T3F = ADD(T1d, ADD(T1g, ADD(T1j, T1m)));
          T31 = FNMS(KP692021471, T30, T1m);
          T32 = FNMS(KP900968867, T31, T1d);
          T43 = FMA(KP554958132, T42, T41);
          T44 = FMA(KP801937735, T43, T40);
        }
      }
      {
        E Tn, TK, T3P, T3N, T3O, T3Q;
        Tn = ADD(Tb, Tm);
        TK = ADD(Ty, TJ);
        T3P = ADD(Tn, TK);
        T3N = ADD(T3E, T3F);
        T3O = ADD(T3H, T3I);
        T3Q = ADD(T3N, T3O);
        Cr[WS(csr, 14)] = SUB(Tn, TK);
        Ci[WS(csi, 14)] = SUB(T3N, T3O);
        Cr[WS(csr, 28)] = SUB(T3P, T3Q);
        Cr[0] = ADD(T3P, T3Q);
      }
      {
        E T3D, T3L, T3K, T3M, T3G, T3J;
        T3D = SUB(Ty, TJ);
        T3L = SUB(Tb, Tm);
        T3G = SUB(T3E, T3F);
        T3J = SUB(T3H, T3I);
        T3K = SUB(T3G, T3J);
        T3M = ADD(T3G, T3J);
        Ci[WS(csi, 7)] = FMA(KP707106781, T3K, T3D);
        Cr[WS(csr, 7)] = FMA(KP707106781, T3M, T3L);
        Ci[WS(csi, 21)] = NEG(FNMS(KP707106781, T3K, T3D));
        Cr[WS(csr, 21)] = FNMS(KP707106781, T3M, T3L);
      }
      {
        E T5a, T5s, TR, T5r, T5E, T5F, TY, T59, T2g, T5G, T5p, T5w,
         T1S, T1T, T5i;
        E T5x, T5j, T5o, T5B;
        T5a = SUB(T1Y, T23);
        T5s = SUB(T29, T2e);
        {
          E TN, TQ, T5C, T5D;
          TN = FNMS(KP900968867, TM, T1);
          TQ = FNMS(KP900968867, TP, Tc);
          TR = ADD(TN, TQ);
          T5r = SUB(TN, TQ);
          T5C = ADD(T5l, T5n);
          T5D = ADD(T5e, T5g);
          T5E = SUB(T5C, T5D);
          T5F = ADD(T5C, T5D);
        }
        {
          E TU, TX, T24, T2f;
          TU = FNMS(KP900968867, TT, To);
          TX = FNMS(KP900968867, TW, Tz);
          TY = ADD(TU, TX);
          T59 = SUB(TU, TX);
          T24 = ADD(T1Y, T23);
          T2f = ADD(T29, T2e);
          T2g = SUB(T24, T2f);
          T5G = ADD(T24, T2f);
        }
        T5j = SUB(T1c, T1p);
        T5o = SUB(T5l, T5n);
        T5p = FMA(KP974927912, T5o, T5j);
        T5w = FNMS(KP974927912, T5o, T5j);
        {
          E T1q, T1R, T5c, T5h;
          T1q = ADD(T1c, T1p);
          T1R = ADD(T1D, T1Q);
          T1S = ADD(T1q, T1R);
          T1T = SUB(T1R, T1q);
          T5c = SUB(T1D, T1Q);
          T5h = SUB(T5e, T5g);
          T5i = FNMS(KP974927912, T5h, T5c);
          T5x = FMA(KP974927912, T5h, T5c);
        }
        Ci[WS(csi, 10)] = FMA(KP974927912, T2g, T1T);
        Ci[WS(csi, 18)] = FNMS(KP974927912, T2g, T1T);
        Ci[WS(csi, 24)] = MUL(KP974927912, ADD(T5G, T5F));
        Ci[WS(csi, 4)] = MUL(KP974927912, SUB(T5F, T5G));
        T5B = SUB(TR, TY);
        Cr[WS(csr, 18)] = FNMS(KP974927912, T5E, T5B);
        Cr[WS(csr, 10)] = FMA(KP974927912, T5E, T5B);
        {
          E TZ, T5z, T5A, T5t, T5u;
          TZ = ADD(TR, TY);
          Cr[WS(csr, 4)] = SUB(TZ, T1S);
          Cr[WS(csr, 24)] = ADD(TZ, T1S);
          T5z = FNMS(KP974927912, T5a, T59);
          T5A = SUB(T5x, T5w);
          Ci[WS(csi, 11)] = FMA(KP707106781, T5A, T5z);
          Ci[WS(csi, 17)] = NEG(FNMS(KP707106781, T5A, T5z));
          T5t = FMA(KP974927912, T5s, T5r);
          T5u = ADD(T5p, T5i);
          Cr[WS(csr, 11)] = FNMS(KP707106781, T5u, T5t);
          Cr[WS(csr, 17)] = FMA(KP707106781, T5u, T5t);
          {
            E T5b, T5q, T5v, T5y;
            T5b = FMA(KP974927912, T5a, T59);
            T5q = SUB(T5i, T5p);
            Ci[WS(csi, 3)] = FMA(KP707106781, T5q, T5b);
            Ci[WS(csi, 25)] = NEG(FNMS(KP707106781, T5q, T5b));
            T5v = FNMS(KP974927912, T5s, T5r);
            T5y = ADD(T5w, T5x);
            Cr[WS(csr, 3)] = FNMS(KP707106781, T5y, T5v);
            Cr[WS(csr, 25)] = FMA(KP707106781, T5y, T5v);
          }
        }
      }
      {
        E T4Q, T50, T2v, T2W, T4X, T55, T2G, T4O, T2U, T4Z, T4U, T54,
         T2N, T4P, T4M;
        E T4N, T4S, T4T, T4B;
        T4Q = SUB(T2E, T2C);
        T50 = SUB(T2z, T2x);
        {
          E T2n, T2u, T4V, T4W;
          T2n = ADD(T2j, T2m);
          T2u = ADD(T2q, T2t);
          T2v = SUB(T2n, T2u);
          T2W = ADD(T2u, T2n);
          T4V = SUB(T2j, T2m);
          T4W = SUB(T4K, T4I);
          T4X = FNMS(KP974927912, T4W, T4V);
          T55 = FMA(KP974927912, T4W, T4V);
        }
        {
          E T2A, T2F, T2Q, T2T;
          T2A = ADD(T2x, T2z);
          T2F = ADD(T2C, T2E);
          T2G = SUB(T2A, T2F);
          T4O = ADD(T2A, T2F);
          T2Q = FNMS(KP900968867, T2P, To);
          T2T = FNMS(KP900968867, T2S, Tz);
          T2U = ADD(T2Q, T2T);
          T4Z = SUB(T2Q, T2T);
        }
        T4S = SUB(T2q, T2t);
        T4T = SUB(T4F, T4D);
        T4U = FMA(KP974927912, T4T, T4S);
        T54 = FNMS(KP974927912, T4T, T4S);
        {
          E T2J, T2M, T4G, T4L;
          T2J = FNMS(KP900968867, T2I, T1);
          T2M = FNMS(KP900968867, T2L, Tc);
          T2N = ADD(T2J, T2M);
          T4P = SUB(T2J, T2M);
          T4G = ADD(T4D, T4F);
          T4L = ADD(T4I, T4K);
          T4M = SUB(T4G, T4L);
          T4N = ADD(T4G, T4L);
        }
        Ci[WS(csi, 2)] = FMA(KP974927912, T2G, T2v);
        Ci[WS(csi, 26)] = FNMS(KP974927912, T2G, T2v);
        Ci[WS(csi, 16)] = MUL(KP974927912, ADD(T4O, T4N));
        Ci[WS(csi, 12)] = MUL(KP974927912, SUB(T4N, T4O));
        T4B = SUB(T2N, T2U);
        Cr[WS(csr, 26)] = FNMS(KP974927912, T4M, T4B);
        Cr[WS(csr, 2)] = FMA(KP974927912, T4M, T4B);
        {
          E T2V, T57, T58, T51, T52;
          T2V = ADD(T2N, T2U);
          Cr[WS(csr, 12)] = SUB(T2V, T2W);
          Cr[WS(csr, 16)] = ADD(T2V, T2W);
          T57 = FNMS(KP974927912, T4Q, T4P);
          T58 = ADD(T54, T55);
          Cr[WS(csr, 19)] = FNMS(KP707106781, T58, T57);
          Cr[WS(csr, 9)] = FMA(KP707106781, T58, T57);
          T51 = FMA(KP974927912, T50, T4Z);
          T52 = SUB(T4X, T4U);
          Ci[WS(csi, 9)] = NEG(FNMS(KP707106781, T52, T51));
          Ci[WS(csi, 19)] = FMA(KP707106781, T52, T51);
          {
            E T4R, T4Y, T53, T56;
            T4R = FMA(KP974927912, T4Q, T4P);
            T4Y = ADD(T4U, T4X);
            Cr[WS(csr, 5)] = FNMS(KP707106781, T4Y, T4R);
            Cr[WS(csr, 23)] = FMA(KP707106781, T4Y, T4R);
            T53 = FNMS(KP974927912, T50, T4Z);
            T56 = SUB(T54, T55);
            Ci[WS(csi, 5)] = NEG(FNMS(KP707106781, T56, T53));
            Ci[WS(csi, 23)] = FMA(KP707106781, T56, T53);
          }
        }
      }
      {
        E T3S, T4m, T3b, T3C, T4y, T4A, T3m, T4z, T3A, T3R, T4j, T4q,
         T3t, T4l, T46;
        E T4r, T47, T4i, T4v;
        T3S = SUB(T3i, T3k);
        T4m = SUB(T3d, T3f);
        {
          E T33, T3a, T4w, T4x;
          T33 = ADD(T2Z, T32);
          T3a = ADD(T36, T39);
          T3b = SUB(T33, T3a);
          T3C = ADD(T33, T3a);
          T4w = ADD(T3Z, T44);
          T4x = ADD(T4c, T4h);
          T4y = SUB(T4w, T4x);
          T4A = ADD(T4w, T4x);
        }
        {
          E T3g, T3l, T3w, T3z;
          T3g = ADD(T3d, T3f);
          T3l = ADD(T3i, T3k);
          T3m = SUB(T3g, T3l);
          T4z = ADD(T3l, T3g);
          T3w = FNMS(KP900968867, T3v, To);
          T3z = FNMS(KP900968867, T3y, Tz);
          T3A = ADD(T3w, T3z);
          T3R = SUB(T3w, T3z);
        }
        T47 = SUB(T36, T39);
        T4i = SUB(T4c, T4h);
        T4j = FNMS(KP974927912, T4i, T47);
        T4q = FMA(KP974927912, T4i, T47);
        {
          E T3p, T3s, T3U, T45;
          T3p = FNMS(KP900968867, T3o, T1);
          T3s = FNMS(KP900968867, T3r, Tc);
          T3t = ADD(T3p, T3s);
          T4l = SUB(T3p, T3s);
          T3U = SUB(T2Z, T32);
          T45 = SUB(T3Z, T44);
          T46 = FMA(KP974927912, T45, T3U);
          T4r = FNMS(KP974927912, T45, T3U);
        }
        Ci[WS(csi, 6)] = FMA(KP974927912, T3m, T3b);
        Ci[WS(csi, 22)] = FNMS(KP974927912, T3m, T3b);
        Ci[WS(csi, 20)] = MUL(KP974927912, SUB(T4A, T4z));
        Ci[WS(csi, 8)] = MUL(KP974927912, ADD(T4z, T4A));
        T4v = SUB(T3t, T3A);
        Cr[WS(csr, 22)] = FNMS(KP974927912, T4y, T4v);
        Cr[WS(csr, 6)] = FMA(KP974927912, T4y, T4v);
        {
          E T3B, T4p, T4s, T4n, T4o;
          T3B = ADD(T3t, T3A);
          Cr[WS(csr, 20)] = SUB(T3B, T3C);
          Cr[WS(csr, 8)] = ADD(T3B, T3C);
          T4p = FNMS(KP974927912, T3S, T3R);
          T4s = SUB(T4q, T4r);
          Ci[WS(csi, 1)] = NEG(FNMS(KP707106781, T4s, T4p));
          Ci[WS(csi, 27)] = FMA(KP707106781, T4s, T4p);
          T4n = FMA(KP974927912, T4m, T4l);
          T4o = ADD(T46, T4j);
          Cr[WS(csr, 27)] = FNMS(KP707106781, T4o, T4n);
          Cr[WS(csr, 1)] = FMA(KP707106781, T4o, T4n);
          {
            E T3T, T4k, T4t, T4u;
            T3T = FMA(KP974927912, T3S, T3R);
            T4k = SUB(T46, T4j);
            Ci[WS(csi, 13)] = NEG(FNMS(KP707106781, T4k, T3T));
            Ci[WS(csi, 15)] = FMA(KP707106781, T4k, T3T);
            T4t = FNMS(KP974927912, T4m, T4l);
            T4u = ADD(T4r, T4q);
            Cr[WS(csr, 13)] = FNMS(KP707106781, T4u, T4t);
            Cr[WS(csr, 15)] = FMA(KP707106781, T4u, T4t);
          }
        }
      }
    }
  }
}
