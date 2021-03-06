/* Generated by: /Users/sfegan/GitHub/fftw3/genfft/gen_r2cf.native -n 15 -standalone -fma -generic-arith -compact -name dft_codelet_r2cf_15 */

/*
 * This function contains 64 FP additions, 35 FP multiplications,
 * (or, 36 additions, 7 multiplications, 28 fused multiply/add),
 * 45 stack variables, 8 constants, and 30 memory accesses
 */
void dft_codelet_r2cf_15(R * R0, R * R1, R * Cr, R * Ci, stride rs, stride csr, stride csi, INT v, INT ivs, INT ovs)
{
  DK(KP910592997, +0.910592997310029334643087372129977886038870291);
  DK(KP951056516, +0.951056516295153572116439333379382143405698634);
  DK(KP823639103, +0.823639103546331925877420039278190003029660514);
  DK(KP559016994, +0.559016994374947424102293417182819058860154590);
  DK(KP250000000, +0.250000000000000000000000000000000000000000000);
  DK(KP618033988, +0.618033988749894848204586834365638117720309180);
  DK(KP866025403, +0.866025403784438646763723170752936183471402627);
  DK(KP500000000, +0.500000000000000000000000000000000000000000000);
  {
    INT i;
    for (i = v; i > 0; i = i - 1, R0 = R0 + ivs, R1 = R1 + ivs, Cr = Cr + ovs, Ci = Ci + ovs, MAKE_VOLATILE_STRIDE(60, rs), MAKE_VOLATILE_STRIDE(60, csr), MAKE_VOLATILE_STRIDE(60, csi)) {
      E Ti, TR, TF, TM, TN, T7, Te, Tf, TV, TW, TX, Ts, Tv, TH, Tl;
      E To, TG, TS, TT, TU;
      {
        E TD, Tg, Th, TE;
        TD = R0[0];
        Tg = R0[WS(rs, 5)];
        Th = R1[WS(rs, 2)];
        TE = ADD(Th, Tg);
        Ti = SUB(Tg, Th);
        TR = ADD(TD, TE);
        TF = FNMS(KP500000000, TE, TD);
      }
      {
        E Tj, Tq, Tt, Tm, T3, Tk, Ta, Tr, Td, Tu, T6, Tn;
        Tj = R1[WS(rs, 1)];
        Tq = R0[WS(rs, 3)];
        Tt = R1[WS(rs, 4)];
        Tm = R0[WS(rs, 6)];
        {
          E T1, T2, T8, T9;
          T1 = R0[WS(rs, 4)];
          T2 = R1[WS(rs, 6)];
          T3 = SUB(T1, T2);
          Tk = ADD(T1, T2);
          T8 = R1[WS(rs, 5)];
          T9 = R1[0];
          Ta = SUB(T8, T9);
          Tr = ADD(T8, T9);
        }
        {
          E Tb, Tc, T4, T5;
          Tb = R0[WS(rs, 7)];
          Tc = R0[WS(rs, 2)];
          Td = SUB(Tb, Tc);
          Tu = ADD(Tb, Tc);
          T4 = R0[WS(rs, 1)];
          T5 = R1[WS(rs, 3)];
          T6 = SUB(T4, T5);
          Tn = ADD(T4, T5);
        }
        TM = SUB(T6, T3);
        TN = SUB(Td, Ta);
        T7 = ADD(T3, T6);
        Te = ADD(Ta, Td);
        Tf = ADD(T7, Te);
        TV = ADD(Tq, Tr);
        TW = ADD(Tt, Tu);
        TX = ADD(TV, TW);
        Ts = FNMS(KP500000000, Tr, Tq);
        Tv = FNMS(KP500000000, Tu, Tt);
        TH = ADD(Ts, Tv);
        Tl = FNMS(KP500000000, Tk, Tj);
        To = FNMS(KP500000000, Tn, Tm);
        TG = ADD(Tl, To);
        TS = ADD(Tj, Tk);
        TT = ADD(Tm, Tn);
        TU = ADD(TS, TT);
      }
      Ci[WS(csi, 5)] = MUL(KP866025403, SUB(Tf, Ti));
      {
        E TK, TQ, TO, TI, TJ, TP, TL;
        TK = SUB(TG, TH);
        TQ = FNMS(KP618033988, TM, TN);
        TO = FMA(KP618033988, TN, TM);
        TI = ADD(TG, TH);
        TJ = FNMS(KP250000000, TI, TF);
        Cr[WS(csr, 5)] = ADD(TF, TI);
        TP = FNMS(KP559016994, TK, TJ);
        Cr[WS(csr, 2)] = FMA(KP823639103, TQ, TP);
        Cr[WS(csr, 7)] = FNMS(KP823639103, TQ, TP);
        TL = FMA(KP559016994, TK, TJ);
        Cr[WS(csr, 1)] = FMA(KP823639103, TO, TL);
        Cr[WS(csr, 4)] = FNMS(KP823639103, TO, TL);
      }
      {
        E T11, T12, T10, TY, TZ;
        T11 = SUB(TW, TV);
        T12 = SUB(TS, TT);
        Ci[WS(csi, 3)] = MUL(KP951056516, FMA(KP618033988, T12, T11));
        Ci[WS(csi, 6)] = NEG(MUL(KP951056516, FNMS(KP618033988, T11, T12)));
        T10 = SUB(TU, TX);
        TY = ADD(TU, TX);
        TZ = FNMS(KP250000000, TY, TR);
        Cr[WS(csr, 3)] = FNMS(KP559016994, T10, TZ);
        Cr[0] = ADD(TR, TY);
        Cr[WS(csr, 6)] = FMA(KP559016994, T10, TZ);
        {
          E Tx, TB, TA, TC;
          {
            E Tp, Tw, Ty, Tz;
            Tp = SUB(Tl, To);
            Tw = SUB(Ts, Tv);
            Tx = FMA(KP618033988, Tw, Tp);
            TB = FNMS(KP618033988, Tp, Tw);
            Ty = FMA(KP250000000, Tf, Ti);
            Tz = SUB(Te, T7);
            TA = FMA(KP559016994, Tz, Ty);
            TC = FNMS(KP559016994, Tz, Ty);
          }
          Ci[WS(csi, 1)] = NEG(MUL(KP951056516, FNMS(KP910592997, TA, Tx)));
          Ci[WS(csi, 7)] = MUL(KP951056516, FMA(KP910592997, TC, TB));
          Ci[WS(csi, 4)] = MUL(KP951056516, FMA(KP910592997, TA, Tx));
          Ci[WS(csi, 2)] = MUL(KP951056516, FNMS(KP910592997, TC, TB));
        }
      }
    }
  }
}
