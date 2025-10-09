# rpeak_jeppesen.py
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from scipy.signal import argrelextrema

@dataclass
class RPeakParams:
    fs: float                 # Hz (samplingsfrekvens EFTER preprocess/resample)
    win_s: float = 2.0        # Thigh/Tlow beregnes pr. 2 sek
    refractory_s: float = 0.25  # min. RR (250 ms)
    # Thigh-opsætning (Jeppesen: adaptiv pr. vindue)
    thigh_scale: float = 0.75
    thigh_peaks: int = 8      # brug top-N lokalmaks i vinduet
    # Tlow (sekundær tærskel; clampes til 0.4*Thigh)
    tlow_scale_posmean: float = 2.0  # 2×mean(positive) i forrige 2 vinduer
    tlow_max_frac: float = 0.40
    # Variabilitet (styrer s2)
    theta_samples_cut: float = 35.0  # ≈ 68 ms @512 Hz; skalerer auto ift. din fs
    s2_high: int = 12
    s2_low: int = 10
    # Searchpoint
    search_30samples_ref_512: int = 30   # “30 samples” regel (skaleres til din fs)
    rshort_n: int = 8
    rrlong_n: int = 34
    rmax_cap_s: float = 1.2              # max Rmax ved lav puls

class JeppesenRPeak:
    def __init__(self, p: RPeakParams):
        self.p = p

    # ---------- hjælpefunktioner ----------
    def _window_edges(self, N: int) -> List[Tuple[int, int]]:
        L = int(round(self.p.win_s * self.p.fs))
        return [(i, min(N, i + L)) for i in range(0, N, L)]

    def _local_maxima(self, x: np.ndarray) -> np.ndarray:
        idx = argrelextrema(x, np.greater, order=1)[0]
        return idx

    def _thigh_series(self, x: np.ndarray) -> np.ndarray:
        """Thigh pr. vindue: 0.75 * median(top-8 lokalmaks) i vinduet.
           (Matcher idéen om adaptiv tærskel pr. 2 s vindue.)"""
        edges = self._window_edges(x.size)
        thigh = np.zeros_like(x, dtype=float)
        for st, en in edges:
            seg = x[st:en]
            lm = self._local_maxima(seg)
            vals = seg[lm] if lm.size else np.asarray([np.max(seg)])  # fallback
            if vals.size > 0:
                vals = np.sort(vals)[-self.p.thigh_peaks:]
                T = self.p.thigh_scale * float(np.median(vals))
            else:
                T = 0.0
            thigh[st:en] = T
        return thigh

    def _variability_mode(self, rr_s: List[float]) -> str:
        """'high' hvis gennemsnitlig afvigelse fra median (efter at fjerne 2 største)
           i de seneste 34 RR overstiger theta_cut (skaleret til din fs)."""
        n = len(rr_s)
        if n < self.p.rrlong_n: 
            return "low"
        rr = np.asarray(rr_s[-self.p.rrlong_n:], float)
        med = np.median(rr)
        dev = np.abs(rr - med)
        if dev.size >= 2:
            mask = np.ones_like(dev, bool)
            mask[np.argsort(dev)[-2:]] = False
            theta = float(np.mean(dev[mask])) if mask.any() else float(np.mean(dev))
        else:
            theta = float(np.mean(dev))
        theta_samples = theta * self.p.fs
        return "high" if theta_samples > self.p.theta_samples_cut * (self.p.fs/512.0) else "low"

    def _tlow_series(self, x: np.ndarray, thigh: np.ndarray, peaks_idx: List[int]) -> np.ndarray:
        edges = self._window_edges(x.size)
        tlow = np.zeros_like(x, float)
        peaks = np.asarray(peaks_idx, int)
        for m, (st, en) in enumerate(edges):
            st2 = edges[max(0, m-2)][0]     # to forrige vinduer → [st2 : st)
            seg = x[st2:st]
            pos = seg[seg > 0]
            mu_pos = float(pos.mean()) if pos.size else 0.0
            if peaks.size:
                s1 = int(((peaks >= st2) & (peaks < st)).sum())
            else:
                s1 = 0
            mode = self._variability_mode(np.diff(peaks)/self.p.fs if peaks.size >= 2 else [])
            s2 = self.p.s2_high if mode == "high" else self.p.s2_low
            T = self.p.tlow_scale_posmean * mu_pos * (s1 / max(1, s2))
            T = min(T, self.p.tlow_max_frac * float(np.mean(thigh[st:en]) if en>st else 0.0))
            tlow[st:en] = T
        return tlow

    def _rmax_seconds(self, rr_s: List[float], mode: str) -> float:
        if not rr_s:
            return 0.8
        rr = np.asarray(rr_s, float)
        rr_long = np.median(rr[-self.p.rrlong_n:]) if rr.size else 0.8
        if mode == "high":
            rr_short = np.median(rr[-self.p.rshort_n:]) if rr.size else rr_long
            rr_temp = min(rr_short, rr_short)  # proxy for “searchback”
        else:
            rr_temp = rr_long
        return float(min(1.2 * rr_temp, self.p.rmax_cap_s))

    def _searchpoint(self, last_idx: int, rr_s: List[float]) -> int:
        fs = self.p.fs
        if not rr_s:
            return last_idx + int(0.8 * fs)
        rr_short = np.median(rr_s[-self.p.rshort_n:])
        rr_samp = int(round(rr_short * fs))
        thr350 = int(round(350 * fs / 512.0))
        thr467 = int(round(467 * fs / 512.0))
        if rr_samp < thr350:
            return last_idx + rr_samp
        elif rr_samp < thr467:
            return last_idx + thr350
        else:
            return last_idx + int(round(0.75 * rr_samp))

    def _localise_max(self, x: np.ndarray, i0: int, radius: int) -> int:
        st = max(0, i0 - radius); en = min(x.size, i0 + radius + 1)
        return st + int(np.argmax(x[st:en]))

    # ---------- hoveddetektor ----------
    def detect(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        fs = self.p.fs
        N = x.size
        peaks: List[int] = []
        rr_s: List[float] = []

        # Primære/sekundære tærskler
        thigh = self._thigh_series(x)
        tlow  = self._tlow_series(x, thigh, peaks)

        min_rr = int(round(self.p.refractory_s * fs))
        rad = int(round(0.05 * fs))  # ±50 ms lokaliseringsradius
        thirty = int(round(self.p.search_30samples_ref_512 * fs / 512.0))

        i = 0
        while i < N:
            # refraktær: spring frem
            if peaks and (i - peaks[-1]) < min_rr:
                i = peaks[-1] + min_rr
                continue

            mode = self._variability_mode(rr_s)
            rmax_samp = int(round(self._rmax_seconds(rr_s, mode) * fs))
            overdue = (not peaks) or ((i - peaks[-1]) >= rmax_samp)

            if not overdue and peaks:
                sp = self._searchpoint(peaks[-1], rr_s)
                left_i = max(0, sp - thirty)
                right_i = min(N-1, sp + thirty)

                # find første tærskelkryds (Thigh) til venstre/højre
                cand = []
                # venstre
                lv = np.where(x[left_i:sp] >= thigh[left_i:sp])[0]
                if lv.size:
                    cand.append(left_i + lv[-1])
                # højre
                rv = np.where(x[sp:right_i+1] >= thigh[sp:right_i+1])[0]
                if rv.size:
                    cand.append(sp + rv[0])

                if cand:
                    # hvis begge inden for 30 samples → vælg højeste lokalmaks
                    if len(cand) == 2 and abs(cand[1] - cand[0]) <= thirty:
                        c0 = self._localise_max(x, cand[0], rad)
                        c1 = self._localise_max(x, cand[1], rad)
                        ci = c0 if x[c0] >= x[c1] else c1
                    else:
                        # ellers lokaliser hver og vælg størst amplitude
                        best = None; best_amp = -1
                        for c in cand:
                            cc = self._localise_max(x, c, rad)
                            amp = x[cc]
                            if amp > best_amp:
                                best_amp = amp; best = cc
                        ci = best
                    # tilføj peak
                    if (not peaks) or (ci - peaks[-1] >= min_rr):
                        peaks.append(ci)
                        if len(peaks) >= 2:
                            rr_s.append((peaks[-1] - peaks[-2]) / fs)
                        # opdatér Tlow med nye peaks
                        tlow = self._tlow_series(x, thigh, peaks)
                        i = ci + min_rr
                        continue

            # “searchback”/Tlow hvis vi er forbi Rmax eller ikke fandt Thigh-kandidat
            if (peaks and (i - peaks[-1]) >= rmax_samp) or not peaks:
                # søg efter første kryds over Tlow fremad
                idx = np.where(x[i:] >= tlow[i:])[0]
                if idx.size:
                    j = i + idx[0]
                    ci = self._localise_max(x, j, rad)
                    if (not peaks) or (ci - peaks[-1] >= min_rr):
                        peaks.append(ci)
                        if len(peaks) >= 2:
                            rr_s.append((peaks[-1] - peaks[-2]) / fs)
                        tlow = self._tlow_series(x, thigh, peaks)
                        i = ci + min_rr
                        continue

                # failsafe: 2 s efter sidste peak → vælg største lokalmaks i interval
                if peaks and (i - peaks[-1]) >= int(round(2.0 * fs)):
                    last = peaks[-1]
                    j0 = last + min(int(round(np.median(rr_s[-self.p.rshort_n:])*fs)) if rr_s else int(0.8*fs),
                                    int(self.p.rmax_cap_s*fs))
                    j1 = last + int(round((self.p.refractory_s + 2.0) * fs))
                    j0 = max(0, j0); j1 = min(N, j1)
                    if j1 > j0:
                        ci = j0 + int(np.argmax(x[j0:j1]))
                        if (not peaks) or (ci - peaks[-1] >= min_rr):
                            peaks.append(ci)
                            if len(peaks) >= 2:
                                rr_s.append((peaks[-1]-peaks[-2])/fs)
                            tlow = self._tlow_series(x, thigh, peaks)
                            i = ci + min_rr
                            continue

            i += 1

        return {
            "peaks_idx": np.asarray(peaks, int),
            "rr_s": np.asarray(rr_s, float),
            "Thigh": thigh,
            "Tlow": tlow,
        }
