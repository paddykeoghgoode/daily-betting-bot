#!/usr/bin/env python3
"""
Football Betting Model v3.0 - Multi-League Multi-Season
========================================================
- Loads 3 seasons of data (2324, 2425, 2526) for better calibration
- Processes ALL leagues independently (separate ratings per league)
- Handles promotion/relegation by country groupings
- INCLUDES: Progress Bar & Optimized Backtesting Speed

NEW (local fixtures + HTML report)
---------------------------------
- Optional local caching of fixtures.xlsx (use --refresh to force re-download)
- Optional HTML report output (use --html)
- HTML groups by Country -> League, and includes filtering controls
- Value bet cards include: market odds, EV, fair odds, model xG (lambdas), and best books

League Codes:
  E0 = England Premier League, E1 = Championship, E2 = League 1, E3 = League 2
  D1 = Germany Bundesliga, D2 = Bundesliga 2
  SP1 = Spain La Liga, SP2 = La Liga 2
  I1 = Italy Serie A, I2 = Serie B
  F1 = France Ligue 1, F2 = Ligue 2
  N1 = Netherlands Eredivisie
  B1 = Belgium Pro League
  P1 = Portugal Primeira Liga
  T1 = Turkey Super Lig
  G1 = Greece Super League
  SC0 = Scotland Premiership, SC1 = Championship

Just run this entire script in Google Colab!
"""

import os
import argparse
from datetime import datetime
import pickle
import re
import urllib.parse

import pandas as pd
import numpy as np
import math
import requests
from scipy.stats import poisson, ttest_1samp
from scipy.optimize import minimize
import scipy.special
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Seasons to load (most recent last)
SEASONS = ['2122', '2223', '2324', '2425', '2526']
SEASON_DECAY = 0.75  # season-level decay (older seasons down-weighted)

# All available leagues grouped by country
LEAGUES = {
    'England': ['E0', 'E1', 'E2', 'E3'],
    'Germany': ['D1', 'D2'],
    'Spain': ['SP1', 'SP2'],
    'Italy': ['I1', 'I2'],
    'France': ['F1', 'F2'],
    'Netherlands': ['N1'],
    'Belgium': ['B1'],
    'Portugal': ['P1'],
    'Turkey': ['T1'],
    'Greece': ['G1'],
    'Scotland': ['SC0', 'SC1'],
}

# Flatten to list of all leagues
ALL_LEAGUES = [lg for country_leagues in LEAGUES.values() for lg in country_leagues]

# Model parameters
DECAY_HALF_LIFE = 8
VALUE_THRESHOLD = 0.05
RHO_BOUNDS = (-0.3, 0.3)
MIN_TRAIN_MATCHES = 50
MIN_FAIR_ODDS = 1.01
MAX_FAIR_ODDS = 50.0

# --- Blending + stability knobs (best-of-both worlds) ---
DC_MIN_WEIGHT = 0.01          # time-decay floor for DC MLE (lower = old matches matter less)
DC_BLEND_BASE = 0.70          # default weight on DC lambdas (structural baseline)
DC_BLEND_MIN  = 0.40          # minimum DC weight (when DC disagrees with macro-xG)
DC_BLEND_MAX  = 0.85          # maximum DC weight
DC_BLEND_SENS = 0.15          # how fast to down-weight DC on disagreement
BAYES_SHRINK_K = 5            # pseudo-match count shrinking team stats toward league averages
DC_MLE_HALFLIFE_DAYS = 180   # time-decay: recent matches weighted more

# League-specific tuning (based on backtests)
LEAGUE_BAYES_SHRINK_K = {
    'E0': 4, 'D1': 4, 'SP1': 4, 'I1': 4, 'F1': 4,
    'E1': 5, 'D2': 5, 'SP2': 5, 'I2': 5, 'F2': 5,
    'E2': 6, 'E3': 6, 'SC1': 6,
    'SC0': 5, 'N1': 5, 'P1': 5, 'B1': 6, 'T1': 6, 'G1': 6,
}
LEAGUE_DC_HALFLIFE_DAYS = {
    'E0': 210, 'D1': 210, 'SP1': 210, 'I1': 210, 'F1': 210,
    'E1': 200, 'D2': 200, 'SP2': 200, 'I2': 200, 'F2': 200,
    'E2': 180, 'E3': 180, 'SC1': 180,
    'SC0': 190, 'N1': 190, 'P1': 190, 'B1': 180, 'T1': 180, 'G1': 180,
}

# Per-league EV threshold overrides (fallback to VALUE_THRESHOLD)
# Keep these conservative; tune with your backtest output.
LEAGUE_EV_THRESH = {
    # Top tiers (generally sharper)
    'E0': 0.04,
    'D1': 0.04,
    'SP1': 0.04,
    'I1': 0.04,
    'F1': 0.045,
    'N1': 0.045,
    'P1': 0.045,
    # Second tiers / mid (a bit less sharp)
    'E1': 0.05,
    'D2': 0.05,
    'SP2': 0.05,
    'I2': 0.05,
    'F2': 0.055,
    'SC0': 0.05,
    'SC1': 0.055,
    # Lower divisions
    'E2': 0.055,
    'E3': 0.06,
    # Smaller leagues
    'B1': 0.05,
    'T1': 0.05,
    'G1': 0.055,
}

# Model caching
MODEL_CACHE_DIR = "output/models"
MODEL_VERSION = "dc_mle_v1"  # bump to invalidate cached models when logic changes

# Fixtures caching
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.xlsx"
DEFAULT_FIXTURES_CACHE = "output/fixtures.xlsx"

def load_cached_model(league_code, cache_dir=MODEL_CACHE_DIR):
    """Load a cached trained LeagueModel from disk. Returns None if not found."""
    path = os.path.join(cache_dir, f"{league_code}_{MODEL_VERSION}.pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"  {league_code}: failed to load cached model ({e}); retraining...")
            return None
    return None

def save_cached_model(league_code, model, cache_dir=MODEL_CACHE_DIR):
    """Save a trained LeagueModel to disk for fast re-use."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{league_code}_{MODEL_VERSION}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


# =============================================================================
# LEAGUE INFO
# =============================================================================

LEAGUE_NAMES = {
    'E0': 'England Premier League',
    'E1': 'England Championship',
    'E2': 'England League 1',
    'E3': 'England League 2',
    'D1': 'Germany Bundesliga',
    'D2': 'Germany 2. Bundesliga',
    'SP1': 'Spain La Liga',
    'SP2': 'Spain La Liga 2',
    'I1': 'Italy Serie A',
    'I2': 'Italy Serie B',
    'F1': 'France Ligue 1',
    'F2': 'France Ligue 2',
    'N1': 'Netherlands Eredivisie',
    'B1': 'Belgium Pro League',
    'P1': 'Portugal Primeira Liga',
    'T1': 'Turkey Super Lig',
    'G1': 'Greece Super League',
    'SC0': 'Scotland Premiership',
    'SC1': 'Scotland Championship',
}

def get_league_name(code):
    return LEAGUE_NAMES.get(code, code)

def get_country(league_code):
    """Get country from league code"""
    for country, leagues in LEAGUES.items():
        if league_code in leagues:
            return country
    return 'Unknown'

def get_ev_threshold(league_code, default=VALUE_THRESHOLD):
    """Get the EV threshold for a league (fallback to default)."""
    try:
        return float(LEAGUE_EV_THRESH.get(league_code, default))
    except Exception:
        return default

def get_league_bayes_k(league_code, default=BAYES_SHRINK_K):
    try:
        return float(LEAGUE_BAYES_SHRINK_K.get(league_code, default))
    except Exception:
        return default

def get_league_dc_halflife(league_code, default=DC_MLE_HALFLIFE_DAYS):
    try:
        return float(LEAGUE_DC_HALFLIFE_DAYS.get(league_code, default))
    except Exception:
        return default

# =============================================================================
# DATA LOADING
# =============================================================================

def load_league_history(league, seasons=SEASONS):
    """Load multiple seasons of data for a single league"""
    all_data = []
    season_weights = {
        season: float(SEASON_DECAY) ** (len(seasons) - 1 - idx)
        for idx, season in enumerate(seasons)
    }

    for season in seasons:
        url = f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
        try:
            df = pd.read_csv(url)
            df['Season'] = season
            df['SeasonWeight'] = float(season_weights.get(season, 1.0))
            all_data.append(df)
        except:
            pass  # Season may not exist for this league

    if not all_data:
        return None

    combined = pd.concat(all_data, ignore_index=True)
    return combined

def load_all_historical_data(leagues=ALL_LEAGUES, seasons=SEASONS):
    """Load historical data for all leagues"""
    print("Loading historical data...")
    print(f"Seasons: {seasons}")
    print(f"Leagues: {len(leagues)}")

    all_data = {}

    for league in leagues:
        df = load_league_history(league, seasons)
        if df is not None and len(df) > 0:
            all_data[league] = df
            print(f"  {league} ({get_league_name(league)}): {len(df)} matches")

    print(f"Loaded {len(all_data)} leagues")
    return all_data

def load_fixtures(cache_path=DEFAULT_FIXTURES_CACHE, refresh=False, url=FIXTURES_URL):
    """Load upcoming fixtures for all leagues (with optional local caching)."""
    print("\nLoading fixtures...")

    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        if refresh or (not os.path.exists(cache_path)):
            print(f"  Downloading fixtures to: {cache_path}")
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(cache_path, "wb") as f:
                f.write(r.content)
        else:
            print(f"  Using cached fixtures: {cache_path}")

        df = pd.read_excel(cache_path, engine='openpyxl')

        # Group by league
        fixtures_by_league = {}
        for league in df['Div'].dropna().unique():
            league_fix = df[df['Div'] == league].copy()
            if len(league_fix) > 0:
                fixtures_by_league[league] = league_fix
                print(f"  {league}: {len(league_fix)} fixtures")

        return fixtures_by_league
    except Exception as e:
        print(f"Error loading fixtures: {e}")
        return {}

# =============================================================================
# XG CALIBRATOR (Per League)
# =============================================================================

class XGCalibrator:
    def __init__(self):
        self.home_coef = {}
        self.away_coef = {}
        self.is_calibrated = False
        self.stats = {}

    def calibrate(self, df, verbose=False):
        cols = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC']
        df_clean = df.dropna(subset=cols).copy()

        if len(df_clean) < 30:
            self.is_calibrated = False
            return

        X_home = pd.DataFrame({
            'sot': df_clean['HST'],
            'soff': df_clean['HS'] - df_clean['HST'],
            'cor': df_clean['HC']
        })
        y_home = df_clean['FTHG']

        X_away = pd.DataFrame({
            'sot': df_clean['AST'],
            'soff': df_clean['AS'] - df_clean['AST'],
            'cor': df_clean['AC']
        })
        y_away = df_clean['FTAG']

        home_model = LinearRegression().fit(X_home, y_home)
        away_model = LinearRegression().fit(X_away, y_away)

        self.home_coef = {
            'sot': home_model.coef_[0],
            'soff': home_model.coef_[1],
            'cor': home_model.coef_[2],
            'int': home_model.intercept_
        }
        self.away_coef = {
            'sot': away_model.coef_[0],
            'soff': away_model.coef_[1],
            'cor': away_model.coef_[2],
            'int': away_model.intercept_
        }

        self.is_calibrated = True
        self.stats = {
            'n': len(df_clean),
            'home_r2': r2_score(y_home, home_model.predict(X_home)),
            'away_r2': r2_score(y_away, away_model.predict(X_away))
        }

        if verbose:
            print(f"  xG Calibrated: n={self.stats['n']}, R2={self.stats['home_r2']:.2f}/{self.stats['away_r2']:.2f}")

    def calc_xg(self, sot, shots, cor, venue='home'):
        if not self.is_calibrated:
            return 0.32*sot + 0.04*(shots-sot) + 0.08*cor + 0.05
        c = self.home_coef if venue == 'home' else self.away_coef
        return max(0, c['sot']*sot + c['soff']*(shots-sot) + c['cor']*cor + c['int'])

# =============================================================================
# TIME DECAY
# =============================================================================

def decay_weight(i, half_life=DECAY_HALF_LIFE):
    return 0.5 ** (i / half_life)

def weighted_avg(vals, half_life=DECAY_HALF_LIFE):
    if not vals:
        return 0
    vals = list(reversed(vals))
    weights = [decay_weight(i, half_life) for i in range(len(vals))]
    return sum(v*w for v,w in zip(vals, weights)) / sum(weights)

# =============================================================================
# DIXON-COLES
# =============================================================================

def dc_adjust(hg, ag, hl, al, rho):
    if hg == 0 and ag == 0:
        return 1 - hl * al * rho
    elif hg == 0 and ag == 1:
        return 1 + hl * rho
    elif hg == 1 and ag == 0:
        return 1 + al * rho
    elif hg == 1 and ag == 1:
        return 1 - rho
    return 1.0

def estimate_rho(df, verbose=False):
    df_c = df.dropna(subset=['FTHG', 'FTAG']).copy()
    if len(df_c) < 50:
        return -0.1  # Default

    avg_h = df_c['FTHG'].mean()
    avg_a = df_c['FTAG'].mean()

    def neg_ll(rho):
        ll = 0
        for _, r in df_c.iterrows():
            h, a = int(r['FTHG']), int(r['FTAG'])
            p = poisson.pmf(h, avg_h) * poisson.pmf(a, avg_a) * dc_adjust(h, a, avg_h, avg_a, rho[0])
            if p > 0:
                ll += np.log(p)
        return -ll

    res = minimize(neg_ll, x0=[0], bounds=[RHO_BOUNDS], method='L-BFGS-B')
    rho = res.x[0]

    if verbose:
        print(f"  Dixon-Coles rho: {rho:.4f}")
    return rho

def poisson_matrix_dc(hl, al, rho=0, max_g=10):
    m = np.zeros((max_g, max_g))
    for h in range(max_g):
        for a in range(max_g):
            m[h,a] = poisson.pmf(h, hl) * poisson.pmf(a, al) * dc_adjust(h, a, hl, al, rho)
    total = m.sum()
    return m / total if total > 0 else m

    
# =============================================================================
# ASIAN LINES / BTTS / SCORELINES (from score probability matrix)
# =============================================================================

def _safe_odds_from_prob(p, floor=1.01):
    p = float(p)
    if p <= 0:
        return 999.0
    o = 1.0 / p
    return max(float(floor), min(999.0, o))



def _asian_fair_odds(W, L, floor=1.01):
    'Fair decimal odds with push returning stake: O = 1 + L/W.'
    W = float(W)
    L = float(L)
    if W <= 1e-12:
        return 999.0
    o = 1.0 + (L / W)
    if o < floor:
        o = floor
    if o > 999.0:
        o = 999.0
    return o
def _total_probs_from_score_matrix(mat):
    # returns dict total_goals -> prob
    totals = {}
    H, A = mat.shape
    for h in range(H):
        for a in range(A):
            t = h + a
            totals[t] = totals.get(t, 0.0) + float(mat[h, a])
    return totals

def _diff_probs_from_score_matrix(mat):
    # returns dict (home-away) -> prob
    diffs = {}
    H, A = mat.shape
    for h in range(H):
        for a in range(A):
            d = h - a
            diffs[d] = diffs.get(d, 0.0) + float(mat[h, a])
    return diffs

def settle_total_over(totp, line):
    # returns (W, P, L) for OVER at Asian total line.
    line = float(line)
    W = P = L = 0.0
    for t, pr in totp.items():
        if t > line:
            W += pr
        elif abs(t - line) < 1e-12:
            P += pr
        else:
            L += pr
    return W, P, L

def settle_handicap_home(diffp, line):
    # returns (W, P, L) for HOME at Asian handicap line (home + line) vs away.
    line = float(line)
    W = P = L = 0.0
    for d, pr in diffp.items():
        adj = d + line
        if adj > 0:
            W += pr
        elif abs(adj) < 1e-12:
            P += pr
        else:
            L += pr
    return W, P, L

def _split_quarter(line):
    """For quarter lines, return the two half-lines.

    Examples:
      2.25 -> (2.0, 2.5)
      2.75 -> (2.5, 3.0)
      -0.75 -> (-1.0, -0.5)

    IMPORTANT: Do not use round() here; Python banker's rounding breaks .75 lines.
    """
    line = float(line)
    lo = math.floor(line * 2.0) / 2.0
    hi = lo + 0.5
    return lo, hi

def fair_totals_ou(totp, line):
    line = float(line)
    frac = abs(line % 1.0)

    def price_over_under(W, L):
        over = _asian_fair_odds(W, L)
        under = _asian_fair_odds(L, W)
        return over, under

    if abs(frac - 0.25) < 1e-9 or abs(frac - 0.75) < 1e-9:
        lo, hi = _split_quarter(line)
        W1, _, L1 = settle_total_over(totp, lo)
        W2, _, L2 = settle_total_over(totp, hi)
        W = 0.5 * W1 + 0.5 * W2
        L = 0.5 * L1 + 0.5 * L2
        return price_over_under(W, L)

    W, _, L = settle_total_over(totp, line)
    return price_over_under(W, L)

def fair_handicap_home_away(diffp, line):
    line = float(line)
    frac = abs(line % 1.0)

    def price_home_away(W, L):
        home = _asian_fair_odds(W, L)
        away = _asian_fair_odds(L, W)
        return home, away

    if abs(frac - 0.25) < 1e-9 or abs(frac - 0.75) < 1e-9:
        lo, hi = _split_quarter(line)
        W1, _, L1 = settle_handicap_home(diffp, lo)
        W2, _, L2 = settle_handicap_home(diffp, hi)
        W = 0.5 * W1 + 0.5 * W2
        L = 0.5 * L1 + 0.5 * L2
        return price_home_away(W, L)

    W, _, L = settle_handicap_home(diffp, line)
    return price_home_away(W, L)

def btts_probs(mat):
    # both teams score >=1
    H, A = mat.shape
    p_yes = 0.0
    for h in range(1, H):
        for a in range(1, A):
            p_yes += float(mat[h, a])
    p_no = max(0.0, 1.0 - p_yes)
    return p_yes, p_no

def top_scorelines(mat, k=5):
    H, A = mat.shape
    items = []
    for h in range(H):
        for a in range(A):
            items.append((h, a, float(mat[h, a])))
    items.sort(key=lambda x: x[2], reverse=True)
    return items[:k]

def most_likely_result_probs(mat):
    # returns (pH, pD, pA)
    H, A = mat.shape
    pH = pD = pA = 0.0
    for h in range(H):
        for a in range(A):
            pr = float(mat[h, a])
            if h > a:
                pH += pr
            elif h == a:
                pD += pr
            else:
                pA += pr
    return pH, pD, pA

def margin_buckets(mat):
    H, A = mat.shape
    p_draw = 0.0
    p_home1 = p_away1 = 0.0
    p_home2p = p_away2p = 0.0
    for h in range(H):
        for a in range(A):
            pr = float(mat[h, a])
            if h == a:
                p_draw += pr
            elif h - a == 1:
                p_home1 += pr
            elif a - h == 1:
                p_away1 += pr
            elif h - a >= 2:
                p_home2p += pr
            elif a - h >= 2:
                p_away2p += pr
    return p_home1, p_home2p, p_draw, p_away1, p_away2p

def recommend_lines(totp, diffp, tot_lines, ah_lines):
    # pick lines closest to 50/50 win/lose
    def wl_total(line):
        line=float(line)
        frac = abs(line % 1.0)
        if abs(frac-0.25) < 1e-9 or abs(frac-0.75) < 1e-9:
            lo, hi = _split_quarter(line)
            W1,_,L1 = settle_total_over(totp, lo)
            W2,_,L2 = settle_total_over(totp, hi)
            return 0.5*W1+0.5*W2, 0.5*L1+0.5*L2
        W,_,L = settle_total_over(totp, line)
        return W, L
    def wl_ah(line):
        line=float(line)
        frac = abs(line % 1.0)
        if abs(frac-0.25) < 1e-9 or abs(frac-0.75) < 1e-9:
            lo, hi = _split_quarter(line)
            W1,_,L1 = settle_handicap_home(diffp, lo)
            W2,_,L2 = settle_handicap_home(diffp, hi)
            return 0.5*W1+0.5*W2, 0.5*L1+0.5*L2
        W,_,L = settle_handicap_home(diffp, line)
        return W, L

    best_tot = None
    best_gap = 1e9
    for L in tot_lines:
        W, Ls = wl_total(L)
        gap = abs(W - Ls)
        if gap < best_gap:
            best_gap = gap
            best_tot = float(L)

    best_ah = None
    best_gap = 1e9
    for L in ah_lines:
        W, Ls = wl_ah(L)
        gap = abs(W - Ls)
        if gap < best_gap:
            best_gap = gap
            best_ah = float(L)

    return best_tot, best_ah



# =============================================================================
# DIXON-COLES MLE (ATTACK/DEFENCE) + SMOOTH ELITE CORRECTION
# =============================================================================

DC_MLE_RIDGE = 0.02          # L2 regularisation for stability
DC_MLE_MAXITER = 300

ELITE_K = 0.18               # strength of elite correction (0 disables)
ELITE_S = 0.55               # softness/scale of elite correction


def _dc_time_weights(df, half_life_days=DC_MLE_HALFLIFE_DAYS):
    if 'Date' not in df.columns:
        base = np.ones(len(df), dtype=float)
        if 'SeasonWeight' in df.columns:
            sw = pd.to_numeric(df['SeasonWeight'], errors='coerce').fillna(1.0).to_numpy()
            base *= sw
        return np.clip(base, DC_MIN_WEIGHT, 1.0)
    dts = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
    if dts.isna().all():
        base = np.ones(len(df), dtype=float)
        if 'SeasonWeight' in df.columns:
            sw = pd.to_numeric(df['SeasonWeight'], errors='coerce').fillna(1.0).to_numpy()
            base *= sw
        return np.clip(base, DC_MIN_WEIGHT, 1.0)
    max_dt = dts.max()
    age_days = (max_dt - dts).dt.days.fillna(0).astype(float).to_numpy()
    tau = float(half_life_days) / np.log(2.0)
    w = np.exp(-age_days / tau)
    if 'SeasonWeight' in df.columns:
        sw = pd.to_numeric(df['SeasonWeight'], errors='coerce').fillna(1.0).to_numpy()
        w = w * sw
    return np.clip(w, DC_MIN_WEIGHT, 1.0)


def fit_dixon_coles_mle(df, rho=0.0, half_life_days=DC_MLE_HALFLIFE_DAYS, ridge=DC_MLE_RIDGE, maxiter=DC_MLE_MAXITER, verbose=False):
    """Time-weighted Dixon–Coles attack/defence MLE.

    lambda_home = exp(gamma + att_home + def_away)
    lambda_away = exp(att_away + def_home)

    Identifiability via sum(att)=0 and sum(def)=0 (parameterise N-1 and derive last).
    """
    cols = ['HomeTeam','AwayTeam','FTHG','FTAG']
    d = df.dropna(subset=cols).copy()
    if len(d) < 30:
        return None

    d['FTHG'] = d['FTHG'].astype(int)
    d['FTAG'] = d['FTAG'].astype(int)

    teams = sorted(set(d['HomeTeam']).union(set(d['AwayTeam'])))
    n = len(teams)
    if n < 4:
        return None

    idx = {t:i for i,t in enumerate(teams)}
    h = d['HomeTeam'].map(idx).to_numpy()
    a = d['AwayTeam'].map(idx).to_numpy()
    hg = d['FTHG'].to_numpy()
    ag = d['FTAG'].to_numpy()

    w = _dc_time_weights(d, half_life_days=half_life_days)

    mean_h = max(hg.mean(), 1e-6)
    mean_a = max(ag.mean(), 1e-6)
    gamma0 = float(np.log(mean_h) - np.log(mean_a)) * 0.45

    p0 = np.zeros(2*(n-1)+1, dtype=float)
    p0[-1] = gamma0

    def unpack(p):
        att = np.zeros(n)
        deff = np.zeros(n)
        att[:-1] = p[:n-1]
        deff[:-1] = p[n-1:2*(n-1)]
        att[-1] = -att[:-1].sum()
        deff[-1] = -deff[:-1].sum()
        gamma = float(p[-1])
        return att, deff, gamma

    def nll(p):
        att, deff, gamma = unpack(p)
        lam = np.exp(gamma + att[h] + deff[a])
        mu  = np.exp(att[a] + deff[h])

        ll = (hg * np.log(lam) - lam - scipy.special.gammaln(hg+1)
              + ag * np.log(mu) - mu - scipy.special.gammaln(ag+1))

        if rho and abs(rho) > 1e-12:
            tau = np.ones_like(lam)
            mask = (hg <= 1) & (ag <= 1)
            if mask.any():
                x = hg[mask]; y = ag[mask]
                lam_m = lam[mask]; mu_m = mu[mask]
                t = np.ones_like(lam_m)
                m00 = (x==0) & (y==0)
                m01 = (x==0) & (y==1)
                m10 = (x==1) & (y==0)
                m11 = (x==1) & (y==1)
                t[m00] = 1.0 - (lam_m[m00] * mu_m[m00] * rho)
                t[m01] = 1.0 + (lam_m[m01] * rho)
                t[m10] = 1.0 + (mu_m[m10] * rho)
                t[m11] = 1.0 - rho
                t = np.clip(t, 1e-9, 10.0)
                tau[mask] = t
            ll = ll + np.log(np.clip(tau, 1e-9, 10.0))

        ll = ll * w
        if ridge and ridge > 0:
            ll = ll - ridge * np.sum(p[:-1]**2)

        return -float(ll.sum())

    res = minimize(nll, p0, method='L-BFGS-B', options={'maxiter': int(maxiter)})
    att, deff, gamma = unpack(res.x)

    out = {
        'teams': teams,
        'attack': {t: float(att[idx[t]]) for t in teams},
        'defence': {t: float(deff[idx[t]]) for t in teams},
        'gamma': float(gamma),
        'success': bool(getattr(res,'success',False)),
        'message': str(getattr(res,'message','') or ''),
    }
    if verbose:
        print(f"  DC MLE: success={out['success']} gamma={out['gamma']:.3f} teams={len(teams)}")
    return out


def elite_multiplier(att_h, def_h, att_a, def_a, k=ELITE_K, s=ELITE_S):
    """Smooth (non-hardcoded) correction based on fitted quality gap.

    quality(team) = attack - defence (log-scale).
    If away quality >> home quality, m > 1. We apply: away_lambda*=m, home_lambda/=m.
    """
    if not k or k == 0:
        return 1.0
    qh = float(att_h - def_h)
    qa = float(att_a - def_a)
    diff = qa - qh
    return float(np.exp(float(k) * np.tanh(diff / float(s))))

# =============================================================================
# TEAM STATS
# =============================================================================

def team_stats(df, team, venue, xg_cal, n=10, hl=DECAY_HALF_LIFE):
    if venue == 'home':
        matches = df[df['HomeTeam'] == team].tail(n).copy()
        if len(matches) == 0:
            return {'xg_for': 0, 'xg_ag': 0, 'n': 0}
        matches['xgf'] = matches.apply(lambda r: xg_cal.calc_xg(r['HST'], r['HS'], r['HC'], 'home'), axis=1)
        matches['xga'] = matches.apply(lambda r: xg_cal.calc_xg(r['AST'], r['AS'], r['AC'], 'away'), axis=1)
    else:
        matches = df[df['AwayTeam'] == team].tail(n).copy()
        if len(matches) == 0:
            return {'xg_for': 0, 'xg_ag': 0, 'n': 0}
        matches['xgf'] = matches.apply(lambda r: xg_cal.calc_xg(r['AST'], r['AS'], r['AC'], 'away'), axis=1)
        matches['xga'] = matches.apply(lambda r: xg_cal.calc_xg(r['HST'], r['HS'], r['HC'], 'home'), axis=1)

    return {
        'xg_for': weighted_avg(matches['xgf'].tolist(), hl),
        'xg_ag': weighted_avg(matches['xga'].tolist(), hl),
        'n': len(matches)
    }


def team_stats_blended(df, team, venue, xg_cal,
                       n_short=8, n_long=25,
                       w_short=0.65, w_long=0.35,
                       hl=DECAY_HALF_LIFE):
    """Blend recent + longer-term team xG stats to reduce noise."""
    short = team_stats(df, team, venue, xg_cal, n=n_short, hl=hl)
    long = team_stats(df, team, venue, xg_cal, n=n_long, hl=hl)

    # If we have very limited data, fall back gracefully.
    if short['n'] == 0 and long['n'] == 0:
        return {'xg_for': 0, 'xg_ag': 0, 'n': 0}
    if short['n'] == 0:
        return long
    if long['n'] == 0:
        return short

    xg_for = w_short * short['xg_for'] + w_long * long['xg_for']
    xg_ag = w_short * short['xg_ag'] + w_long * long['xg_ag']
    return {'xg_for': xg_for, 'xg_ag': xg_ag, 'n': max(short['n'], long['n'])}

def league_avgs(df, xg_cal, hl=DECAY_HALF_LIFE):
    df = df.dropna(subset=['FTHG', 'FTAG']).copy()
    df['hxg'] = df.apply(lambda r: xg_cal.calc_xg(r['HST'], r['HS'], r['HC'], 'home'), axis=1)
    df['axg'] = df.apply(lambda r: xg_cal.calc_xg(r['AST'], r['AS'], r['AC'], 'away'), axis=1)
    return {
        'h_xg': weighted_avg(df['hxg'].tolist(), hl),
        'a_xg': weighted_avg(df['axg'].tolist(), hl),
        'h_goals': df['FTHG'].mean(),
        'a_goals': df['FTAG'].mean()
    }

def build_team_season_stats(df):
    """Build per-team overall/home/away season stats from completed matches."""
    stats = {}

    def _blank():
        return {'mp': 0, 'w': 0, 'd': 0, 'l': 0, 'gf': 0, 'ga': 0, 'gd': 0}

    def _ensure(team):
        if team not in stats:
            stats[team] = {'overall': _blank(), 'home': _blank(), 'away': _blank()}
        return stats[team]

    def _update(bucket, gf, ga):
        bucket['mp'] += 1
        bucket['gf'] += int(gf)
        bucket['ga'] += int(ga)
        if gf > ga:
            bucket['w'] += 1
        elif gf == ga:
            bucket['d'] += 1
        else:
            bucket['l'] += 1
        bucket['gd'] = bucket['gf'] - bucket['ga']

    if df is None or df.empty:
        return stats

    for _, row in df.iterrows():
        try:
            home = row.get('HomeTeam')
            away = row.get('AwayTeam')
            hg = row.get('FTHG')
            ag = row.get('FTAG')
            if pd.isna(home) or pd.isna(away) or pd.isna(hg) or pd.isna(ag):
                continue
            _update(_ensure(home)['overall'], hg, ag)
            _update(_ensure(home)['home'], hg, ag)
            _update(_ensure(away)['overall'], ag, hg)
            _update(_ensure(away)['away'], ag, hg)
        except Exception:
            continue
    return stats

def build_team_recent_snapshot(df, team, xg_cal, n=5):
    """Build recent form + xG snapshot from the last n completed matches."""
    if df is None or df.empty:
        return {'n': 0}

    home = df[df['HomeTeam'] == team].copy()
    away = df[df['AwayTeam'] == team].copy()
    if home.empty and away.empty:
        return {'n': 0}

    home['venue'] = 'home'
    away['venue'] = 'away'
    home['gf'] = home['FTHG']
    home['ga'] = home['FTAG']
    away['gf'] = away['FTAG']
    away['ga'] = away['FTHG']
    matches = pd.concat([home, away], ignore_index=True)
    matches = matches.dropna(subset=['gf', 'ga'])

    if 'Date' in matches.columns:
        matches['Date'] = pd.to_datetime(matches['Date'], errors='coerce')
        matches = matches.sort_values('Date')

    matches = matches.tail(n)
    if matches.empty:
        return {'n': 0}

    w = d = l = 0
    gf = ga = 0
    xg_for_vals = []
    xg_ag_vals = []

    for _, row in matches.iterrows():
        try:
            gf_i = int(row['gf'])
            ga_i = int(row['ga'])
        except Exception:
            continue

        gf += gf_i
        ga += ga_i
        if gf_i > ga_i:
            w += 1
        elif gf_i == ga_i:
            d += 1
        else:
            l += 1

        try:
            if row['venue'] == 'home':
                xgf = xg_cal.calc_xg(row['HST'], row['HS'], row['HC'], 'home')
                xga = xg_cal.calc_xg(row['AST'], row['AS'], row['AC'], 'away')
            else:
                xgf = xg_cal.calc_xg(row['AST'], row['AS'], row['AC'], 'away')
                xga = xg_cal.calc_xg(row['HST'], row['HS'], row['HC'], 'home')
            xg_for_vals.append(float(xgf))
            xg_ag_vals.append(float(xga))
        except Exception:
            continue

    n_played = int(matches.shape[0])
    return {
        'n': n_played,
        'w': w,
        'd': d,
        'l': l,
        'gf': gf,
        'ga': ga,
        'gd': gf - ga,
        'xgf': float(np.mean(xg_for_vals)) if xg_for_vals else 0.0,
        'xga': float(np.mean(xg_ag_vals)) if xg_ag_vals else 0.0,
    }

# =============================================================================
# LEAGUE MODEL (One per league)
# =============================================================================

class LeagueModel:
    def __init__(self, league_code):
        self.league = league_code
        self.name = get_league_name(league_code)
        self.country = get_country(league_code)
        self.xg_cal = XGCalibrator()
        self.rho = -0.1
        self.lavg = {}
        self.trained = False
        self.n_matches = 0
        self.bayes_k = get_league_bayes_k(league_code)
        self.dc_halflife = get_league_dc_halflife(league_code)
        self.calibrators = {}

        # Dixon–Coles MLE params
        self.dc_params = None
        self.att = {}
        self.defn = {}
        self.gamma = 0.0
        self.elite_k = ELITE_K
        self.elite_s = ELITE_S
    def train(self, df, verbose=False):
        df_c = df.dropna(subset=['FTHG','FTAG','HS','AS','HST','AST','HC','AC']).copy()

        if len(df_c) < MIN_TRAIN_MATCHES:
            if verbose:
                print(f"  {self.league}: Insufficient data ({len(df_c)} matches)")
            return False

        self.xg_cal.calibrate(df_c, verbose=verbose)
        self.rho = estimate_rho(df_c, verbose=verbose)
        self.lavg = league_avgs(df_c, self.xg_cal)

        # Fit Dixon–Coles MLE (joint attack/defence). Improves calibration vs elite teams.
        self.dc_params = fit_dixon_coles_mle(
            df_c,
            rho=self.rho,
            half_life_days=self.dc_halflife,
            verbose=verbose
        )
        if self.dc_params:
            self.att = self.dc_params.get('attack', {})
            self.defn = self.dc_params.get('defence', {})
            self.gamma = float(self.dc_params.get('gamma', 0.0) or 0.0)
        self.trained = True
        self.n_matches = len(df_c)

        self._fit_calibration(df_c)

        if verbose:
            print(f"  {self.league}: Trained on {self.n_matches} matches, rho={self.rho:.3f}, avg goals={self.lavg['h_goals']:.2f}-{self.lavg['a_goals']:.2f}")

        return True
    def _fit_calibration(self, df):
        df_c = df.dropna(subset=['FTR', 'HomeTeam', 'AwayTeam']).copy()
        if len(df_c) < 120:
            return
        sample = df_c.tail(700)
        probs = {'H': [], 'D': [], 'A': []}
        outcomes = {'H': [], 'D': [], 'A': []}
        for _, row in sample.iterrows():
            ht = row.get('HomeTeam')
            at = row.get('AwayTeam')
            if pd.isna(ht) or pd.isna(at):
                continue
            pred = self.predict(ht, at, df, apply_calibration=False)
            if pred is None:
                continue
            for out in ['H', 'D', 'A']:
                probs[out].append(float(pred['probs'][out]))
                outcomes[out].append(1.0 if row.get('FTR') == out else 0.0)

        for out in ['H', 'D', 'A']:
            if len(set(outcomes[out])) < 2:
                continue
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(np.array(probs[out]), np.array(outcomes[out]))
            self.calibrators[out] = iso

    def _apply_calibration(self, probs):
        if not self.calibrators:
            return probs
        cal = {}
        for out in ['H', 'D', 'A']:
            model_p = float(probs.get(out, 0.0) or 0.0)
            calib = self.calibrators.get(out)
            cal[out] = float(calib.predict([model_p])[0]) if calib else model_p
        total = sum(cal.values())
        if total > 0:
            cal = {k: v / total for k, v in cal.items()}
        return cal

    def predict(self, ht, at, df, apply_calibration=True):
        if not self.trained:
            return None

        # ---------------------------------------------------------------------
        # 1) Macro-xG (venue-split) lambdas
        #    - Captures team-specific 'better at home / worse away' effects because
        #      we compute home-team stats from HOME matches and away-team stats from
        #      AWAY matches.
        # ---------------------------------------------------------------------
        hs = team_stats_blended(df, ht, 'home', self.xg_cal, n_short=8, n_long=25, w_short=0.65, w_long=0.35)
        aws = team_stats_blended(df, at, 'away', self.xg_cal, n_short=8, n_long=25, w_short=0.65, w_long=0.35)

        hxg = float(self.lavg.get('h_xg', 0) or 0)
        axg = float(self.lavg.get('a_xg', 0) or 0)
        if hxg <= 0 or axg <= 0:
            return None

        # Bayesian shrink: with small samples, pull team stats toward league means
        def shrink(x, n, league_mean):
            try:
                n = float(max(0, int(n)))
                return (float(x) * n + float(league_mean) * float(self.bayes_k)) / (n + float(self.bayes_k))
            except Exception:
                return float(league_mean)

        hs_xgf = shrink(hs.get('xg_for', 0.0), hs.get('n', 0), hxg)
        hs_xga = shrink(hs.get('xg_ag', 0.0),  hs.get('n', 0), axg)
        aw_xgf = shrink(aws.get('xg_for', 0.0), aws.get('n', 0), axg)
        aw_xga = shrink(aws.get('xg_ag', 0.0),  aws.get('n', 0), hxg)

        h_att = hs_xgf / hxg
        h_def = hs_xga / axg
        a_att = aw_xgf / axg
        a_def = aw_xga / hxg

        # Build lambdas from a neutral baseline + explicit home-adv ONCE (avoid double-count)
        base = float(math.sqrt(hxg * axg))
        home_adv = float(np.clip(math.sqrt(hxg / axg), 0.90, 1.25))
        h_lam_ratio = float(h_att * a_def * base * home_adv)
        a_lam_ratio = float(a_att * h_def * base / home_adv)

        # Buchdahl-style recent form tweak (small + capped)
        def buch(team, n=6):
            hm = df[df['HomeTeam'] == team].tail(n)
            am = df[df['AwayTeam'] == team].tail(n)
            return (hm['FTHG'].sum() + am['FTAG'].sum()) - (hm['FTAG'].sum() + am['FTHG'].sum())

        hr, ar = buch(ht), buch(at)
        try:
            adj = float(np.clip((hr - ar) * 0.015, -0.15, 0.15))
            h_lam_ratio *= (1.0 + adj)
            a_lam_ratio *= (1.0 - adj)
        except Exception:
            adj = 0.0

        h_lam_ratio = float(np.clip(h_lam_ratio, 0.05, 4.5))
        a_lam_ratio = float(np.clip(a_lam_ratio, 0.05, 4.5))

        # ---------------------------------------------------------------------
        # 2) Dixon–Coles lambdas (structural baseline)
        #    NOTE: DC has league-wide home advantage via gamma, but it does NOT
        #    have team-specific home/away parameters. That's why we blend.
        # ---------------------------------------------------------------------
        h_lam_dc = a_lam_dc = None
        dc_meta = {'dc': False}
        if self.att and self.defn and (ht in self.att) and (at in self.att) and (ht in self.defn) and (at in self.defn):
            att_h = float(self.att.get(ht, 0.0) or 0.0)
            def_h = float(self.defn.get(ht, 0.0) or 0.0)
            att_a = float(self.att.get(at, 0.0) or 0.0)
            def_a = float(self.defn.get(at, 0.0) or 0.0)

            h_lam_dc = float(np.exp(self.gamma + att_h + def_a))
            a_lam_dc = float(np.exp(att_a + def_h))

            mult = elite_multiplier(att_h, def_h, att_a, def_a, k=self.elite_k, s=self.elite_s)
            a_lam_dc *= mult
            h_lam_dc /= mult

            h_lam_dc = float(np.clip(h_lam_dc, 0.05, 4.5))
            a_lam_dc = float(np.clip(a_lam_dc, 0.05, 4.5))

            dc_meta = {'dc': True, 'att_h': att_h, 'def_h': def_h, 'att_a': att_a, 'def_a': def_a, 'elite_m': mult}

        # ---------------------------------------------------------------------
        # 3) Blend in log space (geometric mean)
        # ---------------------------------------------------------------------
        if h_lam_dc is None or a_lam_dc is None:
            h_lam = h_lam_ratio
            a_lam = a_lam_ratio
            blend_w = 0.0
        else:
            # Down-weight DC when it disagrees strongly with macro-xG
            delta = abs(np.log(h_lam_dc / h_lam_ratio)) + abs(np.log(a_lam_dc / a_lam_ratio))
            blend_w = float(np.clip(DC_BLEND_BASE - DC_BLEND_SENS * delta, DC_BLEND_MIN, DC_BLEND_MAX))
            h_lam = float(np.exp(blend_w * np.log(h_lam_dc) + (1.0 - blend_w) * np.log(h_lam_ratio)))
            a_lam = float(np.exp(blend_w * np.log(a_lam_dc) + (1.0 - blend_w) * np.log(a_lam_ratio)))

        h_lam = float(np.clip(h_lam, 0.05, 4.5))
        a_lam = float(np.clip(a_lam, 0.05, 4.5))

        mat = poisson_matrix_dc(h_lam, a_lam, self.rho)
        probs_raw = {'H': np.tril(mat,-1).sum(), 'D': np.trace(mat), 'A': np.triu(mat,1).sum()}
        probs = self._apply_calibration(probs_raw) if apply_calibration else probs_raw

        def _odds_from_prob(p):
            try:
                p = float(p)
            except Exception:
                return MAX_FAIR_ODDS
            p = min(max(p, 1e-4), 0.999)
            return min(max(1.0 / p, MIN_FAIR_ODDS), MAX_FAIR_ODDS)

        odds_raw = {k: _odds_from_prob(v) for k, v in probs_raw.items()}
        odds_final = {k: _odds_from_prob(v) for k, v in probs.items()}

        return {
            'home': ht, 'away': at, 'league': self.league,
            'h_xg': h_lam, 'a_xg': a_lam,
            'probs': probs,
            'odds': odds_final,
            'probs_raw': probs_raw,
            'odds_raw': odds_raw,
            'odds_final': odds_final,
            'strength': {
                **dc_meta,
                'blend_w': blend_w,
                'lam_ratio': {'h': h_lam_ratio, 'a': a_lam_ratio, 'h_n': hs.get('n', 0), 'a_n': aws.get('n', 0)},
                'lam_dc': {'h': h_lam_dc, 'a': a_lam_dc},
                'h_att': h_att, 'h_def': h_def, 'a_att': a_att, 'a_def': a_def,
            },
            'buch': {'h': hr, 'a': ar, 'diff': hr-ar, 'adj': adj}
        }

    def find_value(self, pred, mkt_odds, thresh=None):
        if pred is None:
            return []
        if thresh is None:
            thresh = get_ev_threshold(self.league, default=VALUE_THRESHOLD)
        vals = []
        for out in ['H', 'D', 'A']:
            mp = pred['probs'][out]
            mo = mkt_odds.get(out, 0)
            if mo > 0 and mo < 50:  # Filter bad odds
                ev = mp * mo - 1
                vals.append({
                    'out': out, 'model_p': mp, 'model_o': pred['odds'][out],
                    'mkt_o': mo, 'ev': ev*100, 'value': ev >= float(thresh)
                })
        return vals

# =============================================================================
# BACKTESTER
# =============================================================================

class Backtester:
    def __init__(self):
        self.results = []
        self.summary = {}

    def run(self, df, model, min_ev=0.05, train_win=60):
        df = df.dropna(subset=['FTHG', 'FTAG', 'B365H', 'B365D', 'B365A']).reset_index(drop=True)
        self.results = []

        if len(df) < train_win + 20:
            return {'n': 0, 'roi': 0, 'p': 1}

        # --- SPEED OPTIMIZATION: Only backtest last 100 matches ---
        # If you want to test the FULL history (very slow), set start_index = train_win
        start_index = max(train_win, len(df) - 300)
        total_to_test = len(df) - start_index

        print(f"    > Backtesting last {total_to_test} matches...", end=' ', flush=True)

        for i in range(start_index, len(df)):
            # --- PROGRESS INDICATOR ---
            if (i - start_index) % 5 == 0:
                pct = ((i - start_index) / total_to_test) * 100
                print(f"\r    > Progress: {i - start_index}/{total_to_test} matches ({pct:.0f}%)", end='', flush=True)
            # ---------------------------

            train = df.iloc[:i].copy()
            test = df.iloc[i]

            # Retrain on expanding window
            temp_model = LeagueModel(model.league)
            if not temp_model.train(train, verbose=False):
                continue

            ht, at = test['HomeTeam'], test['AwayTeam']
            actual = test['FTR']

            pred = temp_model.predict(ht, at, train)
            if pred is None:
                continue

            mkt = {'H': test['B365H'], 'D': test['B365D'], 'A': test['B365A']}
            vals = temp_model.find_value(pred, mkt, thresh=min_ev)

            for v in vals:
                if v['value']:
                    won = 1 if actual == v['out'] else 0
                    profit = (v['mkt_o'] - 1) if won else -1
                    self.results.append({'out': v['out'], 'won': won, 'profit': profit, 'odds': v['mkt_o']})

        print(f"\r    > Done! Processed {total_to_test} matches.              ")
        return self._calc_summary()

    def _calc_summary(self):
        if not self.results:
            return {'n': 0, 'roi': 0, 'p': 1}
        df = pd.DataFrame(self.results)
        n = len(df)
        profit = df['profit'].sum()
        t_stat, p_val = ttest_1samp(df['profit'], 0) if n >= 2 else (0, 1)
        return {
            'n': n, 'wins': df['won'].sum(), 'win_pct': df['won'].mean()*100,
            'profit': profit, 'roi': profit/n*100,
            't': t_stat, 'p': p_val
        }

# =============================================================================
# FIXTURES ODDS HELPERS (multi-bookie)
# =============================================================================



def extract_market_totals_odds(row):
    """Extract O/U market odds from football-data columns.

    Supported patterns include:
      - BOOK>2.5 / BOOK<2.5 (e.g. B365>2.5)
      - BOOKO2.5 / BOOKU2.5 (common alternate naming)
      - O2.5 / U2.5

    Returns dict: {"2.50_over": median_over, "2.50_under": median_under, ...}
    """
    by_line = {}
    try:
        keys = [k for k in list(row.index) if isinstance(k, str)]
        rx_angle = re.compile(r"^[A-Za-z0-9_]*([<>])(\d+(?:\.\d+)?)$")
        rx_ou = re.compile(r"^[A-Za-z0-9_]*([OU])(\d+(?:\.\d+)?)$", re.IGNORECASE)

        for k in keys:
            col = k.strip().replace(' ', '')
            side = None
            line = None

            m1 = rx_angle.match(col)
            if m1:
                side = 'over' if m1.group(1) == '>' else 'under'
                line = float(m1.group(2))
            else:
                m2 = rx_ou.match(col)
                if m2:
                    side = 'over' if m2.group(1).upper() == 'O' else 'under'
                    line = float(m2.group(2))

            if side is None or line is None:
                continue

            try:
                odd = float(row.get(k, 0) or 0)
            except Exception:
                odd = 0
            if not (1.01 < odd < 100):
                continue

            key = f"{line:.2f}_{side}"
            by_line.setdefault(key, []).append(odd)
    except Exception:
        return {}

    out = {}
    for k, vals in by_line.items():
        if vals:
            out[k] = float(np.median(vals))
    return out


def extract_market_btts_odds(row):
    """Extract BTTS odds from columns like B365BTTS/BTTS and yes/no variants."""
    yes_vals, no_vals = [], []
    keys = [k for k in list(row.index) if isinstance(k, str)]

    # direct yes/no columns used by some feeds
    for k in keys:
        ku = k.strip().upper()
        try:
            v = float(row.get(k, 0) or 0)
        except Exception:
            v = 0
        if not (1.01 < v < 100):
            continue
        if 'BTTS' in ku and ('YES' in ku or ku.endswith('Y')):
            yes_vals.append(v)
        elif 'BTTS' in ku and ('NO' in ku or ku.endswith('N')):
            no_vals.append(v)

    # paired form: <BOOK>BTTS + <BOOK>BTTSN
    for k in keys:
        if not isinstance(k, str):
            continue
        ku = k.strip().upper()
        if ku.endswith('BTTS'):
            pref = k[:-4]
            ncol = pref + 'BTTSN'
            if ncol in row.index:
                try:
                    yv = float(row.get(k, 0) or 0)
                    nv = float(row.get(ncol, 0) or 0)
                    if 1.01 < yv < 100:
                        yes_vals.append(yv)
                    if 1.01 < nv < 100:
                        no_vals.append(nv)
                except Exception:
                    pass

    out = {}
    if yes_vals:
        out['yes'] = float(np.median(yes_vals))
    if no_vals:
        out['no'] = float(np.median(no_vals))
    return out


def extract_market_ah_odds(row):
    """Extract AH prices using AH line columns where available.

    Common football-data fields:
      - AHh (line), B365AHH (home), B365AHA (away)
      - AHCh (line), B365CAHH (home), B365CAHA (away)
    Returns dict like {"-0.50_home": 1.95, "-0.50_away": 1.92}
    """
    out = {}
    keys = [k for k in list(row.index) if isinstance(k, str)]

    def _collect(line_col, home_suffix, away_suffix):
        if line_col not in row.index:
            return
        try:
            line = float(row.get(line_col, 0) or 0)
        except Exception:
            return
        hvals, avals = [], []
        for k in keys:
            ku = k.strip().upper()
            try:
                v = float(row.get(k, 0) or 0)
            except Exception:
                v = 0
            if not (1.01 < v < 100):
                continue
            if ku.endswith(home_suffix):
                hvals.append(v)
            elif ku.endswith(away_suffix):
                avals.append(v)
        if hvals:
            out[f"{line:+.2f}_home"] = float(np.median(hvals))
        if avals:
            out[f"{line:+.2f}_away"] = float(np.median(avals))

    _collect('AHh', 'AHH', 'AHA')
    _collect('AHCh', 'CAHH', 'CAHA')
    return out

def extract_all_book_odds(row):
    """
    From a fixtures row, extract all bookie odds that look like <BOOK>H/<BOOK>D/<BOOK>A.
    Returns: dict { 'B365': {'H':..,'D':..,'A':..}, 'BW': {...}, ... }
    """
    odds = {}
    keys = list(row.index)

    for k in keys:
        if not isinstance(k, str) or len(k) < 2:
            continue
        if k.endswith('H'):
            pref = k[:-1]
            d, a = pref + 'D', pref + 'A'
            if d in keys and a in keys:
                try:
                    h = float(row.get(pref + 'H', 0) or 0)
                    dd = float(row.get(pref + 'D', 0) or 0)
                    aa = float(row.get(pref + 'A', 0) or 0)
                    if h > 1.01 or dd > 1.01 or aa > 1.01:
                        odds[pref] = {'H': h, 'D': dd, 'A': aa}
                except:
                    pass

    return odds

def best_books_for_outcome(all_books, outcome, top_n=5):
    """Return list of (book, odds) sorted by best odds for a given outcome."""
    pairs = []
    for book, trip in all_books.items():
        o = trip.get(outcome, 0)
        if o and o > 1.01 and o < 100:
            pairs.append((book, o))
    pairs.sort(key=lambda x: -x[1])
    return pairs[:top_n]

def best_market_price(all_books, outcome, fallback=0):
    best = 0
    for _, trip in all_books.items():
        try:
            best = max(best, float(trip.get(outcome, 0) or 0))
        except:
            pass
    return best if best > 0 else fallback


def median_market_price(all_books, outcome, fallback=0):
    vals=[]
    for _, trip in (all_books or {}).items():
        try:
            o=float(trip.get(outcome, 0) or 0)
            if o and 1.01 < o < 100:
                vals.append(o)
        except Exception:
            pass
    if not vals:
        return fallback
    vals.sort()
    n=len(vals)
    mid=n//2
    return vals[mid] if n%2==1 else (vals[mid-1]+vals[mid])/2

def outcome_label(out):
    return {'H': 'Home', 'D': 'Draw', 'A': 'Away'}.get(out, out)

# =============================================================================
# PRINT HELPERS
# =============================================================================

def print_pred(p, mkt=None, vals=None):
    print(f"\n{p['home']} vs {p['away']}")
    print(f"xG: {p['h_xg']:.2f} - {p['a_xg']:.2f}")

    pr, od = p['probs'], p['odds']
    print(f"Model: H {pr['H']*100:.0f}%={od['H']:.2f} | D {pr['D']*100:.0f}%={od['D']:.2f} | A {pr['A']*100:.0f}%={od['A']:.2f}")

    if mkt and vals:
        print(f"Market: H {mkt['H']:.2f} | D {mkt['D']:.2f} | A {mkt['A']:.2f}")
        for v in vals:
            if v['value']:
                # v['model_o'] is fair odds from model
                print(f"  >>> VALUE: {v['out']} @ {v['mkt_o']:.2f} (EV: {v['ev']:+.1f}%) | Fair: {v['model_o']:.2f}")

# =============================================================================
# HTML REPORT
# =============================================================================

def _html_escape(s):
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))



def _wikidata_logo_filename(entity_id):
    try:
        url = "https://www.wikidata.org/w/api.php"
        params = {'action': 'wbgetclaims', 'entity': entity_id, 'property': 'P154', 'format': 'json'}
        r = requests.get(url, params=params, timeout=6)
        r.raise_for_status()
        data = r.json() or {}
        claims = ((data.get('claims') or {}).get('P154') or [])
        if not claims:
            return None
        val = (((claims[0].get('mainsnak') or {}).get('datavalue') or {}).get('value') or '')
        return str(val).strip() if val else None
    except Exception:
        return None


def _lookup_team_logo_svg(team_name, cache):
    t = str(team_name or '').strip()
    if not t:
        return ''
    if t in cache:
        return cache[t] or ''
    try:
        url = "https://www.wikidata.org/w/api.php"
        params = {'action': 'wbsearchentities', 'search': t, 'language': 'en', 'format': 'json', 'limit': 6}
        r = requests.get(url, params=params, timeout=6)
        r.raise_for_status()
        data = r.json() or {}
        for item in (data.get('search') or []):
            ent = item.get('id')
            if not ent:
                continue
            logo_file = _wikidata_logo_filename(ent)
            if logo_file:
                url = "https://commons.wikimedia.org/wiki/Special:FilePath/" + urllib.parse.quote(logo_file)
                cache[t] = url
                return url
    except Exception:
        pass
    cache[t] = ''
    return ''


def write_html_report(all_value_bets, all_fixtures=None, out_path="output/value_bets.html"):
    """Write a standalone HTML report with tabs + filtering UI.

    Tabs:
      1) Value Bets (Grouped) - Country -> League
      2) Bets by Time - all value bets sorted by date + kickoff
      3) All Fixtures (xG) - every predicted fixture with model xG + fair 1X2

    Filters apply to the currently selected tab.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    logo_cache_path = os.path.join(os.path.dirname(out_path), "team_logo_cache.pkl")
    try:
        with open(logo_cache_path, 'rb') as f:
            logo_cache = pickle.load(f)
            if not isinstance(logo_cache, dict):
                logo_cache = {}
    except Exception:
        logo_cache = {}

    def _time_to_minutes(t):
        try:
            s = str(t).strip()
            if not s or s.lower() in ('nan', 'nat', 'none'):
                return 24 * 60 + 1
            if ':' in s:
                s = s[:5]
                hh, mm = s.split(':')[:2]
                return int(hh) * 60 + int(mm)
        except Exception:
            pass
        return 24 * 60 + 1

    def _stat_val(val):
        try:
            return str(int(val))
        except Exception:
            return "-"

    def _render_team_stat_rows(team_name, team_stats):
        if not team_stats or not team_stats.get('overall'):
            return [
                f"<tr><td class='team'>{_html_escape(team_name)}</td>"
                "<td colspan='8' class='muted small'>No current season stats available.</td></tr>"
            ]

        rows = []
        scopes = [('Overall', 'overall'), ('Home', 'home'), ('Away', 'away')]
        for idx, (label, key) in enumerate(scopes):
            stats = team_stats.get(key, {}) or {}
            team_cell = f"<td class='team' rowspan='3'>{_html_escape(team_name)}</td>" if idx == 0 else ""
            rows.append(
                "<tr>"
                + team_cell
                + f"<td class='scope'>{label}</td>"
                + f"<td>{_stat_val(stats.get('mp'))}</td>"
                + f"<td>{_stat_val(stats.get('w'))}</td>"
                + f"<td>{_stat_val(stats.get('d'))}</td>"
                + f"<td>{_stat_val(stats.get('l'))}</td>"
                + f"<td>{_stat_val(stats.get('gf'))}</td>"
                + f"<td>{_stat_val(stats.get('ga'))}</td>"
                + f"<td>{_stat_val(stats.get('gd'))}</td>"
                + "</tr>"
            )
        return rows

    def _render_season_stats_table(fix):
        team_stats = fix.get('season_stats', {}) or {}
        home_stats = team_stats.get('home', {}) if isinstance(team_stats, dict) else {}
        away_stats = team_stats.get('away', {}) if isinstance(team_stats, dict) else {}
        season_label = fix.get('season_label') or ""

        rows = []
        rows.extend(_render_team_stat_rows(fix.get('home', ''), home_stats))
        rows.extend(_render_team_stat_rows(fix.get('away', ''), away_stats))

        title = "Season stats"
        if season_label:
            title += f" ({_html_escape(season_label)})"

        table_html = (
            f"<h4>{title}</h4>"
            "<table class='table stats-table'>"
            "<thead><tr><th>Team</th><th>Scope</th><th>MP</th><th>W</th><th>D</th><th>L</th><th>GF</th><th>GA</th><th>GD</th></tr></thead>"
            "<tbody>" + "".join(rows) + "</tbody></table>"
        )
        return table_html

    def _confidence_meta(fix):
        team_stats = fix.get('season_stats', {}) or {}
        home_stats = team_stats.get('home', {}) if isinstance(team_stats, dict) else {}
        away_stats = team_stats.get('away', {}) if isinstance(team_stats, dict) else {}
        home_mp = int(((home_stats.get('overall') or {}).get('mp', 0)) or 0)
        away_mp = int(((away_stats.get('overall') or {}).get('mp', 0)) or 0)
        score = min(1.0, (home_mp + away_mp) / 40.0) if (home_mp + away_mp) > 0 else 0.0
        if score >= 0.75:
            label = 'High'
            cls = 'high'
        elif score >= 0.45:
            label = 'Medium'
            cls = 'med'
        else:
            label = 'Low'
            cls = 'low'
        return label, cls, score

    def _teamline_html(home, away):
        h = str(home or '')
        a = str(away or '')
        hl = team_logo_urls.get(h, '')
        al = team_logo_urls.get(a, '')
        hs = (h[:1] or '?').upper()
        as_ = (a[:1] or '?').upper()
        h_logo = f"<img class='teamlogo' src='{_html_escape(hl)}' loading='lazy' onerror='this.style.display=&quot;none&quot;;this.nextElementSibling.style.display=&quot;inline-flex&quot;'><span class='teamfallback'>{_html_escape(hs)}</span>" if hl else f"<span class='teamfallback'>{_html_escape(hs)}</span>"
        a_logo = f"<img class='teamlogo' src='{_html_escape(al)}' loading='lazy' onerror='this.style.display=&quot;none&quot;;this.nextElementSibling.style.display=&quot;inline-flex&quot;'><span class='teamfallback'>{_html_escape(as_)}</span>" if al else f"<span class='teamfallback'>{_html_escape(as_)}</span>"
        return f"<div class='teamline'><span class='logo'>{h_logo}</span><span>{_html_escape(h)}</span><span class='muted'>vs</span><span class='logo'>{a_logo}</span><span>{_html_escape(a)}</span></div>"

    def _format_edge_line(b):
        model_prob = float(b.get('model_prob', 0) or 0)
        market_prob = float(b.get('market_prob', 0) or 0)
        edge_pp = float(b.get('edge_pp', 0) or 0)
        if model_prob <= 0 and market_prob <= 0:
            return ""
        edge_cls = 'edge-pos' if edge_pp >= 0 else 'edge-neg'
        return (
            f"<div class='muted small' style='margin-top:4px'>"
            f"Model P: <b>{model_prob*100:.1f}%</b> · Market Implied: <b>{market_prob*100:.1f}%</b> · "
            f"Edge: <b class='{edge_cls}'>{edge_pp:+.1f}pp</b></div>"
        )

    def _render_recent_form_table(fix):
        recent = fix.get('recent_form', {}) or {}
        home = recent.get('home', {}) or {}
        away = recent.get('away', {}) or {}
        xg = fix.get('recent_xg', {}) or {}
        home_xg5 = (xg.get('home', {}) or {}).get('xg5', {}) or {}
        away_xg5 = (xg.get('away', {}) or {}).get('xg5', {}) or {}
        home_xg10 = (xg.get('home', {}) or {}).get('xg10', {}) or {}
        away_xg10 = (xg.get('away', {}) or {}).get('xg10', {}) or {}

        def _form_row(team, data):
            if not data or not data.get('n'):
                return f"<tr><td>{_html_escape(team)}</td><td colspan='4' class='muted small'>No recent form.</td></tr>"
            return (
                "<tr>"
                f"<td>{_html_escape(team)}</td>"
                f"<td>{_stat_val(data.get('w'))}-{_stat_val(data.get('d'))}-{_stat_val(data.get('l'))}</td>"
                f"<td>{_stat_val(data.get('gf'))}</td>"
                f"<td>{_stat_val(data.get('ga'))}</td>"
                f"<td>{_stat_val(data.get('gd'))}</td>"
                "</tr>"
            )

        def _xg_row(team, data5, data10):
            if not (data5.get('n') or data10.get('n')):
                return f"<tr><td>{_html_escape(team)}</td><td colspan='4' class='muted small'>No xG trend.</td></tr>"
            def _fmt(val):
                try:
                    return f"{float(val):.2f}"
                except Exception:
                    return "-"
            return (
                "<tr>"
                f"<td>{_html_escape(team)}</td>"
                f"<td>{_fmt(data5.get('xgf'))}</td>"
                f"<td>{_fmt(data5.get('xga'))}</td>"
                f"<td>{_fmt(data10.get('xgf'))}</td>"
                f"<td>{_fmt(data10.get('xga'))}</td>"
                "</tr>"
            )

        form_html = (
            "<h4>Recent form (last 5)</h4>"
            "<table class='table stats-table'>"
            "<thead><tr><th>Team</th><th>W-D-L</th><th>GF</th><th>GA</th><th>GD</th></tr></thead>"
            "<tbody>"
            + _form_row(fix.get('home', ''), home)
            + _form_row(fix.get('away', ''), away)
            + "</tbody></table>"
        )

        xg_html = (
            "<h4 style='margin-top:10px'>xG trend (avg)</h4>"
            "<table class='table stats-table'>"
            "<thead><tr><th>Team</th><th>xGF L5</th><th>xGA L5</th><th>xGF L10</th><th>xGA L10</th></tr></thead>"
            "<tbody>"
            + _xg_row(fix.get('home', ''), home_xg5, home_xg10)
            + _xg_row(fix.get('away', ''), away_xg5, away_xg10)
            + "</tbody></table>"
        )
        return form_html + xg_html

    bets = list(all_value_bets or [])
    fixtures = list(all_fixtures or [])

    # Sort bets by date/time for the "Bets by Time" tab
    bets_time_sorted = sorted(
        bets,
        key=lambda x: (
            x.get('date', '9999-12-31'),
            _time_to_minutes(x.get('kickoff', '')),
            -float(x.get('ev', 0) or 0),
            x.get('league_name', ''),
            x.get('home', ''),
        )
    )

    # Sort fixtures by date/time
    fixtures_sorted = sorted(
        fixtures,
        key=lambda x: (
            x.get('date', '9999-12-31'),
            _time_to_minutes(x.get('kickoff', '')),
            x.get('league_name', ''),
            x.get('home', ''),
        )
    )


    all_teams = set()
    for b in bets_time_sorted:
        all_teams.add(str(b.get('home', '')).strip())
        all_teams.add(str(b.get('away', '')).strip())
    for f in fixtures_sorted:
        all_teams.add(str(f.get('home', '')).strip())
        all_teams.add(str(f.get('away', '')).strip())
    all_teams = sorted([t for t in all_teams if t])
    team_logo_urls = {t: _lookup_team_logo_svg(t, logo_cache) for t in all_teams}

    ou_opps = []
    fixtures_with_totals_market = 0
    for f in fixtures_sorted:
        try:
            mkt_tot = f.get('mkt_totals') or {}
            lines = sorted({
                float(str(k).split('_')[0])
                for k in mkt_tot.keys()
                if isinstance(k, str) and '_over' in k
            })
            if not lines:
                continue
            fixtures_with_totals_market += 1

            hxg = float(f.get('h_xg', 0) or 0)
            axg = float(f.get('a_xg', 0) or 0)
            mat = poisson_matrix_dc(hxg, axg, rho=0.0, max_g=11)
            totp = total_probs_from_matrix(mat)

            best = None
            for line in lines:
                mk_over = float(mkt_tot.get(f"{line:.2f}_over") or mkt_tot.get(f"{line}_over") or 0)
                mk_under = float(mkt_tot.get(f"{line:.2f}_under") or mkt_tot.get(f"{line}_under") or 0)
                if mk_over <= 1.01 and mk_under <= 1.01:
                    continue

                fair_over, fair_under = fair_totals_ou(totp, line)
                over_ev = (mk_over / fair_over) - 1.0 if (mk_over > 1.01 and fair_over > 1.01) else -1.0
                under_ev = (mk_under / fair_under) - 1.0 if (mk_under > 1.01 and fair_under > 1.01) else -1.0
                side = 'Over' if over_ev >= under_ev else 'Under'
                edge = max(over_ev, under_ev)

                if (best is None) or (edge > best['edge']):
                    best = {
                        'line': line,
                        'mk_over': mk_over, 'mk_under': mk_under,
                        'fair_over': fair_over, 'fair_under': fair_under,
                        'over_ev': over_ev, 'under_ev': under_ev,
                        'best_side': side,
                        'edge': edge,
                    }

            if not best or best['edge'] < 0.03:
                continue

            side_line = f"{best['best_side']} {best['line']:.2f}"
            pick_odds = best['mk_over'] if best['best_side'] == 'Over' else best['mk_under']
            tip = f"Lean {side_line} if odds stay ≥ {max(pick_odds, 1.01):.2f}"
            ou_opps.append({
                'date': f.get('date', ''), 'kickoff': f.get('kickoff', ''), 'country': f.get('country', ''),
                'league_name': f.get('league_name', ''), 'home': f.get('home', ''), 'away': f.get('away', ''),
                'h_xg': hxg, 'a_xg': axg,
                'line': best['line'],
                'fair_over': best['fair_over'], 'fair_under': best['fair_under'],
                'mk_over': best['mk_over'], 'mk_under': best['mk_under'],
                'over_ev': best['over_ev']*100.0, 'under_ev': best['under_ev']*100.0,
                'best_side': side_line, 'edge': best['edge']*100.0, 'tip': tip,
            })
        except Exception:
            pass
    ou_opps = sorted(ou_opps, key=lambda x: -float(x.get('edge', 0) or 0))

    # Filter controls lists (union from bets + fixtures)
    countries = sorted({*(b.get('country', 'Unknown') for b in bets_time_sorted), *(f.get('country', 'Unknown') for f in fixtures_sorted)})
    leagues = sorted({*(b.get('league_name', b.get('league', '')) for b in bets_time_sorted), *(f.get('league_name', f.get('league', '')) for f in fixtures_sorted)})

    # Best Bets criteria
    BEST_MARKET_MAX = 2.00
    BEST_EV_MIN = 10.0
    best_bets = [
        b for b in bets_time_sorted
        if float(b.get('odds', 999) or 999) <= BEST_MARKET_MAX
        and float(b.get('ev', 0) or 0) >= BEST_EV_MIN
    ]
    best_bets = sorted(best_bets, key=lambda x: -float(x.get('ev', 0) or 0))

    # Group bets by country -> league
    by_country = {}
    for b in bets_time_sorted:
        by_country.setdefault(b.get('country', 'Unknown'), {}).setdefault(
            b.get('league_name', b.get('league', '')),
            []
        ).append(b)

    gen_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    parts = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    parts.append("<title>Football Model Report</title>")

    parts.append("<style>")
    parts.append("/* All Fixtures: highlight biggest market-vs-fair differences */\ntd.bestedge{box-shadow: inset 0 0 0 2px rgba(255,255,255,.28); }\n")
    parts.append(r"""
:root{
  --bg:#f6f7fb;
  --panel:#ffffff;
  --panel2:#f9fafc;
  --text:#1f2937;
  --muted:#6b7280;
  --line:rgba(15,23,42,.12);
  --pill:rgba(15,23,42,.06);
  --shadow: 0 8px 24px rgba(15,23,42,.08);
}
*{box-sizing:border-box;}
body{
  font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;
  margin:0;
  background:var(--bg);
  color:var(--text);
}
.container{max-width:1250px; margin:0 auto; padding:22px 16px 70px;}
h1{margin:0 0 10px 0; font-size:28px; letter-spacing:0.2px}
.muted{color:var(--muted)}

.sticky{position:sticky; top:0; z-index:10; backdrop-filter: blur(10px);}
.toolbar{
  margin:14px 0 14px 0;
  padding:14px;
  border:1px solid var(--line);
  border-radius:14px;
  background:#ffffff;
  box-shadow:var(--shadow);
}
.controls{display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:12px; align-items:flex-end;}
.ctl{display:flex; flex-direction:column; gap:6px}
.ctl.search-ctl{grid-column:span 2}
label{font-size:12px; color:#4b5563}
input,select{
  padding:10px 10px;
  border-radius:12px;
  border:1px solid rgba(15,23,42,.18);
  background:#ffffff;
  color:var(--text);
  min-width:0;
  width:100%;
  outline:none;
}
input::placeholder{color:#94a3b8}
button{
  padding:10px 14px;
  border-radius:12px;
  border:1px solid rgba(15,23,42,.16);
  background:#ffffff;
  color:var(--text);
  cursor:pointer;
}
button:hover{background:#f8fafc}

.tabs{display:flex; gap:10px; flex-wrap:nowrap; overflow-x:auto; margin-top:12px; padding-bottom:2px; -webkit-overflow-scrolling:touch; scrollbar-width:thin}
.tabbtn{
  padding:8px 12px;
  border-radius:999px;
  border:1px solid rgba(15,23,42,.16);
  background:rgba(255,255,255,.06);
  cursor:pointer;
  font-size:13px;
  white-space:nowrap;
  flex:0 0 auto;
}
.tabbtn.active{background:#e8f0ff; border-color:#b7cdfa; color:#1d4ed8; font-weight:700}

.section{margin-top:16px;}
.hidden{display:none !important;}

.group{
  border:1px solid var(--line);
  border-radius:18px;
  background:#ffffff;
  box-shadow:var(--shadow);
  padding:14px;
  margin:16px 0;
}
.grouphead{display:flex; justify-content:space-between; align-items:center; gap:10px; margin-bottom:10px}
.grouphead h2{margin:0; font-size:20px}

.grid{display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:12px}
.card{border:1px solid rgba(15,23,42,.12); border-radius:16px; padding:12px 14px; background:#ffffff}
.card:hover{border-color:rgba(15,23,42,.24)}
.match{font-weight:750; font-size:14px; margin-bottom:6px}
.teamline{display:flex;align-items:center;gap:7px;margin-top:4px;font-size:12px;color:var(--muted)}
.teamlogo{width:18px;height:18px;object-fit:contain;border-radius:50%;background:rgba(255,255,255,.9);padding:1px}
.teamfallback{display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;border-radius:50%;font-size:10px;background:#e5e7eb;color:#374151}
.logo{display:inline-flex;align-items:center}
.row{display:flex; gap:10px; flex-wrap:wrap; align-items:center}
.pill{display:inline-flex; align-items:center; gap:6px; padding:3px 10px; border:1px solid rgba(15,23,42,.14); border-radius:999px; font-size:12px; background:#f3f4f6}
.kv{font-size:13px}
.kv b{font-weight:800}
.books{font-size:12px; color:var(--muted); margin-top:8px}
.badge{display:inline-flex; align-items:center; gap:6px; padding:2px 10px; border-radius:999px; font-size:12px; background:rgba(255, 193, 7, .14); border:1px solid rgba(255, 193, 7, .24);}
.badge.ok{background:rgba(48, 209, 88, .12); border-color:rgba(48, 209, 88, .22);}
.small{font-size:12px}
.quickchips{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
.summarybar{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
.statpill{display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:999px;font-size:12px;border:1px solid rgba(15,23,42,.14);background:#f8fafc;color:#334155}
.statpill b{font-size:12px}
.chipbtn{padding:7px 10px;border-radius:999px;border:1px solid rgba(255,255,255,.16);background:rgba(255,255,255,.06);color:var(--text);font-size:12px;cursor:pointer}
.chipbtn.active{background:#e8f0ff;border-color:#b7cdfa}
.table{width:100%; border-collapse:collapse; margin-top:10px; overflow:auto;}
.table th,.table td{border-bottom:1px solid rgba(255,255,255,.10); padding:8px 10px; text-align:left; font-size:13px;}
.table th{color:#334155; font-weight:700; position:sticky; top:0; background:#f8fafc}
.tablewrap{width:100%; overflow-x:auto}
.table.wide{min-width:1780px}
.stats-table th,.stats-table td{font-size:12px}
.stats-table td.team{font-weight:700}
.stats-table td.scope{color:var(--muted)}
.stats-table{border:1px solid rgba(255,255,255,.18);border-radius:12px;overflow:hidden}
.stats-table thead th{background:#f8fafc}
.stats-table tbody tr{border-bottom:1px solid rgba(255,255,255,.12)}
.stats-table tbody tr:last-child{border-bottom:none}
.stats-table td,.stats-table th{border-right:1px solid rgba(255,255,255,.10)}
.stats-table td:last-child,.stats-table th:last-child{border-right:none}
.value-badge{display:inline-flex;align-items:center;margin-left:6px;padding:1px 6px;border-radius:999px;font-size:10px;font-weight:700;border:1px solid rgba(34,197,94,.35);color:#9ef3c2;background:rgba(34,197,94,.12)}
.value-badge.neg{border-color:rgba(239,68,68,.35);color:#fecaca;background:rgba(239,68,68,.12)}
.edge-pos{color:#9ef3c2}
.edge-neg{color:#fecaca}
.tag{display:inline-flex;align-items:center;border-radius:999px;padding:2px 8px;font-size:11px;border:1px solid rgba(255,255,255,.18);background:rgba(255,255,255,.08)}
.tag.low{background:rgba(239,68,68,.16);border-color:rgba(239,68,68,.32)}
.tag.med{background:rgba(234,179,8,.18);border-color:rgba(234,179,8,.32)}
.tag.high{background:rgba(34,197,94,.16);border-color:rgba(34,197,94,.32)}


.fixrow{cursor:pointer}
.detailrow td{padding:0;border-bottom:none}
.detailwrap{padding:12px 12px 14px 12px;border:1px solid rgba(15,23,42,.10);border-radius:16px;background:#ffffff;margin:10px 0}
.detailgrid{display:grid;grid-template-columns:repeat(auto-fit, minmax(320px, 1fr));gap:12px;margin-top:10px}
.detailbox{border:1px solid rgba(15,23,42,.12);border-radius:16px;padding:12px 14px;background:#ffffff}
.detailbox h4{margin:0 0 6px 0;font-size:14px}
.sg-wrap{margin-top:10px}
.sg-title{font-weight:700;margin:6px 0 8px}
.sg-table{width:100%;border-collapse:separate;border-spacing:6px}
.sg-table th,.sg-table td{padding:8px 10px;text-align:center;border-radius:10px;font-size:12px;border:1px solid rgba(255,255,255,0.08)}
.sg-corner{background:#f8fafc}
.sg-row,.sg-col{font-weight:700;background:#f3f4f6}
.sg-cell{background:#eef2f7}
.sg-axis{display:flex;justify-content:space-between;font-size:11px;opacity:.78;margin-top:6px}

@media (max-width: 900px){
  .sticky{position:static; backdrop-filter:none;}
  .container{padding:16px 12px 56px;}
  h1{font-size:24px;}
  .group{padding:12px; border-radius:14px;}
  .grouphead h2{font-size:18px;}
  .detailgrid{grid-template-columns:1fr;}
}

@media (max-width: 640px){
  .toolbar{padding:12px; border-radius:12px;}
  .controls{grid-template-columns:1fr;}
  .ctl{width:100%;}
  .ctl.search-ctl{grid-column:auto;}
  .tabs{gap:8px; margin-top:10px;}
  .tabbtn{font-size:12px; padding:8px 10px;}
  .grid{grid-template-columns:1fr;}
  .card,.detailbox{padding:10px 11px; border-radius:13px;}
  .match{font-size:13px; line-height:1.35;}
  .table th,.table td{padding:7px 8px; font-size:12px;}
  .table.wide{min-width:1320px}
}

""")
    parts.append("</style>")

    parts.append("</head><body><div class='container'>")

    parts.append("<div class='sticky'>")
    parts.append("<h1>Football Model Report</h1>")
    parts.append(f"<div class='muted'>Generated: {_html_escape(gen_ts)} · Showing: <span id='shownCount'>0</span></div>")

    # Controls
    parts.append("<div class='toolbar'>")
    parts.append("<div class='controls'>")

    parts.append("<div class='ctl'><label>From date</label><input id='fromDate' type='date'></div>")
    parts.append("<div class='ctl'><label>To date</label><input id='toDate' type='date'></div>")
    parts.append("<div class='ctl'><label>Range preset</label><select id='datePreset'><option value='0' selected>Today only</option><option value='2'>Next 2 days</option><option value='3'>Next 3 days</option><option value='7'>Next 7 days</option><option value='custom'>Custom</option></select></div>")

    parts.append("<div class='ctl'><label>Country</label><select id='countrySel'><option value=''>All</option>")
    for c in countries:
        parts.append(f"<option value='{_html_escape(c)}'>{_html_escape(c)}</option>")
    parts.append("</select></div>")

    parts.append("<div class='ctl'><label>League</label><select id='leagueSel'><option value=''>All</option>")
    for lg in leagues:
        parts.append(f"<option value='{_html_escape(lg)}'>{_html_escape(lg)}</option>")
    parts.append("</select></div>")

    parts.append("<div class='ctl'><label>Min EV (%)</label><input id='minEv' type='number' step='0.1' value='5.0'></div>")

    parts.append("<div class='ctl search-ctl'><label>Search</label><input id='search' placeholder='Team / league / country...'></div>")

    parts.append("<div class='ctl'><label>&nbsp;</label><button id='resetBtn'>Reset</button></div>")

    parts.append("</div>")  # controls
    parts.append("<div class='quickchips'>")
    parts.append("<button class='chipbtn' data-days='0'>Today</button>")
    parts.append("<button class='chipbtn' data-days='1'>+1 day</button>")
    parts.append("<button class='chipbtn' data-days='3'>+3 days</button>")
    parts.append("<button class='chipbtn' data-days='7'>+7 days</button>")
    parts.append("<button class='chipbtn' data-ev='3'>EV 3%+</button>")
    parts.append("<button class='chipbtn active' data-ev='5'>EV 5%+</button>")
    parts.append("<button class='chipbtn' data-ev='8'>EV 8%+</button>")
    parts.append("</div>")
    parts.append("<div class='muted small' style='margin-top:8px'>Tip: on mobile, swipe tab buttons and wide tables horizontally.</div>")

    # Tabs
    parts.append("<div class='tabs'>")
    parts.append("<button class='tabbtn active' data-tab='grouped'>Value Bets (Grouped)</button>")
    parts.append("<button class='tabbtn' data-tab='time'>Bets by Time</button>")
    parts.append("<button class='tabbtn' data-tab='fixtures'>All Fixtures (xG)</button>")
    parts.append("<button class='tabbtn' data-tab='totals'>O/U edges</button>")
    parts.append("</div>")

    parts.append("<div class='summarybar'>")
    parts.append("<span class='statpill'>Grouped: <b id='countGrouped'>0</b></span>")
    parts.append("<span class='statpill'>By time: <b id='countTime'>0</b></span>")
    parts.append("<span class='statpill'>O/U: <b id='countTotals'>0</b></span>")
    parts.append("<span class='statpill'>Fixtures: <b id='countFixtures'>0</b></span>")
    parts.append("</div>")

    parts.append("</div>")  # toolbar
    parts.append("</div>")  # sticky

    # --- Best bets (shown in grouped tab only) ---
    parts.append("<div id='tab_grouped' class='section'>")
    if best_bets:
        parts.append("<div class='group'>")
        parts.append("<div class='grouphead'><h2>Best Bets</h2><div class='muted small'>Market ≤ 2.00 & EV ≥ 10%</div></div>")
        parts.append("<div class='grid'>")
        for b in best_bets[:12]:
            ko = (b.get('kickoff','') or '').strip()
            ko_txt = f" {ko}" if ko else ""
            books_total = int(b.get('books_total', b.get('books', 0)) or 0)
            badge = "<span class='badge ok'>Books: %d</span>"%books_total if books_total>=2 else "<span class='badge'>Low books: %d</span>"%books_total
            parts.append(
                f"<div class='card betcard' data-kind='grouped' data-best='1' data-date='{_html_escape(b.get('date',''))}' data-country='{_html_escape(b.get('country',''))}' data-league='{_html_escape(b.get('league_name',''))}' data-ev='{float(b.get('ev',0) or 0):.3f}' data-search='{_html_escape((str(b.get('home',''))+' '+str(b.get('away',''))+' '+str(b.get('league_name',''))+' '+str(b.get('country',''))).lower())}'>"
            )
            parts.append(f"<div class='match'>{_html_escape(b.get('date',''))}{ko_txt}</div>")
            parts.append(_teamline_html(b.get('home',''), b.get('away','')))
            parts.append("<div class='row'>")
            parts.append(f"<span class='pill'>{_html_escape(b.get('bet_label',''))}</span>")
            parts.append(f"<span class='kv'>Market(median): <b>{float(b.get('mkt_median', b.get('odds',0)) or 0):.2f}</b></span>")
            if 'best_odds' in b:
                parts.append(f"<span class='kv'>Best: <b>{float(b.get('best_odds',0) or 0):.2f}</b></span>")
            parts.append(f"<span class='kv'>EV: <b>{float(b.get('ev',0) or 0):+.1f}%</b></span>")
            parts.append(f"{badge}")
            parts.append("</div>")
            parts.append(f"<div class='muted small' style='margin-top:8px'>Fair 1X2: H {float(b.get('fair_odds_h',0) or 0):.2f} · D {float(b.get('fair_odds_d',0) or 0):.2f} · A {float(b.get('fair_odds_a',0) or 0):.2f}</div>")
            parts.append(f"<div class='muted small' style='margin-top:4px'>Bookie 1X2: H {float(b.get('mkt_odds_h',0) or 0):.2f} · D {float(b.get('mkt_odds_d',0) or 0):.2f} · A {float(b.get('mkt_odds_a',0) or 0):.2f}</div>")
            parts.append(_format_edge_line(b))
            parts.append(f"<div class='muted small' style='margin-top:6px'>xG: {_html_escape(b.get('home',''))} <b>{float(b.get('h_xg',0) or 0):.2f}</b> v <b>{float(b.get('a_xg',0) or 0):.2f}</b> {_html_escape(b.get('away',''))}</div>")
            books = b.get('top_books', []) or []
            if books:
                parts.append("<div class='books'>Best books: " + ", ".join([f"{_html_escape(k)} {float(v):.2f}" for k,v in books]) + "</div>")
            parts.append("</div>")
        parts.append("</div></div>")

    # Grouped by country -> league
    for country in sorted(by_country.keys()):
        parts.append("<div class='group'>")
        total_c = sum(len(v) for v in by_country[country].values())
        parts.append(f"<div class='grouphead'><h2>{_html_escape(country)}</h2><div class='muted small'>{total_c} bets</div></div>")
        for league_name in sorted(by_country[country].keys()):
            parts.append(f"<h3>{_html_escape(league_name)}</h3>")
            parts.append("<div class='grid'>")
            for b in by_country[country][league_name]:
                ko = (b.get('kickoff','') or '').strip()
                ko_txt = f" {ko}" if ko else ""
                books_total = int(b.get('books_total', b.get('books', 0)) or 0)
                badge = "<span class='badge ok'>Books: %d</span>"%books_total if books_total>=2 else "<span class='badge'>Low books: %d</span>"%books_total
                parts.append(
                    f"<div class='card betcard' data-kind='grouped' data-best='0' data-date='{_html_escape(b.get('date',''))}' data-country='{_html_escape(country)}' data-league='{_html_escape(league_name)}' data-ev='{float(b.get('ev',0) or 0):.3f}' data-search='{_html_escape((str(b.get('home',''))+' '+str(b.get('away',''))+' '+str(league_name)+' '+str(country)).lower())}'>"
                )
                parts.append(f"<div class='match'>{_html_escape(b.get('date',''))}{ko_txt}</div>")
                parts.append(_teamline_html(b.get('home',''), b.get('away','')))
                parts.append("<div class='row'>")
                parts.append(f"<span class='pill'>{_html_escape(b.get('bet_label',''))}</span>")
                parts.append(f"<span class='kv'>Market(median): <b>{float(b.get('mkt_median', b.get('odds',0)) or 0):.2f}</b></span>")
                if 'best_odds' in b:
                    parts.append(f"<span class='kv'>Best: <b>{float(b.get('best_odds',0) or 0):.2f}</b></span>")
                parts.append(f"<span class='kv'>EV: <b>{float(b.get('ev',0) or 0):+.1f}%</b></span>")
                parts.append(f"{badge}")
                parts.append("</div>")
                parts.append(f"<div class='muted small' style='margin-top:8px'>Fair 1X2: H {float(b.get('fair_odds_h',0) or 0):.2f} · D {float(b.get('fair_odds_d',0) or 0):.2f} · A {float(b.get('fair_odds_a',0) or 0):.2f}</div>")
                parts.append(f"<div class='muted small' style='margin-top:4px'>Bookie 1X2: H {float(b.get('mkt_odds_h',0) or 0):.2f} · D {float(b.get('mkt_odds_d',0) or 0):.2f} · A {float(b.get('mkt_odds_a',0) or 0):.2f}</div>")
                parts.append(_format_edge_line(b))
                parts.append(f"<div class='muted small' style='margin-top:6px'>xG: {_html_escape(b.get('home',''))} <b>{float(b.get('h_xg',0) or 0):.2f}</b> v <b>{float(b.get('a_xg',0) or 0):.2f}</b> {_html_escape(b.get('away',''))}</div>")
                books = b.get('top_books', []) or []
                if books:
                    parts.append("<div class='books'>Best books: " + ", ".join([f"{_html_escape(k)} {float(v):.2f}" for k,v in books]) + "</div>")
                parts.append("</div>")
            parts.append("</div>")
        parts.append("</div>")
    parts.append("</div>")  # tab_grouped

    # --- Bets by time tab ---
    parts.append("<div id='tab_time' class='section hidden'>")
    parts.append("<div class='group'>")
    parts.append("<div class='grouphead'><h2>All Value Bets by Kickoff Time</h2><div class='muted small'>Sorted by date → kickoff</div></div>")
    parts.append("<div class='grid'>")
    for b in bets_time_sorted:
        ko = (b.get('kickoff','') or '').strip()
        ko_txt = f" {ko}" if ko else ""
        books_total = int(b.get('books_total', b.get('books', 0)) or 0)
        badge = "<span class='badge ok'>Books: %d</span>"%books_total if books_total>=2 else "<span class='badge'>Low books: %d</span>"%books_total
        parts.append(
            f"<div class='card betcard' data-kind='time' data-best='0' data-date='{_html_escape(b.get('date',''))}' data-country='{_html_escape(b.get('country',''))}' data-league='{_html_escape(b.get('league_name',''))}' data-ev='{float(b.get('ev',0) or 0):.3f}' data-search='{_html_escape((str(b.get('home',''))+' '+str(b.get('away',''))+' '+str(b.get('league_name',''))+' '+str(b.get('country',''))).lower())}'>"
        )
        parts.append(f"<div class='match'>{_html_escape(b.get('date',''))}{ko_txt}</div>")
        parts.append(_teamline_html(b.get('home',''), b.get('away','')))
        parts.append("<div class='row'>")
        parts.append(f"<span class='pill'>{_html_escape(b.get('bet_label',''))}</span>")
        parts.append(f"<span class='pill'>{_html_escape(b.get('league_name',''))}</span>")
        parts.append(f"<span class='kv'>Market(median): <b>{float(b.get('mkt_median', b.get('odds',0)) or 0):.2f}</b></span>")
        if 'best_odds' in b:
            parts.append(f"<span class='kv'>Best: <b>{float(b.get('best_odds',0) or 0):.2f}</b></span>")
        parts.append(f"<span class='kv'>EV: <b>{float(b.get('ev',0) or 0):+.1f}%</b></span>")
        parts.append(f"{badge}")
        parts.append("</div>")
        parts.append(f"<div class='muted small' style='margin-top:8px'>Fair 1X2: H {float(b.get('fair_odds_h',0) or 0):.2f} · D {float(b.get('fair_odds_d',0) or 0):.2f} · A {float(b.get('fair_odds_a',0) or 0):.2f}</div>")
        parts.append(f"<div class='muted small' style='margin-top:4px'>Bookie 1X2: H {float(b.get('mkt_odds_h',0) or 0):.2f} · D {float(b.get('mkt_odds_d',0) or 0):.2f} · A {float(b.get('mkt_odds_a',0) or 0):.2f}</div>")
        parts.append(_format_edge_line(b))
        parts.append(f"<div class='muted small' style='margin-top:6px'>xG: {_html_escape(b.get('home',''))} <b>{float(b.get('h_xg',0) or 0):.2f}</b> v <b>{float(b.get('a_xg',0) or 0):.2f}</b> {_html_escape(b.get('away',''))}</div>")
        books = b.get('top_books', []) or []
        if books:
            parts.append("<div class='books'>Best books: " + ", ".join([f"{_html_escape(k)} {float(v):.2f}" for k,v in books]) + "</div>")
        parts.append("</div>")
    parts.append("</div></div></div>")

    # --- O/U opportunities tab ---
    parts.append("<div id='tab_totals' class='section hidden'>")
    parts.append("<div class='group'>")
    parts.append("<div class='grouphead'><h2>Bookie vs Model O/U edges</h2><div class='muted small'>Best edge across available totals lines from football-data columns</div></div>")
    if ou_opps:
        parts.append("<div class='grid'>")
        for o in ou_opps:
            ko = (o.get('kickoff','') or '').strip()
            ko_txt = f" {ko}" if ko else ""
            parts.append(
                f"<div class='card' data-kind='totals' data-date='{_html_escape(o.get('date',''))}' data-country='{_html_escape(o.get('country',''))}' data-league='{_html_escape(o.get('league_name',''))}' data-ev='{float(o.get('edge',0) or 0):.3f}' data-search='{_html_escape((str(o.get('home',''))+' '+str(o.get('away',''))+' '+str(o.get('league_name',''))+' '+str(o.get('country',''))).lower())}'>"
            )
            parts.append(f"<div class='match'>{_html_escape(o.get('date',''))}{ko_txt}</div>")
            parts.append(_teamline_html(o.get('home',''), o.get('away','')))
            parts.append(f"<div class='row' style='margin-top:6px'><span class='pill'>{_html_escape(o.get('best_side',''))}</span><span class='kv'>Edge: <b>{float(o.get('edge',0) or 0):+.1f}%</b></span></div>")
            parts.append(f"<div class='muted small' style='margin-top:8px'>Model fair O/U line {float(o.get('line',2.5) or 2.5):.2f}: Over {float(o.get('fair_over',0) or 0):.2f} · Under {float(o.get('fair_under',0) or 0):.2f}</div>")
            parts.append(f"<div class='muted small' style='margin-top:4px'>Market O/U line {float(o.get('line',2.5) or 2.5):.2f}: Over {float(o.get('mk_over',0) or 0):.2f} · Under {float(o.get('mk_under',0) or 0):.2f}</div>")
            parts.append(f"<div class='muted small' style='margin-top:4px'>O EV: {float(o.get('over_ev',0) or 0):+.1f}% · U EV: {float(o.get('under_ev',0) or 0):+.1f}%</div>")
            parts.append(f"<div class='books'>{_html_escape(o.get('tip',''))}</div>")
            parts.append("</div>")
        parts.append("</div>")
    else:
        parts.append(f"<div class='muted'>No O/U opportunities found above edge threshold. Fixtures with totals market: {fixtures_with_totals_market}.</div>")
    parts.append("</div></div>")

    # --- Fixtures tab ---
    parts.append("<div id='tab_fixtures' class='section hidden'>")
    parts.append("<div class='group'>")
    parts.append("<div class='grouphead'><h2>All Fixtures</h2><div class='muted small'>Model xG (drives fair odds), Fair 1X2, Market 1X2</div></div>")

    if fixtures_sorted:
        parts.append("<div class='tablewrap'>")
        parts.append("<table class='table wide'>")
        parts.append("<thead><tr><th>Date</th><th>KO</th><th>League</th><th>Match</th><th>xG</th><th>Fair H</th><th>Fair D</th><th>Fair A</th><th>Mkt H</th><th>Mkt D</th><th>Mkt A</th><th>ΔP H</th><th>ΔP D</th><th>ΔP A</th><th>Mkt O/R</th><th>Conf</th><th>Form L5</th><th>xG Δ</th><th>xG Total</th></tr></thead><tbody>")
        for f in fixtures_sorted:
            ko = (f.get('kickoff','') or '').strip()
            hxg = float(f.get('h_xg',0) or 0)
            axg = float(f.get('a_xg',0) or 0)

            fair_h = float(f.get('fair_odds_h',0) or 0)
            fair_d = float(f.get('fair_odds_d',0) or 0)
            fair_a = float(f.get('fair_odds_a',0) or 0)
            mkt_h = float(f.get('mkt_odds_h',0) or 0)
            mkt_d = float(f.get('mkt_odds_d',0) or 0)
            mkt_a = float(f.get('mkt_odds_a',0) or 0)

            def _ev(mkt, fair):
                try:
                    mkt=float(mkt)
                    fair=float(fair)
                    if mkt>0 and fair>0:
                        return (mkt/fair) - 1.0
                except Exception:
                    pass
                return 0.0

            ev_h = _ev(mkt_h, fair_h)
            ev_d = _ev(mkt_d, fair_d)
            ev_a = _ev(mkt_a, fair_a)

            # biggest absolute discrepancy across 1X2 for this fixture
            best_key, best_ev = max([('H', ev_h), ('D', ev_d), ('A', ev_a)], key=lambda t: abs(t[1]))

            def _cell_style(ev):
                # Color scale: green positive (market bigger than fair), red negative.
                # White/transparent for small differences.
                try:
                    ev=float(ev)
                except Exception:
                    return ''
                if abs(ev) < 0.02:
                    return ''
                cap = 0.25
                intensity = min(abs(ev)/cap, 1.0)
                alpha = 0.10 + 0.35*intensity
                if ev > 0:
                    return f"background-color: rgba(34,197,94,{alpha:.3f});"
                else:
                    return f"background-color: rgba(239,68,68,{alpha:.3f});"

            def _td_market(val, ev, key):
                cls = 'bestedge' if key == best_key else ''
                style = _cell_style(ev)
                style_attr = f" style='{style}'" if style else ''
                cls_attr = f" class='{cls}'" if cls else ''
                return f"<td{cls_attr}{style_attr}>{val:.2f}</td>"

            def _overround():
                odds = [mkt_h, mkt_d, mkt_a]
                inv_sum = 0.0
                for o in odds:
                    try:
                        o = float(o)
                        if o > 0:
                            inv_sum += 1.0 / o
                    except Exception:
                        continue
                if inv_sum <= 0:
                    return "-"
                return f"{inv_sum*100:.1f}%"

            conf_label, conf_cls, conf_score = _confidence_meta(f)
            form = (f.get('recent_form') or {})
            form_h = (form.get('home') or {})
            form_a = (form.get('away') or {})
            if form_h.get('n') and form_a.get('n'):
                form_txt = f"{int(form_h.get('w',0))}-{int(form_h.get('d',0))}-{int(form_h.get('l',0))} | {int(form_a.get('w',0))}-{int(form_a.get('d',0))}-{int(form_a.get('l',0))}"
            else:
                form_txt = "-"

            def _implied_prob(odds):
                try:
                    o = float(odds)
                    return 1.0 / o if o > 0 else 0.0
                except Exception:
                    return 0.0

            prob_h = float(f.get('prob_h', 0) or 0)
            prob_d = float(f.get('prob_d', 0) or 0)
            prob_a = float(f.get('prob_a', 0) or 0)
            prob_raw_h = float(f.get('prob_raw_h', 0) or 0)
            prob_raw_d = float(f.get('prob_raw_d', 0) or 0)
            prob_raw_a = float(f.get('prob_raw_a', 0) or 0)
            imp_h = _implied_prob(mkt_h)
            imp_d = _implied_prob(mkt_d)
            imp_a = _implied_prob(mkt_a)

            def _prob_diff_cell(model_p, implied_p):
                diff = (model_p - implied_p) * 100.0
                cls = "edge-pos" if diff >= 0 else "edge-neg"
                return f"<td class='{cls}'>{diff:+.1f}pp</td>"

            parts.append(
                f"<tr class='fixrow' data-kind='fixtures' data-date='{_html_escape(f.get('date',''))}' data-country='{_html_escape(f.get('country',''))}' data-league='{_html_escape(f.get('league_name',''))}' data-ev='0' data-search='{_html_escape((str(f.get('home',''))+' '+str(f.get('away',''))+' '+str(f.get('league_name',''))+' '+str(f.get('country',''))).lower())}'>"
                f"<td>{_html_escape(f.get('date',''))}</td>"
                f"<td>{_html_escape(ko)}</td>"
                f"<td>{_html_escape(f.get('league_name',''))}</td>"
                f"<td>{_html_escape(f.get('home',''))} vs {_html_escape(f.get('away',''))}</td>"
                f"<td>{hxg:.2f} - {axg:.2f}</td>"
                f"<td>{fair_h:.2f}</td>"
                f"<td>{fair_d:.2f}</td>"
                f"<td>{fair_a:.2f}</td>"
                + _td_market(mkt_h, ev_h, 'H')
                + _td_market(mkt_d, ev_d, 'D')
                + _td_market(mkt_a, ev_a, 'A')
                + _prob_diff_cell(prob_h, imp_h)
                + _prob_diff_cell(prob_d, imp_d)
                + _prob_diff_cell(prob_a, imp_a)
                + f"<td>{_overround()}</td>"
                + f"<td>{conf_label}</td>"
                + f"<td>{form_txt}</td>"
                + f"<td>{(hxg-axg):+.2f}</td>"
                + f"<td>{(hxg+axg):.2f}</td>"
                + "</tr>"
            )


            # Expandable details for this fixture
            # Build probability matrix from xG (rho=0 by default)
            mat = poisson_matrix_dc(hxg, axg, rho=0.0, max_g=11)
            totp = _total_probs_from_score_matrix(mat)
            diffp = _diff_probs_from_score_matrix(mat)

            # BTTS
            btts_yes_p, btts_no_p = btts_probs(mat)
            btts_yes_odds = _safe_odds_from_prob(btts_yes_p)
            btts_no_odds = _safe_odds_from_prob(btts_no_p)

            # Top scorelines + score grid (0–3)
            top5 = top_scorelines(mat, k=5)
            top5_html = " ".join([f"<span class='pill'>{h}-{a} {pr*100:.1f}%</span>" for h,a,pr in top5])

            # Heatmap grid 0-3
            grid_max = max(float(mat[h,a]) for h in range(0,4) for a in range(0,4))
            grid_max = grid_max if grid_max>0 else 1e-9
            sg_head = "<tr><th class='sg-corner'>H\\A</th>" + "".join([f"<th class='sg-col'>{a}</th>" for a in range(0,4)]) + "</tr>"
            sg_rows = []
            for h in range(0,4):
                tds = [f"<th class='sg-row'>{h}</th>"]
                for a in range(0,4):
                    pr = float(mat[h,a])
                    inten = max(0.0, min(1.0, pr / grid_max))
                    alpha = 0.06 + 0.28*inten
                    tds.append(f"<td class='sg-cell' style='background: rgba(255,255,255,{alpha:.3f});'>{pr*100:.1f}%</td>")
                sg_rows.append("<tr>" + "".join(tds) + "</tr>")
            score_grid_html = (
                "<div class='sg-wrap'>"
                "<div class='sg-title'>Score probabilities (0–3)</div>"
                "<table class='table sg-table'><thead>" + sg_head + "</thead><tbody>" + "".join(sg_rows) + "</tbody></table>"
                "<div class='sg-axis'><span><b>Home goals</b> ↓</span><span><b>Away goals</b> →</span></div>"
                "</div>"
            )

            # Result probabilities + most likely
            pH, pD, pA = most_likely_result_probs(mat)
            best_res = max([('H', pH), ('D', pD), ('A', pA)], key=lambda x: x[1])
            mh, ma, mp = top5[0]

            # Winning margin buckets
            pm_h1, pm_h2p, pm_d, pm_a1, pm_a2p = margin_buckets(mat)
            margin_lines = (
                f"Margins: Home by 1 {pm_h1*100:.1f}% · Home by 2+ {pm_h2p*100:.1f}% · "
                f"Draw {pm_d*100:.1f}% · Away by 1 {pm_a1*100:.1f}% · Away by 2+ {pm_a2p*100:.1f}%"
            )

            # Asian lines
            tot_lines = [2.00,2.25,2.50,2.75,3.00,3.25,3.50]
            ah_lines  = [+1.50,+1.25,+1.00,+0.75,+0.50,+0.25,0.00,-0.25,-0.50,-0.75,-1.00,-1.25,-1.50]

            def _edge_cell(fair_odds, mkt_odds):
                # green if market bigger than fair (positive edge), red if smaller
                try:
                    fair_odds=float(fair_odds)
                    mkt_odds=float(mkt_odds)
                except Exception:
                    return '', 0.0
                if fair_odds<=0 or mkt_odds<=0:
                    return '', 0.0
                ev = (mkt_odds/fair_odds) - 1.0
                if abs(ev) < 0.02:
                    return '', ev
                cap=0.25
                inten=min(abs(ev)/cap,1.0)
                alpha=0.10 + 0.35*inten
                if ev>0:
                    return f"background-color: rgba(34,197,94,{alpha:.3f});", ev
                return f"background-color: rgba(239,68,68,{alpha:.3f});", ev

            # Optional market totals/AH odds dicts (if present)
            mkt_totals = f.get('mkt_totals') or {}
            mkt_ah = f.get('mkt_ah') or {}
            mkt_btts = f.get('mkt_btts') or {}

            tot_rows=[]
            best_tot=None
            best_tot_gap=1e9
            for L in tot_lines:
                over, under = fair_totals_ou(totp, L)
                mk_over = mkt_totals.get(f"{L}_over") or mkt_totals.get(f"{L:.2f}_over")
                mk_under = mkt_totals.get(f"{L}_under") or mkt_totals.get(f"{L:.2f}_under")
                st_o, ev_o = _edge_cell(over, mk_over) if mk_over else ('',0)
                st_u, ev_u = _edge_cell(under, mk_under) if mk_under else ('',0)
                badge_o = " <span class='value-badge'>VALUE</span>" if ev_o > 0.02 else ""
                badge_u = " <span class='value-badge'>VALUE</span>" if ev_u > 0.02 else ""
                tot_rows.append(
                    "<tr>"
                    f"<td>O/U {L:.2f}</td>"
                    f"<td style='{st_o}'>{over:.2f}{badge_o}</td>"
                    f"<td style='{st_u}'>{under:.2f}{badge_u}</td>"
                    "</tr>"
                )
                # recommendation: closest to 50/50 (win vs lose)
                W,P,Lose = settle_total_over(totp, L)
                gap = abs(W - Lose)
                if gap < best_tot_gap:
                    best_tot_gap = gap
                    best_tot = L

            ah_rows=[]
            best_ah=None
            best_ah_gap=1e9
            for L in ah_lines:
                home_o, away_o = fair_handicap_home_away(diffp, L)
                mk_h = mkt_ah.get(f"{L}_home") or mkt_ah.get(f"{L:+.2f}_home")
                mk_a = mkt_ah.get(f"{L}_away") or mkt_ah.get(f"{L:+.2f}_away")
                st_h, ev_hh = _edge_cell(home_o, mk_h) if mk_h else ('',0)
                st_a, ev_aa = _edge_cell(away_o, mk_a) if mk_a else ('',0)
                badge_h = " <span class='value-badge'>VALUE</span>" if ev_hh > 0.02 else ""
                badge_a = " <span class='value-badge'>VALUE</span>" if ev_aa > 0.02 else ""
                ah_rows.append(
                    "<tr>"
                    f"<td>AH {L:+.2f}</td>"
                    f"<td style='{st_h}'>{home_o:.2f}{badge_h}</td>"
                    f"<td style='{st_a}'>{away_o:.2f}{badge_a}</td>"
                    "</tr>"
                )
                W,P,Lose = settle_handicap_home(diffp, L)
                gap = abs(W - Lose)
                if gap < best_ah_gap:
                    best_ah_gap = gap
                    best_ah = L

            rec_tot_ov, rec_tot_un = fair_totals_ou(totp, best_tot)
            rec_h, rec_a = fair_handicap_home_away(diffp, best_ah)
            rec_totals = f"Rec O/U line: {best_tot:.2f} (Over {rec_tot_ov:.2f} · Under {rec_tot_un:.2f})"
            rec_ah = f"Rec AH line: {best_ah:+.2f} (Home {rec_h:.2f} · Away {rec_a:.2f})"

            btts_market_yes = mkt_btts.get('yes') or mkt_btts.get('Yes')
            btts_market_no = mkt_btts.get('no') or mkt_btts.get('No')
            btts_yes_badge = ""
            btts_no_badge = ""
            if btts_market_yes:
                _, btts_yes_ev = _edge_cell(btts_yes_odds, btts_market_yes)
                if btts_yes_ev > 0.02:
                    btts_yes_badge = " <span class='value-badge'>VALUE</span>"
            if btts_market_no:
                _, btts_no_ev = _edge_cell(btts_no_odds, btts_market_no)
                if btts_no_ev > 0.02:
                    btts_no_badge = " <span class='value-badge'>VALUE</span>"

            details_html = (
                "<div class='detailwrap'>"
                f"<div class='small muted'>Model probs (cal): H {prob_h*100:.1f}% · D {prob_d*100:.1f}% · A {prob_a*100:.1f}%"
                f" &nbsp;|&nbsp; Most likely result (xG grid): {best_res[0]} ({best_res[1]*100:.1f}%)"
                f" &nbsp;|&nbsp; Most likely score: {mh}-{ma} ({mp*100:.1f}%)"
                f" &nbsp;|&nbsp; Market overround: {_overround()}"
                f" &nbsp;|&nbsp; Confidence: <span class='tag {conf_cls}'>{conf_label}</span>"
                "</div>"
                f"<div class='small muted' style='margin-top:8px'>{margin_lines}</div>"
                f"<div class='small muted' style='margin-top:4px'>{rec_totals}</div>"
                f"<div class='small muted' style='margin-top:4px'>{rec_ah}</div>"
                "<div class='detailgrid'>"
                "<div class='detailbox'>"
                + _render_season_stats_table(f)
                + "</div>"
                "<div class='detailbox'>"
                + _render_recent_form_table(f)
                + "</div>"
                "<div class='detailbox'>"
                "<h4>Implied vs Model (1X2)</h4>"
                "<table class='table stats-table'>"
                "<thead><tr><th>Outcome</th><th>Model %</th><th>Implied %</th><th>ΔP</th></tr></thead><tbody>"
                f"<tr><td>Home</td><td>{prob_h*100:.1f}%</td><td>{imp_h*100:.1f}%</td><td class='{'edge-pos' if (prob_h-imp_h)>=0 else 'edge-neg'}'>{(prob_h-imp_h)*100:+.1f}pp</td></tr>"
                f"<tr><td>Draw</td><td>{prob_d*100:.1f}%</td><td>{imp_d*100:.1f}%</td><td class='{'edge-pos' if (prob_d-imp_d)>=0 else 'edge-neg'}'>{(prob_d-imp_d)*100:+.1f}pp</td></tr>"
                f"<tr><td>Away</td><td>{prob_a*100:.1f}%</td><td>{imp_a*100:.1f}%</td><td class='{'edge-pos' if (prob_a-imp_a)>=0 else 'edge-neg'}'>{(prob_a-imp_a)*100:+.1f}pp</td></tr>"
                "</tbody></table>"
                "</div>"
                "<div class='detailbox'>"
                "<h4>BTTS (Fair Odds)</h4>"
                f"<div class='small'>Yes: <b>{btts_yes_odds:.2f}</b> ({btts_yes_p*100:.1f}%) {btts_yes_badge} · "
                f"No: <b>{btts_no_odds:.2f}</b> ({btts_no_p*100:.1f}%) {btts_no_badge}</div>"
                "<h4 style='margin-top:10px'>Top scorelines</h4>"
                f"<div>{top5_html}</div>"
                + score_grid_html +
                "</div>"
                "<div class='detailbox'>"
                "<h4>Asian Totals (Fair Odds)</h4>"
                "<table class='table'><thead><tr><th>Line</th><th>Over</th><th>Under</th></tr></thead><tbody>" + "".join(tot_rows) + "</tbody></table>"
                "</div>"
                "<div class='detailbox'>"
                "<h4>Asian Handicap (Fair Odds)</h4>"
                "<table class='table'><thead><tr><th>Line</th><th>Home</th><th>Away</th></tr></thead><tbody>" + "".join(ah_rows) + "</tbody></table>"
                "</div>"
                "</div>"
                "</div>"
            )

            parts.append(f"<tr class='detailrow' style='display:none'><td colspan='22'>{details_html}</td></tr>")
        parts.append("</tbody></table></div>")
    else:
        parts.append("<div class='muted'>No fixture predictions available.</div>")

    parts.append("</div></div>")

    # JS: tabs + filters
    parts.append("<script>")
    parts.append(r"""
function todayISO(){
  const d = new Date();
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth()+1).padStart(2,'0');
  const dd = String(d.getDate()).padStart(2,'0');
  return `${yyyy}-${mm}-${dd}`;
}

const fromDate = document.getElementById('fromDate');
const toDate = document.getElementById('toDate');
const datePreset = document.getElementById('datePreset');
const countrySel = document.getElementById('countrySel');
const leagueSel = document.getElementById('leagueSel');
const minEv = document.getElementById('minEv');
const search = document.getElementById('search');
const shownCount = document.getElementById('shownCount');
const countGrouped = document.getElementById('countGrouped');
const countTime = document.getElementById('countTime');
const countTotals = document.getElementById('countTotals');
const countFixtures = document.getElementById('countFixtures');
const chipButtons = Array.from(document.querySelectorAll('.chipbtn'));

let activeTab = 'grouped';

// default dates to today
const t = todayISO();
if(fromDate && !fromDate.value) fromDate.value = t;
if(toDate && !toDate.value) toDate.value = t;


function addDaysISO(iso, n){
  const d = parseDate(iso) || new Date();
  d.setDate(d.getDate()+n);
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth()+1).padStart(2,'0');
  const dd = String(d.getDate()).padStart(2,'0');
  return `${yyyy}-${mm}-${dd}`;
}


function setDatePreset(v){
  if(!fromDate || !toDate) return;
  if(v === 'custom') return;
  const n = parseInt(v || '0', 10) || 0;
  const t0 = todayISO();
  fromDate.value = t0;
  toDate.value = addDaysISO(t0, n);
}

if(datePreset){
  setDatePreset(datePreset.value || '0');
  datePreset.addEventListener('change', ()=>{
    setDatePreset(datePreset.value || '0');
    applyFilters();
  });
}

for(const chip of chipButtons){
  chip.addEventListener('click', ()=>{
    if(chip.dataset.days !== undefined){
      const base = todayISO();
      const days = parseInt(chip.dataset.days || '0', 10) || 0;
      if(fromDate) fromDate.value = base;
      if(toDate) toDate.value = addDaysISO(base, days);
      for(const c of chipButtons.filter(x=>x.dataset.days!==undefined)){ c.classList.remove('active'); }
      chip.classList.add('active');
    }
    if(chip.dataset.ev !== undefined){
      if(minEv) minEv.value = chip.dataset.ev;
      for(const c of chipButtons.filter(x=>x.dataset.ev!==undefined)){ c.classList.remove('active'); }
      chip.classList.add('active');
    }
    applyFilters();
  });
}

function parseDate(s){
  if(!s) return null;
  const d = new Date(s+'T00:00:00');
  return isNaN(d.getTime()) ? null : d;
}

function setTab(tab){
  activeTab = tab;
  document.getElementById('tab_grouped').classList.toggle('hidden', tab !== 'grouped');
  document.getElementById('tab_time').classList.toggle('hidden', tab !== 'time');
  document.getElementById('tab_fixtures').classList.toggle('hidden', tab !== 'fixtures');
  document.getElementById('tab_totals').classList.toggle('hidden', tab !== 'totals');

  for(const btn of document.querySelectorAll('.tabbtn')){
    btn.classList.toggle('active', btn.dataset.tab === tab);
  }

  // Min EV only relevant for bets tabs
  if(minEv){
    minEv.closest('.ctl').style.display = (tab === 'fixtures') ? 'none' : '';
  }

  applyFilters();
}

for(const btn of document.querySelectorAll('.tabbtn')){
  btn.addEventListener('click', ()=> setTab(btn.dataset.tab));
}

function applyFilters(){
  const f = parseDate(fromDate ? fromDate.value : '');
  const t2 = parseDate(toDate ? toDate.value : '');
  const c = (countrySel ? countrySel.value : '').toLowerCase();
  const l = (leagueSel ? leagueSel.value : '').toLowerCase();
  const q = (search ? search.value : '').trim().toLowerCase();
  const evMin = (minEv ? (parseFloat(minEv.value || '5') || 0) : 0);

  let shown = 0;
  const kindCounts = {grouped:0, time:0, totals:0, fixtures:0};

  // collapse all fixture detail rows whenever filters change
  for(const dr of document.querySelectorAll('tr.detailrow')){ dr.style.display = 'none'; }

  const nodes = Array.from(document.querySelectorAll('[data-kind]'));
  for(const el of nodes){
    const kind = el.dataset.kind;
    if(kind !== activeTab){
      // also hide grouped/time/fixtures content not in active tab
      if(kind === 'grouped' || kind === 'time' || kind === 'fixtures'){
        el.style.display = 'none';
      }
      continue;
    }

    const d = parseDate(el.dataset.date);
    const ev = parseFloat(el.dataset.ev || '0');
    const country = (el.dataset.country || '').toLowerCase();
    const league = (el.dataset.league || '').toLowerCase();
    const hay = (el.dataset.search || '');

    let ok = true;
    if(f && d && d < f) ok = false;
    if(t2 && d && d > t2) ok = false;
    if(c && country !== c) ok = false;
    if(l && league !== l) ok = false;
    if(activeTab !== 'fixtures' && ev < evMin) ok = false;
    if(q && !hay.includes(q)) ok = false;

    el.style.display = ok ? '' : 'none';
    if(ok){
      shown++;
      if(kindCounts[kind] !== undefined) kindCounts[kind]++;
    }
  }

  if(shownCount) shownCount.textContent = String(shown);
  if(countGrouped) countGrouped.textContent = String(kindCounts.grouped);
  if(countTime) countTime.textContent = String(kindCounts.time);
  if(countTotals) countTotals.textContent = String(kindCounts.totals);
  if(countFixtures) countFixtures.textContent = String(kindCounts.fixtures);
}

for(const ctl of [fromDate, toDate, countrySel, leagueSel, minEv, search]){
  if(!ctl) continue;
  ctl.addEventListener('input', ()=>{ if((ctl===fromDate || ctl===toDate) && datePreset) datePreset.value='custom'; applyFilters(); });
  ctl.addEventListener('change', ()=>{ if((ctl===fromDate || ctl===toDate) && datePreset) datePreset.value='custom'; applyFilters(); });
}

const resetBtn = document.getElementById('resetBtn');
if(resetBtn){
  resetBtn.addEventListener('click', ()=>{
    const t0 = todayISO();
    if(fromDate) fromDate.value = t0;
    if(toDate) toDate.value = t0;
    if(datePreset) datePreset.value = '0';
    if(countrySel) countrySel.value = '';
    if(leagueSel) leagueSel.value = '';
    if(minEv) minEv.value = '5.0';
    if(search) search.value = '';
    applyFilters();
  });
}


// Toggle expandable rows on fixtures table
document.addEventListener('click', (ev)=>{
  const tr = ev.target.closest('tr.fixrow');
  if(!tr) return;
  const next = tr.nextElementSibling;
  if(next && next.classList.contains('detailrow')){
    const isHidden = (next.style.display === 'none' || next.style.display === '');
    // if currently hidden -> show, else hide
    next.style.display = (next.style.display === 'none') ? '' : 'none';
  }
});

setTab('grouped');
""")
    parts.append("</script>")

    parts.append("</div></body></html>")

    try:
        with open(logo_cache_path, 'wb') as cf:
            pickle.dump(logo_cache, cf)
    except Exception:
        pass

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(parts))

    return out_path


# =============================================================================
# MAIN
# =============================================================================

parser = argparse.ArgumentParser(description="Football Betting Model v3.0")
parser.add_argument("--refresh", action="store_true", help="Force re-download fixtures.xlsx")
parser.add_argument("--fixtures-cache", default=DEFAULT_FIXTURES_CACHE, help="Path to cached fixtures.xlsx")
parser.add_argument("--retrain", action="store_true", help="Force retraining models (ignore cached pickles)")
parser.add_argument("--backtest", action="store_true", help="Run backtests (can be slow)")
parser.add_argument("--html", action="store_true", help="Write HTML report to output/value_bets.html")
parser.add_argument("--html-out", default="output/value_bets.html", help="HTML output file")
args = parser.parse_args()

print("="*60)
print("FOOTBALL BETTING MODEL v3.0")
print("Multi-League, Multi-Season Edition")
print("="*60)

# Load all data
historical_data = load_all_historical_data()
fixtures_data = load_fixtures(cache_path=args.fixtures_cache, refresh=args.refresh)

# Train models for each league
print("\n" + "="*60)
print("TRAINING MODELS (per league)")
print("="*60)

models = {}
backtest_results = {}

for league, df in historical_data.items():
    df_done = df[df['FTR'].notna()].copy()

    # Try load cached model (fast path)
    model = None
    if not args.retrain:
        model = load_cached_model(league)
        if model is not None:
            models[league] = model
            print(f"  {league} ({get_league_name(league)}): loaded cached model")

    # Train + cache model if not loaded or forced retrain
    if model is None:
        model = LeagueModel(league)
        if model.train(df_done, verbose=True):
            models[league] = model
            save_cached_model(league, model)

    # Optional backtest
    if args.backtest and league in models:
        bt = Backtester()
        bt_res = bt.run(df_done, models[league], min_ev=get_ev_threshold(league), train_win=60)
        backtest_results[league] = bt_res

        if bt_res['n'] > 0:
            sig = "SIG" if bt_res['p'] < 0.05 else "n.s."
            print(f"    Backtest: {bt_res['n']} bets, ROI={bt_res['roi']:+.1f}%, p={bt_res['p']:.3f} [{sig}]")

# Generate predictions
print("\n" + "="*60)
print("PREDICTIONS BY LEAGUE")
print("="*60)

all_value_bets = []
all_fixture_preds = []

for league, fixtures in fixtures_data.items():
    if league not in models:
        continue

    model = models[league]
    df_hist = historical_data.get(league)
    if df_hist is None:
        continue

    df_done = df_hist[df_hist['FTR'].notna()].copy()
    current_season = ''
    season_df = df_done
    if 'Season' in df_done.columns:
        season_vals = sorted([s for s in df_done['Season'].dropna().unique() if str(s).strip() != ''])
        if season_vals:
            current_season = season_vals[-1]
            season_df = df_done[df_done['Season'] == current_season].copy()
    season_stats = build_team_season_stats(season_df)

    print(f"\n{'='*60}")
    print(f"{get_league_name(league)} ({league})")
    print(f"{'='*60}")

    for _, row in fixtures.iterrows():
        ht, at = row.get('HomeTeam'), row.get('AwayTeam')
        if pd.isna(ht) or pd.isna(at):
            continue

        date = str(row.get('Date', '')).split(' ')[0]
        # Kickoff time (fixtures.xlsx varies by column name)
        ko_raw = row.get('Time', '')
        if not ko_raw or (isinstance(ko_raw, float) and ko_raw != ko_raw):
            ko_raw = row.get('KO', '')
        if not ko_raw or (isinstance(ko_raw, float) and ko_raw != ko_raw):
            ko_raw = row.get('Kickoff', '')
        ko = str(ko_raw).strip()
        if ko.lower() in ('nan', 'nat', 'none'):
            ko = ''
        # Normalize common formats like '15:00:00' -> '15:00'
        if len(ko) >= 5 and ':' in ko:
            ko = ko[:5]


        # Collect all book odds (if present) and choose best available market price per outcome
        all_books = extract_all_book_odds(row)
        fallback_mkt = {
            'H': row.get('B365H', 0) or 0,
            'D': row.get('B365D', 0) or 0,
            'A': row.get('B365A', 0) or 0
        }
        mkt = {
            'H': best_market_price(all_books, 'H', fallback=float(fallback_mkt['H'] or 0)),
            'D': best_market_price(all_books, 'D', fallback=float(fallback_mkt['D'] or 0)),
            'A': best_market_price(all_books, 'A', fallback=float(fallback_mkt['A'] or 0)),
        }

        try:
            pred = model.predict(ht, at, df_done)
            if pred is None:
                continue

            recent_home_5 = build_team_recent_snapshot(season_df, ht, model.xg_cal, n=5)
            recent_away_5 = build_team_recent_snapshot(season_df, at, model.xg_cal, n=5)
            recent_home_10 = build_team_recent_snapshot(season_df, ht, model.xg_cal, n=10)
            recent_away_10 = build_team_recent_snapshot(season_df, at, model.xg_cal, n=10)

            # Store fixture-level prediction for the Fixtures tab
            all_fixture_preds.append({
                'date': date,
                'kickoff': ko,
                'country': model.country,
                'league': league,
                'league_name': model.name,
                'home': ht,
                'away': at,
                'h_xg': pred['h_xg'],
                'a_xg': pred['a_xg'],
                'prob_h': pred.get('probs', {}).get('H', 0),
                'prob_d': pred.get('probs', {}).get('D', 0),
                'prob_a': pred.get('probs', {}).get('A', 0),
                'prob_raw_h': pred.get('probs_raw', {}).get('H', 0),
                'prob_raw_d': pred.get('probs_raw', {}).get('D', 0),
                'prob_raw_a': pred.get('probs_raw', {}).get('A', 0),
                'fair_odds_h': pred['odds'].get('H', 0),
                'fair_odds_d': pred['odds'].get('D', 0),
                'fair_odds_a': pred['odds'].get('A', 0),
                'fair_odds_raw_h': pred.get('odds_raw', {}).get('H', 0),
                'fair_odds_raw_d': pred.get('odds_raw', {}).get('D', 0),
                'fair_odds_raw_a': pred.get('odds_raw', {}).get('A', 0),
                'mkt_odds_h': median_market_price(all_books, 'H', fallback=float(fallback_mkt['H'] or 0)),
                'mkt_odds_d': median_market_price(all_books, 'D', fallback=float(fallback_mkt['D'] or 0)),
                'mkt_odds_a': median_market_price(all_books, 'A', fallback=float(fallback_mkt['A'] or 0)),
                'season_label': current_season,
                'season_stats': {
                    'home': season_stats.get(ht, {}),
                    'away': season_stats.get(at, {}),
                },
                'recent_form': {
                    'home': recent_home_5,
                    'away': recent_away_5,
                },
                'recent_xg': {
                    'home': {'xg5': recent_home_5, 'xg10': recent_home_10},
                    'away': {'xg5': recent_away_5, 'xg10': recent_away_10},
                },
                'mkt_totals': extract_market_totals_odds(row),
                'mkt_btts': extract_market_btts_odds(row),
                'mkt_ah': extract_market_ah_odds(row),
            })

            vals = model.find_value(pred, mkt)
            print_pred(pred, mkt, vals)

            for v in vals:
                if v['value']:
                    top_books = best_books_for_outcome(all_books, v['out'], top_n=5)
                    all_value_bets.append({
                        'date': date,
                        'kickoff': ko,
                        'country': model.country,
                        'league': league,
                        'league_name': model.name,
                        'home': ht,
                        'away': at,
                        'bet': v['out'],
                        'bet_label': outcome_label(v['out']),
                        'odds': v['mkt_o'],
                        'mkt_median': median_market_price(all_books, v['out'], fallback=float(fallback_mkt[v['out']] or 0)),
                        'ev': v['ev'],
                        'fair_odds': v['model_o'],
                        'model_odds': v['model_o'],
                        'h_xg': pred['h_xg'],
                        'a_xg': pred['a_xg'],
                        'fair_odds_h': (pred.get('odds_final', pred.get('odds', {})) or {}).get('H', 0),
                        'fair_odds_d': (pred.get('odds_final', pred.get('odds', {})) or {}).get('D', 0),
                        'fair_odds_a': (pred.get('odds_final', pred.get('odds', {})) or {}).get('A', 0),
                        'mkt_odds_h': median_market_price(all_books, 'H', fallback=float(fallback_mkt['H'] or 0)),
                        'mkt_odds_d': median_market_price(all_books, 'D', fallback=float(fallback_mkt['D'] or 0)),
                        'mkt_odds_a': median_market_price(all_books, 'A', fallback=float(fallback_mkt['A'] or 0)),
                        'top_books': top_books,
                        'model_prob': float((pred.get('probs', {}) or {}).get(v['out'], 0) or 0),
                        'market_prob': (1.0 / float(v.get('mkt_o', 0) or 0)) if float(v.get('mkt_o', 0) or 0) > 0 else 0.0,
                        'edge_pp': (float((pred.get('probs', {}) or {}).get(v['out'], 0) or 0) - ((1.0 / float(v.get('mkt_o', 0) or 0)) if float(v.get('mkt_o', 0) or 0) > 0 else 0.0)) * 100.0,
                    })
        except Exception:
            pass

# Final Summary
print("\n" + "="*60)
print("ALL VALUE BETS (EV >= 5%)")
print("="*60)

if all_value_bets:
    # Sort by EV
    all_value_bets.sort(key=lambda x: -x['ev'])

    # Group by country -> league
    by_country = {}
    for vb in all_value_bets:
        c = vb.get('country', 'Unknown')
        lg = vb.get('league', '')
        by_country.setdefault(c, {}).setdefault(lg, []).append(vb)

    for country in sorted(by_country.keys()):
        print(f"\n{country}:")
        for lg in sorted(by_country[country].keys()):
            print(f"  {get_league_name(lg)} ({lg}):")
            for vb in by_country[country][lg]:
                print(
                    f"    {vb['date']} {vb['home']} vs {vb['away']}: "
                    f"{vb['bet']} @ {vb['odds']:.2f} (EV: {vb['ev']:+.1f}%) "
                    f"| Fair: {vb.get('fair_odds', vb.get('model_odds', 0)):.2f} "
                    f"| xG: {vb.get('h_xg', 0):.2f} v {vb.get('a_xg', 0):.2f}"
                )

    print(f"\nTotal value bets found: {len(all_value_bets)}")
else:
    print("No value bets found.")

# Optional HTML output (always write when requested, even if no bets)
if args.html:
    # Force the output to 'index.html' so it works with GitHub Pages
    target_path = "output/index.html"
    out_file = write_html_report(all_value_bets, all_fixture_preds, out_path=target_path)
    print(f"\nHTML report written to: {out_file}")

# Backtest Summary
print("\n" + "="*60)
print("BACKTEST SUMMARY BY LEAGUE")
print("="*60)

print(f"{'League':<8} {'Name':<25} {'Bets':>6} {'ROI':>8} {'p-value':>8} {'Sig?':<5}")
print("-"*60)

for league in sorted(backtest_results.keys()):
    res = backtest_results[league]
    if res['n'] > 0:
        sig = "YES" if res['p'] < 0.05 else "no"
        print(f"{league:<8} {get_league_name(league)[:25]:<25} {res['n']:>6} {res['roi']:>+7.1f}% {res['p']:>8.4f} {sig:<5}")

print("\n" + "="*60)
print("DONE")
print("="*60)
