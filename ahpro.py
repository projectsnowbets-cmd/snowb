import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import math

# Page configuration
st.set_page_config(page_title="Asian Handicap Predictor", page_icon="⚽", layout="wide")

# -----------------------
# Custom CSS
# -----------------------
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1.5rem;
    }
    .prediction-box {
        padding: 1.2rem;
        border-radius: 8px;
        margin: 0.8rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.06);
    }
    .high-confidence {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .medium-confidence {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    .low-confidence {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .suspicious {
        background-color: #ffe6e6;
        border-left: 5px solid #dc3545;
        animation: pulse 2s infinite;
    }
    .neutral {
        background-color: #e7f3ff;
        border-left: 5px solid #6c757d;
    }
    .ah-explain {
        border: 1px solid #ddd;
        padding: 0.8rem;
        border-radius: 6px;
        background: #fafafa;
        margin-top: 0.6rem;
    }
    .ah-explain table { width: 100%; border-collapse: collapse; }
    .ah-explain td, .ah-explain th { padding: 6px 8px; vertical-align: top; border-bottom: 1px solid #eee; }
    .ah-explain th { text-align: left; font-weight: 700; }
    .ou-explain {
        border: 1px solid #ddd;
        padding: 0.8rem;
        border-radius: 6px;
        background: #fff;
        margin-top: 0.6rem;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.75; }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⚽ Asian Handicap & Over/Under Predictor + Anomaly Detector</div>', unsafe_allow_html=True)

# -----------------------
# Session initialization
# -----------------------
if 'matches' not in st.session_state:
    st.session_state.matches = []

# -----------------------
# Sidebar: Inputs & Config
# -----------------------
with st.sidebar:
    st.header("🔍 Match Data Entry")

    # Fixed team names per user's request
    st.markdown("**Home Team:** `HOME`  \n**Away Team:** `AWAY`")
    team_home = "HOME"
    team_away = "AWAY"

    st.subheader("Asian Handicap Data")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Opening AH**")
        open_ah_line = st.number_input("Line", value=0.0, step=0.25, key="open_line", format="%.2f")
        open_ah_home = st.number_input("Home Odds", value=1.90, step=0.01, key="open_home", format="%.2f")
        open_ah_away = st.number_input("Away Odds", value=1.90, step=0.01, key="open_away", format="%.2f")
    with col2:
        st.markdown("**Pre-match AH**")
        pre_ah_line = st.number_input("Line", value=0.0, step=0.25, key="pre_line", format="%.2f")
        pre_ah_home = st.number_input("Home Odds", value=1.85, step=0.01, key="pre_home", format="%.2f")
        pre_ah_away = st.number_input("Away Odds", value=1.95, step=0.01, key="pre_away", format="%.2f")

    # Live AH moved into a collapsed expander so it's hidden by default
    with st.expander("Live AH (Optional)", expanded=False):
        live_enabled = st.checkbox("Match has started (use live odds)")
        live_ah_line = st.number_input("Line", value=0.0, step=0.25, key="live_line", format="%.2f")
        live_ah_home = st.number_input("Home Odds", value=1.80, step=0.01, key="live_home", format="%.2f")
        live_ah_away = st.number_input("Away Odds", value=2.00, step=0.01, key="live_away", format="%.2f")

    st.markdown("---")

    # Put thresholds & league inside a collapsed expander by default
    with st.expander("⚙️ Thresholds & League (advanced)", expanded=False):
        # League presets to adjust volatility / thresholds
        league = st.selectbox("League / Market profile", ["Top leagues (low vol)", "Mid leagues (med vol)", "Low leagues (high vol)"], index=1)
        # User-configurable thresholds (defaults set per user's request)
        st.markdown("Odds & line thresholds (editable):")
        reverse_line_threshold = st.number_input("Reverse correlation line shift (goals)", value=0.25, step=0.25, format="%.2f")
        reverse_odds_pct = st.number_input("Reverse correlation odds rise (%)", value=6.0, step=0.5, format="%.1f")
        violent_drop_pct = st.number_input("Violent odds drop threshold (%)", value=18.0, step=1.0, format="%.1f")
        steam_odds_pct = st.number_input("Steam odds movement (%)", value=6.0, step=0.5, format="%.1f")
        steam_time_window = st.number_input("Steam time window (min)", value=12, min_value=1, step=1)
        margin_threshold = st.number_input("Margin change threshold (%)", value=2.5, step=0.5, format="%.1f")
        override_extreme_drop = st.number_input("Override lock if live drop > (%)", value=22.0, step=1.0, format="%.1f")

        st.markdown("---")
        st.markdown("1X2 blending & nudge settings (tuneable without historical data):")
        w_model = st.slider("Weight for Poisson-model (Home/Draw/Away) in 1X2 blend", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
        w_market = st.slider("Weight for market two-way in 1X2 blend (allocates across home/away)", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
        max_nudge = st.slider("Max nudge toward suspected manipulated side (fraction of prob)", min_value=0.0, max_value=0.5, value=0.25, step=0.01)

        st.markdown("---")
        st.markdown("Override behavior when system is extremely confident:")
        override_fix_threshold = st.slider("Override 1X2 when fix probability ≥ (%)", min_value=50, max_value=99, value=90, step=1)
        override_strength_pct = st.slider("Override strength (%) - favored side probability when overriding", min_value=75, max_value=99, value=95, step=1)

    st.markdown("---")
    if st.button("🔍 Analyze Match", type="primary", use_container_width=True):
        # Validations
        def is_valid_ah_line(line):
            return abs(line * 4 - round(line * 4)) < 1e-8

        if not (is_valid_ah_line(open_ah_line) and is_valid_ah_line(pre_ah_line) and is_valid_ah_line(live_ah_line)):
            st.error("Invalid Asian Handicap line(s). Lines must be multiples of 0.25 (quarter lines).")
        else:
            match_data = {
                'home_team': team_home,
                'away_team': team_away,
                'timestamp': datetime.now(),  # treat as opening/record time for steam detection
                'open_line': open_ah_line,
                'open_home': open_ah_home,
                'open_away': open_ah_away,
                'pre_line': pre_ah_line,
                'pre_home': pre_ah_home,
                'pre_away': pre_ah_away,
                'live_line': live_ah_line,
                'live_home': live_ah_home,
                'live_away': live_ah_away,
                'live_enabled': bool(live_enabled),
                'league_profile': league
            }
            st.session_state.matches.append(match_data)
            st.success("✅ Match added for analysis!")

    if st.button("🗑️ Clear All Data", use_container_width=True):
        st.session_state.matches = []
        st.experimental_rerun()

# -----------------------
# Cached utility functions
# -----------------------
@st.cache_data
def calculate_implied_probability(odds):
    """Convert odds to implied probability"""
    if odds <= 1:
        return 0.0
    return 1.0 / odds

@st.cache_data
def calculate_margin(home_odds, away_odds):
    """Approximate bookmaker margin"""
    prob_home = calculate_implied_probability(home_odds)
    prob_away = calculate_implied_probability(away_odds)
    return (prob_home + prob_away - 1.0) * 100.0

# -----------------------
# Helper functions / refactor
# -----------------------
def determine_favorite(line, home_odds, away_odds):
    """
    Improved favorite determination:
    - Treat lines +/-0.25 as minimal, +/-0.75+ as strong.
    - Always corroborate with odds (home_odds < away_odds => home favored).
    - Return: ('HOME','STRONG') / ('HOME','WEAK') / ('AWAY','STRONG') / ('NEUTRAL','NONE')
    """
    # Normalize tiny float noise
    def approx(x): return round(x * 4) / 4.0
    line = approx(line)

    # decide by line first with thresholds
    if line <= -0.75 and home_odds < away_odds - 0.02:
        return 'HOME', 'STRONG'
    if line <= -0.25 and home_odds < away_odds - 0.02:
        return 'HOME', 'WEAK'
    if line >= 0.75 and away_odds < home_odds - 0.02:
        return 'AWAY', 'STRONG'
    if line >= 0.25 and away_odds < home_odds - 0.02:
        return 'AWAY', 'WEAK'

    # fallback to odds only with tolerance
    if home_odds < away_odds - 0.05:
        return 'HOME', 'WEAK'
    if away_odds < home_odds - 0.05:
        return 'AWAY', 'WEAK'

    return 'NEUTRAL', 'NONE'

def interpret_line_movement(open_line, pre_line, favorite_side, current_line=None):
    """
    Interprets line movement direction and magnitude.
    Returns: ('HOME'|'AWAY'|'NEUTRAL', abs_change)
    """
    if current_line is not None:
        delta = current_line - pre_line
    else:
        delta = pre_line - open_line

    delta = round(delta * 4) / 4.0  # quarter precision
    if abs(delta) < 0.125:
        return 'NEUTRAL', 0.0

    # For favorite context: negative => HOME strengthening
    if favorite_side == 'HOME':
        if delta < 0:
            return 'HOME', abs(delta)
        else:
            return 'AWAY', abs(delta)
    elif favorite_side == 'AWAY':
        if delta > 0:
            return 'AWAY', abs(delta)
        else:
            return 'HOME', abs(delta)
    else:
        if delta < 0:
            return 'HOME', abs(delta)
        else:
            return 'AWAY', abs(delta)

def calculate_directional_odds_movement(open_odds, pre_odds, live_odds=None):
    """
    Returns ('DROP'|'RISE'|'NEUTRAL', pct_change)
    Measures open->pre unless live provided (then pre->live).
    """
    if open_odds <= 0 or pre_odds <= 0:
        return 'NEUTRAL', 0.0

    if live_odds and live_odds > 0:
        base = pre_odds
        final = live_odds
    else:
        base = open_odds
        final = pre_odds

    pct = ((final - base) / base) * 100.0
    if pct < -1e-6 and abs(pct) >= 0.1:
        return 'DROP', abs(pct)
    elif pct > 3.0:
        return 'RISE', abs(pct)
    else:
        return 'NEUTRAL', abs(pct)

def league_volatility_pct(league_profile):
    """Return a volatility baseline (%) based on selected league profile."""
    if league_profile.startswith("Top"):
        return 10.0  # big leagues: smaller typical moves
    if league_profile.startswith("Mid"):
        return 12.5
    return 15.0  # low leagues: higher volatility

# -----------------------
# Suspicion detection helpers
# -----------------------
def detect_reverse_correlation(match, line_direction, line_strength, home_dir, home_str, away_dir, away_str, cfg):
    flags = []
    inds = []
    score_add = 0
    conf_add = 0
    reverse = False
    result = None

    thr_line = cfg['reverse_line_threshold']
    thr_odds = cfg['reverse_odds_pct']

    if line_direction == 'HOME' and line_strength >= thr_line and home_dir == 'RISE' and home_str >= thr_odds:
        flags.append("🔥 CRITICAL: Line favors HOME but home odds INCREASED - HOME LIKELY ARRANGED TO UNDERPERFORM")
        inds.append("⚠️ HOME TEAM SUSPECTED TO LOSE (reverse correlation)")
        score_add += 40
        conf_add += 50
        reverse = True
        result = 'HOME_FAIL'

    if line_direction == 'AWAY' and line_strength >= thr_line and away_dir == 'RISE' and away_str >= thr_odds:
        flags.append("🔥 CRITICAL: Line favors AWAY but away odds INCREASED - AWAY LIKELY ARRANGED TO UNDERPERFORM")
        inds.append("⚠️ AWAY TEAM SUSPECTED TO LOSE (reverse correlation)")
        score_add += 40
        conf_add += 50
        reverse = True
        result = 'AWAY_FAIL'

    return reverse, result, flags, inds, score_add, conf_add

def detect_violent_odds(match, home_dir, home_str, away_dir, away_str, cfg, league_vol):
    flags = []
    inds = []
    score_add = 0
    conf_add = 0
    result = None
    dyn_thr = cfg['violent_drop_pct']
    dyn_thr = dyn_thr * (league_vol / 10.0)

    if home_dir == 'DROP' and home_str >= dyn_thr:
        flags.append(f"⚠️ Violent odds drop on HOME ({home_str:.1f}%) - Heavy money backing home")
        inds.append("Massive betting volume on home win")
        score_add += 30
        conf_add += 40
        result = 'HOME_WIN'
    if away_dir == 'DROP' and away_str >= dyn_thr:
        flags.append(f"⚠️ Violent odds drop on AWAY ({away_str:.1f}%) - Heavy money backing away")
        inds.append("Massive betting volume on away win")
        score_add += 30
        conf_add += 40
        result = 'AWAY_WIN'

    return result, flags, inds, score_add, conf_add

def detect_steam_moves(match, line_direction, line_strength, home_dir, home_str, away_dir, away_str, cfg, minutes_since_open, reverse_detected):
    flags = []
    inds = []
    conf_add = 0
    if minutes_since_open <= cfg['steam_time_window'] and not reverse_detected:
        if line_strength >= 0.25 and ((line_direction == 'HOME' and home_dir == 'DROP') or (line_direction == 'AWAY' and away_dir == 'DROP')):
            inds.append("🔥 STEAM MOVE - synchronized sharp betting (line + odds)")
            conf_add += 10
        if home_str >= cfg['steam_odds_pct'] and minutes_since_open <= cfg['steam_time_window']:
            inds.append("🔥 STEAM MOVE on HOME - rapid odds move")
            conf_add += 5
        if away_str >= cfg['steam_odds_pct'] and minutes_since_open <= cfg['steam_time_window']:
            inds.append("🔥 STEAM MOVE on AWAY - rapid odds move")
            conf_add += 5
    return flags, inds, conf_add

def calculate_margin_change_alert(match, cfg, line_direction, home_dir, away_dir):
    flags = []
    inds = []
    add_score = 0
    add_conf = 0
    margin_open = calculate_margin(match['open_home'], match['open_away'])
    margin_pre = calculate_margin(match['pre_home'], match['pre_away'])
    diff = abs(margin_pre - margin_open)
    if diff > cfg['margin_threshold']:
        if line_direction == 'HOME' or home_dir == 'DROP':
            flags.append(f"💰 Unusual margin change (approx. {diff:.1f}%) aligning with HOME movement")
            inds.append("Bookmaker risk management active (HOME)")
            add_score += 10
            add_conf += 10
        elif line_direction == 'AWAY' or away_dir == 'DROP':
            flags.append(f"💰 Unusual margin change (approx. {diff:.1f}%) aligning with AWAY movement")
            inds.append("Bookmaker risk management active (AWAY)")
            add_score += 10
            add_conf += 10
    return flags, inds, add_score, add_conf

# -----------------------
# Main suspicious detector (refactored flow)
# -----------------------
def detect_suspicious_patterns(match, cfg):
    flags = []
    manipulation_indicators = []
    risk_score = 0
    fix_confidence = 0
    suspected_result = None
    reverse_detected = False

    favorite_side, favorite_strength = determine_favorite(match['open_line'], match['open_home'], match['open_away'])

    use_live = bool(match.get('live_enabled'))
    home_live = match.get('live_home') if use_live else None
    away_live = match.get('live_away') if use_live else None
    use_live_line = use_live and (match.get('live_line') != match.get('pre_line'))

    line_direction, line_strength = interpret_line_movement(match['open_line'], match['pre_line'], favorite_side,
                                                           current_line=(match.get('live_line') if use_live_line else None))

    home_dir, home_str = calculate_directional_odds_movement(match['open_home'], match['pre_home'], home_live)
    away_dir, away_str = calculate_directional_odds_movement(match['open_away'], match['pre_away'], away_live)

    lv = league_volatility_pct(match.get('league_profile', 'Top leagues (low vol)'))

    rev_detected, rev_result, rev_flags, rev_inds, rev_score, rev_conf = detect_reverse_correlation(
        match, line_direction, line_strength, home_dir, home_str, away_dir, away_str, cfg
    )
    if rev_detected:
        flags.extend(rev_flags)
        manipulation_indicators.extend(rev_inds)
        risk_score += rev_score
        fix_confidence += rev_conf
        suspected_result = rev_result
        reverse_detected = True

    if not reverse_detected:
        vo_result, vo_flags, vo_inds, vo_score, vo_conf = detect_violent_odds(match, home_dir, home_str, away_dir, away_str, cfg, lv)
        if vo_result:
            flags.extend(vo_flags)
            manipulation_indicators.extend(vo_inds)
            risk_score += vo_score
            fix_confidence += vo_conf
            suspected_result = vo_result
            outcome_locked = True
        else:
            outcome_locked = False
    else:
        outcome_locked = True

    if not outcome_locked:
        if home_dir == 'DROP' and home_str > 8:
            flags.append(f"⚡ Significant HOME odds drop ({home_str:.1f}%)")
            if not suspected_result:
                suspected_result = 'HOME_WIN'
            fix_confidence += 20
        if away_dir == 'DROP' and away_str > 8:
            flags.append(f"⚡ Significant AWAY odds drop ({away_str:.1f}%)")
            if not suspected_result:
                suspected_result = 'AWAY_WIN'
            fix_confidence += 20

    if line_strength >= 0.5:
        flags.append(f"🚨 Major line shift toward {line_direction} ({line_strength:.2f} goals)")
        manipulation_indicators.append(f"Bookmakers adjusted expectations toward {line_direction}")
        risk_score += 25
        fix_confidence += 25
        if not outcome_locked and not suspected_result:
            suspected_result = f"{line_direction}_WIN"

    m_flags, m_inds, m_score, m_conf = calculate_margin_change_alert(match, cfg, line_direction, home_dir, away_dir)
    flags.extend(m_flags)
    manipulation_indicators.extend(m_inds)
    risk_score += m_score
    fix_confidence += m_conf

    if use_live and match.get('live_home') and match.get('live_away'):
        live_home_dir, live_home_str = calculate_directional_odds_movement(match['open_home'], match['pre_home'], match.get('live_home'))
        live_away_dir, live_away_str = calculate_directional_odds_movement(match['open_away'], match['pre_away'], match.get('live_away'))

        if not outcome_locked:
            if live_home_dir == 'DROP' and live_home_str > cfg['violent_drop_pct']:
                flags.append(f"🔴 Extreme live HOME backing ({live_home_str:.1f}% odds drop) - in-play money detected")
                manipulation_indicators.append("In-play money flooding home team")
                fix_confidence += 35
                if not suspected_result:
                    suspected_result = 'HOME_WIN'
            if live_away_dir == 'DROP' and live_away_str > cfg['violent_drop_pct']:
                flags.append(f"🔴 Extreme live AWAY backing ({live_away_str:.1f}% odds drop) - in-play money detected")
                manipulation_indicators.append("In-play money flooding away team")
                fix_confidence += 35
                if not suspected_result:
                    suspected_result = 'AWAY_WIN'
        else:
            if live_home_dir == 'DROP' and live_home_str > cfg['override_extreme_drop']:
                flags.append(f"🔁 OVERRIDE: live HOME drop {live_home_str:.1f}% exceeds override threshold")
                suspected_result = 'HOME_WIN'
                fix_confidence = min(100, fix_confidence + 40)
                risk_score = min(100, risk_score + 20)
            if live_away_dir == 'DROP' and live_away_str > cfg['override_extreme_drop']:
                flags.append(f"🔁 OVERRIDE: live AWAY drop {live_away_str:.1f}% exceeds override threshold")
                suspected_result = 'AWAY_WIN'
                fix_confidence = min(100, fix_confidence + 40)
                risk_score = min(100, risk_score + 20)

    minutes_since_open = (datetime.now() - match.get('timestamp', datetime.now())).total_seconds() / 60.0
    s_flags, s_inds, s_conf = detect_steam_moves(match, line_direction, line_strength, home_dir, home_str, away_dir, away_str,
                                                 cfg, minutes_since_open, reverse_detected)
    flags.extend(s_flags)
    manipulation_indicators.extend(s_inds)
    fix_confidence += s_conf

    corr_bonus = 0
    corr_penalty = 0
    if line_direction == 'HOME' and home_dir == 'DROP':
        corr_bonus += 10
    if line_direction == 'AWAY' and away_dir == 'DROP':
        corr_bonus += 10
    if (line_direction == 'HOME' and home_dir == 'RISE') or (line_direction == 'AWAY' and away_dir == 'RISE'):
        corr_penalty += 10

    risk_score = min(100, risk_score + corr_bonus)
    fix_confidence = min(100, fix_confidence + corr_bonus)
    fix_confidence = max(0, fix_confidence - corr_penalty)

    risk_score = min(100, risk_score)
    fix_confidence = min(100, fix_confidence)

    return {
        'flags': flags,
        'risk_score': int(risk_score),
        'suspected_result': suspected_result,
        'manipulation_indicators': manipulation_indicators,
        'fix_confidence': int(fix_confidence),
        'reverse_detected': reverse_detected,
        'favorite': favorite_side,
        'line_direction': line_direction,
        'home_odds_direction': home_dir,
        'away_odds_direction': away_dir,
        'line_strength': line_strength,
        'home_odds_change_pct': home_str,
        'away_odds_change_pct': away_str,
        'minutes_since_open': round(minutes_since_open, 1)
    }

# -----------------------
# Prediction function (improved stability & weighting)
# -----------------------
def predict_outcome(match, cfg):
    lv = league_volatility_pct(match.get('league_profile', 'Top leagues (low vol)'))

    favorite_side, fav_strength = determine_favorite(match['pre_line'], match['pre_home'], match['pre_away'])

    home_live = match.get('live_home') if match.get('live_enabled') else None
    away_live = match.get('live_away') if match.get('live_enabled') else None
    use_live_line = bool(match.get('live_enabled')) and match.get('live_line') != match.get('pre_line')

    line_dir, line_str = interpret_line_movement(match['open_line'], match['pre_line'], favorite_side,
                                                current_line=(match.get('live_line') if use_live_line else None))

    home_dir, home_str = calculate_directional_odds_movement(match['open_home'], match['pre_home'], home_live)
    away_dir, away_str = calculate_directional_odds_movement(match['open_away'], match['pre_away'], away_live)

    home_score = 0.0
    away_score = 0.0

    W_LINE = 40.0
    W_ODDS = 35.0
    W_MARKET = 25.0

    if line_dir == 'HOME':
        home_score += W_LINE
    elif line_dir == 'AWAY':
        away_score += W_LINE
    else:
        home_score += W_LINE / 2.0
        away_score += W_LINE / 2.0

    home_score += W_ODDS * min(home_str / lv, 1.0) if home_dir == 'DROP' else 0.0
    away_score += W_ODDS * min(away_str / lv, 1.0) if away_dir == 'DROP' else 0.0

    if match['pre_home'] < match['pre_away']:
        home_score += W_MARKET
    else:
        away_score += W_MARKET

    if line_dir == 'HOME' and home_dir == 'DROP':
        home_score += 10.0
    if line_dir == 'AWAY' and away_dir == 'DROP':
        away_score += 10.0
    if (line_dir == 'HOME' and home_dir == 'RISE') or (line_dir == 'AWAY' and away_dir == 'RISE'):
        if line_dir == 'HOME':
            home_score -= 8.0
        else:
            away_score -= 8.0

    home_score = max(0.0, home_score)
    away_score = max(0.0, away_score)

    home_score = min(home_score, 70.0)
    away_score = min(away_score, 70.0)

    total = home_score + away_score
    if total > 0:
        home_confidence = (home_score / total) * 100.0
        away_confidence = (away_score / total) * 100.0
    else:
        home_confidence = away_confidence = 50.0

    if abs(match['pre_line']) <= 0.25 and abs(match['pre_home'] - match['pre_away']) < 0.05:
        home_confidence = max(30.0, home_confidence * 0.6)
        away_confidence = max(30.0, away_confidence * 0.6)

    if home_confidence > away_confidence:
        predicted = "HOME"
        confidence = home_confidence
        ah_pick = f"Home {match['pre_line']:+.2f}"
    else:
        predicted = "AWAY"
        confidence = away_confidence
        ah_pick = f"Away {(-match['pre_line']):+.2f}"

    return {
        'predicted_winner': predicted,
        'confidence': float(confidence),
        'ah_pick': ah_pick,
        'favorite': favorite_side,
        'line_direction': line_dir,
        'home_odds_direction': home_dir,
        'away_odds_direction': away_dir,
        'line_strength': line_str,
        'home_odds_change_pct': home_str,
        'away_odds_change_pct': away_str,
        'home_confidence': float(home_confidence),
        'away_confidence': float(away_confidence)
    }

# -----------------------
# Over/Under prediction helpers
# -----------------------
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def predict_over_under(match, cfg):
    lv = league_volatility_pct(match.get('league_profile', 'Top leagues (low vol)'))

    home_live = match.get('live_home') if match.get('live_enabled') else None
    away_live = match.get('live_away') if match.get('live_enabled') else None

    home_dir, home_str = calculate_directional_odds_movement(match['open_home'], match['pre_home'], home_live)
    away_dir, away_str = calculate_directional_odds_movement(match['open_away'], match['pre_away'], away_live)

    fav_side, _ = determine_favorite(match['open_line'], match['open_home'], match['open_away'])
    line_dir, line_str = interpret_line_movement(match['open_line'], match['pre_line'], fav_side)

    expected_goals = 2.5

    avg_move = (home_str + away_str) / 2.0
    move_adj = (avg_move - (lv * 0.5)) / (lv if lv > 0 else 10)
    expected_goals += move_adj * 1.2

    expected_goals += min(1.0, line_str * 0.6)

    margin_open = calculate_margin(match['open_home'], match['open_away'])
    margin_pre = calculate_margin(match['pre_home'], match['pre_away'])
    margin_diff = margin_pre - margin_open
    expected_goals -= (margin_diff / 10.0)

    expected_goals = max(0.5, min(6.0, expected_goals))

    thresholds = [0.5, 1.5, 2.5, 3.5, 4.5]
    k = 1.2
    probs = []
    for t in thresholds:
        p_over = _sigmoid((expected_goals - t) * k)
        probs.append((t, p_over))

    best = max(probs, key=lambda x: abs(x[1] - 0.5))
    th, p_over = best
    if p_over > 0.5:
        pick = f"OVER {th:.1f}"
        confidence = (p_over - 0.5) * 2.0
    else:
        pick = f"UNDER {th:.1f}"
        confidence = (0.5 - p_over) * 2.0

    conf_pct = float(min(99.0, max(25.0, confidence * 100.0 + (abs(line_str) * 8.0))))
    expl = f"Expected goals ≈ {expected_goals:.2f}. Based on AH line shift ({line_str:+.2f}) and avg odds movement ({avg_move:.2f}%)."
    return {
        'ou_pick': pick,
        'ou_confidence': conf_pct,
        'expected_goals': expected_goals,
        'explanation': expl,
        'threshold_probability': float(p_over)
    }

# -----------------------
# Poisson / 3-way helper (NEW)
# -----------------------
def poisson_pmf(k, lam):
    if lam <= 0:
        return 0.0 if k > 0 else 1.0
    try:
        return (lam ** k) * math.exp(-lam) / math.factorial(k)
    except OverflowError:
        return 0.0

def three_way_from_expected_goals(lambda_h, lambda_a, max_goals=8):
    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0

    for i in range(max_goals + 1):
        ph = poisson_pmf(i, lambda_h)
        for j in range(max_goals + 1):
            pa = poisson_pmf(j, lambda_a)
            prob = ph * pa
            if i > j:
                p_home += prob
            elif i == j:
                p_draw += prob
            else:
                p_away += prob

    total = p_home + p_draw + p_away
    if total <= 0:
        return 1/3, 1/3, 1/3
    return p_home / total, p_draw / total, p_away / total

def split_expected_goals_to_team_lambdas(expected_goals, pre_line):
    goal_diff = -pre_line
    lambda_h = (expected_goals + goal_diff) / 2.0
    lambda_a = (expected_goals - goal_diff) / 2.0
    lambda_h = max(0.05, lambda_h)
    lambda_a = max(0.05, lambda_a)
    return lambda_h, lambda_a

# -----------------------
# 1X2 Prediction helper (WITH OVERRIDE)
# -----------------------
def predict_1x2(match, cfg, analysis=None):
    """
    Produce a 1X2 probability distribution (Home / Draw / Away) in percentages.

    - Uses Poisson-derived 3-way (from OU expected goals).
    - Blends model with market two-way using user-configured weights.
    - If fix probability >= override threshold, applies a controlled override that heavily favors the suspected side.
    - Otherwise applies a bounded nudge proportional to fix probability.
    """
    try:
        global_w_model = float(w_model)
        global_w_market = float(w_market)
        global_max_nudge = float(max_nudge)
    except Exception:
        global_w_model = 0.6
        global_w_market = 0.4
        global_max_nudge = 0.25

    try:
        ov_threshold = float(override_fix_threshold) / 100.0
        ov_strength = float(override_strength_pct) / 100.0
    except Exception:
        ov_threshold = 0.90
        ov_strength = 0.95

    w_model = max(0.0, min(1.0, global_w_model))
    w_market = max(0.0, min(1.0, global_w_market))

    ou = predict_over_under(match, cfg)
    expected_goals = ou.get('expected_goals', 2.5)

    lambda_h, lambda_a = split_expected_goals_to_team_lambdas(expected_goals, match.get('pre_line', 0.0))

    model_home, model_draw, model_away = three_way_from_expected_goals(lambda_h, lambda_a, max_goals=8)

    ph = calculate_implied_probability(match.get('pre_home', 1.0))
    pa = calculate_implied_probability(match.get('pre_away', 1.0))
    denom = ph + pa
    if denom > 0:
        market_home_two_way = ph / denom
        market_away_two_way = pa / denom
    else:
        market_home_two_way = market_away_two_way = 0.5

    blended_home = w_model * model_home + w_market * (market_home_two_way * (1.0 - model_draw))
    blended_away = w_model * model_away + w_market * (market_away_two_way * (1.0 - model_draw))
    blended_draw = w_model * model_draw + w_market * model_draw

    total = blended_home + blended_draw + blended_away
    if total <= 0:
        blended_home = blended_draw = blended_away = 1.0 / 3.0
        total = 1.0
    blended_home /= total
    blended_draw /= total
    blended_away /= total

    if analysis is None:
        analysis = detect_suspicious_patterns(match, cfg)
    risk_score = analysis.get('risk_score', 0)
    fix_conf = analysis.get('fix_confidence', 0)

    fix_prob = (0.6 * (risk_score / 100.0)) + (0.4 * (fix_conf / 100.0))
    fix_prob = max(0.0, min(1.0, fix_prob))

    suspected = analysis.get('suspected_result')

    if suspected:
        if suspected.endswith('_FAIL'):
            fail_side = suspected.split('_')[0]
            favored_side = 'AWAY' if fail_side == 'HOME' else 'HOME'
        else:
            favored_side = suspected.split('_')[0]

        if fix_prob >= ov_threshold:
            rem = 1.0 - ov_strength
            draw_share = rem * 0.20
            other_share = rem - draw_share

            if favored_side == 'HOME':
                final_home = ov_strength
                final_draw = draw_share
                final_away = other_share
            else:
                final_away = ov_strength
                final_draw = draw_share
                final_home = other_share

            s = final_home + final_draw + final_away
            if s <= 0:
                final_home = final_draw = final_away = 1.0 / 3.0
            else:
                final_home /= s
                final_draw /= s
                final_away /= s

            blended_home, blended_draw, blended_away = final_home, final_draw, final_away

        else:
            alpha = fix_prob * global_max_nudge
            if favored_side == 'HOME':
                prior = np.array([1.0, 0.0, 0.0])
            else:
                prior = np.array([0.0, 0.0, 1.0])

            blended = np.array([blended_home, blended_draw, blended_away])
            final = (1.0 - alpha) * blended + alpha * prior
            s = final.sum()
            if s <= 0:
                final = np.array([1/3, 1/3, 1/3])
            else:
                final = final / s
            blended_home, blended_draw, blended_away = float(final[0]), float(final[1]), float(final[2])

    home_pct = float(round(blended_home * 100.0, 1))
    draw_pct = float(round(blended_draw * 100.0, 1))
    away_pct = float(round(blended_away * 100.0, 1))
    fix_probability_pct = float(round(fix_prob * 100.0, 1))

    explanation = (
        f"1X2 uses a Poisson model (from OU expected goals) to estimate Home/Draw/Away. "
        f"Home/Away distribution is blended with market two-way (weights: model={w_model}, market={w_market}). "
        f"If fix probability ≥ {int(ov_threshold*100)}% the distribution is overridden so the suspected side has {int(ov_strength*100)}% share."
    )

    return {
        'home_pct': home_pct,
        'draw_pct': draw_pct,
        'away_pct': away_pct,
        'fix_probability_pct': fix_probability_pct,
        'explanation': explanation,
        'raw': {
            'model_home': model_home,
            'model_draw': model_draw,
            'model_away': model_away,
            'blended_home': blended_home,
            'blended_draw': blended_draw,
            'blended_away': blended_away,
            'fix_prob': fix_prob
        }
    }

# -----------------------
# Asian Handicap explanation helper
# -----------------------
def generate_ah_explanation_html(ah_pick):
    try:
        parts = ah_pick.split()
        side = parts[0]
        line_val = float(parts[1])
    except Exception:
        return ""

    sign = -1 if line_val < 0 else 1
    abs_line = abs(line_val)
    quarters = int(round(abs_line * 4)) % 4
    int_part = int(abs_line)

    def row(outcome, label, explanation):
        return f"<tr><th style='width:34%'>{outcome}</th><td style='width:14%'>{label}</td><td>{explanation}</td></tr>"

    pick_wins_label = "✅ Full Win"
    pick_wins_expl = "Both halves / full stake win (depending on split)."

    draw_label = "➖ Push / Half"
    draw_expl = "Depends on split: may be push, half-win, or half-loss."

    pick_loses_label = "❌ Full Loss"
    pick_loses_expl = "Both halves / full stake lose."

    if quarters == 0:
        if int_part == 0:
            pick_wins_label = "✅ Full Win"
            pick_wins_expl = "If the picked team wins the match, bet wins; draw is a push; loss is a full loss."
            draw_label = "➖ Push"
            draw_expl = "Exact refund of stake if match ends level."
            pick_loses_label = "❌ Full Loss"
            pick_loses_expl = "Bet loses if picked team loses."
        else:
            pick_wins_label = "✅ Full Win (if win by more than {})".format(int_part)
            pick_wins_expl = f"Picked team must win by more than {int_part} goal(s) for a full win. If they win by exactly {int_part} it's a push (refund)."
            draw_label = "❌ Full Loss"
            draw_expl = "Draw is a loss for handicaps of whole numbers >0 for the giving side."
            pick_loses_label = "❌ Full Loss"
            pick_loses_expl = "Picked team loses => full loss."
    elif quarters == 1:
        pick_wins_label = "✅ Full Win"
        pick_wins_expl = "If the picked team wins the match (any margin), the bet is a full win."
        if sign < 0:
            draw_label = "➖ Half Loss"
            draw_expl = f"The bet is split: the '0' half is refunded, the '-0.5' half loses."
        else:
            draw_label = "➕ Half Win"
            draw_expl = f"The bet is split: the '0' half is refunded, the '+0.5' half wins."
        pick_loses_label = "❌ Full Loss"
        pick_loses_expl = "If the picked team loses the match, both halves lose."
    elif quarters == 2:
        pick_wins_label = "✅ Full Win"
        pick_wins_expl = "Picked team wins = full win."
        if sign < 0:
            draw_label = "❌ Full Loss"
            draw_expl = "Draw results in full loss for a negative (giving) -0.5 handicap."
        else:
            draw_label = "✅ Full Win"
            draw_expl = "Draw results in full win for a positive (receiving) +0.5 handicap."
        pick_loses_label = "❌ Full Loss"
        pick_loses_expl = "Picked team loses => full loss."
    elif quarters == 3:
        pick_wins_label = "✅ Full Win / Half Win"
        if sign < 0:
            pick_wins_expl = "If picked team wins by 2+ goals => full win. If they win by exactly 1 => half win (one half wins, one half pushes)."
            draw_label = "❌ Full Loss"
            draw_expl = "Draw results in full loss for a -0.75 giving handicap."
        else:
            pick_wins_expl = "If picked team wins by 2+ goals => full win. If they win by exactly 1 => half win (one half wins, one half pushes)."
            draw_label = "➕ Half Win"
            draw_expl = "Draw results in a half win for +0.75 (the +0.5 half wins, the +1.0 half pushes)."
        pick_loses_label = "❌ Full Loss"
        pick_loses_expl = "Picked team loses => full loss."

    side_label = side
    html = f"""
    <div class='ah-explain'>
      <strong>What '{ah_pick}' means</strong>
      <p style="margin:6px 0 10px 0;color:#444;">Pick: <strong>{side_label} {line_val:+.2f}</strong></p>
      <table>
        {row(f"{side_label} team wins", pick_wins_label, pick_wins_expl)}
        {row("Match ends in a draw", draw_label, draw_expl)}
        {row(f"{side_label} team loses", pick_loses_label, pick_loses_expl)}
      </table>
    </div>
    """
    return html

def generate_ou_explanation_html(ou_pick, expected_goals, explanation):
    try:
        side, thr = ou_pick.split()
    except Exception:
        side = ou_pick
        thr = ""
    html = f"""
      <div class='ou-explain'>
        <strong>What '{ou_pick}' means</strong>
        <p style="margin:6px 0 8px 0;color:#444;">This prediction estimates total goals in the match.</p>
        <ul>
          <li><strong>Expected goals:</strong> {expected_goals:.2f}</li>
          <li><strong>Interpretation:</strong> If '{ou_pick}' = OVER {thr}, we expect the match to have more than {thr} goals (combined). UNDER means fewer than {thr} goals.</li>
        </ul>
        <p style="margin:6px 0 0 0;color:#666;"><em>Reasoning:</em> {explanation}</p>
      </div>
    """
    return html

# -----------------------
# Main UI / Analysis flow
# -----------------------
if st.session_state.matches:
    st.header("📊 Match Analysis Dashboard")
    last_match = st.session_state.matches[-1]

    cfg = {
        'reverse_line_threshold': float(reverse_line_threshold),
        'reverse_odds_pct': float(reverse_odds_pct),
        'violent_drop_pct': float(violent_drop_pct),
        'steam_odds_pct': float(steam_odds_pct),
        'steam_time_window': int(steam_time_window),
        'margin_threshold': float(margin_threshold),
        'override_extreme_drop': float(override_extreme_drop)
    }

    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Predictions", "⚠️ Anomaly Detection", "📈 Odds Movement", "📋 History"])

    with tab1:
        st.subheader(f"🏟️ {last_match['home_team']} vs {last_match['away_team']}")
        prediction = predict_outcome(last_match, cfg)
        ou_prediction = predict_over_under(last_match, cfg)

        analysis_for_1x2 = detect_suspicious_patterns(last_match, cfg)
        one_x_two = predict_1x2(last_match, cfg, analysis=analysis_for_1x2)

        confidence_level = "high" if prediction['confidence'] > 75 else "medium" if prediction['confidence'] > 60 else "low"
        box_class = f"{confidence_level}-confidence"

        st.markdown(f"""
        <div class="prediction-box {box_class}">
            <h3>🎯 Market Prediction: {prediction['predicted_winner']} Team</h3>
            <h2>Confidence: {prediction['confidence']:.1f}%</h2>
            <p><strong>Asian Handicap Pick:</strong> {prediction['ah_pick']}</p>
            <p><strong>Market Favorite:</strong> {prediction['favorite']}</p>
            <p><strong>Line Movement:</strong> Toward {prediction['line_direction']}, change {prediction['line_strength']}</p>
        </div>
        """, unsafe_allow_html=True)

        ah_html = generate_ah_explanation_html(prediction['ah_pick'])
        if ah_html:
            st.markdown(ah_html, unsafe_allow_html=True)

        ou_conf_level = "high" if ou_prediction['ou_confidence'] > 75 else "medium" if ou_prediction['ou_confidence'] > 60 else "low"
        ou_box_class = f"{ou_conf_level}-confidence"
        st.markdown(f"""
        <div class="prediction-box {ou_box_class}">
            <h3>📊 Over/Under Prediction: {ou_prediction['ou_pick']}</h3>
            <h2>Confidence: {ou_prediction['ou_confidence']:.1f}%</h2>
            <p><strong>Estimated total goals:</strong> {ou_prediction['expected_goals']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

        max_pct = max(one_x_two['home_pct'], one_x_two['draw_pct'], one_x_two['away_pct'])
        one_class = "high-confidence" if max_pct > 60 else "medium-confidence" if max_pct > 45 else "low-confidence"
        st.markdown(f"""
        <div class="prediction-box {one_class}">
            <h3>1X2 Prediction</h3>
            <h2>Home: {one_x_two['home_pct']}%  —  Draw: {one_x_two['draw_pct']}%  —  Away: {one_x_two['away_pct']}%</h2>
            <p><strong>Anomaly / Fix Probability:</strong> {one_x_two['fix_probability_pct']}%</p>
            <p style="margin-top:6px;color:#444;"><em>{one_x_two['explanation']}</em></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(generate_ou_explanation_html(ou_prediction['ou_pick'], ou_prediction['expected_goals'], ou_prediction['explanation']), unsafe_allow_html=True)

        with st.expander("Why this prediction? (explain metrics)"):
            st.write("- 1X2 now uses a Poisson model derived from the Over/Under expected goals to compute Home/Draw/Away probabilities (more consistent draw estimation).")
            st.write("- Home/Away split is derived purely from market AH line (no team stats available).")
            st.write("- You can tune blending weights between the Poisson model and the market two-way in the sidebar.")
            st.write("- If anomaly detection suggests a manipulated side, a bounded nudge (user-tunable) moves probabilities toward the suspected side. Extremely high fix probability can trigger an override (also tunable).")
            st.write("- Probabilities are normalized and guarded to avoid numeric instability.")

    with tab2:
        st.subheader("🚨 Match Integrity Analysis")
        analysis = detect_suspicious_patterns(last_match, cfg)

        risk_score = analysis['risk_score']
        fix_confidence = analysis['fix_confidence']
        suspected_result = analysis['suspected_result']

        if risk_score >= 70:
            risk_level = "CRITICAL - HIGHLY SUSPICIOUS"
            risk_emoji = "🚨"
        elif risk_score >= 40:
            risk_level = "ELEVATED - SUSPICIOUS PATTERNS"
            risk_emoji = "⚠️"
        else:
            risk_level = "NORMAL MARKET BEHAVIOR"
            risk_emoji = "✅"

        st.markdown(f"""
        <div class="prediction-box {'suspicious' if risk_score >= 70 else 'medium-confidence' if risk_score >= 40 else 'high-confidence'}">
            <h2>{risk_emoji} Risk Level: {risk_level}</h2>
            <h3>Manipulation Score: {risk_score}/100</h3>
        </div>
        """, unsafe_allow_html=True)

        if suspected_result:
            if suspected_result.endswith('_FAIL'):
                fail_side = suspected_result.split('_')[0]
                failing_team = last_match['home_team'] if fail_side == 'HOME' else last_match['away_team']
                value_side = 'AWAY' if fail_side == 'HOME' else 'HOME'
                value_team = last_match['away_team'] if fail_side == 'HOME' else last_match['home_team']
                ah_pick = f"{'Away' if value_side=='AWAY' else 'Home'} {(-last_match['pre_line'] if value_side=='AWAY' else last_match['pre_line']):+.2f}"
                st.markdown(f"""
                <div class="prediction-box {'suspicious' if fix_confidence >= 70 else 'medium-confidence' if fix_confidence >= 40 else 'high-confidence'}">
                    <h2>🎯 TEAM EXPECTED TO FAIL</h2>
                    <h1 style='color: {"darkred" if fix_confidence >= 70 else "darkorange" if fix_confidence >= 40 else "darkgreen"}; font-size: 2.4rem; text-align: center;'>
                        {failing_team} (EXPECTED TO FAIL)
                    </h1>
                    <h3 style='text-align: center;'>Value Side / Opposite: {value_team}</h3>
                    <h2 style='text-align: center;'>Fix Confidence: {fix_confidence}%</h2>
                    <p style='text-align: center;'><strong>Asian Handicap (OPPOSITE SIDE VALUE):</strong> {ah_pick}</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(generate_ah_explanation_html(ah_pick), unsafe_allow_html=True)
            else:
                win_side = suspected_result.split('_')[0]
                winning_team = last_match['home_team'] if win_side == 'HOME' else last_match['away_team']
                ah_pick = f"{'Home' if win_side=='HOME' else 'Away'} {last_match['pre_line']:+.2f}" if win_side=='HOME' else f"Away {(-last_match['pre_line']):+.2f}"
                st.markdown(f"""
                <div class="prediction-box {'suspicious' if fix_confidence >= 70 else 'medium-confidence' if fix_confidence >= 40 else 'high-confidence'}">
                    <h2>🎯 TEAM EXPECTED TO WIN</h2>
                    <h1 style='color: {"darkred" if fix_confidence >= 70 else "darkorange" if fix_confidence >= 40 else "darkgreen"}; font-size: 2.8rem; text-align: center;'>
                        {winning_team} (EXPECTED WIN)
                    </h1>
                    <h2 style='text-align: center;'>Fix Confidence: {fix_confidence}%</h2>
                    <p style='text-align: center;'><strong>Asian Handicap:</strong> {ah_pick}</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(generate_ah_explanation_html(ah_pick), unsafe_allow_html=True)

            if fix_confidence >= 70:
                stake = "x1.5–2.0 (aggressive multiplier)"
                rec_type = "STRONG"
            elif fix_confidence >= 40:
                stake = "x0.75–1.0 (moderate multiplier)"
                rec_type = "MODERATE"
            else:
                stake = "x0.5–0.75 (cautious multiplier)"
                rec_type = "LIGHT"

            st.markdown(f"""
            <div class="prediction-box {'suspicious' if fix_confidence >= 70 else 'medium-confidence'}">
                <h3>💡 {rec_type} RECOMMENDATION</h3>
                <p><strong>Suggested confidence multiplier:</strong> {stake}</p>
                <p><strong>Fix confidence:</strong> {fix_confidence}%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="prediction-box neutral">
                <h3>ℹ️ NO CLEAR MANIPULATION SIGNALS</h3>
                <p>Market appears normal. Signals too weak or contradictory to determine manipulation.</p>
                <p><strong>Recommendation:</strong> Use standard betting analysis. No special manipulation detected.</p>
            </div>
            """, unsafe_allow_html=True)

        if analysis['flags']:
            st.subheader("📋 Detected Patterns")
            for flag in analysis['flags']:
                if "CRITICAL" in flag or "🔥" in flag:
                    st.error(flag)
                elif "Major" in flag or "Violent" in flag or "Extreme" in flag or "🔁" in flag:
                    st.warning(flag)
                else:
                    st.info(flag)

        if analysis['manipulation_indicators']:
            st.subheader("🔍 Manipulation Evidence")
            for indicator in analysis['manipulation_indicators']:
                st.warning(indicator)

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                title={'text': "Risk Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred" if risk_score >= 70 else "orange" if risk_score >= 40 else "green"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fix_confidence,
                title={'text': "Fix Confidence"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred" if fix_confidence >= 70 else "orange" if fix_confidence >= 40 else "blue"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightblue"},
                        {'range': [40, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ]
                }
            ))
            st.plotly_chart(fig2, use_container_width=True)

        with st.expander("What do these metrics mean?"):
            st.write("- Risk Score: aggregated suspicion level (higher = more suspicious).")
            st.write("- Fix Confidence: how confident the system is that the market reflects manipulation/fixing.")
            st.write("- Reverse correlation: when bookmakers shift line toward one side but that side's odds increase (money avoiding).")
            st.write("- Steam moves: rapid, synchronized movement suggesting sharp consensus (not necessarily a fix).")
            st.write("- Over/Under predictions are supplementary and derived heuristically from AH and odds movement; treat cautiously.")

    with tab3:
        st.subheader("📈 Odds & Line Movement Visualization")
        stages = ['Opening', 'Pre-match', 'Live']
        if last_match.get('live_enabled'):
            home_odds = [last_match['open_home'], last_match['pre_home'], last_match['live_home']]
            away_odds = [last_match['open_away'], last_match['pre_away'], last_match['live_away']]
            lines = [last_match['open_line'], last_match['pre_line'], last_match['live_line']]
        else:
            home_odds = [last_match['open_home'], last_match['pre_home'], last_match['pre_home']]
            away_odds = [last_match['open_away'], last_match['pre_away'], last_match['pre_away']]
            lines = [last_match['open_line'], last_match['pre_line'], last_match['pre_line']]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stages, y=home_odds, mode='lines+markers', name='Home Odds', line=dict(color='blue', width=3)))
        fig.add_trace(go.Scatter(x=stages, y=away_odds, mode='lines+markers', name='Away Odds', line=dict(color='red', width=3)))
        fig.update_layout(title="Odds Movement", xaxis_title="Stage", yaxis_title="Odds", height=420)
        st.plotly_chart(fig, use_container_width=True)

        colors = []
        annotations = []
        for i, l in enumerate(lines):
            if l < 0:
                colors.append('rgba(0,153,51,0.6)')
                arrows = "⬇️" if l < 0 else ""
            elif l > 0:
                colors.append('rgba(204,0,0,0.6)')
                arrows = "⬆️" if l > 0 else ""
            else:
                colors.append('rgba(120,120,120,0.5)')
                arrows = ""
            annotations.append(f"{l:+.2f} {arrows}")

        fig2 = go.Figure(go.Bar(x=stages, y=lines, text=[f"{l:+.2f}" for l in lines], marker_color=colors, textposition='auto'))
        fig2.update_layout(
            title="Line Movement (Negative = Home Fav, Positive = Away Fav)",
            xaxis_title="Stage",
            yaxis_title="AH Line",
            height=360
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        st.subheader("📋 Analysis History")
        if len(st.session_state.matches) > 0:
            history_data = []
            for match in st.session_state.matches:
                analysis = detect_suspicious_patterns(match, cfg)
                pred = predict_outcome(match, cfg)
                ou_pred = predict_over_under(match, cfg)
                one_x_two_row = predict_1x2(match, cfg, analysis=analysis)

                suspected_txt = "None"
                if analysis['suspected_result']:
                    sr = analysis['suspected_result']
                    if sr.endswith('_FAIL'):
                        suspected_txt = f"{match['home_team'] if sr.startswith('HOME') else match['away_team']} (EXPECTED TO FAIL)"
                    else:
                        suspected_txt = f"{match['home_team'] if sr.startswith('HOME') else match['away_team']} (EXPECTED WIN)"

                history_data.append({
                    'Match': f"{match['home_team']} vs {match['away_team']}",
                    'Suspected Result': suspected_txt,
                    'Fix Confidence': f"{analysis['fix_confidence']}%",
                    'Risk Score': f"{analysis['risk_score']}/100",
                    'Market Prediction': pred['predicted_winner'],
                    'AH Pick': pred['ah_pick'],
                    'AH Confidence': f"{pred['confidence']:.1f}%",
                    'OU Prediction': ou_pred['ou_pick'],
                    'OU Confidence': f"{ou_pred['ou_confidence']:.1f}%",
                    '1X2 Home %': f"{one_x_two_row['home_pct']:.1f}",
                    '1X2 Draw %': f"{one_x_two_row['draw_pct']:.1f}",
                    '1X2 Away %': f"{one_x_two_row['away_pct']:.1f}",
                    '1X2 Fix %': f"{one_x_two_row['fix_probability_pct']:.1f}",
                    'Line Strength': analysis.get('line_strength', 0),
                    'Home Odds Change %': f"{analysis.get('home_odds_change_pct', 0):.1f}",
                    'Away Odds Change %': f"{analysis.get('away_odds_change_pct', 0):.1f}",
                    'Minutes Since Open': analysis.get('minutes_since_open', 0),
                    'Flags': " | ".join(analysis.get('flags', [])),
                    'Time': match['timestamp'].strftime("%Y-%m-%d %H:%M")
                })

            df = pd.DataFrame(history_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download History",
                data=csv,
                file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
else:
    st.info("👈 Enter match data in the sidebar to start")
    st.markdown("""
    ### Key Improvements (summary)
    - 1X2 uses a Poisson-based approach (from expected goals) to compute Home/Draw/Away probabilities, replacing the previous simple draw heuristic.
    - Home/Away split is derived from the AH pre_line to stay within your constraint of only using odds data.
    - Blending weights between model and market two-way are user-tunable in the sidebar (no historical data needed).
    - Override behavior: when fix probability is very high (user-configurable), 1X2 can force a strong recommendation toward the suspected side while preserving a small residual for other outcomes.
    - The manipulation nudge is bounded and applied in a stable additive way to avoid numeric instability.
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>⚠️ Responsible Gambling Disclaimer</strong></p>
    <p>This tool is for educational and analytical purposes only. Always gamble responsibly.</p>
    <p>Past performance does not guarantee future results. Bet only what you can afford to lose.</p>
</div>
""", unsafe_allow_html=True)