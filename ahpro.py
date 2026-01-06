import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

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
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.75; }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⚽ Asian Handicap Predictor & Anomaly Detector</div>', unsafe_allow_html=True)

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

    team_home = st.text_input("Home Team", placeholder="e.g., Manchester City")
    team_away = st.text_input("Away Team", placeholder="e.g., Liverpool")

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
    if st.button("🔍 Analyze Match", type="primary", use_container_width=True):
        # Validations
        def is_valid_ah_line(line):
            return abs(line * 4 - round(line * 4)) < 1e-8

        if not (team_home and team_away):
            st.error("Please enter both team names!")
        elif not (is_valid_ah_line(open_ah_line) and is_valid_ah_line(pre_ah_line) and is_valid_ah_line(live_ah_line)):
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
        if pct < -3.0:
            return 'DROP', abs(pct)
        else:
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
    """Return (detected_bool, result_string, flags, indicators, scores_to_add)"""
    flags = []
    inds = []
    score_add = 0
    conf_add = 0
    reverse = False
    result = None

    # Require meaningful line shift and meaningful opposite odds move (>= configured)
    thr_line = cfg['reverse_line_threshold']
    thr_odds = cfg['reverse_odds_pct']

    # Home reverse: line toward HOME but HOME odds rise
    if line_direction == 'HOME' and line_strength >= thr_line and home_dir == 'RISE' and home_str >= thr_odds:
        flags.append("🔥 CRITICAL: Line favors HOME but home odds INCREASED - HOME LIKELY ARRANGED TO UNDERPERFORM")
        inds.append("⚠️ HOME TEAM SUSPECTED TO LOSE (reverse correlation)")
        score_add += 40
        conf_add += 50
        reverse = True
        result = 'HOME_FAIL'

    # Away reverse
    if line_direction == 'AWAY' and line_strength >= thr_line and away_dir == 'RISE' and away_str >= thr_odds:
        flags.append("🔥 CRITICAL: Line favors AWAY but away odds INCREASED - AWAY LIKELY ARRANGED TO UNDERPERFORM")
        inds.append("⚠️ AWAY TEAM SUSPECTED TO LOSE (reverse correlation)")
        score_add += 40
        conf_add += 50
        reverse = True
        result = 'AWAY_FAIL'

    # Also check subtle reverse where both teams show contradictory small signals
    return reverse, result, flags, inds, score_add, conf_add

def detect_violent_odds(match, home_dir, home_str, away_dir, away_str, cfg, league_vol):
    flags = []
    inds = []
    score_add = 0
    conf_add = 0
    result = None
    # dynamic threshold: use configured violent_drop_pct but allow scaling by league volatility
    dyn_thr = cfg['violent_drop_pct']  # user supplied baseline
    # If league more volatile, require slightly higher threshold
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
    # Steam: rapid movement within configured window and exceeding configured pct/line
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
        # only mark if aligns with other signals
        # align with side that shows strength
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
    """
    Main orchestrator - uses helper detectors.
    Returns dictionary with detailed metrics for storage.
    """

    flags = []
    manipulation_indicators = []
    risk_score = 0
    fix_confidence = 0
    suspected_result = None
    reverse_detected = False

    # Favorite using pre-market (open is used for initial favorite decision to interpret line movement direction)
    favorite_side, favorite_strength = determine_favorite(match['open_line'], match['open_home'], match['open_away'])

    # Decide whether to use live line/odds (live_enabled must be true)
    use_live = bool(match.get('live_enabled'))
    home_live = match.get('live_home') if use_live else None
    away_live = match.get('live_away') if use_live else None
    use_live_line = use_live and (match.get('live_line') != match.get('pre_line'))

    # Interpret movements
    line_direction, line_strength = interpret_line_movement(match['open_line'], match['pre_line'], favorite_side,
                                                           current_line=(match.get('live_line') if use_live_line else None))

    home_dir, home_str = calculate_directional_odds_movement(match['open_home'], match['pre_home'], home_live)
    away_dir, away_str = calculate_directional_odds_movement(match['open_away'], match['pre_away'], away_live)

    # league volatility baseline
    lv = league_volatility_pct(match.get('league_profile', 'Top leagues (low vol)'))

    # 1) Reverse correlation
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

    # 2) Violent odds drops (only if not reverse-locked)
    if not reverse_detected:
        vo_result, vo_flags, vo_inds, vo_score, vo_conf = detect_violent_odds(match, home_dir, home_str, away_dir, away_str, cfg, lv)
        if vo_result:
            flags.extend(vo_flags)
            manipulation_indicators.extend(vo_inds)
            risk_score += vo_score
            fix_confidence += vo_conf
            suspected_result = vo_result
            # lock outcome (can be overridden later by extreme live)
            outcome_locked = True
        else:
            outcome_locked = False
    else:
        outcome_locked = True

    # 3) Moderate movements (add smaller confidence if not locked)
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

    # 4) Major line movements
    if line_strength >= 0.5:
        flags.append(f"🚨 Major line shift toward {line_direction} ({line_strength:.2f} goals)")
        manipulation_indicators.append(f"Bookmakers adjusted expectations toward {line_direction}")
        risk_score += 25
        fix_confidence += 25
        if not outcome_locked and not suspected_result:
            suspected_result = f"{line_direction}_WIN"

    # 5) Margin change (reduced weight and only if aligned)
    m_flags, m_inds, m_score, m_conf = calculate_margin_change_alert(match, cfg, line_direction, home_dir, away_dir)
    flags.extend(m_flags)
    manipulation_indicators.extend(m_inds)
    risk_score += m_score
    fix_confidence += m_conf

    # 6) Live movements (only if live enabled and not locked; severe live drops can override)
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
            # If locked by reverse but extreme live movement occurs, allow override
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

    # 7) Steam moves detection (timing sensitive)
    minutes_since_open = (datetime.now() - match.get('timestamp', datetime.now())).total_seconds() / 60.0
    s_flags, s_inds, s_conf = detect_steam_moves(match, line_direction, line_strength, home_dir, home_str, away_dir, away_str,
                                                 cfg, minutes_since_open, reverse_detected)
    flags.extend(s_flags)
    manipulation_indicators.extend(s_inds)
    fix_confidence += s_conf

    # 8) Correlation bonus between line and odds:
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

    # Final bounding
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
    """
    Prediction with:
    - weighted contributions: line movement 40, odds movement 35, market position 25
    - normalization of odds movement by league volatility baseline
    - correlation bonus/penalty
    - neutral-market cautious handling
    """
    # league volatility baseline
    lv = league_volatility_pct(match.get('league_profile', 'Top leagues (low vol)'))

    # Favorite (pre-market)
    favorite_side, fav_strength = determine_favorite(match['pre_line'], match['pre_home'], match['pre_away'])

    # use live if enabled
    home_live = match.get('live_home') if match.get('live_enabled') else None
    away_live = match.get('live_away') if match.get('live_enabled') else None
    use_live_line = bool(match.get('live_enabled')) and match.get('live_line') != match.get('pre_line')

    # Movements
    line_dir, line_str = interpret_line_movement(match['open_line'], match['pre_line'], favorite_side,
                                                current_line=(match.get('live_line') if use_live_line else None))

    home_dir, home_str = calculate_directional_odds_movement(match['open_home'], match['pre_home'], home_live)
    away_dir, away_str = calculate_directional_odds_movement(match['open_away'], match['pre_away'], away_live)

    # Scores base
    home_score = 0.0
    away_score = 0.0

    # Weight constants
    W_LINE = 40.0
    W_ODDS = 35.0
    W_MARKET = 25.0

    # Line contribution
    if line_dir == 'HOME':
        home_score += W_LINE
    elif line_dir == 'AWAY':
        away_score += W_LINE
    else:
        home_score += W_LINE / 2.0
        away_score += W_LINE / 2.0

    # Odds contribution normalized by league volatility baseline
    # full W_ODDS when change >= league_volatility_pct
    home_score += W_ODDS * min(home_str / lv, 1.0) if home_dir == 'DROP' else 0.0
    away_score += W_ODDS * min(away_str / lv, 1.0) if away_dir == 'DROP' else 0.0

    # Market position
    if match['pre_home'] < match['pre_away']:
        home_score += W_MARKET
    else:
        away_score += W_MARKET

    # Correlation bonus: if line and odds align, small bonus
    if line_dir == 'HOME' and home_dir == 'DROP':
        home_score += 10.0
    if line_dir == 'AWAY' and away_dir == 'DROP':
        away_score += 10.0
    # correlation penalty
    if (line_dir == 'HOME' and home_dir == 'RISE') or (line_dir == 'AWAY' and away_dir == 'RISE'):
        if line_dir == 'HOME':
            home_score -= 8.0
        else:
            away_score -= 8.0

    # Avoid negative
    home_score = max(0.0, home_score)
    away_score = max(0.0, away_score)

    # Cap each side
    home_score = min(home_score, 70.0)
    away_score = min(away_score, 70.0)

    # Determine confidences
    total = home_score + away_score
    if total > 0:
        home_confidence = (home_score / total) * 100.0
        away_confidence = (away_score / total) * 100.0
    else:
        home_confidence = away_confidence = 50.0

    # Handle neutral markets conservatively
    if abs(match['pre_line']) <= 0.25 and abs(match['pre_home'] - match['pre_away']) < 0.05:
        # lower final confidence to reflect uncertainty
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
        'away_odds_change_pct': away_str
    }

# -----------------------
# Main UI / Analysis flow
# -----------------------
if st.session_state.matches:
    st.header("📊 Match Analysis Dashboard")
    last_match = st.session_state.matches[-1]

    # Prepare config dict from sidebar inputs (read current values from session-like variables)
    # If the expander wasn't opened, the variables still exist with defaults defined above.
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

        # Explanations (tooltips) in an expander
        with st.expander("Why this prediction? (explain metrics)"):
            st.write("- Line movement weighted 40% of score.")
            st.write("- Odds movement normalized by league volatility and weighted 35%.")
            st.write("- Current market position (which side has lower odds) weighted 25%.")
            st.write("- Correlation between line and odds adds a small reinforcement bonus; misalignment penalizes slightly.")
            st.write("- Neutral markets (small line and tiny odds difference) reduce final confidence to avoid overcommitting.")

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

        # Suspected result display with WIN/FAIL semantics
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
                    <h1 style='color: {'darkred' if fix_confidence >= 70 else 'darkorange' if fix_confidence >= 40 else 'darkgreen'}; font-size: 2.4rem; text-align: center;'>
                        {failing_team} (EXPECTED TO FAIL)
                    </h1>
                    <h3 style='text-align: center;'>Value Side / Opposite: {value_team}</h3>
                    <h2 style='text-align: center;'>Fix Confidence: {fix_confidence}%</h2>
                    <p style='text-align: center;'><strong>Asian Handicap (OPPOSITE SIDE VALUE):</strong> {ah_pick}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                win_side = suspected_result.split('_')[0]
                winning_team = last_match['home_team'] if win_side == 'HOME' else last_match['away_team']
                ah_pick = f"{'Home' if win_side=='HOME' else 'Away'} {last_match['pre_line']:+.2f}" if win_side=='HOME' else f"Away {(-last_match['pre_line']):+.2f}"
                st.markdown(f"""
                <div class="prediction-box {'suspicious' if fix_confidence >= 70 else 'medium-confidence' if fix_confidence >= 40 else 'high-confidence'}">
                    <h2>🎯 TEAM EXPECTED TO WIN</h2>
                    <h1 style='color: {'darkred' if fix_confidence >= 70 else 'darkorange' if fix_confidence >= 40 else 'darkgreen'}; font-size: 2.8rem; text-align: center;'>
                        {winning_team} (EXPECTED WIN)
                    </h1>
                    <h2 style='text-align: center;'>Fix Confidence: {fix_confidence}%</h2>
                    <p style='text-align: center;'><strong>Asian Handicap:</strong> {ah_pick}</p>
                </div>
                """, unsafe_allow_html=True)

            # Betting recommendation wording
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

        # Show flags and indicators
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

        # Gauges
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

        # Explanations
        with st.expander("What do these metrics mean?"):
            st.write("- Risk Score: aggregated suspicion level (higher = more suspicious).")
            st.write("- Fix Confidence: how confident the system is that the market reflects manipulation/fixing.")
            st.write("- Reverse correlation: when bookmakers shift line toward one side but that side's odds increase (money avoiding).")
            st.write("- Steam moves: rapid, synchronized movement suggesting sharp consensus (not necessarily a fix).")

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

        # Odds chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stages, y=home_odds, mode='lines+markers', name='Home Odds', line=dict(color='blue', width=3)))
        fig.add_trace(go.Scatter(x=stages, y=away_odds, mode='lines+markers', name='Away Odds', line=dict(color='red', width=3)))
        fig.update_layout(title="Odds Movement", xaxis_title="Stage", yaxis_title="Odds", height=420)
        st.plotly_chart(fig, use_container_width=True)

        # Line movement bar with polarity coloring and arrows
        colors = []
        annotations = []
        for i, l in enumerate(lines):
            # negative => home favorite (green tint), positive => away favorite (red tint)
            if l < 0:
                colors.append('rgba(0,153,51,0.6)')  # greenish
                arrows = "⬇️" if l < 0 else ""
            elif l > 0:
                colors.append('rgba(204,0,0,0.6)')  # reddish
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

                suspected_txt = "None"
                if analysis['suspected_result']:
                    sr = analysis['suspected_result']
                    if sr.endswith('_FAIL'):
                        suspected_txt = f"{match['home_team'] if sr.startswith('HOME') else match['away_team']} (EXPECTED TO FAIL)"
                    else:
                        suspected_txt = f"{match['home_team'] if sr.startswith('HOME') else match['away_team']} (EXPECTED WIN)"

                # store detailed metrics for backtesting / logging
                history_data.append({
                    'Match': f"{match['home_team']} vs {match['away_team']}",
                    'Suspected Result': suspected_txt,
                    'Fix Confidence': f"{analysis['fix_confidence']}%",
                    'Risk Score': f"{analysis['risk_score']}/100",
                    'Market Prediction': pred['predicted_winner'],
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
    - Favorite detection uses meaningful thresholds (±0.25, ±0.75) and corroborates with odds.
    - Odds movement normalized by league volatility; thresholds configurable per league/profile.
    - Line/odds correlation reinforced with small bonuses; misalignment penalizes.
    - Reverse correlation detection requires both meaningful line shift and opposite odds rise.
    - Steam moves consider timing (within configurable window after record time).
    - Live odds logic requires explicit 'Match has started' checkbox to avoid false pre-kickoff live feeds.
    - Margin alerts only flagged when aligned with other suspicious signals and with reduced weight.
    - Outcome locking on reverse detection; override only allowed by extreme live movements.
    - Refactored detection into helper functions for maintainability.
    - Thresholds exposed in sidebar in a collapsed expander so power users can tune per league/market.
    - History stores additional metrics for post-analysis/backtesting.
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>⚠️ Responsible Gambling Disclaimer</strong></p>
    <p>This tool is for educational and analytical purposes only. Always gamble responsibly.</p>
    <p>Past performance does not guarantee future results. Bet only what you can afford to lose.</p>
</div>
""", unsafe_allow_html=True)