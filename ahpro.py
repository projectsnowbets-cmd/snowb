import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(page_title="Asian Handicap Predictor", page_icon="⚽", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
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
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">⚽ Asian Handicap Betting Predictor & Anomaly Detector</div>', unsafe_allow_html=True)

# Initialize session state
if 'matches' not in st.session_state:
    st.session_state.matches = []

# Sidebar - Match Input
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
    
    st.markdown("**Live AH (Optional)**")
    live_ah_line = st.number_input("Line", value=0.0, step=0.25, key="live_line", format="%.2f")
    live_ah_home = st.number_input("Home Odds", value=1.80, step=0.01, key="live_home", format="%.2f")
    live_ah_away = st.number_input("Away Odds", value=2.00, step=0.01, key="live_away", format="%.2f")
    
    if st.button("🔍 Analyze Match", type="primary", use_container_width=True):
        if team_home and team_away:
            match_data = {
                'home_team': team_home,
                'away_team': team_away,
                'timestamp': datetime.now(),
                'open_line': open_ah_line,
                'open_home': open_ah_home,
                'open_away': open_ah_away,
                'pre_line': pre_ah_line,
                'pre_home': pre_ah_home,
                'pre_away': pre_ah_away,
                'live_line': live_ah_line,
                'live_home': live_ah_home,
                'live_away': live_ah_away
            }
            st.session_state.matches.append(match_data)
            st.success("✅ Match added for analysis!")
        else:
            st.error("Please enter both team names!")
    
    if st.button("🗑️ Clear All Data", use_container_width=True):
        st.session_state.matches = []
        st.rerun()

# Helper Functions
def calculate_implied_probability(odds):
    """Convert odds to implied probability"""
    if odds <= 1:
        return 0
    return 1 / odds

def calculate_margin(home_odds, away_odds):
    """Calculate bookmaker margin"""
    prob_home = calculate_implied_probability(home_odds)
    prob_away = calculate_implied_probability(away_odds)
    return (prob_home + prob_away - 1) * 100

def calculate_movement_score(open_val, pre_val, live_val=None):
    """Calculate odds movement severity"""
    movements = []
    
    # Opening to Pre-match
    if open_val != 0:
        pct_change = abs((pre_val - open_val) / open_val) * 100
        movements.append(pct_change)
    
    # Pre-match to Live
    if live_val and live_val != 0 and pre_val != 0:
        pct_change = abs((live_val - pre_val) / pre_val) * 100
        movements.append(pct_change)
    
    return sum(movements) / len(movements) if movements else 0

def detect_line_movement_pattern(open_line, pre_line, live_line):
    """Detect significant line movements"""
    total_movement = abs(live_line - open_line) if live_line else abs(pre_line - open_line)
    
    if total_movement >= 0.5:
        return "MAJOR", total_movement
    elif total_movement >= 0.25:
        return "MODERATE", total_movement
    else:
        return "MINOR", total_movement

def analyze_odds_direction(open_home, pre_home, live_home, open_away, pre_away, live_away):
    """Analyze if odds favor home or away based on movements"""
    home_movement = (pre_home - open_home) + (live_home - pre_home if live_home else 0)
    away_movement = (pre_away - open_away) + (live_away - pre_away if live_away else 0)
    
    # Lower odds = more money on that team
    if home_movement < -0.1 and away_movement > 0.1:
        return "HOME", abs(home_movement)
    elif away_movement < -0.1 and home_movement > 0.1:
        return "AWAY", abs(away_movement)
    else:
        return "NEUTRAL", 0

def detect_suspicious_patterns(match):
    """Advanced algorithm to detect match-fixing indicators with clear outcome predictions"""
    suspicious_flags = []
    risk_score = 0
    suspected_outcome = None  # Will be 'HOME', 'AWAY', or None
    manipulation_indicators = []
    fix_confidence = 0  # Confidence that match is fixed (0-100)
    
    # 1. Violent odds swings - WHO is being backed?
    home_move = calculate_movement_score(match['open_home'], match['pre_home'], match['live_home'])
    away_move = calculate_movement_score(match['open_away'], match['pre_away'], match['live_away'])
    
    # Lower odds = more money backing that team
    home_odds_dropped = match['pre_home'] < match['open_home']
    away_odds_dropped = match['pre_away'] < match['open_away']
    
    if home_move > 15 or away_move > 15:
        if home_move > away_move and home_odds_dropped:
            suspicious_flags.append(f"⚠️ Violent odds swing on HOME ({home_move:.1f}% drop) - Heavy money backing home team")
            suspected_outcome = 'HOME'
            manipulation_indicators.append("Massive betting volume on home win")
            fix_confidence += 40
        elif away_move > home_move and away_odds_dropped:
            suspicious_flags.append(f"⚠️ Violent odds swing on AWAY ({away_move:.1f}% drop) - Heavy money backing away team")
            suspected_outcome = 'AWAY'
            manipulation_indicators.append("Massive betting volume on away win")
            fix_confidence += 40
        risk_score += 30
    elif home_move > 8 or away_move > 8:
        if home_move > away_move and home_odds_dropped:
            suspicious_flags.append(f"⚡ Significant HOME odds drop ({home_move:.1f}%)")
            if not suspected_outcome:
                suspected_outcome = 'HOME'
            fix_confidence += 20
        elif away_move > home_move and away_odds_dropped:
            suspicious_flags.append(f"⚡ Significant AWAY odds drop ({away_move:.1f}%)")
            if not suspected_outcome:
                suspected_outcome = 'AWAY'
            fix_confidence += 20
        risk_score += 15
    
    # 2. Line movement - What do bookmakers expect?
    line_pattern, line_move = detect_line_movement_pattern(match['open_line'], match['pre_line'], match['live_line'])
    line_moved_to_home = match['pre_line'] < match['open_line']  # Line decreased = favoring home more
    
    if line_pattern == "MAJOR":
        if line_moved_to_home:
            suspicious_flags.append(f"🚨 Major line shift TOWARDS HOME ({line_move:.2f} goals) - Bookmakers expect home dominance")
            manipulation_indicators.append("Bookmakers drastically adjusted line favoring home")
            if not suspected_outcome:
                suspected_outcome = 'HOME'
            fix_confidence += 25
        else:
            suspicious_flags.append(f"🚨 Major line shift TOWARDS AWAY ({line_move:.2f} goals) - Bookmakers expect away dominance")
            manipulation_indicators.append("Bookmakers drastically adjusted line favoring away")
            if not suspected_outcome:
                suspected_outcome = 'AWAY'
            fix_confidence += 25
        risk_score += 25
    
    # 3. CRITICAL: Reverse movement (STRONGEST MATCH-FIXING INDICATOR)
    # Line moves one way but odds move opposite = market manipulation
    reverse_detected = False
    if match['pre_line'] < match['open_line'] and match['pre_home'] > match['open_home']:
        suspicious_flags.append("🔄 CRITICAL: Line favors HOME but home odds INCREASED - Possible manipulation to lose")
        suspected_outcome = 'AWAY'  # Opposite of what line suggests
        manipulation_indicators.append("⚠️ HOME TEAM SUSPECTED TO UNDERPERFORM/LOSE")
        risk_score += 35
        fix_confidence += 50  # HIGHEST indicator
        reverse_detected = True
    elif match['pre_line'] > match['open_line'] and match['pre_away'] > match['open_away']:
        suspicious_flags.append("🔄 CRITICAL: Line favors AWAY but away odds INCREASED - Possible manipulation to lose")
        suspected_outcome = 'HOME'  # Opposite of what line suggests
        manipulation_indicators.append("⚠️ AWAY TEAM SUSPECTED TO UNDERPERFORM/LOSE")
        risk_score += 35
        fix_confidence += 50  # HIGHEST indicator
        reverse_detected = True
    
    # 4. Unbalanced market - Bookmaker uncertainty
    margin_open = calculate_margin(match['open_home'], match['open_away'])
    margin_pre = calculate_margin(match['pre_home'], match['pre_away'])
    
    if abs(margin_pre - margin_open) > 3:
        suspicious_flags.append(f"💰 Unusual margin change ({abs(margin_pre - margin_open):.1f}%) - Bookmaker uncertainty or risk management")
        manipulation_indicators.append("Bookmakers protecting themselves from informed money")
        risk_score += 15
        fix_confidence += 15
    
    # 5. Live odds dramatically different - In-play manipulation
    if match['live_home'] and match['live_away']:
        live_home_change = (match['live_home'] - match['pre_home']) / match['pre_home'] * 100
        live_away_change = (match['live_away'] - match['pre_away']) / match['pre_away'] * 100
        
        if abs(live_home_change) > 20 or abs(live_away_change) > 20:
            if live_home_change < -20:  # Home odds dropped significantly
                suspicious_flags.append(f"🔴 Extreme live HOME backing ({abs(live_home_change):.1f}% odds drop)")
                if not reverse_detected:  # Don't override reverse correlation
                    suspected_outcome = 'HOME'
                manipulation_indicators.append("In-play money flooding home team")
                fix_confidence += 35
            elif live_away_change < -20:  # Away odds dropped significantly
                suspicious_flags.append(f"🔴 Extreme live AWAY backing ({abs(live_away_change):.1f}% odds drop)")
                if not reverse_detected:
                    suspected_outcome = 'AWAY'
                manipulation_indicators.append("In-play money flooding away team")
                fix_confidence += 35
            risk_score += 35
    
    # 6. Steam move detection (rapid odds change with line movement in same direction)
    if line_move > 0.25:
        if line_moved_to_home and home_odds_dropped:
            manipulation_indicators.append("🔥 STEAM MOVE on HOME - Synchronized sharp betting")
            fix_confidence += 10
        elif not line_moved_to_home and away_odds_dropped:
            manipulation_indicators.append("🔥 STEAM MOVE on AWAY - Synchronized sharp betting")
            fix_confidence += 10
    
    # If no clear outcome detected but there's suspicious activity, use odds movement
    if not suspected_outcome and risk_score > 0:
        if home_odds_dropped and home_move > away_move:
            suspected_outcome = 'HOME'
        elif away_odds_dropped and away_move > home_move:
            suspected_outcome = 'AWAY'
        else:
            # Use the team with lower final odds
            suspected_outcome = 'HOME' if match['pre_home'] < match['pre_away'] else 'AWAY'
    
    # Calculate final fix confidence (higher risk = higher fix confidence)
    fix_confidence = min(fix_confidence, 100)
    
    return suspicious_flags, min(risk_score, 100), suspected_outcome, manipulation_indicators, fix_confidence

def predict_outcome(match):
    """Intelligent prediction algorithm based on Asian Handicap movements"""
    # Calculate various factors
    
    # 1. Market efficiency score (lower margin = sharper line)
    margin_score = 100 - calculate_margin(match['pre_home'], match['pre_away']) * 10
    
    # 2. Odds direction analysis
    direction, strength = analyze_odds_direction(
        match['open_home'], match['pre_home'], match['live_home'],
        match['open_away'], match['pre_away'], match['live_away']
    )
    
    # 3. Line movement
    line_pattern, line_move = detect_line_movement_pattern(
        match['open_line'], match['pre_line'], match['live_line']
    )
    
    # 4. Implied probability shift
    open_home_prob = calculate_implied_probability(match['open_home']) * 100
    pre_home_prob = calculate_implied_probability(match['pre_home']) * 100
    live_home_prob = calculate_implied_probability(match['live_home']) * 100 if match['live_home'] else pre_home_prob
    
    open_away_prob = calculate_implied_probability(match['open_away']) * 100
    pre_away_prob = calculate_implied_probability(match['pre_away']) * 100
    live_away_prob = calculate_implied_probability(match['live_away']) * 100 if match['live_away'] else pre_away_prob
    
    # Calculate confidence based on multiple factors
    home_confidence = 0
    away_confidence = 0
    
    # Factor 1: Line movement (40% weight)
    if match['pre_line'] < match['open_line']:  # Line moved towards home
        home_confidence += 40
    elif match['pre_line'] > match['open_line']:  # Line moved towards away
        away_confidence += 40
    else:
        home_confidence += 20
        away_confidence += 20
    
    # Factor 2: Odds trends (35% weight)
    if direction == "HOME":
        home_confidence += 35 * (strength / 0.3)
    elif direction == "AWAY":
        away_confidence += 35 * (strength / 0.3)
    else:
        home_confidence += 17.5
        away_confidence += 17.5
    
    # Factor 3: Live market validation (25% weight)
    if match['live_home'] and match['live_away']:
        if live_home_prob > pre_home_prob:
            home_confidence += 25
        elif live_away_prob > pre_away_prob:
            away_confidence += 25
        else:
            home_confidence += 12.5
            away_confidence += 12.5
    else:
        # Use pre-match data
        if pre_home_prob > open_home_prob:
            home_confidence += 25
        elif pre_away_prob > open_away_prob:
            away_confidence += 25
        else:
            home_confidence += 12.5
            away_confidence += 12.5
    
    # Normalize confidence scores
    total = home_confidence + away_confidence
    home_confidence = (home_confidence / total) * 100
    away_confidence = (away_confidence / total) * 100
    
    # Determine prediction
    if home_confidence > away_confidence:
        predicted = "HOME"
        confidence = home_confidence
        ah_pick = f"Home {match['pre_line']}"
    else:
        predicted = "AWAY"
        confidence = away_confidence
        ah_pick = f"Away +{abs(match['pre_line'])}" if match['pre_line'] != 0 else "Away 0.0"
    
    # Additional betting suggestions
    suggestions = []
    
    # Over/Under suggestion based on line movement
    if line_move > 0.5:
        suggestions.append({
            'market': 'Over/Under',
            'pick': f"{'Over' if match['pre_line'] < match['open_line'] else 'Under'} 2.5",
            'confidence': min(60 + (line_move * 20), 85),
            'reason': 'Large line movement indicates goals expectation change'
        })
    
    # 1X2 suggestion
    if confidence > 70:
        suggestions.append({
            'market': '1X2',
            'pick': f"{'Home Win' if predicted == 'HOME' else 'Away Win'}",
            'confidence': confidence * 0.85,  # Slightly lower for 1X2
            'reason': 'Strong market consensus on winner'
        })
    
    # Correct score suggestion
    if abs(match['pre_line']) >= 1.5 and confidence > 75:
        if predicted == "HOME":
            score = "2-0 or 3-1"
        else:
            score = "0-1 or 1-2"
        suggestions.append({
            'market': 'Correct Score',
            'pick': score,
            'confidence': confidence * 0.6,
            'reason': 'High handicap line suggests clear favorite'
        })
    
    return {
        'predicted_winner': predicted,
        'confidence': confidence,
        'ah_pick': ah_pick,
        'market_efficiency': margin_score,
        'suggestions': suggestions,
        'analysis': {
            'home_prob_shift': live_home_prob - open_home_prob,
            'away_prob_shift': live_away_prob - open_away_prob,
            'line_movement': line_move,
            'direction_strength': strength
        }
    }

# Main Analysis Section
if st.session_state.matches:
    st.header("📊 Match Analysis Dashboard")
    
    # Analyze last match
    last_match = st.session_state.matches[-1]
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Predictions", "⚠️ Anomaly Detection", "📈 Odds Movement", "📋 History"])
    
    with tab1:
        st.subheader(f"🏟️ {last_match['home_team']} vs {last_match['away_team']}")
        
        prediction = predict_outcome(last_match)
        
        confidence_level = "high" if prediction['confidence'] > 75 else "medium" if prediction['confidence'] > 60 else "low"
        box_class = f"{confidence_level}-confidence"
        
        st.markdown(f"""
        <div class="prediction-box {box_class}">
            <h3>🎯 Primary Prediction: {prediction['predicted_winner']} Team</h3>
            <h2>Confidence: {prediction['confidence']:.1f}%</h2>
            <p><strong>Asian Handicap Pick:</strong> {prediction['ah_pick']} @ {last_match['pre_home'] if prediction['predicted_winner'] == 'HOME' else last_match['pre_away']}</p>
            <p><strong>Market Efficiency Score:</strong> {prediction['market_efficiency']:.1f}/100</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional Suggestions
        if prediction['suggestions']:
            st.subheader("💡 Additional Betting Suggestions")
            
            for sug in prediction['suggestions']:
                sug_confidence_level = "high" if sug['confidence'] > 75 else "medium" if sug['confidence'] > 60 else "low"
                sug_box_class = f"{sug_confidence_level}-confidence"
                
                st.markdown(f"""
                <div class="prediction-box {sug_box_class}">
                    <h4>{sug['market']}: {sug['pick']}</h4>
                    <p><strong>Confidence:</strong> {sug['confidence']:.1f}%</p>
                    <p><em>{sug['reason']}</em></p>
                </div>
                """, unsafe_allow_html=True)
        
        # Detailed Analysis
        with st.expander("🔎 Detailed Technical Analysis"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Home Probability Shift", f"{prediction['analysis']['home_prob_shift']:+.1f}%")
            with col2:
                st.metric("Away Probability Shift", f"{prediction['analysis']['away_prob_shift']:+.1f}%")
            with col3:
                st.metric("Line Movement", f"{prediction['analysis']['line_movement']:.2f}")
    
    with tab2:
        st.subheader("🚨 Match Integrity Analysis")
        
        flags, risk_score, suspected_outcome, manipulation_indicators, fix_confidence = detect_suspicious_patterns(last_match)
        
        # ALWAYS ensure we have a suspected outcome
        if not suspected_outcome:
            # Default to the team with better odds
            suspected_outcome = 'HOME' if last_match['pre_home'] < last_match['pre_away'] else 'AWAY'
            fix_confidence = 5  # Very low confidence if no clear signals
        
        team_name = last_match['home_team'] if suspected_outcome == 'HOME' else last_match['away_team']
        
        # Risk Assessment
        if risk_score >= 70:
            risk_level = "CRITICAL - HIGHLY SUSPICIOUS"
            risk_emoji = "🚨"
            confidence_label = "VERY HIGH FIX PROBABILITY"
        elif risk_score >= 40:
            risk_level = "ELEVATED - SUSPICIOUS PATTERNS"
            risk_emoji = "⚠️"
            confidence_label = "MODERATE FIX PROBABILITY"
        else:
            risk_level = "NORMAL MARKET BEHAVIOR"
            risk_emoji = "✅"
            confidence_label = "LOW/NO FIX PROBABILITY"
        
        # ALWAYS show suspected outcome prominently - even for "normal" matches
        st.markdown(f"""
        <div class="prediction-box {'suspicious' if risk_score >= 70 else 'medium-confidence' if risk_score >= 40 else 'high-confidence'}">
            <h2>{risk_emoji} Risk Level: {risk_level}</h2>
            <h3>Match Manipulation Score: {risk_score}/100</h3>
            <h3>{confidence_label}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # CRITICAL: Always show suspected outcome with confidence
        st.markdown(f"""
        <div class="prediction-box {'suspicious' if fix_confidence >= 70 else 'medium-confidence' if fix_confidence >= 40 else 'high-confidence'}">
            <h2>🎯 SUSPECTED FIXED OUTCOME</h2>
            <h1 style='color: {'darkred' if fix_confidence >= 70 else 'darkorange' if fix_confidence >= 40 else 'darkgreen'}; font-size: 3rem; text-align: center;'>
                {team_name} ({suspected_outcome}) TO WIN
            </h1>
            <h2 style='text-align: center; color: {'darkred' if fix_confidence >= 70 else 'darkorange' if fix_confidence >= 40 else 'darkgreen'};'>
                Fix Confidence: {fix_confidence}%
            </h2>
            <p style='font-size: 1.3rem; text-align: center;'>
                <strong>{'🚨 EXTREMELY HIGH MANIPULATION SIGNALS' if fix_confidence >= 70 else '⚠️ Moderate manipulation signals detected' if fix_confidence >= 40 else 'ℹ️ Normal market - low manipulation signals'}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show manipulation indicators if any
        if manipulation_indicators:
            st.subheader("🔍 Manipulation Evidence")
            for indicator in manipulation_indicators:
                if "SUSPECTED TO" in indicator:
                    st.error(indicator)
                elif "STEAM MOVE" in indicator or "Massive betting" in indicator:
                    st.warning(indicator)
                else:
                    st.info(indicator)
        
        # ALWAYS show betting recommendation with confidence percentages
        st.subheader("💡 Betting Recommendation")
        
        if fix_confidence >= 70:
            st.markdown(f"""
            <div class="prediction-box suspicious">
                <h3>🎯 STRONG BET RECOMMENDATION</h3>
                <h2 style='color: darkred;'>Bet on: {team_name} ({suspected_outcome})</h2>
                <p><strong>Fix Confidence: {fix_confidence}%</strong> - Very high manipulation signals detected</p>
                <p><strong>Recommended Stake:</strong> 150-200% of normal stake (aggressive)</p>
                <p><strong>Asian Handicap:</strong> {team_name} {last_match['pre_line'] if suspected_outcome == 'HOME' else f"+{abs(last_match['pre_line'])}"}</p>
                <p><strong>1X2:</strong> {team_name} to Win</p>
                <p><em>⚠️ High confidence but remember: no bet is 100% certain. This is based on market manipulation signals, not inside information.</em></p>
            </div>
            """, unsafe_allow_html=True)
        elif fix_confidence >= 40:
            st.markdown(f"""
            <div class="prediction-box medium-confidence">
                <h3>⚠️ MODERATE BET RECOMMENDATION</h3>
                <h2 style='color: darkorange;'>Consider betting: {team_name} ({suspected_outcome})</h2>
                <p><strong>Fix Confidence: {fix_confidence}%</strong> - Moderate manipulation signals</p>
                <p><strong>Recommended Stake:</strong> 75-100% of normal stake</p>
                <p><strong>Asian Handicap:</strong> {team_name} {last_match['pre_line'] if suspected_outcome == 'HOME' else f"+{abs(last_match['pre_line'])}"}</p>
                <p><em>Market shows some unusual patterns. Bet with caution.</em></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-box high-confidence">
                <h3>ℹ️ STANDARD BET (Normal Market)</h3>
                <h2 style='color: darkgreen;'>Market suggests: {team_name} ({suspected_outcome})</h2>
                <p><strong>Fix Confidence: {fix_confidence}%</strong> - Normal market behavior, no manipulation detected</p>
                <p><strong>Recommended Stake:</strong> Standard stake (no special signals)</p>
                <p><strong>Asian Handicap:</strong> {team_name} {last_match['pre_line'] if suspected_outcome == 'HOME' else f"+{abs(last_match['pre_line'])}"}</p>
                <p><em>This is a normal market. Bet based on your own analysis and standard betting strategy.</em></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show all detected anomalies
        if flags:
            st.subheader("📋 Detected Patterns")
            for flag in flags:
                if "CRITICAL" in flag:
                    st.error(flag)
                elif "Major" in flag or "Violent" in flag or "Extreme" in flag:
                    st.warning(flag)
                else:
                    st.info(flag)
        else:
            st.success("✅ No unusual patterns detected. Market appears clean.")
        
        # Visual Risk Gauge
        col1, col2 = st.columns(2)
        
        with col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk/Suspicion Score"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred" if risk_score >= 70 else "orange" if risk_score >= 40 else "darkgreen"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            fig_confidence = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fix_confidence,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Fix Confidence"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred" if fix_confidence >= 70 else "orange" if fix_confidence >= 40 else "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightblue"},
                        {'range': [40, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            st.plotly_chart(fig_confidence, use_container_width=True)
        
        with st.expander("ℹ️ Understanding the Scores"):
            st.markdown(f"""
            **📊 Current Analysis:**
            - **Suspected Winner:** {team_name} ({suspected_outcome})
            - **Fix Confidence:** {fix_confidence}% - {'VERY HIGH - Strong manipulation signals' if fix_confidence >= 70 else 'MODERATE - Some suspicious patterns' if fix_confidence >= 40 else 'LOW - Normal market behavior'}
            - **Risk Score:** {risk_score}/100 - {'CRITICAL - High manipulation detected' if risk_score >= 70 else 'ELEVATED - Unusual patterns' if risk_score >= 40 else 'NORMAL - Clean market'}
            
            ---
            
            **Understanding Fix Confidence:**
            - **70-100%**: Very high probability of match manipulation. STRONG BET on suspected outcome.
            - **40-69%**: Moderate manipulation signals. Bet with caution on suspected outcome.
            - **0-39%**: Normal market. No clear manipulation. Use standard betting strategy.
            
            **Key Indicators (ordered by importance):**
            
            1. **🔄 REVERSE CORRELATION (50 points)** - STRONGEST INDICATOR
               - Line moves one way, odds move opposite
               - Example: Line favors home but home odds increase
               - This is unnatural and suggests one team is "arranged" to lose
               - **Winner = opposite of what line suggests**
            
            2. **🔴 Violent Odds Swings (40 points)**
               - Odds drop >15% = massive money on that team
               - Usually from insider/informed betting
               - **Winner = team whose odds dropped dramatically**
            
            3. **💥 Extreme Live Movements (35 points)**
               - Live odds change >20% = in-play manipulation
               - **Winner = team with dropping live odds**
            
            4. **🚨 Major Line Shifts (25 points)**
               - Line moves ≥0.5 goals
               - Bookmakers drastically adjusting expectations
               - **Winner = team line moved toward**
            
            5. **⚡ Significant Odds Movement (20 points)**
               - 8-15% odds changes
               - Sharp money detected
            
            6. **💰 Margin Changes (15 points)**
               - Bookmaker protecting from informed bets
            
            7. **🔥 Steam Moves (10 points)**
               - Synchronized betting across multiple books
            
            **How to Use:**
            - **Fix Confidence 70+**: Aggressive bet recommended
            - **Fix Confidence 40-69**: Moderate bet with caution
            - **Fix Confidence 0-39**: Standard analysis, no special signals
            
            **Remember:** High confidence means strong market manipulation signals are detected, but doesn't guarantee outcome. Always bet responsibly.
            """)
    
    with tab3:
        st.subheader("📈 Odds Movement Visualization")
        
        # Create odds movement chart
        stages = ['Opening', 'Pre-match', 'Live']
        home_odds = [last_match['open_home'], last_match['pre_home'], last_match['live_home'] or last_match['pre_home']]
        away_odds = [last_match['open_away'], last_match['pre_away'], last_match['live_away'] or last_match['pre_away']]
        lines = [last_match['open_line'], last_match['pre_line'], last_match['live_line'] or last_match['pre_line']]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=stages, y=home_odds,
            mode='lines+markers',
            name='Home Odds',
            line=dict(color='blue', width=3),
            marker=dict(size=10)
        ))
        
        fig.add_trace(go.Scatter(
            x=stages, y=away_odds,
            mode='lines+markers',
            name='Away Odds',
            line=dict(color='red', width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title="Odds Movement Timeline",
            xaxis_title="Stage",
            yaxis_title="Odds",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Line movement chart
        fig2 = go.Figure()
        
        fig2.add_trace(go.Bar(
            x=stages,
            y=lines,
            text=[f"{l:+.2f}" for l in lines],
            textposition='auto',
            marker_color=['lightblue', 'royalblue', 'darkblue']
        ))
        
        fig2.update_layout(
            title="Handicap Line Movement",
            xaxis_title="Stage",
            yaxis_title="Asian Handicap Line",
            height=350
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Statistics table
        st.subheader("📊 Movement Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Home Team**")
            home_change = last_match['pre_home'] - last_match['open_home']
            home_pct = (home_change / last_match['open_home']) * 100
            st.metric("Opening → Pre-match", f"{last_match['pre_home']:.2f}", f"{home_change:+.2f} ({home_pct:+.1f}%)")
            
            if last_match['live_home']:
                live_change = last_match['live_home'] - last_match['pre_home']
                live_pct = (live_change / last_match['pre_home']) * 100
                st.metric("Pre-match → Live", f"{last_match['live_home']:.2f}", f"{live_change:+.2f} ({live_pct:+.1f}%)")
        
        with col2:
            st.markdown("**Away Team**")
            away_change = last_match['pre_away'] - last_match['open_away']
            away_pct = (away_change / last_match['open_away']) * 100
            st.metric("Opening → Pre-match", f"{last_match['pre_away']:.2f}", f"{away_change:+.2f} ({away_pct:+.1f}%)")
            
            if last_match['live_away']:
                live_away_change = last_match['live_away'] - last_match['pre_away']
                live_away_pct = (live_away_change / last_match['pre_away']) * 100
                st.metric("Pre-match → Live", f"{last_match['live_away']:.2f}", f"{live_away_change:+.2f} ({live_away_pct:+.1f}%)")
    
    with tab4:
        st.subheader("📋 Analysis History")
        
        if len(st.session_state.matches) > 0:
            history_data = []
            
            for match in st.session_state.matches:
                pred = predict_outcome(match)
                flags, risk, suspected_outcome, manip_indicators, fix_conf = detect_suspicious_patterns(match)
                
                # Ensure suspected_outcome exists
                if not suspected_outcome:
                    suspected_outcome = 'HOME' if match['pre_home'] < match['pre_away'] else 'AWAY'
                
                team_name = match['home_team'] if suspected_outcome == 'HOME' else match['away_team']
                
                history_data.append({
                    'Match': f"{match['home_team']} vs {match['away_team']}",
                    'Suspected Winner': f"{team_name} ({suspected_outcome})",
                    'Fix Confidence': f"{fix_conf}%",
                    'Risk Score': f"{risk}/100",
                    'Prediction': pred['predicted_winner'],
                    'Pred. Confidence': f"{pred['confidence']:.1f}%",
                    'AH Pick': pred['ah_pick'],
                    'Time': match['timestamp'].strftime("%Y-%m-%d %H:%M")
                })
            
            df = pd.DataFrame(history_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Export option
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Analysis History (CSV)",
                data=csv,
                file_name=f"betting_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No match history yet. Add matches using the sidebar.")

else:
    st.info("👈 Enter match data in the sidebar to start analysis")
    
    st.markdown("""
    ### How This Predictor Works
    
    This advanced Asian Handicap predictor uses multiple intelligent algorithms:
    
    #### 🎯 Prediction Engine
    - **Market Movement Analysis**: Tracks line and odds changes from opening to live
    - **Implied Probability Shifts**: Calculates true probability changes
    - **Sharp Money Detection**: Identifies where professional bettors are placing money
    - **Multi-Factor Confidence Scoring**: Weighs various indicators for accuracy
    
    #### 🚨 Anomaly Detection & Fix Detection
    - **Violent Swing Detection**: Flags unusual odds movements (>15%)
    - **Reverse Correlation Analysis**: Identifies line-odds contradictions (strongest fix indicator)
    - **Market Efficiency Tracking**: Monitors bookmaker margin changes
    - **Live Market Validation**: Compares live behavior to pre-match expectations
    - **Fix Confidence Score**: Calculates probability of match manipulation (0-100%)
    
    #### 💡 Betting Recommendations
    - **70%+ Fix Confidence**: STRONG BET - Aggressive stake (150-200%)
    - **40-69% Fix Confidence**: MODERATE BET - Cautious stake (75-100%)
    - **0-39% Fix Confidence**: STANDARD BET - Normal stake
    - Asian Handicap picks with precise confidence levels
    - Over/Under recommendations based on line movement
    - 1X2 predictions when market consensus is strong
    
    **Start by entering your match data in the sidebar!**
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>⚠️ Responsible Gambling Disclaimer</strong></p>
    <p>This tool is for educational and analytical purposes only. Always gamble responsibly.</p>
    <p>Past performance does not guarantee future results. Bet only what you can afford to lose.</p>
</div>
""", unsafe_allow_html=True)