import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import io
import re
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Fixed Match Detector", layout="wide", page_icon="⚽")

# Title and description
st.title("⚽ Fixed Match Detector App")
st.markdown("""
This app analyzes betting odds movements, volumes, and moneyway distributions to detect potential match-fixing patterns.
Paste your betting data below and click **Analyze Data** to begin.
""")

# Sidebar for thresholds
st.sidebar.header("⚙️ Detection Thresholds")
st.sidebar.markdown("Adjust sensitivity for anomaly detection:")

# Preset configurations
preset = st.sidebar.selectbox(
    "🎯 Threshold Preset",
    ["Custom", "Conservative (Low False Positives)", "Balanced (Recommended)", "Aggressive (High Sensitivity)"],
    index=2,
    help="Choose a preset or customize manually"
)

# Set default values based on preset
if preset == "Conservative (Low False Positives)":
    default_odd_change = 0.8
    default_volume_spike = 7.0
    default_moneyway = 85
    default_zscore = 3.5
elif preset == "Balanced (Recommended)":
    default_odd_change = 0.5
    default_volume_spike = 5.0
    default_moneyway = 75
    default_zscore = 3.0
elif preset == "Aggressive (High Sensitivity)":
    default_odd_change = 0.3
    default_volume_spike = 3.0
    default_moneyway = 65
    default_zscore = 2.5
else:  # Custom
    default_odd_change = 0.5
    default_volume_spike = 5.0
    default_moneyway = 80
    default_zscore = 3.0

odd_change_threshold = st.sidebar.slider(
    "Odd Change Threshold", 
    min_value=0.1, max_value=2.0, value=default_odd_change, step=0.1,
    help="Lower = more sensitive. 0.5 means flag changes >0.5 odds units",
    disabled=(preset != "Custom")
)

volume_spike_multiplier = st.sidebar.slider(
    "Volume Spike Multiplier", 
    min_value=2.0, max_value=10.0, value=default_volume_spike, step=0.5,
    help="Lower = more sensitive. 5x means flag volumes 5× above average",
    disabled=(preset != "Custom")
)

moneyway_threshold = st.sidebar.slider(
    "Moneyway Imbalance (%)", 
    min_value=60, max_value=95, value=default_moneyway, step=5,
    help="Lower = more sensitive. 75% means flag when one outcome gets >75% of bets",
    disabled=(preset != "Custom")
)

z_score_threshold = st.sidebar.slider(
    "Z-Score Threshold (σ)", 
    min_value=2.0, max_value=4.0, value=default_zscore, step=0.5,
    help="Lower = more sensitive. 3σ captures ~99.7% of normal distribution",
    disabled=(preset != "Custom")
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 💡 Threshold Guide

**For Fixed Match Detection:**

**Conservative** - Best when:
- You want high confidence
- Avoiding false accusations
- Need strong evidence

**Balanced** ✅ - Best when:
- General analysis
- Most match-fixing cases
- Good accuracy/sensitivity trade-off

**Aggressive** - Best when:
- Early warning system
- Research/investigation
- Don't mind false positives

**Key Indicators:**
- 🔴 Odd drops >20% without score change
- 🔴 Late pre-match movements
- 🔴 Volume spikes with odd decreases
- 🟡 High moneyway concentration (>75%)
""")

st.sidebar.markdown("---")
st.sidebar.info("""
**Recommended for Fixed Matches:**
Use **Balanced** preset and look for:
- Suspicion Score >60
- Prediction Confidence >70%
- Multiple high-severity flags
""")

# Main input area
st.header("📊 Data Input")
data_input = st.text_area(
    "Paste Betting Data Here",
    height=300,
    placeholder="Paste your betting data (tab-separated format)...",
    help="Copy and paste betting odds data from your source"
)

analyze_button = st.button("🔍 Analyze Data", type="primary")


def parse_betting_data(raw_text):
    """Parse raw betting data text into a structured DataFrame."""
    try:
        lines = raw_text.strip().split('\n')
        
        if len(lines) < 4:
            st.error("Not enough data. Please paste at least 4 rows including headers.")
            return None
        
        # Find where actual data starts (skip headers: 1, X, 2, TimeScore row)
        data_start_idx = 0
        for i, line in enumerate(lines):
            # Look for the header line that contains "TimeScore" or similar
            if 'Time' in line or 'Score' in line:
                data_start_idx = i + 1
                break
            # Or if we see just "1", "X", "2" in first few lines
            if i < 5 and line.strip() in ['1', 'X', '2']:
                continue
            else:
                data_start_idx = i
                break
        
        # If no clear header found, assume first 3 lines are headers (1, X, 2)
        if data_start_idx == 0:
            data_start_idx = 3
        
        # Get data lines
        data_lines = [line for line in lines[data_start_idx:] if line.strip()]
        
        if not data_lines:
            st.error("No data rows found after headers. Please check the format.")
            return None
        
        # Parse each line
        rows = []
        for line in data_lines:
            # Try different splitting methods
            parts = re.split(r'\t+', line.strip())
            
            if len(parts) < 15:
                parts = re.split(r'\s{2,}', line.strip())
            
            if len(parts) < 15:
                parts = line.strip().split('\t')
            
            # Handle both 16 and 17 column formats
            # Format with Time+Score combined (16 cols): Time, Odd1, Change1, Volume1, Agg1, Pct1, ...
            # Format with Time and Score separate (17 cols): Time, Score, Odd1, Change1, ...
            
            if len(parts) == 16:
                # Split first column into Time and Score if it contains a score pattern
                time_col = parts[0]
                
                # Check if this is a time with embedded score (like "01-04 11:55" - no score in pre-match)
                # or live time with score (like "45'" with score in next data)
                # For pre-match, there's usually no score, so we add empty score
                if re.match(r'^\d{2}-\d{2}\s+\d{1,2}:\d{2}', time_col):
                    # Pre-match format: just time, no score yet
                    rows.append([time_col, '0-0'] + parts[1:15])
                elif "'" in time_col or time_col in ['HT', 'Inplay']:
                    # Live format: time might have score in data
                    rows.append([time_col, '0-0'] + parts[1:15])
                else:
                    rows.append([time_col, '0-0'] + parts[1:15])
                    
            elif len(parts) >= 17:
                rows.append(parts[:17])
            else:
                # Skip lines with too few columns (like header rows)
                if len(parts) > 10:  # Only warn about data-like rows
                    st.warning(f"Skipping line with {len(parts)} columns: {line[:80]}...")
        
        if not rows:
            st.error("Could not parse any valid data rows.")
            return None
        
        # Create DataFrame
        columns = [
            'Time', 'Score', 
            'Odd1', 'Change1', 'Volume1', 'Agg1', 'Pct1',
            'OddX', 'ChangeX', 'VolumeX', 'AggX', 'PctX',
            'Odd2', 'Change2', 'Volume2', 'Agg2', 'Pct2'
        ]
        
        df = pd.DataFrame(rows, columns=columns)
        
        # Clean and convert data types
        df = clean_data(df)
        
        # Reverse to chronological order (earliest first)
        df = df.iloc[::-1].reset_index(drop=True)
        
        # Add phase column
        def detect_phase(time_str):
            time_str = str(time_str).strip()
            
            # Half-time marker
            if time_str.upper() == 'HT':
                return 'Half-time'
            
            # Inplay marker
            if time_str.lower() == 'inplay':
                return 'Transition'
            
            # Pre-match: DD-MM HH:MM format
            if re.match(r'^\d{2}-\d{2}\s+\d{1,2}:\d{2}', time_str):
                return 'Pre-match'
            
            # Also without space
            if re.match(r'^\d{2}-\d{2}\d{1,2}:\d{2}', time_str):
                return 'Pre-match'
            
            # Live: apostrophe format
            if "'" in time_str:
                return 'Live'
            
            return 'Unknown'
        
        df['Phase'] = df['Time'].apply(detect_phase)
        
        # Show phase distribution
        phase_counts = df['Phase'].value_counts()
        if 'Unknown' in phase_counts.index and phase_counts['Unknown'] > 0:
            st.warning(f"⚠️ {phase_counts['Unknown']} rows with unknown phase")
            unknown_samples = df[df['Phase'] == 'Unknown']['Time'].head(5).tolist()
            st.code(f"Unknown times: {unknown_samples}")
        
        # Add total aggregated volume
        df['Total_Agg'] = df['Agg1'] + df['AggX'] + df['Agg2']
        
        # Add row index
        df['Index'] = range(len(df))
        
        # Success message
        prematch_count = len(df[df['Phase'] == 'Pre-match'])
        live_count = len(df[df['Phase'] == 'Live'])
        st.success(f"✅ Parsed {len(df)} rows: **{prematch_count} pre-match**, {live_count} live")
        
        return df
    
    except Exception as e:
        st.error(f"Error parsing data: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None


def clean_data(df):
    """Clean and convert data types."""
    # Remove currency symbols and convert to float
    for col in ['Volume1', 'VolumeX', 'Volume2', 'Agg1', 'AggX', 'Agg2']:
        df[col] = df[col].str.replace('€', '').str.replace(' ', '').replace('-', '0').replace('', '0')
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Convert percentages
    for col in ['Pct1', 'PctX', 'Pct2']:
        df[col] = df[col].str.replace('%', '').replace('-', '0').replace('', '0')
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) / 100
    
    # Convert odds
    for col in ['Odd1', 'OddX', 'Odd2']:
        df[col] = df[col].replace('-', np.nan)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert odd changes
    for col in ['Change1', 'ChangeX', 'Change2']:
        df[col] = df[col].replace('-', '0').replace('', '0')
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df


def detect_anomalies(df, thresholds):
    """Detect suspicious patterns using multiple methods."""
    flags = []
    suspicion_score = 0
    
    # Initialize pick confidence tracking
    pick_signals = {'1': 0, 'X': 0, '2': 0}  # Tracks signals for each outcome
    
    # Prepare features for IsolationForest
    features = df[['Change1', 'ChangeX', 'Change2', 'Volume1', 'VolumeX', 'Volume2', 
                    'Pct1', 'PctX', 'Pct2']].fillna(0)
    
    # Apply IsolationForest
    if len(df) > 10:
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        df['Anomaly_IF'] = iso_forest.fit_predict(features)
        anomaly_indices = df[df['Anomaly_IF'] == -1].index.tolist()
        
        for idx in anomaly_indices:
            flags.append({
                'Time': df.loc[idx, 'Time'],
                'Score': df.loc[idx, 'Score'],
                'Reason': 'Statistical anomaly detected (IsolationForest)',
                'Severity': 'Medium'
            })
            suspicion_score += 10
    
    # Calculate z-scores for odd changes
    for outcome in ['1', 'X', '2']:
        change_col = f'Change{outcome}'
        df[f'Z_Change{outcome}'] = np.abs((df[change_col] - df[change_col].mean()) / df[change_col].std())
        
        extreme_changes = df[df[f'Z_Change{outcome}'] > thresholds['z_score']]
        for idx in extreme_changes.index:
            if df.loc[idx, change_col] != 0:
                flags.append({
                    'Time': df.loc[idx, 'Time'],
                    'Score': df.loc[idx, 'Score'],
                    'Reason': f'Extreme odd change on {outcome}: {df.loc[idx, change_col]:.2f} ({df.loc[idx, f"Z_Change{outcome}"]:.1f}σ)',
                    'Severity': 'High'
                })
                suspicion_score += 15
    
    # Detect sudden odd drops
    for i in range(1, len(df)):
        prev_score = df.loc[i-1, 'Score']
        curr_score = df.loc[i, 'Score']
        
        # Check if score hasn't changed
        if prev_score == curr_score:
            for outcome in ['1', 'X', '2']:
                odd_col = f'Odd{outcome}'
                change_col = f'Change{outcome}'
                
                if pd.notna(df.loc[i, odd_col]) and pd.notna(df.loc[i-1, odd_col]):
                    pct_change = (df.loc[i, odd_col] - df.loc[i-1, odd_col]) / df.loc[i-1, odd_col]
                    
                    if pct_change < -0.2:  # 20% drop
                        flags.append({
                            'Time': df.loc[i, 'Time'],
                            'Score': df.loc[i, 'Score'],
                            'Reason': f'Sudden odd drop on {outcome}: {pct_change*100:.1f}% without score change',
                            'Severity': 'High'
                        })
                        suspicion_score += 20
                        pick_signals[outcome] += 25  # Strong signal for this outcome
    
    # Detect volume spikes
    for outcome in ['1', 'X', '2']:
        vol_col = f'Volume{outcome}'
        avg_volume = df[vol_col].mean()
        
        if avg_volume > 0:
            spikes = df[df[vol_col] > avg_volume * thresholds['volume_spike']]
            for idx in spikes.index:
                flags.append({
                    'Time': df.loc[idx, 'Time'],
                    'Score': df.loc[idx, 'Score'],
                    'Reason': f'Volume spike on {outcome}: {df.loc[idx, vol_col]:.0f}€ ({df.loc[idx, vol_col]/avg_volume:.1f}x average)',
                    'Severity': 'Medium'
                })
                suspicion_score += 12
                pick_signals[outcome] += 15  # Volume spike indicates insider action
    
    # Detect moneyway distortions
    for idx in df.index:
        for outcome, pct_col in [('1', 'Pct1'), ('X', 'PctX'), ('2', 'Pct2')]:
            if df.loc[idx, pct_col] > thresholds['moneyway'] / 100:
                flags.append({
                    'Time': df.loc[idx, 'Time'],
                    'Score': df.loc[idx, 'Score'],
                    'Reason': f'Moneyway imbalance: {df.loc[idx, pct_col]*100:.0f}% on {outcome}',
                    'Severity': 'Medium'
                })
                suspicion_score += 10
                pick_signals[outcome] += 20  # Heavy moneyway suggests insider knowledge
    
    # Detect odds moving against volume
    for i in range(1, len(df)):
        for outcome in ['1', 'X', '2']:
            vol_col = f'Volume{outcome}'
            odd_col = f'Odd{outcome}'
            
            if df.loc[i, vol_col] > 50:  # Significant volume
                if pd.notna(df.loc[i, odd_col]) and pd.notna(df.loc[i-1, odd_col]):
                    odd_change = df.loc[i, odd_col] - df.loc[i-1, odd_col]
                    
                    # High volume should decrease odds, but if odds increase, it's suspicious
                    if odd_change > 0.1 and df.loc[i, vol_col] > df[vol_col].mean() * 2:
                        flags.append({
                            'Time': df.loc[i, 'Time'],
                            'Score': df.loc[i, 'Score'],
                            'Reason': f'Odds moving against volume on {outcome}: high volume ({df.loc[i, vol_col]:.0f}€) but odds increased',
                            'Severity': 'High'
                        })
                        suspicion_score += 18
                        pick_signals[outcome] += 22  # Counter-intuitive movement is very suspicious
    
    # Analyze late pre-match movements (strong indicator)
    prematch_df = df[df['Phase'] == 'Pre-match']
    if len(prematch_df) > 5:
        # Get last 10 pre-match records (or all if less than 10)
        last_prematch = prematch_df.tail(min(10, len(prematch_df)))
        
        # Focus on the last hour of pre-match (most critical period)
        for outcome in ['1', 'X', '2']:
            odd_col = f'Odd{outcome}'
            vol_col = f'Volume{outcome}'
            pct_col = f'Pct{outcome}'
            
            valid_odds = last_prematch[odd_col].dropna()
            if len(valid_odds) >= 3:
                first_odd = valid_odds.iloc[0]
                last_odd = valid_odds.iloc[-1]
                
                if first_odd > 0:
                    pct_change = (last_odd - first_odd) / first_odd
                    
                    # Sharp drop in late pre-match is VERY suspicious
                    if pct_change < -0.15:  # 15% drop
                        pick_signals[outcome] += 35  # Strongest signal
                        flags.append({
                            'Time': 'Late Pre-match',
                            'Score': last_prematch['Score'].iloc[-1],
                            'Reason': f'Sharp late pre-match odd drop on {outcome}: {pct_change*100:.1f}% (from {first_odd:.2f} to {last_odd:.2f})',
                            'Severity': 'High'
                        })
                        suspicion_score += 30
                    elif pct_change < -0.10:  # 10% drop still notable
                        pick_signals[outcome] += 20
                        flags.append({
                            'Time': 'Late Pre-match',
                            'Score': last_prematch['Score'].iloc[-1],
                            'Reason': f'Moderate late pre-match odd drop on {outcome}: {pct_change*100:.1f}%',
                            'Severity': 'Medium'
                        })
                        suspicion_score += 15
            
            # Check for volume concentration in late pre-match
            late_volume = last_prematch[vol_col].sum()
            total_prematch_volume = prematch_df[vol_col].sum()
            if total_prematch_volume > 0:
                volume_concentration = late_volume / total_prematch_volume
                if volume_concentration > 0.5 and late_volume > 20:  # >50% of volume in last moments
                    pick_signals[outcome] += 15
                    flags.append({
                        'Time': 'Late Pre-match',
                        'Score': last_prematch['Score'].iloc[-1],
                        'Reason': f'Heavy late betting on {outcome}: {volume_concentration*100:.0f}% of pre-match volume',
                        'Severity': 'Medium'
                    })
                    suspicion_score += 12
    
    # Analyze live betting patterns
    live_df = df[df['Phase'] == 'Live']
    if len(live_df) > 3:
        for outcome in ['1', 'X', '2']:
            pct_col = f'Pct{outcome}'
            avg_pct = live_df[pct_col].mean()
            if avg_pct > 0.6:  # Consistently high moneyway during live
                pick_signals[outcome] += 18
    
    # Determine pick and confidence
    max_signal = max(pick_signals.values())
    predicted_outcome = max(pick_signals, key=pick_signals.get)
    
    # Calculate confidence score (0-100)
    if max_signal > 0:
        # Normalize to 0-100 scale
        confidence_score = min(100, int((max_signal / 150) * 100))  # 150 is max possible signal
        
        # Adjust based on signal difference
        sorted_signals = sorted(pick_signals.values(), reverse=True)
        if len(sorted_signals) > 1 and sorted_signals[0] > sorted_signals[1]:
            signal_gap = sorted_signals[0] - sorted_signals[1]
            confidence_score = min(100, confidence_score + int(signal_gap / 5))
    else:
        confidence_score = 0
        predicted_outcome = None
    
    # Outcome name mapping
    outcome_names = {'1': 'Home Win', 'X': 'Draw', '2': 'Away Win'}
    
    # Cap suspicion score at 100
    suspicion_score = min(suspicion_score, 100)
    
    pick_info = {
        'predicted_outcome': predicted_outcome,
        'outcome_name': outcome_names.get(predicted_outcome, 'Unknown'),
        'confidence': confidence_score,
        'signals': pick_signals
    }
    
    return flags, suspicion_score, df, pick_info


def plot_odds_over_time(df):
    """Create interactive line chart for odds over time."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Index'], y=df['Odd1'], mode='lines+markers',
        name='Home Win (1)', line=dict(color='#2ecc71', width=2),
        hovertemplate='Time: %{text}<br>Odd: %{y:.2f}<extra></extra>',
        text=df['Time']
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Index'], y=df['OddX'], mode='lines+markers',
        name='Draw (X)', line=dict(color='#f39c12', width=2),
        hovertemplate='Time: %{text}<br>Odd: %{y:.2f}<extra></extra>',
        text=df['Time']
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Index'], y=df['Odd2'], mode='lines+markers',
        name='Away Win (2)', line=dict(color='#e74c3c', width=2),
        hovertemplate='Time: %{text}<br>Odd: %{y:.2f}<extra></extra>',
        text=df['Time']
    ))
    
    # Mark pre-match to live transition
    transition_idx = df[df['Phase'] == 'Transition'].index
    if len(transition_idx) > 0:
        trans_idx = transition_idx[0]
        fig.add_vline(x=trans_idx, line_dash="dash", line_color="purple", line_width=3, opacity=0.7)
        fig.add_annotation(
            x=trans_idx, y=df[['Odd1', 'OddX', 'Odd2']].max().max(),
            text="⚽ KICKOFF",
            showarrow=False, yshift=15, font=dict(size=14, color="purple", family="Arial Black")
        )
    
    # Add score changes as annotations (only for actual goals, not 0-0)
    prev_score = None
    for idx in df.index:
        curr_score = df.loc[idx, 'Score']
        
        # Only annotate if:
        # 1. Score has changed from previous
        # 2. Score is not "0-0" (not the initial state)
        # 3. We're in live phase
        if (prev_score is not None and 
            curr_score != prev_score and 
            curr_score != "0-0" and
            df.loc[idx, 'Phase'] == 'Live'):
            
            fig.add_vline(x=idx, line_dash="dash", line_color="red", opacity=0.4)
            fig.add_annotation(
                x=idx, y=df[['Odd1', 'OddX', 'Odd2']].max().max() * 0.95,
                text=f"⚽ GOAL! {curr_score}",
                showarrow=False, yshift=5, font=dict(size=10, color="red")
            )
        
        prev_score = curr_score
    
    fig.update_layout(
        title="Odds Movement Over Time (Pre-match → Live)",
        xaxis_title="Time Progression",
        yaxis_title="Odds",
        hovermode='x unified',
        height=500
    )
    
    return fig


def plot_volume_changes(df):
    """Create bar chart for volume changes."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['Index'], y=df['Volume1'],
        name='Volume 1', marker_color='#2ecc71'
    ))
    
    fig.add_trace(go.Bar(
        x=df['Index'], y=df['VolumeX'],
        name='Volume X', marker_color='#f39c12'
    ))
    
    fig.add_trace(go.Bar(
        x=df['Index'], y=df['Volume2'],
        name='Volume 2', marker_color='#e74c3c'
    ))
    
    fig.update_layout(
        title="Betting Volume Changes",
        xaxis_title="Time Progression",
        yaxis_title="Volume (€)",
        barmode='group',
        height=500
    )
    
    return fig


def plot_moneyway(df):
    """Create stacked bar chart for moneyway percentages."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['Index'], y=df['Pct1']*100,
        name='% on 1', marker_color='#2ecc71'
    ))
    
    fig.add_trace(go.Bar(
        x=df['Index'], y=df['PctX']*100,
        name='% on X', marker_color='#f39c12'
    ))
    
    fig.add_trace(go.Bar(
        x=df['Index'], y=df['Pct2']*100,
        name='% on 2', marker_color='#e74c3c'
    ))
    
    fig.update_layout(
        title="Moneyway Distribution (% of Total Bets)",
        xaxis_title="Time Progression",
        yaxis_title="Percentage (%)",
        barmode='stack',
        height=500
    )
    
    return fig


def plot_odd_changes(df, threshold):
    """Create scatter plot for odd changes with highlights."""
    fig = go.Figure()
    
    # Plot normal changes
    for outcome, color in [('1', '#2ecc71'), ('X', '#f39c12'), ('2', '#e74c3c')]:
        change_col = f'Change{outcome}'
        
        # Normal changes
        normal = df[np.abs(df[change_col]) <= threshold]
        fig.add_trace(go.Scatter(
            x=normal['Index'], y=normal[change_col],
            mode='markers', name=f'Change {outcome}',
            marker=dict(color=color, size=6)
        ))
        
        # Suspicious changes
        suspicious = df[np.abs(df[change_col]) > threshold]
        fig.add_trace(go.Scatter(
            x=suspicious['Index'], y=suspicious[change_col],
            mode='markers', name=f'⚠️ Large {outcome}',
            marker=dict(color=color, size=12, symbol='diamond', line=dict(color='red', width=2))
        ))
    
    fig.update_layout(
        title=f"Odd Changes (Highlights: |change| > {threshold})",
        xaxis_title="Time Progression",
        yaxis_title="Odd Change",
        hovermode='closest',
        height=500
    )
    
    return fig


# Main analysis logic
if analyze_button and data_input:
    with st.spinner("Analyzing data..."):
        # Parse data
        df = parse_betting_data(data_input)
        
        if df is not None and len(df) > 0:
            # Prepare thresholds
            thresholds = {
                'odd_change': odd_change_threshold,
                'volume_spike': volume_spike_multiplier,
                'moneyway': moneyway_threshold,
                'z_score': z_score_threshold
            }
            
            # Detect anomalies
            flags, suspicion_score, df_analyzed, pick_info = detect_anomalies(df, thresholds)
            
            # Display summary
            st.header("📈 Data Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                prematch_count = len(df[df['Phase'] == 'Pre-match'])
                st.metric("Pre-match Records", prematch_count)
            with col3:
                live_count = len(df[df['Phase'] == 'Live'])
                st.metric("Live Records", live_count)
            with col4:
                st.metric("Suspicion Score", f"{suspicion_score}/100", 
                         delta="High Risk" if suspicion_score > 60 else "Medium Risk" if suspicion_score > 30 else "Low Risk")
            
            # Fixed Match Pick Section
            st.header("🎯 Fixed Match Prediction")
            
            if pick_info['confidence'] >= 40:
                # Determine color and emoji based on confidence
                if pick_info['confidence'] >= 80:
                    conf_color = "green"
                    conf_emoji = "🟢"
                    conf_level = "VERY HIGH"
                elif pick_info['confidence'] >= 60:
                    conf_color = "orange"
                    conf_emoji = "🟠"
                    conf_level = "HIGH"
                else:
                    conf_color = "blue"
                    conf_emoji = "🔵"
                    conf_level = "MODERATE"
                
                st.markdown(f"""
                <div style='background-color: rgba(46, 204, 113, 0.1); padding: 20px; border-radius: 10px; border-left: 5px solid {conf_color};'>
                    <h2 style='margin: 0; color: {conf_color};'>{conf_emoji} PREDICTED OUTCOME: {pick_info['outcome_name']} ({pick_info['predicted_outcome']})</h2>
                    <h3 style='margin: 10px 0; color: {conf_color};'>Confidence Level: {conf_level} ({pick_info['confidence']}%)</h3>
                    <p style='font-size: 16px; margin: 10px 0;'>
                        <strong>Signal Breakdown:</strong><br>
                        • Home Win (1): {pick_info['signals']['1']} points<br>
                        • Draw (X): {pick_info['signals']['X']} points<br>
                        • Away Win (2): {pick_info['signals']['2']} points
                    </p>
                    <p style='font-size: 14px; margin: 15px 0 0 0; color: #555;'>
                        <em>⚠️ This prediction is based on suspicious betting patterns and should be used for analytical purposes only.</em>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show current threshold being used
                st.caption(f"Using **{preset}** thresholds for analysis")
                
                st.info("""
                **How to interpret this prediction:**
                - **80-100% confidence**: Extremely strong evidence of fixing toward this outcome
                - **60-79% confidence**: Strong suspicious patterns favoring this outcome  
                - **40-59% confidence**: Moderate indicators pointing to this outcome
                
                **Key factors considered:**
                - 🔴 Late pre-match odd movements (strongest indicator - 30-35 points)
                - 🟡 Volume spikes and concentration patterns (15 points)
                - 🟠 Moneyway imbalances and distribution (20 points)
                - 🔵 Counter-intuitive odds vs. volume behavior (22 points)
                """)
                
                # Show the final odds for the predicted outcome
                if not df_analyzed.empty:
                    last_row = df_analyzed.iloc[-1]
                    predicted_odd = last_row[f'Odd{pick_info["predicted_outcome"]}']
                    if pd.notna(predicted_odd):
                        potential_return = (predicted_odd - 1) * 100
                        st.success(f"**Current odds for {pick_info['outcome_name']}: {predicted_odd:.2f}** (Potential return: {potential_return:.0f}% on stake)")
                
                # Show comprehensive pre-match and live analysis breakdown
                prematch_df = df_analyzed[df_analyzed['Phase'] == 'Pre-match']
                live_df = df_analyzed[df_analyzed['Phase'] == 'Live']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📊 Pre-match Analysis")
                    if len(prematch_df) > 0:
                        st.success(f"✅ **{len(prematch_df)} pre-match records** analyzed")
                        
                        # Calculate pre-match movements
                        for outcome in ['1', 'X', '2']:
                            odd_col = f'Odd{outcome}'
                            vol_col = f'Volume{outcome}'
                            agg_col = f'Agg{outcome}'
                            
                            valid_odds = prematch_df[odd_col].dropna()
                            if len(valid_odds) >= 2:
                                first = valid_odds.iloc[0]
                                last = valid_odds.iloc[-1]
                                change_pct = ((last - first) / first) * 100
                                
                                # Show with appropriate emoji
                                if change_pct < -10:
                                    emoji = "🔴"
                                    status = "Sharp drop"
                                elif change_pct < -5:
                                    emoji = "🟠"
                                    status = "Moderate drop"
                                elif change_pct > 5:
                                    emoji = "🔵"
                                    status = "Rising"
                                else:
                                    emoji = "➡️"
                                    status = "Stable"
                                
                                st.write(f"{emoji} **Outcome {outcome}**: {first:.2f} → {last:.2f} ({change_pct:+.1f}%) - {status}")
                        
                        # Show volume analysis
                        st.markdown("**Volume Pattern:**")
                        for outcome in ['1', 'X', '2']:
                            agg_col = f'Agg{outcome}'
                            if prematch_df[agg_col].sum() > 0:
                                pct_of_total = (prematch_df[agg_col].sum() / prematch_df['Total_Agg'].sum()) * 100
                                st.write(f"• Outcome {outcome}: {pct_of_total:.0f}% of pre-match volume")
                    else:
                        st.info("No pre-match data available")
                
                with col2:
                    st.markdown("### ⚽ Live Analysis")
                    if len(live_df) > 0:
                        st.success(f"✅ **{len(live_df)} live records** analyzed")
                        
                        # Calculate live movements
                        for outcome in ['1', 'X', '2']:
                            odd_col = f'Odd{outcome}'
                            pct_col = f'Pct{outcome}'
                            
                            valid_odds = live_df[odd_col].dropna()
                            if len(valid_odds) >= 2:
                                first = valid_odds.iloc[0]
                                last = valid_odds.iloc[-1]
                                change_pct = ((last - first) / first) * 100
                                
                                if change_pct < -5:
                                    emoji = "📉"
                                elif change_pct > 5:
                                    emoji = "📈"
                                else:
                                    emoji = "➡️"
                                
                                st.write(f"{emoji} **Outcome {outcome}**: {first:.2f} → {last:.2f} ({change_pct:+.1f}%)")
                        
                        # Show moneyway distribution
                        st.markdown("**Average Moneyway (Live):**")
                        for outcome in ['1', 'X', '2']:
                            pct_col = f'Pct{outcome}'
                            avg_pct = live_df[pct_col].mean() * 100
                            st.write(f"• Outcome {outcome}: {avg_pct:.0f}% of bets")
                    else:
                        st.info("No live data available yet")
                
                st.markdown("---")
                
                # Show detection sources
                with st.expander("🔍 What signals contributed to this prediction?"):
                    st.markdown("""
                    **Analysis covers BOTH pre-match and live phases:**
                    
                    **Pre-match Signals (Strongest):**
                    - ✅ Late pre-match odd drops (30-35 points) - Most reliable indicator
                    - ✅ Volume concentration in final hour before kickoff (15 points)
                    - ✅ Pre-match moneyway imbalances (20 points)
                    
                    **Live Signals:**
                    - ✅ Sudden odd changes without score changes (25 points)
                    - ✅ Volume spikes during play (15 points)
                    - ✅ Sustained moneyway pressure (18 points)
                    
                    **Cross-phase Signals:**
                    - ✅ Odds moving against volume (22 points)
                    - ✅ Statistical anomalies (IsolationForest - 10 points)
                    - ✅ Extreme z-score deviations (15 points)
                    
                    The more signals from BOTH phases, the higher the confidence!
                    """)
            
            else:
                st.info("""
                ### ℹ️ No Clear Fixed Outcome Detected
                
                The confidence level is too low to make a reliable prediction. This could mean:
                - The match doesn't show clear signs of fixing
                - Suspicious patterns are distributed across multiple outcomes
                - Insufficient data to determine the intended result
                
                **Signal Breakdown:**
                - Home Win (1): {0} points
                - Draw (X): {1} points  
                - Away Win (2): {2} points
                """.format(pick_info['signals']['1'], pick_info['signals']['X'], pick_info['signals']['2']))
            
            st.markdown("---")
            
            # Visualizations
            st.header("📊 Visualizations")
            
            # Add phase distribution info
            phase_info = df['Phase'].value_counts()
            if 'Pre-match' in phase_info.index and 'Live' in phase_info.index:
                st.info(f"📊 Data contains **{phase_info.get('Pre-match', 0)} pre-match** and **{phase_info.get('Live', 0)} live** betting records")
            
            tab1, tab2, tab3, tab4 = st.tabs(["Odds Movement", "Volume Changes", "Moneyway", "Odd Changes"])
            
            with tab1:
                st.plotly_chart(plot_odds_over_time(df), width='stretch')
            
            with tab2:
                st.plotly_chart(plot_volume_changes(df), width='stretch')
            
            with tab3:
                st.plotly_chart(plot_moneyway(df), width='stretch')
            
            with tab4:
                st.plotly_chart(plot_odd_changes(df, odd_change_threshold), width='stretch')
            
            # Anomaly detection results
            st.header("🚨 Anomaly Detection Results")
            
            if flags:
                st.warning(f"**{len(flags)} suspicious patterns detected!**")
                
                # Group by severity
                high_severity = [f for f in flags if f['Severity'] == 'High']
                medium_severity = [f for f in flags if f['Severity'] == 'Medium']
                
                if high_severity:
                    st.subheader("🔴 High Severity Flags")
                    for flag in high_severity[:10]:  # Limit to 10
                        st.error(f"**{flag['Time']}** (Score: {flag['Score']}) - {flag['Reason']}")
                
                if medium_severity:
                    st.subheader("🟡 Medium Severity Flags")
                    with st.expander(f"Show {len(medium_severity)} medium severity flags"):
                        for flag in medium_severity:
                            st.warning(f"**{flag['Time']}** (Score: {flag['Score']}) - {flag['Reason']}")
            else:
                st.success("No significant anomalies detected.")
            
            # Suspicion report
            st.header("📋 Suspicion Report")
            
            if suspicion_score >= 60:
                st.error(f"""
                ### ⚠️ HIGH SUSPICION OF MATCH FIXING
                
                **Suspicion Score: {suspicion_score}/100**
                
                This match shows multiple red flags consistent with potential match fixing:
                - **{len(flags)} anomalous patterns detected**
                - **{len([f for f in flags if f['Severity'] == 'High'])} high-severity warnings**
                
                **Recommendation:** Further investigation strongly recommended.
                """)
            elif suspicion_score >= 30:
                st.warning(f"""
                ### ⚠️ MODERATE SUSPICION
                
                **Suspicion Score: {suspicion_score}/100**
                
                Some suspicious patterns detected. While not conclusive, the betting behavior warrants attention.
                
                **Recommendation:** Monitor for additional evidence.
                """)
            else:
                st.success(f"""
                ### ✅ LOW SUSPICION
                
                **Suspicion Score: {suspicion_score}/100**
                
                Betting patterns appear mostly normal. No strong indicators of match fixing detected.
                """)
            
            # Export functionality
            st.header("💾 Export Data")
            
            # Prepare CSV
            csv_buffer = io.StringIO()
            df_analyzed.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Download Analyzed Data (CSV)",
                data=csv_data,
                file_name="analyzed_betting_data.csv",
                mime="text/csv"
            )
            
            # Show raw data table
            with st.expander("📋 View Full Data Table"):
                st.dataframe(df_analyzed, width='stretch')
        
        else:
            st.error("Failed to parse data. Please check the format and try again.")

elif analyze_button:
    st.warning("Please paste betting data before analyzing.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Fixed Match Detector v1.0 | For educational and analytical purposes only</p>
    <p>⚠️ This tool provides statistical analysis and should not be the sole basis for accusations of match fixing.</p>
</div>
""", unsafe_allow_html=True)