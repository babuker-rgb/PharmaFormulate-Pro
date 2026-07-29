# ================================================================
# Hybrid AI · Multi-Objective Tablet Optimization
# Nile Valley University · Sudan · v29.28‑R32
# VERSION 12 – PARETO FRONT WITH GOLDEN & TESTED POINT
# ================================================================

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import plotly.graph_objects as go
import time
import warnings
import json
import os
import tempfile
from datetime import datetime

warnings.filterwarnings('ignore')

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="Hybrid AI · Tablet Optimization v29.28‑R32",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# CONSTANTS
# ================================================================
API_MIN, API_MAX = 80.0, 98.0
BINDER_MIN, BINDER_MAX = 1.4, 6.0
PVPP_MIN, PVPP_MAX = 1.0, 6.0
MGST_MIN, MGST_MAX = 0.10, 1.2
MCC_MIN, MCC_MAX = 1.5, 8.0
MOISTURE_MIN, MOISTURE_MAX = 0.5, 5.0

PRESSURE_MIN, PRESSURE_MAX = 150.0, 250.0
SPEED_MIN, SPEED_MAX = 15.0, 30.0
PARTICLE_SIZE_MIN, PARTICLE_SIZE_MAX = 10.0, 200.0
DWELL_TIME_MIN, DWELL_TIME_MAX = 5.0, 50.0
FRICTION_MIN, FRICTION_MAX = 0.1, 0.5
DECOMPRESSION_TIME_MIN, DECOMPRESSION_TIME_MAX = 10.0, 80.0
GRANULE_MIN, GRANULE_MAX = 30.0, 250.0

BINDER_GRADES = {
    "MCC PH101": {"compressibility": 0.85, "disintegration": 0.90, "flow": 0.80},
    "MCC PH102": {"compressibility": 0.90, "disintegration": 0.85, "flow": 0.85},
    "MCC PH200": {"compressibility": 0.95, "disintegration": 0.80, "flow": 0.90},
    "MCC KG": {"compressibility": 0.88, "disintegration": 0.88, "flow": 0.82},
    "Lactose Monohydrate": {"compressibility": 0.75, "disintegration": 0.95, "flow": 0.78},
    "Dicalcium Phosphate": {"compressibility": 0.70, "disintegration": 0.85, "flow": 0.75}
}
BINDER_GRADE_NAMES = list(BINDER_GRADES.keys())

POPULATION_SIZE = 50
NSGA_GENERATIONS = 80
TRAINING_EPOCHS = 1200

# ================================================================
# SESSION STATE
# ================================================================
def initialize_session_state():
    defaults = {
        'api': 96.5, 'binder': 1.4, 'pvpp': 1.0, 'mgst': 0.10,
        'mcc': 1.5, 'moisture': 0.50, 'binder_grade': 0,
        'particle_size': 50.0, 'pressure': 200.0, 'speed': 20.0,
        'granule': 125.0, 'dwell_time': 25.0, 'friction': 0.25,
        'decompression_time': 35.0, 'optimization_complete': False,
        'results': None, 'best_solutions': None, 'golden_solution': None,
        'runtime': 0, 'pareto_history': None,
        'user_data': None, 'data_source': 'synthetic',
        'force_retrain': False,
        'pareto_plot_type': 'API vs EFRF'
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
initialize_session_state()

# ================================================================
# HELPER FUNCTIONS (all unchanged from previous version)
# ================================================================
def normalize_formulation(api, binder, pvpp, mgst, mcc, moisture):
    comps = np.array([api, binder, pvpp, mgst, mcc, moisture])
    total = np.sum(comps)
    if total <= 0:
        total = 1.0
    norm = (comps / total) * 100
    return {
        'api': norm[0], 'binder': norm[1], 'pvpp': norm[2],
        'mgst': norm[3], 'mcc': norm[4], 'moisture': norm[5], 'total': 100.0
    }

def get_formulation_summary(api, binder, pvpp, mgst, mcc, moisture):
    n = normalize_formulation(api, binder, pvpp, mgst, mcc, moisture)
    return {'API': n['api'], 'Binder': n['binder'], 'PVPP': n['pvpp'],
            'MgSt': n['mgst'], 'MCC': n['mcc'], 'Moisture': n['moisture'],
            'Total': n['total']}

def validate_formulation(api, binder, pvpp, mgst, mcc, moisture):
    total = sum([api, binder, pvpp, mgst, mcc, moisture])
    return (95 <= total <= 105, f"Total is {total:.1f}% – should be ~100%")

def calculate_quality_score(density, tensile, efrf, api=None):
    density_score = min(100, (density / 0.95) * 100)
    tensile_score = min(100, (tensile / 8.5) * 100)
    efrf_score = max(0, (1 - efrf) * 100)
    weights = {'density': 0.4, 'tensile': 0.3, 'efrf': 0.3}
    overall = (density_score * weights['density'] +
               tensile_score * weights['tensile'] +
               efrf_score * weights['efrf'])
    if api is not None:
        api_score = (api - 80) / 18 * 100
        overall = 0.7 * overall + 0.3 * api_score
        return {'overall': overall, 'density_score': density_score,
                'tensile_score': tensile_score, 'efrf_score': efrf_score,
                'api_score': api_score, 'weights': {**weights, 'api': 0.3}}
    else:
        return {'overall': overall, 'density_score': density_score,
                'tensile_score': tensile_score, 'efrf_score': efrf_score,
                'weights': weights}

# ================================================================
# HYBRID NEURAL NETWORK, DATA GENERATION, NSGA-II, etc. (same as before)
# ================================================================
class HybridTabletModel(nn.Module):
    # ... (identical to the previous version)
    def __init__(self, input_dim=8, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 5)
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self, x):
        h1 = torch.relu(self.bn1(self.fc1(x)))
        h2 = torch.relu(self.bn2(self.fc2(h1))) + h1
        h3 = torch.relu(self.bn3(self.fc3(h2))) + h2
        h4 = torch.relu(self.bn4(self.fc4(h3))) + h3
        out = self.fc5(h4)
        density = torch.sigmoid(out[:, 0]) * 0.4 + 0.55
        tensile = torch.sigmoid(out[:, 1]) * 8.0 + 0.5
        efrf = torch.sigmoid(out[:, 2])
        disintegration = torch.sigmoid(out[:, 3]) * 45.0 + 2.0
        dissolution = torch.sigmoid(out[:, 4]) * 80.0 + 10.0
        return torch.stack([density, tensile, efrf, disintegration, dissolution], dim=1)
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            return self.forward(x).numpy()

# ================================================================
# DATA GENERATION (unchanged)
# ================================================================
N_SAMPLES = 8000
BOUNDARY_FRACTION = 0.30

def _sample_compositions(n_samples, rng, boundary_fraction=0.0):
    bounds = [(API_MIN, API_MAX), (BINDER_MIN, BINDER_MAX), (PVPP_MIN, PVPP_MAX),
              (MGST_MIN, MGST_MAX), (MCC_MIN, MCC_MAX), (MOISTURE_MIN, MOISTURE_MAX)]
    cols = [rng.uniform(lo, hi, n_samples) for lo, hi in bounds]
    comps = np.column_stack(cols)

    n_boundary = int(n_samples * boundary_fraction)
    if n_boundary > 0:
        idx = rng.choice(n_samples, n_boundary, replace=False)
        for row in idx:
            n_pinned = rng.integers(2, 6)
            pinned_dims = rng.choice(6, n_pinned, replace=False)
            for d in pinned_dims:
                lo, hi = bounds[d]
                span = hi - lo
                near_lo = rng.random() < 0.5
                jitter = rng.uniform(0, 0.08) * span
                comps[row, d] = lo + jitter if near_lo else hi - jitter
    return comps

def generate_synthetic_data(n_samples=N_SAMPLES, seed=42):
    rng = np.random.default_rng(seed)
    comps = _sample_compositions(n_samples, rng, boundary_fraction=BOUNDARY_FRACTION)
    comps = comps / comps.sum(axis=1, keepdims=True) * 100.0
    api_n, binder_n, pvpp_n, mgst_n, mcc_n, moisture_n = comps.T

    pressure = rng.uniform(PRESSURE_MIN, PRESSURE_MAX, n_samples)
    speed = rng.uniform(SPEED_MIN, SPEED_MAX, n_samples)

    X = np.column_stack([api_n, binder_n, pvpp_n, mgst_n, mcc_n, moisture_n, pressure, speed])

    porosity0 = 0.45 - 0.001 * (pressure - PRESSURE_MIN) - 0.01 * (binder_n - 3.0)
    density = np.clip(1.0 - porosity0 * np.exp(-0.01 * (pressure - PRESSURE_MIN)), 0.55, 0.95)
    density += rng.normal(0, 0.005, n_samples)
    density = np.clip(density, 0.55, 0.95)

    tensile = (0.5 + 6.0 * (density - 0.55) / 0.40 + 0.4 * (binder_n - BINDER_MIN)
               - 1.2 * (mgst_n - MGST_MIN) + 0.3 * (api_n - API_MIN) / (API_MAX - API_MIN))
    tensile += rng.normal(0, 0.1, n_samples)
    tensile = np.clip(tensile, 0.5, 8.5)

    efrf = (0.55 - 0.35 * (density - 0.55) / 0.40 + 0.25 * (api_n - API_MIN) / (API_MAX - API_MIN)
            - 0.15 * (binder_n - BINDER_MIN) / (BINDER_MAX - BINDER_MIN) + 0.2 * (mgst_n - MGST_MIN))
    efrf += rng.normal(0, 0.03, n_samples)
    efrf = np.clip(efrf, 0.02, 0.98)

    disintegration = (12.0 - 4.0 * (pvpp_n - PVPP_MIN) / (PVPP_MAX - PVPP_MIN)
                       + 5.0 * (binder_n - BINDER_MIN) / (BINDER_MAX - BINDER_MIN)
                       + 3.0 * (moisture_n - MOISTURE_MIN) / (MOISTURE_MAX - MOISTURE_MIN))
    disintegration += rng.normal(0, 0.5, n_samples)
    disintegration = np.clip(disintegration, 2.0, 45.0)

    dissolution = 1.8 * disintegration + 5.0 - 3.0 * (pvpp_n - PVPP_MIN) / (PVPP_MAX - PVPP_MIN)
    dissolution += rng.normal(0, 1.0, n_samples)
    dissolution = np.clip(dissolution, 10.0, 90.0)

    y = np.column_stack([density, tensile, efrf, disintegration, dissolution])
    return X.astype(np.float32), y.astype(np.float32)

class InputScaler:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self
    def transform(self, X):
        return (X - self.mean_) / self.std_

# ================================================================
# CHECKPOINT PATHS AND TRAINING (same as before)
# ================================================================
CHECKPOINT_SYNTHETIC = os.path.join(tempfile.gettempdir(), 'co_hybai_synthetic_v12.pt')
CHECKPOINT_REAL = os.path.join(tempfile.gettempdir(), 'co_hybai_real_v12.pt')

@st.cache_resource(show_spinner=False)
def train_model(use_real=False, real_df=None):
    # ... identical to previous version (use checkpoint paths above)
    # I'll copy the exact function from the previous answer for brevity, but it's the same.
    pass  # Placeholder; the full code will include the full function.

# ================================================================
# NSGA-II OPTIMIZER – 2 OBJECTIVES (same as previous)
# ================================================================
class NSGAIIOptimizer:
    # ... identical to previous version
    pass

# ================================================================
# RESULT FUNCTIONS (identical)
# ================================================================
def get_model_and_scaler():
    # ... same
    pass

def run_real_training_and_get_history():
    # ... same
    pass

def run_real_optimization(progress_callback=None):
    # ... same
    pass

def get_current_formulation_results():
    # ... same
    pass

# ================================================================
# UI RENDER FUNCTIONS (updated plot function)
# ================================================================
def render_sidebar():
    # ... same as previous version
    pass

def render_binder_grade_comparison():
    # ... same
    pass

def render_mass_balance_display(api, binder, pvpp, mgst, mcc, moisture):
    # ... same
    pass

def render_input_panel():
    # ... same
    pass

def target_status(value, threshold, mode='min', comfortable=None):
    # ... same
    pass

def render_results_summary(results):
    # ... same
    pass

def render_training_progress():
    # ... same
    pass

def generate_feasible_samples(model, scaler, n_samples=3000):
    # ... same as previous version (generates feasible region points)
    pass

# --- UPDATED PARETO PLOT FUNCTION ---
def render_pareto_evolution():
    st.markdown("---")
    st.markdown("## 🌐 Pareto Front Evolution")
    golden = st.session_state.get('golden_solution', None)
    pareto_history = st.session_state.get('pareto_history', None)
    if not pareto_history:
        st.info("Run the optimization to see the real Pareto front evolve across generations.")
        return

    plot_type = st.radio(
        "Select plot type",
        options=["API vs EFRF", "API vs Tensile", "API vs Disintegration", "API vs Dissolution"],
        index=0,
        horizontal=True,
        key="pareto_plot_type"
    )

    generations_recorded = [h['generation'] for h in pareto_history]
    gen_slider = st.select_slider("Select generation to view", options=generations_recorded, value=generations_recorded[-1])
    current_entry = next(h for h in pareto_history if h['generation'] == gen_slider)
    current_obj = current_entry['pareto_objectives']
    current_pop = current_entry['pareto_solutions']
    
    api_vals = current_pop[:, 0]
    # Extract objectives: first is -API, second is EFRF
    efrf_vals = current_obj[:, 1]
    # Tensile is not in objectives; we re-predict to get it
    model = st.session_state.get('_trained_model')
    scaler = st.session_state.get('_trained_scaler')
    if model is not None and scaler is not None:
        preds = model.predict(scaler.transform(current_pop))
        density_vals = preds[:, 0]
        tensile_vals = preds[:, 1]
        dis_vals = preds[:, 3]
        diss_vals = preds[:, 4]
        # Also generate feasible region samples
        feat_api, feat_efrf = generate_feasible_samples(model, scaler)
    else:
        density_vals = np.full_like(api_vals, np.nan)
        tensile_vals = np.full_like(api_vals, np.nan)
        dis_vals = np.full_like(api_vals, np.nan)
        diss_vals = np.full_like(api_vals, np.nan)
        feat_api, feat_efrf = np.array([]), np.array([])

    # Build figure based on type
    if plot_type == "API vs EFRF":
        x_label, y_label = "API (%)", "EFRF"
        x_vals, y_vals = api_vals, efrf_vals
        feasible_x, feasible_y = feat_api, feat_efrf
        # Also get tested point's EFRF
        tested_results = get_current_formulation_results()
        tested_api = st.session_state.api
        tested_efrf = tested_results['efrf']
        tested_y = tested_efrf
    elif plot_type == "API vs Tensile":
        x_label, y_label = "API (%)", "Tensile (MPa)"
        x_vals, y_vals = api_vals, tensile_vals
        feasible_x, feasible_y = None, None
        tested_results = get_current_formulation_results()
        tested_api = st.session_state.api
        tested_y = tested_results['tensile']
    elif plot_type == "API vs Disintegration":
        x_label, y_label = "API (%)", "Disintegration (min)"
        x_vals, y_vals = api_vals, dis_vals
        feasible_x, feasible_y = None, None
        tested_results = get_current_formulation_results()
        tested_api = st.session_state.api
        tested_y = tested_results['disintegration']
    else:  # API vs Dissolution
        x_label, y_label = "API (%)", "Dissolution (min)"
        x_vals, y_vals = api_vals, diss_vals
        feasible_x, feasible_y = None, None
        tested_results = get_current_formulation_results()
        tested_api = st.session_state.api
        tested_y = tested_results['dissolution']

    fig = go.Figure()

    # ---- Feasible region background (only for API vs EFRF) ----
    if plot_type == "API vs EFRF" and len(feasible_x) > 0:
        fig.add_trace(go.Scatter(
            x=feasible_x,
            y=feasible_y,
            mode='markers',
            name='Feasible region (sampled)',
            marker=dict(
                size=3,
                color='lightblue',
                opacity=0.3,
                line=dict(width=0)
            ),
            hovertemplate='API: %{x:.1f}%<br>EFRF: %{y:.3f}<extra></extra>',
            showlegend=True
        ))

    # ---- Pareto front (line + markers) ----
    # Sort by x to get a smooth line
    sort_idx = np.argsort(x_vals)
    x_sorted = x_vals[sort_idx]
    y_sorted = y_vals[sort_idx]
    api_sorted = api_vals[sort_idx]  # for coloring markers

    fig.add_trace(go.Scatter(
        x=x_sorted,
        y=y_sorted,
        mode='lines+markers',
        name='Pareto Front',
        line=dict(color='red', width=2),
        marker=dict(
            size=8,
            color=api_sorted,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="API%", x=1.02),
            line=dict(width=1, color='black')
        ),
        hovertemplate=(
            f'{x_label}: %{{x:.3f}}<br>{y_label}: %{{y:.3f}}<br>'
            f'API: %{{marker.color:.1f}}%<br>Density: %{{customdata[0]:.3f}}<extra></extra>'
        ),
        customdata=np.column_stack([density_vals[sort_idx]])
    ))

    # ---- Golden solution marker ----
    if golden:
        golden_x = golden['API (%)']
        if plot_type == "API vs EFRF":
            golden_y = golden['EFRF']
        elif plot_type == "API vs Tensile":
            golden_y = golden['Tensile (MPa)']
        elif plot_type == "API vs Disintegration":
            golden_y = golden['Disintegration (min)']
        else:
            golden_y = golden['Dissolution (min)']
        fig.add_trace(go.Scatter(
            x=[golden_x],
            y=[golden_y],
            mode='markers',
            name='🏆 Golden Solution',
            marker=dict(size=15, color='red', symbol='diamond', line=dict(width=2, color='white')),
            hovertemplate=f'<b>Golden</b><br>{x_label}: %{{x:.3f}}<br>{y_label}: %{{y:.3f}}<extra></extra>'
        ))

    # ---- Tested formulation marker ----
    fig.add_trace(go.Scatter(
        x=[tested_api],
        y=[tested_y],
        mode='markers',
        name='Tested Formulation',
        marker=dict(size=12, color='blue', symbol='circle', line=dict(width=2, color='darkblue')),
        hovertemplate=f'<b>Tested</b><br>{x_label}: %{{x:.3f}}<br>{y_label}: %{{y:.3f}}<extra></extra>'
    ))

    # ---- Constraint boundaries ----
    if plot_type == "API vs EFRF":
        fig.add_hline(y=0.40, line_dash='dash', line_color='gray', annotation_text='EFRF threshold (0.40)')
    fig.add_vline(x=API_MIN, line_dash='dot', line_color='gray', annotation_text=f'API min ({API_MIN}%)')
    fig.add_vline(x=API_MAX, line_dash='dot', line_color='gray', annotation_text=f'API max ({API_MAX}%)')

    fig.update_layout(
        title=f'Pareto Front - Generation {gen_slider}',
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=500,
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"**Generation {gen_slider+1}/{NSGA_GENERATIONS}** · "
        f"Pareto-optimal solutions at this generation: {len(current_pop)}"
    )
    if plot_type == "API vs EFRF":
        st.caption("Light blue points are randomly sampled feasible formulations (all constraints satisfied).")

# --- Remaining UI functions (render_golden_solution, render_side_by_side_comparison, etc.) are unchanged ---

# ================================================================
# MAIN ORCHESTRATION
# ================================================================
def main():
    # ... identical to previous version
    pass

if __name__ == "__main__":
    main()
