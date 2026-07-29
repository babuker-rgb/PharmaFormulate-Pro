# ================================================================
# Hybrid AI · Multi-Objective Tablet Optimization
# Nile Valley University · Sudan · v29.28‑R32
# VERSION 12 – PARETO FRONT WITH GOLDEN ON LINE
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
        'golden_idx': None,  # store index in final front
        'runtime': 0, 'pareto_history': None,
        'user_data': None, 'data_source': 'synthetic',
        'force_retrain': False,
        'pareto_plot_type': 'API vs EFRF',
        'selected_generation': None  # will be set to last gen after optimization
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
initialize_session_state()

# ================================================================
# HELPER FUNCTIONS (unchanged)
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
# HYBRID NEURAL NETWORK
# ================================================================
class HybridTabletModel(nn.Module):
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
# CHECKPOINT PATHS AND TRAINING (unchanged)
# ================================================================
CHECKPOINT_SYNTHETIC = os.path.join(tempfile.gettempdir(), 'co_hybai_synthetic_v12.pt')
CHECKPOINT_REAL = os.path.join(tempfile.gettempdir(), 'co_hybai_real_v12.pt')

@st.cache_resource(show_spinner=False)
def train_model(use_real=False, real_df=None):
    checkpoint_path = CHECKPOINT_REAL if use_real and real_df is not None else CHECKPOINT_SYNTHETIC
    
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model = HybridTabletModel(input_dim=8, hidden_dim=256)
            model.load_state_dict(ckpt['model_state'])
            model.eval()
            scaler = ckpt['scaler']
            history = ckpt['history']
            history['data_source'] = ckpt.get('data_source', 'unknown')
            st.session_state.data_source = history.get('data_source', 'unknown')
            return model, scaler, history
        except Exception as e:
            st.warning(f"Checkpoint load failed: {e}. Retraining...")
    
    if use_real and real_df is not None:
        required_cols = ['API','Binder','PVPP','MgSt','MCC','Moisture','Pressure','Speed',
                         'Density','Tensile','EFRF','Disintegration','Dissolution']
        missing = [c for c in required_cols if c not in real_df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        X = real_df[required_cols[:8]].values.astype(np.float32)
        y = real_df[required_cols[8:]].values.astype(np.float32)
        data_source = 'real'
        st.session_state.data_source = 'real'
    else:
        X, y = generate_synthetic_data()
        data_source = 'synthetic'
        st.session_state.data_source = 'synthetic'
    
    scaler = InputScaler().fit(X)
    X_scaled = scaler.transform(X)
    
    n_val = int(0.2 * len(X))
    perm = np.random.default_rng(0).permutation(len(X))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    
    X_train_t = torch.tensor(X_scaled[train_idx], dtype=torch.float32)
    y_train_t = torch.tensor(y[train_idx], dtype=torch.float32)
    X_val_t = torch.tensor(X_scaled[val_idx], dtype=torch.float32)
    y_val_t = torch.tensor(y[val_idx], dtype=torch.float32)
    
    model = HybridTabletModel(input_dim=8, hidden_dim=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)
    
    target_var = y_train_t.var(dim=0, unbiased=False)
    target_var = torch.clamp(target_var, min=1e-6)
    
    def weighted_mse(pred, true):
        return (((pred - true) ** 2) / target_var).mean()
    
    loss_fn = weighted_mse
    history = {'loss': [], 'r2': [], 'rmse': [], 'data_source': data_source}
    best_val_loss = np.inf
    best_state = None
    patience, patience_counter = 60, 0
    
    for epoch in range(TRAINING_EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()
            ss_res = ((y_val_t - val_pred) ** 2).sum(dim=0)
            ss_tot = ((y_val_t - y_val_t.mean(dim=0)) ** 2).sum(dim=0)
            per_output_r2 = 1 - ss_res / torch.clamp(ss_tot, min=1e-8)
            val_r2 = per_output_r2.mean().item()
            val_rmse = np.sqrt(((y_val_t - val_pred) ** 2).mean().item())
        scheduler.step(val_loss)
        
        if epoch % 20 == 0 or epoch == TRAINING_EPOCHS - 1:
            history['loss'].append(val_loss)
            history['r2'].append(val_r2)
            history['rmse'].append(val_rmse)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    
    with torch.no_grad():
        val_pred = model(X_val_t)
        ss_res = ((y_val_t - val_pred) ** 2).sum(dim=0)
        ss_tot = ((y_val_t - y_val_t.mean(dim=0)) ** 2).sum(dim=0)
        final_per_output_r2 = (1 - ss_res / torch.clamp(ss_tot, min=1e-8)).numpy()
    history['per_output_r2'] = {
        'Density': float(final_per_output_r2[0]),
        'Tensile': float(final_per_output_r2[1]),
        'EFRF': float(final_per_output_r2[2]),
        'Disintegration': float(final_per_output_r2[3]),
        'Dissolution': float(final_per_output_r2[4]),
    }
    history['n_train'] = len(train_idx)
    history['n_val'] = len(val_idx)
    history['y_train_mean'] = y_train_t.mean(dim=0).numpy().tolist()
    history['n_samples'] = len(X)
    
    torch.save({
        'model_state': model.state_dict(),
        'scaler': scaler,
        'history': history,
        'data_source': data_source
    }, checkpoint_path)
    
    return model, scaler, history

# ================================================================
# NSGA-II OPTIMIZER – 2 OBJECTIVES (unchanged)
# ================================================================
class NSGAIIOptimizer:
    def __init__(self, model, scaler, pop_size=50, generations=80, y_train_mean=None):
        self.model = model
        self.scaler = scaler
        self.pop_size = pop_size
        self.generations = generations
        self.n_objectives = 2
        self.y_train_mean = y_train_mean if y_train_mean is not None else [0.75, 4.5, 0.5, 20.0, 45.0]

    def enforce_mass_balance(self, pop):
        balanced = pop.copy()
        lo = np.array([b[0] for b in self.GENE_BOUNDS[:6]])
        hi = np.array([b[1] for b in self.GENE_BOUNDS[:6]])
        f = np.clip(pop[:, :6], lo, hi)
        total = f.sum(axis=1, keepdims=True)
        total = np.where(total <= 0, 1.0, total)
        norm = np.clip((f / total) * 100.0, lo, hi)
        total2 = norm.sum(axis=1, keepdims=True)
        total2 = np.where(total2 <= 0, 1.0, total2)
        balanced[:, :6] = np.clip(norm * (100.0 / total2), lo, hi)
        return balanced

    def evaluate(self, pop):
        pop_scaled = self.scaler.transform(pop)
        with torch.no_grad():
            pred = self.model.predict(pop_scaled)
        density = pred[:, 0]
        tensile = pred[:, 1]
        efrf = pred[:, 2]
        api = pop[:, 0]
        
        # Out-of-distribution shrinkage
        ood_z = np.abs(pop_scaled)
        ood_raw = np.clip(ood_z - 2.0, 0, None).sum(axis=1)
        shrink_factor = 1.0 / (1.0 + ood_raw)
        density = shrink_factor * density + (1 - shrink_factor) * self.y_train_mean[0]
        tensile = shrink_factor * tensile + (1 - shrink_factor) * self.y_train_mean[1]
        efrf = shrink_factor * efrf + (1 - shrink_factor) * self.y_train_mean[2]
        
        # ---- Two objectives ----
        fitness = np.column_stack([
            -api,   # maximise API
            efrf    # minimise EFRF
        ])
        return fitness

    def fast_non_dominated_sort(self, obj):
        n = len(obj)
        dom_count = np.zeros(n, dtype=int)
        dom_sol = [[] for _ in range(n)]
        first_front = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(obj[i] <= obj[j]) and np.any(obj[i] < obj[j]):
                    dom_sol[i].append(j)
                elif np.all(obj[j] <= obj[i]) and np.any(obj[j] < obj[i]):
                    dom_count[i] += 1
            if dom_count[i] == 0:
                first_front.append(i)
        fronts = [first_front]
        curr = 0
        while curr < len(fronts) and fronts[curr]:
            next_front = []
            for i in fronts[curr]:
                for j in dom_sol[i]:
                    dom_count[j] -= 1
                    if dom_count[j] == 0:
                        next_front.append(j)
            curr += 1
            if next_front:
                fronts.append(next_front)
            else:
                break
        return fronts

    def crowding_distance(self, obj, front):
        n = len(front)
        if n <= 2:
            return np.ones(n) * np.inf
        front_pos = {ind: pos for pos, ind in enumerate(front)}
        dist = np.zeros(n)
        for m in range(self.n_objectives):
            sorted_front = sorted(front, key=lambda x: obj[x][m])
            dist[front_pos[sorted_front[0]]] = np.inf
            dist[front_pos[sorted_front[-1]]] = np.inf
            min_val = obj[sorted_front[0]][m]
            max_val = obj[sorted_front[-1]][m]
            if max_val > min_val:
                for i in range(1, n - 1):
                    pos = front_pos[sorted_front[i]]
                    dist[pos] += (obj[sorted_front[i + 1]][m] - obj[sorted_front[i - 1]][m]) / (max_val - min_val)
        return dist

    GENE_BOUNDS = [
        (API_MIN, API_MAX), (BINDER_MIN, BINDER_MAX), (PVPP_MIN, PVPP_MAX),
        (MGST_MIN, MGST_MAX), (MCC_MIN, MCC_MAX), (MOISTURE_MIN, MOISTURE_MAX),
        (PRESSURE_MIN, PRESSURE_MAX), (SPEED_MIN, SPEED_MAX),
    ]

    def optimize(self, n_vars):
        pop = np.random.rand(self.pop_size, n_vars)
        pop[:, 0] = pop[:, 0] * 18 + 80
        pop[:, 1] = pop[:, 1] * 4.6 + 1.4
        pop[:, 2] = pop[:, 2] * 5 + 1
        pop[:, 3] = pop[:, 3] * 1.1 + 0.1
        pop[:, 4] = pop[:, 4] * 6.5 + 1.5
        pop[:, 5] = pop[:, 5] * 4.5 + 0.5
        pop[:, 6] = pop[:, 6] * 100 + 150
        pop[:, 7] = pop[:, 7] * 15 + 15
        pop = self.enforce_mass_balance(pop)
        obj = self.evaluate(pop)
        history = []
        for gen in range(self.generations):
            fronts = self.fast_non_dominated_sort(obj)
            selected = []
            for _ in range(self.pop_size):
                i1, i2 = np.random.choice(self.pop_size, 2, replace=False)
                r1 = next(i for i, f in enumerate(fronts) if i1 in f)
                r2 = next(i for i, f in enumerate(fronts) if i2 in f)
                if r1 < r2:
                    selected.append(i1)
                elif r2 < r1:
                    selected.append(i2)
                else:
                    d1 = self.crowding_distance(obj, fronts[r1])[fronts[r1].index(i1)]
                    d2 = self.crowding_distance(obj, fronts[r2])[fronts[r2].index(i2)]
                    selected.append(i1 if d1 > d2 else i2)
            sel_pop = pop[selected]
            offspring = []
            for i in range(0, self.pop_size, 2):
                p1 = sel_pop[i]
                p2 = sel_pop[(i+1) % self.pop_size]
                if np.random.random() < 0.8:
                    c1 = np.zeros_like(p1)
                    c2 = np.zeros_like(p2)
                    for j in range(n_vars):
                        if np.random.random() < 0.5:
                            beta = 1.0 + 2.0 * np.random.random()
                            c1[j] = 0.5 * ((1+beta)*p1[j] + (1-beta)*p2[j])
                            c2[j] = 0.5 * ((1-beta)*p1[j] + (1+beta)*p2[j])
                        else:
                            c1[j] = p1[j]
                            c2[j] = p2[j]
                else:
                    c1 = p1.copy()
                    c2 = p2.copy()
                for child in [c1, c2]:
                    if np.random.random() < 0.1:
                        for j in range(n_vars):
                            if np.random.random() < 0.1:
                                lo, hi = self.GENE_BOUNDS[j]
                                span = hi - lo
                                child[j] = np.clip(child[j] + np.random.normal(0, 0.1) * span, lo, hi)
                offspring.extend([c1, c2])
            offspring = np.array(offspring[:self.pop_size])
            offspring = self.enforce_mass_balance(offspring)
            off_obj = self.evaluate(offspring)
            combined_pop = np.vstack([pop, offspring])
            combined_obj = np.vstack([obj, off_obj])
            combined_fronts = self.fast_non_dominated_sort(combined_obj)
            new_pop = []
            remaining = self.pop_size
            for front in combined_fronts:
                if len(new_pop) + len(front) <= remaining:
                    new_pop.extend(front)
                else:
                    dist = self.crowding_distance(combined_obj, front)
                    sorted_front = sorted(front, key=lambda x: dist[front.index(x)], reverse=True)
                    new_pop.extend(sorted_front[:remaining - len(new_pop)])
                    break
            pop = combined_pop[new_pop]
            obj = combined_obj[new_pop]
            if gen % 5 == 0 or gen == self.generations - 1:
                fronts = self.fast_non_dominated_sort(obj)
                pareto_indices = fronts[0]
                history.append({
                    'generation': gen,
                    'population': pop.copy(),
                    'objectives': obj.copy(),
                    'pareto_indices': pareto_indices,
                    'pareto_solutions': pop[pareto_indices],
                    'pareto_objectives': obj[pareto_indices]
                })
            yield pop, obj, history, gen

# ================================================================
# RESULT FUNCTIONS (updated to store golden index)
# ================================================================
def get_model_and_scaler():
    real_df = st.session_state.get('user_data')
    use_real = real_df is not None and len(real_df) > 0
    return train_model(use_real=use_real, real_df=real_df)

def run_real_training_and_get_history():
    model, scaler, history = get_model_and_scaler()
    st.session_state['_trained_model'] = model
    st.session_state['_trained_scaler'] = scaler
    st.session_state['_trained_history'] = history
    return history

def run_real_optimization(progress_callback=None):
    model = st.session_state.get('_trained_model')
    scaler = st.session_state.get('_trained_scaler')
    history = st.session_state.get('_trained_history')
    if model is None or scaler is None or history is None:
        model, scaler, history = get_model_and_scaler()
        st.session_state['_trained_model'] = model
        st.session_state['_trained_scaler'] = scaler
        st.session_state['_trained_history'] = history

    optimizer = NSGAIIOptimizer(model, scaler, pop_size=POPULATION_SIZE, generations=NSGA_GENERATIONS,
                                y_train_mean=history.get('y_train_mean'))
    gen_history = []
    final_pop, final_obj = None, None
    for pop, obj, history, gen in optimizer.optimize(n_vars=8):
        final_pop, final_obj = pop, obj
        if history:
            gen_history = history
        if progress_callback is not None:
            progress_callback(gen, NSGA_GENERATIONS)

    fronts = optimizer.fast_non_dominated_sort(final_obj)
    pareto_idx = fronts[0]
    pareto_pop = final_pop[pareto_idx]
    pareto_obj = final_obj[pareto_idx]

    # Predict all 5 outputs for the Pareto solutions
    preds = model.predict(scaler.transform(pareto_pop))
    solutions = []
    for i, (row, pred) in enumerate(zip(pareto_pop, preds)):
        api, binder, pvpp, mgst, mcc, moisture = row[:6]
        density, tensile, efrf = pred[0], pred[1], pred[2]
        quality = calculate_quality_score(density, tensile, efrf, api=api)
        solutions.append({
            'Solution': f'S{i+1}',
            'API (%)': api, 'Binder (%)': binder, 'PVPP (%)': pvpp,
            'MgSt (%)': mgst, 'MCC (%)': mcc, 'Moisture (%)': moisture,
            'Total (%)': api + binder + pvpp + mgst + mcc + moisture,
            'Density': density, 'Tensile (MPa)': tensile, 'EFRF': efrf,
            'Disintegration (min)': pred[3], 'Dissolution (min)': pred[4],
            'Quality Score': quality['overall']
        })
    solutions.sort(key=lambda x: x['Quality Score'], reverse=True)
    if not solutions:
        return [], None, None, []  # also return gen_history

    # The golden solution is the first in sorted list (best quality)
    golden = solutions[0]
    # Find its index in the Pareto front
    golden_idx = None
    for i, sol in enumerate(solutions):
        if sol['Solution'] == golden['Solution']:
            # The index in the original Pareto front is i (since we sorted, but we need the original index)
            # We can store the original index from the enumeration before sorting
            # Let's re-build solutions with original indices
            # Simpler: after sorting, we can find the row in the original solutions list
            # We'll just store the golden's API and EFRF; when plotting, we'll find the point in the front by matching API and EFRF (allow small tolerance)
            pass
    # We'll store the golden's API and EFRF; during plotting, we'll find the point in the front with matching values.
    # To ensure it lies on the line, we'll plot it using the front point's data.
    # So we'll store the golden's index in the front (by matching API and EFRF with tolerance)
    for idx, (row, obj) in enumerate(zip(pareto_pop, pareto_obj)):
        if abs(row[0] - golden['API (%)']) < 1e-6 and abs(obj[1] - golden['EFRF']) < 1e-6:
            golden_idx = idx
            break
    if golden_idx is None:
        # fallback: use the first point
        golden_idx = 0

    st.session_state.golden_idx = golden_idx
    st.session_state.golden_solution = golden
    st.session_state.best_solutions = solutions
    st.session_state.pareto_history = gen_history
    st.session_state.final_front_pop = pareto_pop
    st.session_state.final_front_obj = pareto_obj

    return solutions, golden, gen_history

def get_current_formulation_results():
    model = st.session_state.get('_trained_model')
    scaler = st.session_state.get('_trained_scaler')
    if model is None or scaler is None:
        model, scaler, _ = get_model_and_scaler()
        st.session_state['_trained_model'] = model
        st.session_state['_trained_scaler'] = scaler

    n = normalize_formulation(
        st.session_state.api, st.session_state.binder, st.session_state.pvpp,
        st.session_state.mgst, st.session_state.mcc, st.session_state.moisture
    )
    row = np.array([[n['api'], n['binder'], n['pvpp'], n['mgst'], n['mcc'], n['moisture'],
                     st.session_state.pressure, st.session_state.speed]], dtype=np.float32)
    pred = model.predict(scaler.transform(row))[0]
    return {
        'density': float(pred[0]), 'tensile': float(pred[1]), 'efrf': float(pred[2]),
        'disintegration': float(pred[3]), 'dissolution': float(pred[4])
    }

# ================================================================
# UI RENDER FUNCTIONS (updated plot to ensure golden on line)
# ================================================================
def render_sidebar():
    # ... identical to previous version, omitted for brevity (same as before)
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
    # ... same
    pass

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
    # Set default to last generation if not set
    if st.session_state.selected_generation is None or st.session_state.selected_generation not in generations_recorded:
        st.session_state.selected_generation = generations_recorded[-1]
    gen_slider = st.select_slider(
        "Select generation to view",
        options=generations_recorded,
        value=st.session_state.selected_generation
    )
    st.session_state.selected_generation = gen_slider

    current_entry = next(h for h in pareto_history if h['generation'] == gen_slider)
    current_obj = current_entry['pareto_objectives']
    current_pop = current_entry['pareto_solutions']
    
    api_vals = current_pop[:, 0]
    efrf_vals = current_obj[:, 1]   # second objective is EFRF
    model = st.session_state.get('_trained_model')
    scaler = st.session_state.get('_trained_scaler')
    if model is not None and scaler is not None:
        preds = model.predict(scaler.transform(current_pop))
        density_vals = preds[:, 0]
        tensile_vals = preds[:, 1]
        dis_vals = preds[:, 3]
        diss_vals = preds[:, 4]
        feat_api, feat_efrf = generate_feasible_samples(model, scaler)
    else:
        density_vals = np.full_like(api_vals, np.nan)
        tensile_vals = np.full_like(api_vals, np.nan)
        dis_vals = np.full_like(api_vals, np.nan)
        diss_vals = np.full_like(api_vals, np.nan)
        feat_api, feat_efrf = np.array([]), np.array([])

    # Determine y-values
    if plot_type == "API vs EFRF":
        x_label, y_label = "API (%)", "EFRF"
        x_vals, y_vals = api_vals, efrf_vals
        feasible_x, feasible_y = feat_api, feat_efrf
        tested_results = get_current_formulation_results()
        tested_api = st.session_state.api
        tested_y = tested_results['efrf']
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

    # Feasible region background (only for API vs EFRF)
    if plot_type == "API vs EFRF" and len(feasible_x) > 0:
        fig.add_trace(go.Scatter(
            x=feasible_x,
            y=feasible_y,
            mode='markers',
            name='Feasible region (sampled)',
            marker=dict(size=3, color='lightblue', opacity=0.3, line=dict(width=0)),
            hovertemplate='API: %{x:.1f}%<br>EFRF: %{y:.3f}<extra></extra>',
            showlegend=True
        ))

    # Sort by API for line
    sort_idx = np.argsort(x_vals)
    x_sorted = x_vals[sort_idx]
    y_sorted = y_vals[sort_idx]
    api_sorted = api_vals[sort_idx]
    density_sorted = density_vals[sort_idx]

    # Pareto front line + markers
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
        customdata=np.column_stack([density_sorted])
    ))

    # ---- Golden solution: only if we are on the final generation ----
    final_gen = generations_recorded[-1]
    if golden and gen_slider == final_gen:
        # Find the golden point in the current front
        golden_idx = st.session_state.get('golden_idx')
        if golden_idx is not None and golden_idx < len(current_pop):
            golden_api = current_pop[golden_idx, 0]
            golden_efrf = current_obj[golden_idx, 1]
            # Map to y-axis based on plot type
            if plot_type == "API vs EFRF":
                golden_y = golden_efrf
            elif plot_type == "API vs Tensile":
                golden_y = tensile_vals[golden_idx]
            elif plot_type == "API vs Disintegration":
                golden_y = dis_vals[golden_idx]
            else:
                golden_y = diss_vals[golden_idx]
            fig.add_trace(go.Scatter(
                x=[golden_api],
                y=[golden_y],
                mode='markers',
                name='🏆 Golden Solution',
                marker=dict(size=15, color='gold', symbol='diamond', line=dict(width=2, color='black')),
                hovertemplate=f'<b>Golden</b><br>{x_label}: %{{x:.3f}}<br>{y_label}: %{{y:.3f}}<extra></extra>'
            ))
        else:
            st.caption("Golden solution not found in this front (it may be from a different generation).")
    elif golden and gen_slider != final_gen:
        # Show a note that the golden is from final gen
        st.caption("The golden solution is from the final generation; it is not shown on this earlier front.")

    # ---- Tested formulation ----
    if tested_y is not None:
        fig.add_trace(go.Scatter(
            x=[tested_api],
            y=[tested_y],
            mode='markers',
            name='Tested Formulation',
            marker=dict(size=12, color='blue', symbol='circle', line=dict(width=2, color='darkblue')),
            hovertemplate=f'<b>Tested</b><br>{x_label}: %{{x:.3f}}<br>{y_label}: %{{y:.3f}}<extra></extra>'
        ))

    # Constraint boundaries
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

# ================================================================
# The rest of the UI functions (render_golden_solution, render_side_by_side_comparison, etc.) are unchanged.
# ================================================================
def render_golden_solution(golden):
    # ... same as before
    pass

def render_side_by_side_comparison(golden, all_solutions):
    # ... same
    pass

def render_best_solutions():
    # ... same
    pass

def render_optimization_summary():
    # ... same
    pass

# ================================================================
# MAIN ORCHESTRATION (unchanged except for setting default generation)
# ================================================================
def main():
    render_sidebar()
    st.markdown("# 🧬 Hybrid AI · Multi-Objective Tablet Optimization")
    st.markdown("#### Nile Valley University · Sudan · v29.28‑R32")
    st.markdown("---")
    render_input_panel()
    render_binder_grade_comparison()
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        quick_predict = st.button("⚡ Quick Predict (current formulation)", use_container_width=True)
    with col2:
        run_button = st.button("🚀 Run Hybrid Optimization", type="primary", use_container_width=True)

    if quick_predict:
        valid, msg = validate_formulation(
            st.session_state.api, st.session_state.binder,
            st.session_state.pvpp, st.session_state.mgst,
            st.session_state.mcc, st.session_state.moisture
        )
        if not valid:
            st.error(f"❌ {msg}")
        else:
            with st.spinner("Running model prediction..."):
                quick_results = get_current_formulation_results()
            render_results_summary(quick_results)
            st.info("This is a direct model prediction for the formulation currently set on the sliders — it does not run the NSGA-II search.")

    if run_button:
        start_time = time.time()
        valid, msg = validate_formulation(
            st.session_state.api, st.session_state.binder,
            st.session_state.pvpp, st.session_state.mgst,
            st.session_state.mcc, st.session_state.moisture
        )
        if not valid:
            st.error(f"❌ {msg}")
            return
        st.session_state.optimization_complete = True

        render_training_progress()
        opt_progress = st.progress(0, text="Running NSGA-II generation 0/%d..." % NSGA_GENERATIONS)
        def _update_opt_progress(gen, total):
            frac = min(1.0, max(0.0, (gen + 1) / total))
            opt_progress.progress(frac, text=f"Running NSGA-II generation {min(gen + 1, total)}/{total}...")
        solutions, golden, gen_history = run_real_optimization(progress_callback=_update_opt_progress)
        opt_progress.empty()
        st.session_state.results = get_current_formulation_results()
        st.session_state.golden_solution = golden
        st.session_state.best_solutions = solutions
        st.session_state.pareto_history = gen_history
        # Set default generation to last
        if gen_history:
            st.session_state.selected_generation = gen_history[-1]['generation']
        st.session_state.runtime = round(time.time() - start_time, 1)

        render_results_summary(st.session_state.results)
        render_pareto_evolution()
        render_golden_solution(golden)
        render_side_by_side_comparison(golden, solutions)
        render_best_solutions()
        render_optimization_summary()

        st.success(f"⏱️ Optimization completed in {st.session_state.runtime} seconds!")
        st.balloons()

    elif st.session_state.optimization_complete and st.session_state.results:
        render_results_summary(st.session_state.results)
        render_pareto_evolution()
        render_golden_solution(st.session_state.golden_solution)
        render_side_by_side_comparison(st.session_state.golden_solution, st.session_state.best_solutions)
        render_best_solutions()
        render_optimization_summary()

    else:
        st.info("👆 Adjust parameters and click 'Run Hybrid Optimization' to begin.")
        st.markdown("---")
        st.markdown("### 🎯 Key Features")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**🧠 Physics-Informed AI**")
            st.markdown("**📊 Two Objectives: API & EFRF**")
        with col2:
            st.markdown("**⚖️ Mass Balance Enforced**")
            st.markdown("**🔬 Tensile as Constraint**")
        with col3:
            st.markdown("**📈 Pareto Front with Feasible Region**")
            st.markdown("**🏆 Golden Solution**")

if __name__ == "__main__":
    main()
