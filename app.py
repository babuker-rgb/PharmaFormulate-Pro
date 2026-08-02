# ================================================================
# Hybrid AI · Multi-Objective Tablet Optimization (Integrated v32.1)
# Nile Valley University · Sudan · v29.28‑R32 (Heckel PINN + v32.1 Features)
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="Hybrid AI · Tablet Optimization v32.1",
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
        'runtime': 0, 'train_time': None, 'nsga_time': None, 'pareto_history': None,
        'user_data': None, 'data_source': 'synthetic',
        'force_retrain': False,
        '_trained_model': None, '_trained_scaler': None, '_trained_history': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
initialize_session_state()

# ================================================================
# HELPER FUNCTIONS
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
# HYBRID NEURAL NETWORK (ADDED DROPOUT & UNCERTAINTY)
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
        self.dropout = nn.Dropout(0.1)  # NEW: Added for Uncertainty
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                
    def forward(self, x):
        h1 = torch.relu(self.bn1(self.fc1(x)))
        h1 = self.dropout(h1) # NEW
        h2 = torch.relu(self.bn2(self.fc2(h1))) + h1
        h2 = self.dropout(h2) # NEW
        h3 = torch.relu(self.bn3(self.fc3(h2))) + h2
        h3 = self.dropout(h3) # NEW
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

    # NEW: Monte Carlo Dropout for Uncertainty
    def predict_with_uncertainty(self, x, n_samples=20):
        self.train()
        with torch.no_grad():
            if not torch.is_tensor(x): x = torch.tensor(x, dtype=torch.float32)
            x_repeat = x.repeat(n_samples, 1)
            preds = self.forward(x_repeat).numpy().reshape(n_samples, -1, 5)
        self.eval()
        return np.mean(preds, 0), np.std(preds, 0)

# ================================================================
# DATA GENERATION
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

def apply_ood_shrinkage(pred, pop_scaled, y_train_mean):
    """Shrinks density/tensile/EFRF predictions toward the training-target
    mean, proportional to how far out-of-distribution the (scaled) input
    is. BUGFIX: this exact correction previously existed only inside
    NSGAIIOptimizer.evaluate() — the "Quick Predict" / current-formulation
    path (get_current_formulation_results()) called model.predict()
    directly with no correction at all. For a formulation sitting in a
    sparsely-trained region, that produced simultaneous saturation on
    multiple outputs at once (density crashing to its floor while tensile/
    EFRF/disintegration all hit their own extremes for the same input) —
    exactly the surrogate-exploitation failure mode the optimizer's
    shrinkage was built to prevent, just reachable through a path that
    never got the fix. Extracted into one shared function so the two
    prediction paths can't silently drift apart again.
    Returns (density, tensile, efrf) — matches what NSGAIIOptimizer.evaluate()
    corrects; disintegration/dissolution are left as raw predictions,
    consistent with the optimizer's existing (not-yet-extended) scope.
    """
    density, tensile, efrf = pred[..., 0], pred[..., 1], pred[..., 2]
    ood_z = np.abs(pop_scaled)
    ood_raw = np.clip(ood_z - 2.0, 0, None).sum(axis=-1)
    shrink_factor = 1.0 / (1.0 + ood_raw)
    density = shrink_factor * density + (1 - shrink_factor) * y_train_mean[0]
    tensile = shrink_factor * tensile + (1 - shrink_factor) * y_train_mean[1]
    efrf = shrink_factor * efrf + (1 - shrink_factor) * y_train_mean[2]
    return density, tensile, efrf

# ================================================================
# CHECKPOINT PATHS & ATOMIC SAVE (From v29.28)
# ================================================================
CHECKPOINT_SYNTHETIC = os.path.join(tempfile.gettempdir(), 'co_hybai_synthetic_v10_physics.pt')

def _data_fingerprint(df):
    try:
        row_hash = int(pd.util.hash_pandas_object(df, index=False).sum())
    except Exception:
        row_hash = hash(tuple(df.shape))
    return f"{len(df)}_{row_hash & 0xFFFFFFFF}"

@st.cache_resource(show_spinner=False)
def train_model(use_real=False, _real_df=None, data_fingerprint=None):
    if use_real and _real_df is not None and data_fingerprint:
        checkpoint_path = os.path.join(tempfile.gettempdir(), f'co_hybai_real_{data_fingerprint}.pt')
    else:
        checkpoint_path = CHECKPOINT_SYNTHETIC

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
            try:
                os.remove(checkpoint_path)
            except OSError:
                pass
    
    if use_real and _real_df is not None:
        required_cols = ['API','Binder','PVPP','MgSt','MCC','Moisture','Pressure','Speed',
                         'Density','Tensile','EFRF','Disintegration','Dissolution']
        missing = [c for c in required_cols if c not in _real_df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        X = _real_df[required_cols[:8]].values.astype(np.float32)
        y = _real_df[required_cols[8:]].values.astype(np.float32)
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

    # Physics Variables (Heckel)
    pressure_train_t = torch.tensor(X[train_idx, 6], dtype=torch.float32)
    binder_train_t = torch.tensor(X[train_idx, 1], dtype=torch.float32)

    def calculate_heckel_density_torch(pressure, binder):
        porosity0 = 0.45 - 0.001 * (pressure - PRESSURE_MIN) - 0.01 * (binder - 3.0)
        return torch.clamp(1.0 - porosity0 * torch.exp(-0.01 * (pressure - PRESSURE_MIN)), 0.55, 0.95)
    
    model = HybridTabletModel(input_dim=8, hidden_dim=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)
    
    target_var = y_train_t.var(dim=0, unbiased=False)
    target_var = torch.clamp(target_var, min=1e-6)
    
    def weighted_mse(pred, true):
        return (((pred - true) ** 2) / target_var).mean()
    
    loss_fn = weighted_mse
    PHYSICS_LOSS_WEIGHT = 0.1
    history = {'loss': [], 'r2': [], 'rmse': [], 'data_source': data_source}
    best_val_loss = np.inf
    best_state = None
    patience, patience_counter = 60, 0
    
    for epoch in range(TRAINING_EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)
        # Physics-residual term
        physical_density = calculate_heckel_density_torch(pressure_train_t, binder_train_t)
        physics_loss = torch.mean((pred[:, 0] - physical_density) ** 2) * PHYSICS_LOSS_WEIGHT
        loss = loss + physics_loss
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
    
    # ATOMIC SAVE
    tmp_path = checkpoint_path + f'.tmp{os.getpid()}'
    torch.save({
        'model_state': model.state_dict(),
        'scaler': scaler,
        'history': history,
        'data_source': data_source
    }, tmp_path)
    os.replace(tmp_path, checkpoint_path)
    
    return model, scaler, history

# ================================================================
# NSGA-II OPTIMIZER
# ================================================================
class NSGAIIOptimizer:
    def __init__(self, model, scaler, pop_size=50, generations=80, y_train_mean=None):
        self.model = model
        self.scaler = scaler
        self.pop_size = pop_size
        self.generations = generations
        self.n_objectives = 3
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
        api = pop[:, 0]

        density, tensile, efrf = apply_ood_shrinkage(pred, pop_scaled, self.y_train_mean)
        
        fitness = np.column_stack([-density, -tensile, efrf])
        api_norm = np.clip((api - 80) / 18, 0.0, 1.0)
        tensile_norm = np.clip(tensile / 8.5, 0.0, 1.0)
        penalty_api = 0.08 * (1 - api_norm)
        penalty_tensile = 0.05 * (1 - tensile_norm)
        fitness[:, 0] += penalty_api
        fitness[:, 1] += penalty_tensile
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
# NEW: ANALYSIS FUNCTIONS (Sensitivity & 3D/Radar)
# ================================================================
def perform_sensitivity_analysis(model, scaler, ref_solution):
    try:
        rf = RandomForestRegressor(n_estimators=50)
        X_local = np.random.normal(loc=ref_solution, scale=0.05*np.abs(ref_solution), size=(500, 8))
        bounds_min = np.array([API_MIN, BINDER_MIN, PVPP_MIN, MGST_MIN, MCC_MIN, MOISTURE_MIN, PRESSURE_MIN, SPEED_MIN])
        bounds_max = np.array([API_MAX, BINDER_MAX, PVPP_MAX, MGST_MAX, MCC_MAX, MOISTURE_MAX, PRESSURE_MAX, SPEED_MAX])
        X_local = np.clip(X_local, bounds_min, bounds_max)
        X_scaled = scaler.transform(X_local)
        y_local = model.predict(X_scaled)[:, 0] # Using Density as target
        rf.fit(X_scaled, y_local)
        perm_importance = permutation_importance(rf, X_scaled, y_local)
        feature_names = ['API', 'Binder', 'PVPP', 'MgSt', 'MCC', 'Moisture', 'Pressure', 'Speed']
        return dict(zip(feature_names, perm_importance.importances_mean))
    except:
        return None

def render_3d_pareto(pop, obj, golden_idx, tested_data=None):
    fig = go.Figure(data=[go.Scatter3d(x=pop[:, 0], y=obj[:, 2], z=-obj[:, 1], mode='markers', marker=dict(size=4, color=pop[:, 0], colorscale='Viridis'), name='Pareto')])
    if golden_idx is not None:
        fig.add_trace(go.Scatter3d(x=[pop[golden_idx, 0]], y=[obj[golden_idx, 2]], z=[-obj[golden_idx, 1]], mode='markers', marker=dict(size=15, color='gold', symbol='diamond'), name='Golden'))
    if tested_data is not None:
        fig.add_trace(go.Scatter3d(x=[tested_data['api']], y=[tested_data['efrf']], z=[tested_data['tensile']], mode='markers', marker=dict(size=12, color='blue', symbol='circle'), name='Tested Formulation'))
    fig.update_layout(scene=dict(xaxis_title='API (%)', yaxis_title='EFRF', zaxis_title='Tensile (MPa)'), height=450)
    st.plotly_chart(fig, use_container_width=True)

# ================================================================
# RESULT FUNCTIONS (Feasible Filtering)
# ================================================================
def get_model_and_scaler():
    real_df = st.session_state.get('user_data')
    use_real = real_df is not None and len(real_df) > 0
    fingerprint = _data_fingerprint(real_df) if use_real else None
    return train_model(use_real=use_real, _real_df=real_df, data_fingerprint=fingerprint)

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

    preds = model.predict(scaler.transform(pareto_pop))
    solutions = []
    for i, (row, pred) in enumerate(zip(pareto_pop, preds)):
        api, binder, pvpp, mgst, mcc, moisture = row[:6]
        density, tensile = pred[0], pred[1]
        efrf = float(pareto_obj[i, 2])
        if efrf < 0.40:
            quality = calculate_quality_score(density, tensile, efrf, api=api)
            solutions.append({
                'Solution': f'S{i+1}',
                'API (%)': api, 'Binder (%)': binder, 'PVPP (%)': pvpp,
                'MgSt (%)': mgst, 'MCC (%)': mcc, 'Moisture (%)': moisture,
                'Total (%)': api + binder + pvpp + mgst + mcc + moisture,
                'Density': density, 'Tensile (MPa)': tensile, 'EFRF': efrf,
                'Quality Score': quality['overall']
            })
    solutions.sort(key=lambda x: x['Quality Score'], reverse=True)
    if not solutions:
        return [], None, [], None, None
    return solutions, solutions[0], gen_history, pareto_pop, pareto_obj

def get_current_formulation_results():
    model = st.session_state.get('_trained_model')
    scaler = st.session_state.get('_trained_scaler')
    history = st.session_state.get('_trained_history')
    if model is None or scaler is None or history is None:
        model, scaler, history = get_model_and_scaler()
        st.session_state['_trained_model'] = model
        st.session_state['_trained_scaler'] = scaler
        st.session_state['_trained_history'] = history

    n = normalize_formulation(
        st.session_state.api, st.session_state.binder, st.session_state.pvpp,
        st.session_state.mgst, st.session_state.mcc, st.session_state.moisture
    )
    row = np.array([[n['api'], n['binder'], n['pvpp'], n['mgst'], n['mcc'], n['moisture'],
                     st.session_state.pressure, st.session_state.speed]], dtype=np.float32)
    row_scaled = scaler.transform(row)
    pred = model.predict(row_scaled)[0]
    # BUGFIX: previously used this raw, unshrunk prediction directly — see
    # apply_ood_shrinkage() for why that let this specific path produce
    # simultaneously-saturated outputs (density at its floor, tensile/EFRF
    # at their extremes) for formulations in sparsely-trained regions,
    # while the NSGA-II optimizer's own predictions were already corrected
    # for exactly this.
    y_train_mean = history.get('y_train_mean', [0.75, 4.5, 0.5, 20.0, 45.0])
    density, tensile, efrf = apply_ood_shrinkage(pred[np.newaxis, :], row_scaled, y_train_mean)
    return {
        'api': float(n['api']),
        'density': float(density[0]), 'tensile': float(tensile[0]), 'efrf': float(efrf[0]),
        'disintegration': float(pred[3]), 'dissolution': float(pred[4])
    }

# ================================================================
# UI RENDER FUNCTIONS
# ================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🧬 Hybrid AI Framework")
        st.markdown("---")
        st.markdown(f"**Version:** v32.1-Integrated")
        st.markdown(f"**Institution:** Nile Valley University")
        st.markdown(f"**Department:** Pharmaceutical Engineering")
        st.markdown("---")
        
        st.markdown("### 📂 Data Source")
        uploaded_file = st.file_uploader(
            "Upload your dataset (CSV)",
            type=["csv"],
            help="Required columns: API, Binder, PVPP, MgSt, MCC, Moisture, Pressure, Speed, Density, Tensile, EFRF, Disintegration, Dissolution"
        )
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                required_cols = ['API','Binder','PVPP','MgSt','MCC','Moisture','Pressure','Speed',
                                 'Density','Tensile','EFRF','Disintegration','Dissolution']
                missing = [c for c in required_cols if c not in df.columns]
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    numeric_df = df[required_cols].apply(pd.to_numeric, errors='coerce')
                    bad_cols = [c for c in required_cols if numeric_df[c].isna().any()]
                    if bad_cols:
                        st.error(f"Non-numeric or missing values found in: {bad_cols}. "
                                f"Please clean the data and re-upload.")
                    elif len(df) < 20:
                        st.error(f"Only {len(df)} rows found — at least 20 are needed for a "
                                f"meaningful train/validation split.")
                    else:
                        st.session_state.user_data = df
                        st.success(f"✅ Loaded {len(df)} samples")
                        st.session_state.force_retrain = True
            except Exception as e:
                st.error(f"Error reading file: {e}")
        else:
            if st.session_state.data_source == 'real':
                st.info(f"🔵 Using real data ({len(st.session_state.get('user_data', []))} samples)")
            else:
                st.info("🟢 Using synthetic data (fallback)")

        if st.session_state.get('force_retrain'):
            st.info("ℹ️ New data uploaded — the model will train on it "
                    "(first run only; cached afterward) next time you click "
                    "Quick Predict or Run Hybrid Optimization.")

        if st.button("🔄 Force Retrain", use_container_width=True):
            import glob
            checkpoints_to_remove = [CHECKPOINT_SYNTHETIC] + glob.glob(
                os.path.join(tempfile.gettempdir(), 'co_hybai_real_*.pt'))
            for checkpoint in checkpoints_to_remove:
                if os.path.exists(checkpoint):
                    try:
                        os.remove(checkpoint)
                    except OSError:
                        pass
            train_model.clear()
            for key in ('_trained_model', '_trained_scaler', '_trained_history'):
                st.session_state.pop(key, None)
            st.session_state.optimization_complete = False
            st.session_state.force_retrain = False
            st.success("Cache cleared — model will retrain on the next run.")
        
        st.markdown("---")
        
        with st.expander("📊 Optimization Objectives", expanded=True):
            st.markdown("1. **Maximize API%** (penalised low‑API)")
            st.markdown("2. **Maximize Tensile** (penalised low‑tensile)")
            st.markdown("3. **Maximize Density** → Better tablet quality")
            st.markdown("4. **Minimize EFRF** → Better powder flow")
        with st.expander("⚙️ Algorithm Settings", expanded=False):
            st.markdown(f"**Population:** {POPULATION_SIZE}")
            st.markdown(f"**Generations:** {NSGA_GENERATIONS}")
            st.markdown(f"**Training Epochs:** {TRAINING_EPOCHS}")
            st.markdown("**Algorithm:** NSGA‑II (3 obj + API & Tensile penalties)")
            st.markdown("**Model:** Physics‑Informed Neural Network")
            st.markdown("**Constraint:** Mass Balance (Σ = 100%)")
            st.markdown(f"**Runtime:** {st.session_state.runtime}s" if st.session_state.runtime else "**Runtime:** Pending")
        st.markdown("---")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("↺ Reset Sliders", use_container_width=True):
                defaults = {
                    'api': 96.5, 'binder': 1.4, 'pvpp': 1.0, 'mgst': 0.10,
                    'mcc': 1.5, 'moisture': 0.50, 'binder_grade': 0,
                    'particle_size': 50.0, 'pressure': 200.0, 'speed': 20.0,
                    'granule': 125.0, 'dwell_time': 25.0, 'friction': 0.25,
                    'decompression_time': 35.0,
                }
                for k, v in defaults.items():
                    st.session_state[k] = v
                st.rerun()
        st.markdown("---")
        st.caption("© 2024 Nile Valley University · Sudan")

    # Return weights to main scope
    w_api = st.sidebar.slider("Weight for API", 0.0, 1.0, 0.4)
    w_quality = st.sidebar.slider("Weight for Quality", 0.0, 1.0, 0.6)
    return w_api, w_quality

def render_binder_grade_comparison():
    st.markdown("---")
    st.markdown("## 🔬 Binder Grade Impact")
    df = pd.DataFrame([
        {"Binder Grade": name,
         "Compressibility": p["compressibility"]*100,
         "Disintegration": p["disintegration"]*100,
         "Flowability": p["flow"]*100}
        for name, p in BINDER_GRADES.items()
    ])
    fig = go.Figure()
    for col in ["Compressibility", "Disintegration", "Flowability"]:
        fig.add_trace(go.Bar(
            x=df["Binder Grade"], y=df[col], name=col,
            text=[f"{v:.0f}%" for v in df[col]], textposition="outside"
        ))
    fig.update_layout(
        barmode="group",
        title="Binder Grade Properties",
        yaxis=dict(title="Score (%)", range=[0, 100]),
        height=350,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

def render_mass_balance_display(api, binder, pvpp, mgst, mcc, moisture):
    raw_total = api + binder + pvpp + mgst + mcc + moisture
    summary = get_formulation_summary(api, binder, pvpp, mgst, mcc, moisture)
    st.markdown("### 📊 Formulation Mass Balance")
    components = [
        ('API', summary['API'], '#ff6b6b'),
        ('Binder', summary['Binder'], '#4ecdc4'),
        ('PVPP', summary['PVPP'], '#45b7d1'),
        ('MgSt', summary['MgSt'], '#96ceb4'),
        ('MCC', summary['MCC'], '#ffeaa7'),
        ('Moisture', summary['Moisture'], '#dfe6e9')
    ]
    fig = go.Figure()
    for name, value, color in components:
        fig.add_trace(go.Bar(
            y=[name], x=[value], orientation='h',
            name=name, marker_color=color,
            text=f'{value:.1f}%', textposition='outside'
        ))
    fig.update_layout(
        xaxis=dict(title='Percentage (%)', range=[0, 105]),
        height=250, showlegend=False, barmode='stack',
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    col1, col2 = st.columns([3, 1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        deviation = abs(raw_total - 100.0)
        if deviation < 2.0:
            status = "✅ Close to 100%"
        elif deviation < 10.0:
            status = "⚠️ Adjusted to fit"
        else:
            status = "🔴 Large adjustment"
        st.metric("**Raw Total (before normalization)**", f"{raw_total:.1f}%", status)
        st.caption("Normalized formulation used for prediction:")
        for name in ['API', 'Binder', 'PVPP', 'MgSt', 'MCC', 'Moisture']:
            st.caption(f"{name}: {summary[name]:.1f}%")

def render_input_panel():
    st.markdown("## 🧪 Formulation Parameters")
    st.info("⚠️ Components will be automatically normalized to sum to 100%.")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.api = st.slider("**API Content (%)**", API_MIN, API_MAX, st.session_state.api, step=0.5)
        st.session_state.binder = st.slider("**Binder (%)**", BINDER_MIN, BINDER_MAX, st.session_state.binder, step=0.1)
        st.session_state.pvpp = st.slider("**PVPP (%)**", PVPP_MIN, PVPP_MAX, st.session_state.pvpp, step=0.1)
        st.session_state.mgst = st.slider("**MgSt (%)**", MGST_MIN, MGST_MAX, st.session_state.mgst, step=0.05)
    with col2:
        st.session_state.mcc = st.slider("**MCC (%)**", MCC_MIN, MCC_MAX, st.session_state.mcc, step=0.1)
        st.session_state.moisture = st.slider("**Moisture Content (%)**", MOISTURE_MIN, MOISTURE_MAX, st.session_state.moisture, step=0.1)
        grade_idx = st.session_state.get('binder_grade', 0)
        if not isinstance(grade_idx, int) or grade_idx >= len(BINDER_GRADE_NAMES):
            grade_idx = 0
        selected = st.selectbox("**Binder Grade**", BINDER_GRADE_NAMES, index=grade_idx)
        st.session_state.binder_grade = BINDER_GRADE_NAMES.index(selected)
        props = BINDER_GRADES[selected]
        st.caption(f"🔍 **{selected} Properties:**")
        st.caption(f"• Compressibility: {props['compressibility']:.0%}")
        st.caption(f"• Disintegration: {props['disintegration']:.0%}")
        st.caption(f"• Flowability: {props['flow']:.0%}")
        st.session_state.particle_size = st.slider("**Particle Size (µm)**", PARTICLE_SIZE_MIN, PARTICLE_SIZE_MAX, st.session_state.particle_size, step=5.0)
    render_mass_balance_display(
        st.session_state.api, st.session_state.binder,
        st.session_state.pvpp, st.session_state.mgst,
        st.session_state.mcc, st.session_state.moisture
    )
    st.markdown("---")
    st.markdown("## ⚙️ Process Parameters")
    st.caption("ℹ️ Only **Compression Pressure** and **Tableting Speed** currently feed into the model's predictions.")
    col3, col4 = st.columns(2)
    with col3:
        st.session_state.pressure = st.slider("**Compression Pressure (MPa)**", PRESSURE_MIN, PRESSURE_MAX, st.session_state.pressure, step=2.0)
        st.session_state.speed = st.slider("**Tableting Speed (rpm)**", SPEED_MIN, SPEED_MAX, st.session_state.speed, step=0.5)
        st.session_state.granule = st.slider("**Granule Size (µm)**", GRANULE_MIN, GRANULE_MAX, st.session_state.granule, step=5.0)
    with col4:
        st.session_state.dwell_time = st.slider("**Dwell Time (ms)**", DWELL_TIME_MIN, DWELL_TIME_MAX, st.session_state.dwell_time, step=1.0)
        st.session_state.friction = st.slider("**Friction Coefficient**", FRICTION_MIN, FRICTION_MAX, st.session_state.friction, step=0.01)
        st.session_state.decompression_time = st.slider("**Decompression Time (ms)**", DECOMPRESSION_TIME_MIN, DECOMPRESSION_TIME_MAX, st.session_state.decompression_time, step=2.0)

def target_status(value, threshold, mode='min', comfortable=None):
    if mode == 'min':
        if value < threshold:
            return "🔴 Below target"
        if comfortable is not None and value >= comfortable:
            return "✅ Excellent"
        return "✅ Passes (near limit)"
    else:
        if value > threshold:
            return "🔴 Exceeds limit"
        if comfortable is not None and value <= comfortable:
            return "✅ Excellent"
        return "⚠️ Passes (near limit)"

def render_results_summary(results):
    st.markdown("---")
    st.markdown("## 📊 Optimization Results")
    api_val = st.session_state.api
    quality = calculate_quality_score(results['density'], results['tensile'], results['efrf'], api=api_val)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("**API%**", f"{api_val:.1f}%", "🎯 Target: maximize")
        st.metric("**Density**", f"{results['density']:.3f}",
                 target_status(results['density'], 0.80, mode='min', comfortable=0.85))
    with col2:
        st.metric("**Tensile Strength**", f"{results['tensile']:.2f} MPa",
                 target_status(results['tensile'], 1.5, mode='min', comfortable=3.0))
        st.metric("**EFRF**", f"{results['efrf']:.3f}",
                 target_status(results['efrf'], 0.40, mode='max', comfortable=0.30))
    with col3:
        st.metric("**Disintegration Time**", f"{results['disintegration']:.1f} min",
                 target_status(results['disintegration'], 15.0, mode='max', comfortable=10.0))
        st.metric("**Overall Quality Score**", f"{quality['overall']:.1f}%",
                 "Good" if quality['overall'] > 60 else "Needs Improvement")
    with st.expander("📊 Quality Score Breakdown", expanded=False):
        st.markdown(f"""
        | Component | Score | Weight | Contribution |
        |-----------|-------|--------|--------------|
        | API%      | {quality.get('api_score', 0):.1f}% | 30% | {quality.get('api_score', 0) * 0.3:.1f}% |
        | Density   | {quality['density_score']:.1f}% | {quality['weights']['density']:.0%} | {quality['density_score']*quality['weights']['density']:.1f}% |
        | Tensile   | {quality['tensile_score']:.1f}% | {quality['weights']['tensile']:.0%} | {quality['tensile_score']*quality['weights']['tensile']:.1f}% |
        | EFRF      | {quality['efrf_score']:.1f}% | {quality['weights']['efrf']:.0%} | {quality['efrf_score']*quality['weights']['efrf']:.1f}% |
        | **Total** | - | - | **{quality['overall']:.1f}%** |
        """)

def render_training_progress():
    st.markdown("---")
    st.markdown("## 🔍 Training Progress")
    with st.spinner("Training physics-informed model on formulation data..."):
        history = run_real_training_and_get_history()
    if not history['loss']:
        st.warning("No training history available.")
        return
    
    data_source = history.get('data_source', 'unknown')
    st.info(f"📊 Model trained on: **{data_source.upper()}** data ({history.get('n_samples', '?')} samples)")
    
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(y=history['loss'], mode='lines', name='Validation Loss', line=dict(color='#ff6b6b', width=2)))
    fig_loss.update_layout(title='Loss Evolution (real validation loss, recorded every 20 epochs)',
                           xaxis_title='Recorded checkpoint', yaxis_title='MSE Loss', height=250)
    st.plotly_chart(fig_loss, use_container_width=True)
    fig_metrics = go.Figure()
    fig_metrics.add_trace(go.Scatter(y=history['r2'], mode='lines', name='R² Score', line=dict(color='#51cf66', width=2)))
    fig_metrics.add_trace(go.Scatter(y=history['rmse'], mode='lines', name='RMSE', line=dict(color='#5c7cfa', width=2)))
    fig_metrics.update_layout(title='Model Performance (real validation metrics)',
                              xaxis_title='Recorded checkpoint', yaxis_title='Metric Value', height=250)
    st.plotly_chart(fig_metrics, use_container_width=True)
    st.success(f"✅ Training complete! Final validation R² (macro-average across 5 outputs) = "
              f"{history['r2'][-1]:.3f}, RMSE = {history['rmse'][-1]:.3f}")

    per_output = history.get('per_output_r2')
    if per_output:
        with st.expander("🔬 Model Diagnostics — per-property fit quality", expanded=False):
            diag_df = pd.DataFrame({
                'Property': list(per_output.keys()),
                'R²': [f"{v:.3f}" for v in per_output.values()],
                'Fit quality': ['✅ Good' if v > 0.7 else ('⚠️ Moderate' if v > 0.4 else '🔴 Poor')
                                for v in per_output.values()]
            })
            st.dataframe(diag_df, hide_index=True, use_container_width=True)
            st.caption(
                f"Trained on {history.get('n_train', '?')} samples, validated on "
                f"{history.get('n_val', '?')} held-out samples."
            )

def render_pareto_evolution():
    st.markdown("---")
    st.markdown("## 🌐 Pareto Front Evolution: API% vs EFRF")
    golden = st.session_state.get('golden_solution', None)
    pareto_history = st.session_state.get('pareto_history', None)
    if not pareto_history:
        st.info("Run the optimization to see the real Pareto front evolve across generations.")
        return

    generations_recorded = [h['generation'] for h in pareto_history]
    gen_slider = st.select_slider("Select generation to view", options=generations_recorded, value=generations_recorded[-1])
    current_entry = next(h for h in pareto_history if h['generation'] == gen_slider)
    current_obj = current_entry['pareto_objectives']
    current_pop = current_entry['pareto_solutions']

    # Extract data
    api_vals = current_pop[:, 0]
    efrf_vals = current_obj[:, 2]

    # Filter feasible (EFRF < 0.40)
    feasible_mask = efrf_vals < 0.40
    api_feas = api_vals[feasible_mask]
    efrf_feas = efrf_vals[feasible_mask]

    # Sort by API
    sort_idx = np.argsort(api_feas)
    api_sorted = api_feas[sort_idx]
    efrf_sorted = efrf_feas[sort_idx]

    # Enforce monotonic (cumulative maximum) for smooth curve
    if len(efrf_sorted) > 0:
        cummax_efrf = np.maximum.accumulate(efrf_sorted)
    else:
        cummax_efrf = efrf_sorted

    fig = go.Figure()

    fig.add_hrect(
        y0=0, y1=0.40, x0=API_MIN, x1=API_MAX,
        fillcolor='rgba(144, 238, 144, 0.25)', line_width=0,
        layer='below'
    )
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=12, symbol='square', color='rgba(144, 238, 144, 0.5)'),
        name='Feasible region (EFRF < 0.40)'
    ))

    # Smooth Pareto front line + markers (using cummax_efrf)
    fig.add_trace(go.Scatter(
        x=api_sorted,
        y=cummax_efrf,
        mode='lines+markers',
        name='Pareto Front',
        line=dict(color='red', width=2),
        marker=dict(size=8, color='#a3c4f3', line=dict(width=1, color='#4a6fa5')),
        hovertemplate='API: %{x:.2f}%<br>EFRF: %{y:.3f}<extra></extra>'
    ))

    # ---- EXTENSION TO LIMIT 0.40 ----
    if len(api_sorted) > 0 and cummax_efrf[-1] < 0.40:
        last_api = api_sorted[-1]
        last_efrf = cummax_efrf[-1]
        # Draw a dashed line from the last Pareto point to the limit at 0.40
        fig.add_trace(go.Scatter(
            x=[last_api, last_api],
            y=[last_efrf, 0.40],
            mode='lines',
            name='Front extension to limit',
            line=dict(color='red', width=2, dash='dash'),
            showlegend=True,
            hovertemplate='Extension to EFRF=0.40<extra></extra>'
        ))
        # Add a marker at the limit point
        fig.add_trace(go.Scatter(
            x=[last_api],
            y=[0.40],
            mode='markers',
            name='EFRF limit point',
            marker=dict(size=8, color='red', symbol='cross'),
            showlegend=True,
            hovertemplate=f'API: {last_api:.2f}%<br>EFRF: 0.40<extra></extra>'
        ))
    # --------------------------------------

    # Golden solution (already feasible)
    if golden:
        fig.add_trace(go.Scatter(
            x=[golden['API (%)']],
            y=[golden['EFRF']],
            mode='markers',
            name='🏆 Golden Solution',
            marker=dict(size=22, color='gold', symbol='star', line=dict(width=1.5, color='#8a6d00')),
            hovertemplate=f"<b>🏆 Golden Solution</b><br>API: {golden['API (%)']:.2f}%<br>EFRF: {golden['EFRF']:.3f}<extra></extra>"
        ))

    # Tested formulation
    tested = st.session_state.get('results')
    if tested and 'api' in tested and 'efrf' in tested:
        fig.add_trace(go.Scatter(
            x=[tested['api']],
            y=[tested['efrf']],
            mode='markers',
            name='🔵 Tested Formulation',
            marker=dict(size=14, color='blue', symbol='circle', line=dict(width=1.5, color='white')),
            hovertemplate=f"<b>Tested Formulation</b><br>API: {tested['api']:.2f}%<br>EFRF: {tested['efrf']:.3f}<extra></extra>"
        ))

    # Boundaries
    fig.add_hline(y=0.40, line_dash='dash', line_color='gray',
                  annotation_text='EFRF limit (0.40)', annotation_position='top left')
    fig.add_vline(x=API_MIN, line_dash='dash', line_color='gray',
                  annotation_text=f'API min ({API_MIN}%)', annotation_position='bottom left')
    fig.add_vline(x=API_MAX, line_dash='dash', line_color='gray',
                  annotation_text=f'API max ({API_MAX}%)', annotation_position='bottom right')

    fig.update_layout(
        title=f'Pareto Front - Generation {gen_slider}',
        xaxis_title='API (%)',
        yaxis_title='EFRF',
        height=500,
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"**Generation {gen_slider+1}/{NSGA_GENERATIONS}** · "
        f"Pareto-optimal solutions at this generation: {len(api_sorted)}"
    )

def render_golden_solution(golden):
    if not golden:
        return
    st.markdown("---")
    st.markdown("## 🏆 Golden Solution (Balanced Trade-off)")
    density_status = target_status(golden['Density'], 0.80, mode='min', comfortable=0.85)
    tensile_status = target_status(golden['Tensile (MPa)'], 1.5, mode='min', comfortable=3.0)
    efrf_status = target_status(golden['EFRF'], 0.40, mode='max', comfortable=0.30)
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px; border-radius: 12px; color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
        <h3 style="color: white;">✨ Optimal Formulation</h3>
        <p><b>API:</b> {golden['API (%)']:.1f}% &nbsp;|&nbsp;
           <b>Binder:</b> {golden['Binder (%)']:.1f}% &nbsp;|&nbsp;
           <b>PVPP:</b> {golden['PVPP (%)']:.1f}% &nbsp;|&nbsp;
           <b>MgSt:</b> {golden['MgSt (%)']:.2f}% &nbsp;|&nbsp;
           <b>MCC:</b> {golden['MCC (%)']:.1f}% &nbsp;|&nbsp;
           <b>Moisture:</b> {golden['Moisture (%)']:.1f}%</p>
        <div style="display: flex; gap: 20px; flex-wrap: wrap; margin-top: 10px;">
            <div><b>API%:</b> {golden['API (%)']:.1f}% 🎯 High</div>
            <div><b>Density:</b> {golden['Density']:.3f} {density_status}</div>
            <div><b>Tensile:</b> {golden['Tensile (MPa)']:.2f} MPa {tensile_status}</div>
            <div><b>EFRF:</b> {golden['EFRF']:.3f} {efrf_status}</div>
            <div><b>Quality Score:</b> {golden['Quality Score']:.1f}% 🏆 Best</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    status_map = {'Density': density_status, 'Tensile': tensile_status, 'EFRF': efrf_status}
    flagged = {name: s for name, s in status_map.items()
              if "near limit" in s or "Below target" in s or "Exceeds limit" in s}
    if flagged:
        details = "; ".join(f"**{name}** ({s.split(' ', 1)[1] if ' ' in s else s})" for name, s in flagged.items())
        st.warning(f"⚠️ This is the best available trade-off among the Pareto-optimal solutions found, "
                   f"but {details} — worth reviewing before committing to this formulation.")
    else:
        st.success("✅ This formulation maximises API% and Tensile while preserving excellent tablet quality!")

# ================================================================
# DYNAMIC RADAR IMPLEMENTATION
# ================================================================
def render_side_by_side_comparison(golden, all_solutions):
    if not golden or not all_solutions:
        return
    st.markdown("---")
    st.markdown("## 📊 Side‑by‑Side Comparison")
    top = all_solutions[:3]
    df = pd.DataFrame(top)
    # FIX: ensure all expected columns exist; if not, we fill with NaN or skip
    expected_cols = ['Solution','API (%)','Binder (%)','PVPP (%)','MgSt (%)',
                     'MCC (%)','Moisture (%)','Density','Tensile (MPa)',
                     'EFRF','Quality Score']
    # If any column is missing, add it with NaN
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan
    st.dataframe(df[expected_cols], use_container_width=True)
    
    st.markdown("### 🎯 Dynamic Performance Radar")
    # allow user to select which solutions to compare
    selected = st.multiselect(
        "Select solutions to compare in Radar chart:",
        options=[s['Solution'] for s in all_solutions],
        default=[all_solutions[0]['Solution'], all_solutions[1]['Solution']] if len(all_solutions) > 1 else [all_solutions[0]['Solution']]
    )
    
    if selected:
        categories = ["API%", "Density", "Tensile (MPa)", "EFRF (inverted)", "Quality Score"]
        fig = go.Figure()
        for sol in all_solutions:
            if sol['Solution'] in selected:
                fig.add_trace(go.Scatterpolar(
                    r=[
                        (sol["API (%)"] - 80) / 18,
                        sol["Density"] / 0.95,
                        sol["Tensile (MPa)"] / 8.5,
                        1 - sol["EFRF"],
                        sol["Quality Score"] / 100
                    ],
                    theta=categories,
                    fill='toself',
                    name=sol["Solution"]
                ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,1])),
            showlegend=True,
            height=400,
            margin=dict(l=40, r=40, t=40, b=40),
            title="Performance Comparison Across Selected Solutions"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select at least one solution to display the radar chart.")

def render_best_solutions():
    solutions = st.session_state.get('best_solutions')
    golden = st.session_state.get('golden_solution')
    if not solutions or not golden:
        return
    st.markdown("---")
    st.markdown("## 🏆 Optimal Solutions (Mass Balance Ensured)")
    st.info("✅ All formulations are normalized to sum to 100%")

    df = pd.DataFrame(solutions)
    df_display = df.copy()
    for col in ['API (%)', 'Binder (%)', 'PVPP (%)', 'MCC (%)', 'Moisture (%)']:
        if col in df_display.columns:
            df_display[col] = df_display[col].round(1)
    if 'Total (%)' in df_display.columns:
        df_display['Total (%)'] = df_display['Total (%)'].round(1)
    df_display['MgSt (%)'] = df_display['MgSt (%)'].round(2)
    df_display['Density'] = df_display['Density'].round(3)
    df_display['Tensile (MPa)'] = df_display['Tensile (MPa)'].round(2)
    df_display['EFRF'] = df_display['EFRF'].round(3)
    df_display['Quality Score'] = df_display['Quality Score'].round(1)
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download Optimization Report (CSV)",
                           data=csv,
                           file_name=f"results_{timestamp}.csv",
                           mime="text/csv",
                           use_container_width=True)
    with col2:
        json_report = {
            'timestamp': timestamp,
            'golden_solution': golden,
            'all_solutions': df.to_dict('records'),
            'data_source': st.session_state.data_source,
            'parameters': {
                'population': POPULATION_SIZE,
                'generations': NSGA_GENERATIONS,
                'epochs': TRAINING_EPOCHS,
                'runtime_seconds': st.session_state.runtime,
                'api_penalty': 0.08,
                'tensile_penalty': 0.05
            }
        }
        st.download_button("📥 Download Full Report (JSON)",
                           data=json.dumps(json_report, indent=2, default=str),
                           file_name=f"report_{timestamp}.json",
                           mime="application/json",
                           use_container_width=True)

def render_optimization_summary():
    st.markdown("---")
    st.markdown("## 📈 Optimization Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("⏱️ Total Runtime", f"{st.session_state.runtime}s" if st.session_state.runtime else "—")
        train_t = st.session_state.get('train_time')
        nsga_t = st.session_state.get('nsga_time')
        if train_t is not None and nsga_t is not None:
            # NEW: "Runtime" previously bundled model load/train time and
            # the NSGA-II optimization time into one number, with no way to
            # tell which phase was actually slow when the total was
            # unexpectedly high (as it was on this run). Broken out here.
            st.caption(f"Model load/train: {train_t}s · NSGA-II search: {nsga_t}s")
    with col2:
        # BUGFIX: this was previously computed from the TOTAL runtime
        # (including training), which understates true optimization
        # throughput whenever training time is non-trivial — "evaluations
        # per second" should reflect the NSGA-II loop alone.
        nsga_time_for_calc = st.session_state.get('nsga_time') or st.session_state.runtime
        evals_per_sec = (POPULATION_SIZE * NSGA_GENERATIONS) / max(1, nsga_time_for_calc)
        st.metric("⚡ Evaluations/Second (NSGA-II only)", f"{evals_per_sec:.0f}")

    solutions = st.session_state.get('best_solutions') or []
    col3, col4 = st.columns([2, 1])
    with col3:
        st.markdown("### Key Statistics")
        if solutions:
            sol_df = pd.DataFrame(solutions)
            stats = pd.DataFrame({
                'Metric': [
                    'Data Source',
                    'Total Solutions Evaluated',
                    'Pareto Solutions Found',
                    'Best Density',
                    'Best Tensile',
                    'Best EFRF',
                    'Best API%',
                    'Mass Balance',
                    'Penalties'
                ],
                'Value': [
                    st.session_state.data_source.upper(),
                    f'{POPULATION_SIZE * NSGA_GENERATIONS:,}',
                    f'{len(sol_df)}',
                    f'{sol_df["Density"].max():.3f}',
                    f'{sol_df["Tensile (MPa)"].max():.2f} MPa',
                    f'{sol_df["EFRF"].min():.3f}',
                    f'{sol_df["API (%)"].max():.1f}%',
                    '✅ 100% (Enforced)',
                    'API: 0.08 | Tensile: 0.05'
                ]
            })
            st.dataframe(stats, hide_index=True, use_container_width=True)
        else:
            st.info("Run the optimization to see real statistics here.")
    with col4:
        st.markdown("### Status Indicators")
        st.success("✅ Algorithm: NSGA‑II + dual penalty")
        st.success("✅ Model: Physics‑Informed Neural Network")
        st.success("✅ Constraint: Mass Balance")
        st.info("📊 Pareto Front: Optimized")
        st.info("🎯 Objectives: 3 + API/Tensile bias")
        if st.session_state.data_source == 'real':
            st.info("📂 Data: Real (user uploaded)")
        else:
            st.info("🔄 Data: Synthetic (fallback)")

# ================================================================
# MAIN ORCHESTRATION
# ================================================================
def main():
    # FIX: Assign the return values from render_sidebar correctly!
    w_api, w_quality = render_sidebar()
    
    st.markdown("# 🧬 Hybrid AI · Multi-Objective Tablet Optimization")
    st.markdown("#### Nile Valley University · Sudan · v32.1-Integrated")
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
            st.info("This is a direct model prediction for the formulation currently set on the "
                    "sliders — it does not run the NSGA-II search. Click **Run Hybrid Optimization** "
                    "for a full Pareto-front search across the design space.")

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

        # NEW: previously start_time covered render_training_progress()
        # (model load/train) AND the NSGA-II run combined into one
        # "Runtime" / "Evaluations per second" figure — which is
        # misleading (evaluations/second implies pure optimization
        # throughput) and made two consecutive ~50s runs impossible to
        # diagnose, since there was no way to tell whether training or
        # optimization was the slow part. Split so the next run shows both.
        train_start = time.time()
        render_training_progress()
        train_elapsed = round(time.time() - train_start, 1)

        opt_progress = st.progress(0, text="Running NSGA-II generation 0/%d..." % NSGA_GENERATIONS)
        def _update_opt_progress(gen, total):
            frac = min(1.0, max(0.0, (gen + 1) / total))
            opt_progress.progress(frac, text=f"Running NSGA-II generation {min(gen + 1, total)}/{total}...")
        
        nsga_start = time.time()
        solutions, golden_from_optimizer, gen_history, pareto_pop, pareto_obj = run_real_optimization(progress_callback=_update_opt_progress)
        nsga_elapsed = round(time.time() - nsga_start, 1)
        opt_progress.empty()
        
        st.session_state.results = get_current_formulation_results()
        # We will recompute golden based on custom weights, so we don't use golden_from_optimizer
        st.session_state.pareto_history = gen_history
        st.session_state.runtime = round(time.time() - start_time, 1)
        st.session_state.train_time = train_elapsed
        st.session_state.nsga_time = nsga_elapsed

        # Store weights safely for future use
        st.session_state.sidebar_w_api = w_api
        st.session_state.sidebar_w_quality = w_quality
        
        # FIX: Verify model and scaler are available
        model = st.session_state.get('_trained_model')
        scaler = st.session_state.get('_trained_scaler')
        if model is None or scaler is None:
            st.error("Model or Scaler not found. Please ensure training completed successfully.")
            st.stop()

        # ---- FIX: Filter Pareto front to only feasible solutions (EFRF < 0.40) ----
        feasible_mask = pareto_obj[:, 2] < 0.40
        if not np.any(feasible_mask):
            st.error("No feasible Pareto solutions found (EFRF >= 0.40 for all). Try adjusting constraints.")
            st.stop()
        pareto_pop = pareto_pop[feasible_mask]
        pareto_obj = pareto_obj[feasible_mask]
        # ------------------------------------------------------------------------

        # Calculate Golden solution ONLY from feasible Pareto front solutions using custom weights
        weights = np.array([w_api, w_quality])
        scores = []
        for i in range(len(pareto_pop)):
            s = (pareto_pop[i,0]/100 * weights[0]) + ((1 - pareto_obj[i].sum()/4) * weights[1])
            scores.append(s)
        golden_idx = np.argmax(scores)
        best_sol = pareto_pop[golden_idx]
        
        # Predict uncertainty for the golden solution
        pop_scaled = scaler.transform([best_sol])
        preds, unc = model.predict_with_uncertainty(torch.tensor(pop_scaled, dtype=torch.float32))
        preds, unc = preds[0], unc[0]
        
        # Reconstruct the golden dictionary from the chosen solution
        api_val = best_sol[0]
        binder_val = best_sol[1]
        pvpp_val = best_sol[2]
        mgst_val = best_sol[3]
        mcc_val = best_sol[4]
        moisture_val = best_sol[5]
        density_val = preds[0]
        tensile_val = preds[1]
        efrf_val = preds[2]
        quality = calculate_quality_score(density_val, tensile_val, efrf_val, api=api_val)

        golden = {
            'Solution': f'S{golden_idx+1}',
            'API (%)': api_val,
            'Binder (%)': binder_val,
            'PVPP (%)': pvpp_val,
            'MgSt (%)': mgst_val,
            'MCC (%)': mcc_val,
            'Moisture (%)': moisture_val,
            'Total (%)': api_val + binder_val + pvpp_val + mgst_val + mcc_val + moisture_val,
            'Density': density_val,
            'Tensile (MPa)': tensile_val,
            'EFRF': efrf_val,
            'Quality Score': quality['overall']
        }

        # Update session state with the unified golden solution
        st.session_state.golden_solution = golden

        st.success(f"🏆 Golden Solution Found!\nAPI: {golden['API (%)']:.2f}% | EFRF: {golden['EFRF']:.3f} ± {unc[2]:.3f}")
        st.caption(f"Optimization took {st.session_state.runtime:.2f} seconds.")

        # Compute Tested Formulation Data using st.session_state
        slider_form = np.array([[st.session_state.api, st.session_state.binder, st.session_state.pvpp,
                                 st.session_state.mgst, st.session_state.mcc, st.session_state.moisture,
                                 st.session_state.pressure, st.session_state.speed]], dtype=np.float32)
        slider_preds, _ = model.predict_with_uncertainty(torch.tensor(scaler.transform(slider_form), dtype=torch.float32))
        tested_data = {'api': float(st.session_state.api), 'efrf': float(slider_preds[0][2]), 'tensile': float(slider_preds[0][1])}

        # ---- FIX: Use the full solutions list, sorted by weighted scores ----
        # Compute scores for all feasible solutions using the same weights
        scores_all = []
        for i in range(len(pareto_pop)):
            s = (pareto_pop[i,0]/100 * weights[0]) + ((1 - pareto_obj[i].sum()/4) * weights[1])
            scores_all.append(s)
        sorted_indices = np.argsort([-scores_all[i] for i in range(len(pareto_pop))])
        
        # Rebuild the full solutions list from the original 'solutions' (which contain all columns)
        # We need to map the sorted indices to the original solutions list
        # Since 'solutions' corresponds to pareto_pop (feasible), we can sort it accordingly
        # But 'solutions' was already filtered to feasible; we'll re-sort it.
        # We need to get the full solutions list from the optimizer (which is returned as 'solutions')
        # and we have 'solutions' variable already.
        # We'll create a list of (score, solution) and sort.
        # However, 'solutions' is already a list of dicts; we can sort it directly.
        # To keep it simple, we'll create a sorted list from 'solutions' using the scores computed.
        # We'll pair each solution with its score from scores_all.
        paired = list(zip(scores_all, solutions))
        paired.sort(key=lambda x: -x[0])  # sort descending
        sorted_solutions = [sol for _, sol in paired]
        st.session_state.best_solutions = sorted_solutions
        # ----------------------------------------------------------------

        # Build a limited DataFrame for radar (if needed) - but we already have full solutions.
        # We'll keep the radar using the full solutions.
        # The radar uses only a subset of columns anyway, so it's fine.

        # Render All Visualizations
        st.subheader("🌐 2D Pareto Front (API vs EFRF)")
        render_pareto_evolution()
        
        st.subheader("🌐 3D Pareto Front (API - EFRF - Tensile)")
        # Re-create final population from gen_history for 3D plot (if needed)
        last_entry = gen_history[-1]
        final_pop_hist = last_entry['population']
        final_obj_hist = last_entry['objectives']
        # Find golden index in this population for 3D plot marker
        golden_idx_hist = None
        for i, row in enumerate(final_pop_hist):
            if abs(row[0] - golden['API (%)']) < 0.01:
                golden_idx_hist = i
                break
        render_3d_pareto(final_pop_hist, final_obj_hist, golden_idx_hist, tested_data=tested_data)

        render_golden_solution(golden)
        render_side_by_side_comparison(golden, st.session_state.best_solutions)
        render_best_solutions()
        render_optimization_summary()

        with st.expander("🔬 Sensitivity Analysis (Local)", expanded=False):
            model = st.session_state.get('_trained_model')
            scaler = st.session_state.get('_trained_scaler')
            if model and scaler and golden:
                sens_data = perform_sensitivity_analysis(model, scaler, np.array([golden['API (%)'], golden['Binder (%)'], golden['PVPP (%)'], golden['MgSt (%)'], golden['MCC (%)'], golden['Moisture (%)'], 200.0, 20.0]))
                if sens_data:
                    st.bar_chart(pd.Series(sens_data))
                else:
                    st.warning("Could not compute local sensitivity.")
            else:
                st.warning("Model or Golden Solution missing for Sensitivity Analysis.")

        st.success(f"⏱️ Optimization completed in {st.session_state.runtime} seconds!")
        st.balloons()
        st.rerun()

    elif st.session_state.optimization_complete and st.session_state.results:
        render_results_summary(st.session_state.results)
        render_pareto_evolution()
        
        with st.expander("🌐 3D Pareto Front (API - EFRF - Tensile)", expanded=False):
            gen_history = st.session_state.get('pareto_history')
            if gen_history:
                last_entry = gen_history[-1]
                final_pop = last_entry['population']
                final_obj = last_entry['objectives']
                golden = st.session_state.get('golden_solution')
                golden_idx = None
                if golden:
                    for i, row in enumerate(final_pop):
                        if abs(row[0] - golden['API (%)']) < 0.01:
                            golden_idx = i
                            break
                render_3d_pareto(final_pop, final_obj, golden_idx, tested_data=st.session_state.results)

        render_golden_solution(st.session_state.golden_solution)
        render_side_by_side_comparison(st.session_state.golden_solution, st.session_state.best_solutions)
        render_best_solutions()
        render_optimization_summary()

        with st.expander("🔬 Sensitivity Analysis (Local)", expanded=False):
            model = st.session_state.get('_trained_model')
            scaler = st.session_state.get('_trained_scaler')
            golden = st.session_state.get('golden_solution')
            if model and scaler and golden:
                sens_data = perform_sensitivity_analysis(model, scaler, np.array([golden['API (%)'], golden['Binder (%)'], golden['PVPP (%)'], golden['MgSt (%)'], golden['MCC (%)'], golden['Moisture (%)'], 200.0, 20.0]))
                if sens_data:
                    st.bar_chart(pd.Series(sens_data))
                else:
                    st.warning("Could not compute local sensitivity.")
            else:
                st.warning("Model or Golden Solution missing for Sensitivity Analysis.")

    else:
        st.info("👆 Adjust parameters and click 'Run Hybrid Optimization' to begin.")
        st.markdown("---")
        st.markdown("### 🎯 Key Features")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**🧠 Physics-Informed AI**")
            st.markdown("**📊 API & Tensile Penalties**")
        with col2:
            st.markdown("**⚖️ Mass Balance Enforced**")
            st.markdown("**🔬 PINN Constraints**")
        with col3:
            st.markdown("**📈 Pareto Front**")
            st.markdown("**🏆 Golden Solution**")

if __name__ == "__main__":
    main()
