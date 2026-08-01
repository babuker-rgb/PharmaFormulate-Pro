# ================================================================
# Hybrid AI v32.2-Ultimate-Pro-Full · Integrated UI/Physics/3D/Radar
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
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="Hybrid AI v32.2-Pro-Full", page_icon="🧬", layout="wide"
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

POPULATION_SIZE = 50
NSGA_GENERATIONS = 80
TRAINING_EPOCHS = 1200
EARLY_STOPPING_PATIENCE = 60
HIDDEN_SIZE = 256
N_SAMPLES = 8000
PHYSICS_LOSS_WEIGHT = 0.1
BOUNDARY_FRACTION = 0.30
EFRF_THRESHOLD = 0.40

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
    if total <= 0: total = 1.0
    norm = (comps / total) * 100
    return {'api': norm[0], 'binder': norm[1], 'pvpp': norm[2],
            'mgst': norm[3], 'mcc': norm[4], 'moisture': norm[5], 'total': 100.0}

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
                'tensile_score': tensile_score, 'efrf_score': efrf_score, 'weights': weights}

# ================================================================
# SCALER & HYBRID PINN MODEL (WITH UNCERTAINTY / DROPOUT)
# ================================================================
class InputScaler:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self
    def transform(self, X):
        return (X - self.mean_) / self.std_

class HybridTabletModel(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=HIDDEN_SIZE):
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
        self.dropout = nn.Dropout(0.1) # For Monte Carlo Dropout Uncertainty
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
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
    def predict_with_uncertainty(self, x, n_samples=20):
        self.train() # Activate Dropout
        with torch.no_grad():
            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=torch.float32)
            x_repeat = x.repeat(n_samples, 1)
            preds = self.forward(x_repeat).numpy().reshape(n_samples, -1, 5)
        self.eval()
        return np.mean(preds, axis=0), np.std(preds, axis=0)

# ================================================================
# DATA GENERATION & PHYSICS CONSTRAINTS
# ================================================================
def generate_synthetic_data(n_samples=N_SAMPLES, seed=42):
    rng = np.random.default_rng(seed)
    bounds = [(API_MIN, API_MAX), (BINDER_MIN, BINDER_MAX), (PVPP_MIN, PVPP_MAX),
              (MGST_MIN, MGST_MAX), (MCC_MIN, MCC_MAX), (MOISTURE_MIN, MOISTURE_MAX)]
    cols = [rng.uniform(lo, hi, n_samples) for lo, hi in bounds]
    comps = np.column_stack(cols)
    n_boundary = int(n_samples * BOUNDARY_FRACTION)
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
    comps = comps / comps.sum(axis=1, keepdims=True) * 100.0
    api, binder, pvpp, mgst, mcc, moisture = comps.T
    pressure = rng.uniform(PRESSURE_MIN, PRESSURE_MAX, n_samples)
    speed = rng.uniform(SPEED_MIN, SPEED_MAX, n_samples)
    X = np.column_stack([api, binder, pvpp, mgst, mcc, moisture, pressure, speed]).astype(np.float32)
    
    porosity0 = 0.45 - 0.001 * (pressure - PRESSURE_MIN) - 0.01 * (binder - 3.0)
    density = np.clip(1.0 - porosity0 * np.exp(-0.01 * (pressure - PRESSURE_MIN)), 0.55, 0.95)
    density += rng.normal(0, 0.005, n_samples)
    density = np.clip(density, 0.55, 0.95)
    tensile = (0.5 + 6.0 * (density - 0.55) / 0.40 + 0.4 * (binder - BINDER_MIN) - 1.2 * (mgst - MGST_MIN) + 0.3 * (api - API_MIN) / (API_MAX - API_MIN))
    tensile += rng.normal(0, 0.1, n_samples)
    tensile = np.clip(tensile, 0.5, 8.5)
    efrf = (0.55 - 0.35 * (density - 0.55) / 0.40 + 0.25 * (api - API_MIN) / (API_MAX - API_MIN) - 0.15 * (binder - BINDER_MIN) / (BINDER_MAX - BINDER_MIN) + 0.2 * (mgst - MGST_MIN))
    efrf += rng.normal(0, 0.03, n_samples)
    efrf = np.clip(efrf, 0.02, 0.98)
    disintegration = (12.0 - 4.0 * (pvpp - PVPP_MIN) / (PVPP_MAX - PVPP_MIN) + 5.0 * (binder - BINDER_MIN) / (BINDER_MAX - BINDER_MIN) + 3.0 * (moisture - MOISTURE_MIN) / (MOISTURE_MAX - MOISTURE_MIN))
    disintegration += rng.normal(0, 0.5, n_samples)
    disintegration = np.clip(disintegration, 2.0, 45.0)
    dissolution = 1.8 * disintegration + 5.0 - 3.0 * (pvpp - PVPP_MIN) / (PVPP_MAX - PVPP_MIN)
    dissolution += rng.normal(0, 1.0, n_samples)
    dissolution = np.clip(dissolution, 10.0, 90.0)
    y = np.column_stack([density, tensile, efrf, disintegration, dissolution])
    return X.astype(np.float32), y.astype(np.float32)

# ================================================================
# ROBUST TRAINING LOOP (ATOMIC SAVE + PHYSICS LOSS)
# ================================================================
CHECKPOINT_SYNTHETIC = os.path.join(tempfile.gettempdir(), 'co_hybai_ultimate_v32.pt')

def _data_fingerprint(df):
    try: row_hash = int(pd.util.hash_pandas_object(df, index=False).sum())
    except: row_hash = hash(tuple(df.shape))
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
            model = HybridTabletModel(input_dim=8, hidden_dim=HIDDEN_SIZE)
            model.load_state_dict(ckpt['model_state'])
            model.eval()
            scaler = ckpt['scaler']
            history = ckpt['history']
            history['data_source'] = ckpt.get('data_source', 'unknown')
            st.session_state.data_source = history.get('data_source', 'unknown')
            return model, scaler, history
        except Exception as e:
            st.warning(f"Checkpoint load failed: {e}. Retraining...")
            try: os.remove(checkpoint_path)
            except OSError: pass
    
    if use_real and _real_df is not None:
        required_cols = ['API','Binder','PVPP','MgSt','MCC','Moisture','Pressure','Speed',
                         'Density','Tensile','EFRF','Disintegration','Dissolution']
        missing = [c for c in required_cols if c not in _real_df.columns]
        if missing: raise ValueError(f"Missing columns: {missing}")
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

    # Physics tensors for PINN
    pressure_train_t = torch.tensor(X[train_idx, 6], dtype=torch.float32)
    binder_train_t = torch.tensor(X[train_idx, 1], dtype=torch.float32)
    def calculate_heckel_density_torch(pressure, binder):
        porosity0 = 0.45 - 0.001 * (pressure - PRESSURE_MIN) - 0.01 * (binder - 3.0)
        return torch.clamp(1.0 - porosity0 * torch.exp(-0.01 * (pressure - PRESSURE_MIN)), 0.55, 0.95)

    model = HybridTabletModel(input_dim=8, hidden_dim=HIDDEN_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)
    target_var = y_train_t.var(dim=0, unbiased=False)
    target_var = torch.clamp(target_var, min=1e-6)
    def weighted_mse(pred, true): return (((pred - true) ** 2) / target_var).mean()

    history = {'loss': [], 'r2': [], 'rmse': [], 'data_source': data_source}
    best_val_loss = np.inf
    best_state = None
    patience_counter = 0
    
    for epoch in range(TRAINING_EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = weighted_mse(pred, y_train_t)
        # Physics Constraint (Heckel equation)
        physical_density = calculate_heckel_density_torch(pressure_train_t, binder_train_t)
        physics_loss = torch.mean((pred[:, 0] - physical_density) ** 2) * PHYSICS_LOSS_WEIGHT
        loss += physics_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = weighted_mse(val_pred, y_val_t).item()
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
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                break
    
    if best_state is not None: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        ss_res = ((y_val_t - val_pred) ** 2).sum(dim=0)
        ss_tot = ((y_val_t - y_val_t.mean(dim=0)) ** 2).sum(dim=0)
        final_per_output_r2 = (1 - ss_res / torch.clamp(ss_tot, min=1e-8)).numpy()
    history['per_output_r2'] = {
        'Density': float(final_per_output_r2[0]), 'Tensile': float(final_per_output_r2[1]),
        'EFRF': float(final_per_output_r2[2]), 'Disintegration': float(final_per_output_r2[3]),
        'Dissolution': float(final_per_output_r2[4])
    }
    history['n_train'] = len(train_idx)
    history['n_val'] = len(val_idx)
    history['y_train_mean'] = y_train_t.mean(dim=0).numpy().tolist()
    history['n_samples'] = len(X)

    # ATOMIC SAVE: Write to temp then os.replace
    tmp_path = checkpoint_path + f'.tmp{os.getpid()}'
    torch.save({'model_state': model.state_dict(), 'scaler': scaler, 'history': history, 'data_source': data_source}, tmp_path)
    os.replace(tmp_path, checkpoint_path)
    return model, scaler, history

# ================================================================
# NSGA-II OPTIMIZER
# ================================================================
class NSGAIIOptimizer:
    def __init__(self, model, scaler, pop_size=POPULATION_SIZE, generations=NSGA_GENERATIONS, y_train_mean=None):
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
        density, tensile, efrf = pred[:, 0], pred[:, 1], pred[:, 2]
        api = pop[:, 0]
        ood_z = np.abs(pop_scaled)
        ood_raw = np.clip(ood_z - 2.0, 0, None).sum(axis=1)
        shrink_factor = 1.0 / (1.0 + ood_raw)
        density = shrink_factor * density + (1 - shrink_factor) * self.y_train_mean[0]
        tensile = shrink_factor * tensile + (1 - shrink_factor) * self.y_train_mean[1]
        efrf = shrink_factor * efrf + (1 - shrink_factor) * self.y_train_mean[2]
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
                if i == j: continue
                if np.all(obj[i] <= obj[j]) and np.any(obj[i] < obj[j]):
                    dom_sol[i].append(j)
                elif np.all(obj[j] <= obj[i]) and np.any(obj[j] < obj[i]):
                    dom_count[i] += 1
            if dom_count[i] == 0: first_front.append(i)
        fronts = [first_front]
        curr = 0
        while curr < len(fronts) and fronts[curr]:
            next_front = []
            for i in fronts[curr]:
                for j in dom_sol[i]:
                    dom_count[j] -= 1
                    if dom_count[j] == 0: next_front.append(j)
            curr += 1
            if next_front: fronts.append(next_front)
            else: break
        return fronts

    def crowding_distance(self, obj, front):
        n = len(front)
        if n <= 2: return np.ones(n) * np.inf
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

    GENE_BOUNDS = [(API_MIN, API_MAX), (BINDER_MIN, BINDER_MAX), (PVPP_MIN, PVPP_MAX),
                   (MGST_MIN, MGST_MAX), (MCC_MIN, MCC_MAX), (MOISTURE_MIN, MOISTURE_MAX),
                   (PRESSURE_MIN, PRESSURE_MAX), (SPEED_MIN, SPEED_MAX)]

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
                if r1 < r2: selected.append(i1)
                elif r2 < r1: selected.append(i2)
                else:
                    d1 = self.crowding_distance(obj, fronts[r1])[fronts[r1].index(i1)]
                    d2 = self.crowding_distance(obj, fronts[r2])[fronts[r2].index(i2)]
                    selected.append(i1 if d1 > d2 else i2)
            sel_pop = pop[selected]
            offspring = []
            for i in range(0, self.pop_size, 2):
                p1, p2 = sel_pop[i], sel_pop[(i+1) % self.pop_size]
                if np.random.random() < 0.8:
                    c1, c2 = np.zeros_like(p1), np.zeros_like(p2)
                    for j in range(n_vars):
                        if np.random.random() < 0.5:
                            beta = 1.0 + 2.0 * np.random.random()
                            c1[j] = 0.5 * ((1+beta)*p1[j] + (1-beta)*p2[j])
                            c2[j] = 0.5 * ((1-beta)*p1[j] + (1+beta)*p2[j])
                        else:
                            c1[j], c2[j] = p1[j], p2[j]
                else:
                    c1, c2 = p1.copy(), p2.copy()
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
                history.append({'generation': gen, 'population': pop.copy(), 'objectives': obj.copy(),
                                'pareto_indices': pareto_indices, 'pareto_solutions': pop[pareto_indices],
                                'pareto_objectives': obj[pareto_indices]})
            yield pop, obj, history, gen

# ================================================================
# ADVANCED UI PLOTTING & ANALYSIS FUNCTIONS
# ================================================================
def generate_feasible_samples(model, scaler, n_samples=3000):
    if model is None or scaler is None: return np.array([]), np.array([])
    try:
        rng = np.random.default_rng(42)
        api = rng.uniform(API_MIN, API_MAX, n_samples)
        binder = rng.uniform(BINDER_MIN, BINDER_MAX, n_samples)
        pvpp = rng.uniform(PVPP_MIN, PVPP_MAX, n_samples)
        mgst = rng.uniform(MGST_MIN, MGST_MAX, n_samples)
        mcc = rng.uniform(MCC_MIN, MCC_MAX, n_samples)
        moisture = rng.uniform(MOISTURE_MIN, MOISTURE_MAX, n_samples)
        pressure = rng.uniform(PRESSURE_MIN, PRESSURE_MAX, n_samples)
        speed = rng.uniform(SPEED_MIN, SPEED_MAX, n_samples)
        comps = np.column_stack([api, binder, pvpp, mgst, mcc, moisture])
        lo = np.array([API_MIN, BINDER_MIN, PVPP_MIN, MGST_MIN, MCC_MIN, MOISTURE_MIN])
        hi = np.array([API_MAX, BINDER_MAX, PVPP_MAX, MGST_MAX, MCC_MAX, MOISTURE_MAX])
        comps = np.clip(comps, lo, hi)
        total = comps.sum(axis=1, keepdims=True)
        norm = comps / total * 100.0
        X = np.column_stack([norm, pressure, speed])
        preds = model.predict(scaler.transform(X))
        density, tensile, efrf = preds[:, 0], preds[:, 1], preds[:, 2]
        feasible_mask = (efrf < 0.40) & (density >= 0.7) & (tensile >= 1.5)
        return norm[feasible_mask, 0], efrf[feasible_mask]
    except: return np.array([]), np.array([])

def perform_sensitivity_analysis(model, scaler, ref_solution):
    try:
        rf = RandomForestRegressor(n_estimators=50)
        X_local = np.random.normal(loc=ref_solution, scale=0.05*np.abs(ref_solution), size=(500, 8))
        bounds_min = np.array([API_MIN, BINDER_MIN, PVPP_MIN, MGST_MIN, MCC_MIN, MOISTURE_MIN, PRESSURE_MIN, SPEED_MIN])
        bounds_max = np.array([API_MAX, BINDER_MAX, PVPP_MAX, MGST_MAX, MCC_MAX, MOISTURE_MAX, PRESSURE_MAX, SPEED_MAX])
        X_local = np.clip(X_local, bounds_min, bounds_max)
        X_scaled = scaler.transform(X_local)
        y_local = model(torch.tensor(X_scaled, dtype=torch.float32)).numpy()[:, 0]
        rf.fit(X_scaled, y_local)
        perm_importance = permutation_importance(rf, X_scaled, y_local)
        feature_names = ['API', 'Binder', 'PVPP', 'MgSt', 'MCC', 'Moisture', 'Pressure', 'Speed']
        return dict(zip(feature_names, perm_importance.importances_mean))
    except: return None

def render_3d_pareto(pop, obj, golden_idx, tested_data=None):
    fig = go.Figure(data=[go.Scatter3d(x=pop[:, 0], y=obj[:, 2], z=-obj[:, 1], mode='markers', marker=dict(size=4, color=pop[:, 0], colorscale='Viridis'), name='Pareto')])
    if golden_idx is not None:
        fig.add_trace(go.Scatter3d(x=[pop[golden_idx, 0]], y=[obj[golden_idx, 2]], z=[-obj[golden_idx, 1]], mode='markers', marker=dict(size=15, color='gold', symbol='diamond'), name='🏆 Golden'))
    if tested_data is not None:
        fig.add_trace(go.Scatter3d(x=[tested_data['api']], y=[tested_data['efrf']], z=[tested_data['tensile']], mode='markers', marker=dict(size=12, color='blue', symbol='circle', line=dict(color='white', width=1)), name='🔵 Tested'))
    fig.update_layout(scene=dict(xaxis_title='API (%)', yaxis_title='EFRF', zaxis_title='Tensile (MPa)'), height=450)
    st.plotly_chart(fig, use_container_width=True)

def render_dynamic_radar(solutions_df, selected_solutions):
    if not selected_solutions: return
    fig = go.Figure()
    for i, row in solutions_df.iterrows():
        if row['Solution'] in selected_solutions:
            fig.add_trace(go.Scatterpolar(r=[(row['API (%)']-80)/18, row['Density']/0.95, row['Tensile (MPa)']/8.5, 1-row['EFRF'], row['Quality Score']/100], theta=['API%', 'Density', 'Tensile', 'EFRF (Inv)', 'Quality'], fill='toself', name=row['Solution']))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True, height=380)
    st.plotly_chart(fig, use_container_width=True)

def render_results_summary(results):
    st.markdown("---"); st.markdown("## 📊 Optimization Results")
    api_val = st.session_state.api
    quality = calculate_quality_score(results['density'], results['tensile'], results['efrf'], api=api_val)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("**API%**", f"{api_val:.1f}%", "🎯 Target: maximize")
        st.metric("**Density**", f"{results['density']:.3f}", "Target: 0.80+")
    with col2:
        st.metric("**Tensile Strength**", f"{results['tensile']:.2f} MPa", "Target: 1.5+")
        st.metric("**EFRF**", f"{results['efrf']:.3f}", "Limit: 0.40")
    with col3:
        st.metric("**Disintegration Time**", f"{results['disintegration']:.1f} min", "Limit: 15.0")
        st.metric("**Overall Quality Score**", f"{quality['overall']:.1f}%", "Good if > 60")

def render_pareto_evolution():
    st.markdown("---"); st.markdown("## 🌐 Pareto Front Evolution: API% vs EFRF")
    golden = st.session_state.get('golden_solution', None)
    pareto_history = st.session_state.get('pareto_history', None)
    if not pareto_history: return st.info("Run the optimization to see the Pareto front.")

    generations_recorded = [h['generation'] for h in pareto_history]
    gen_slider = st.select_slider("Select generation to view", options=generations_recorded, value=generations_recorded[-1])
    current_entry = next(h for h in pareto_history if h['generation'] == gen_slider)
    current_obj = current_entry['pareto_objectives']; current_pop = current_entry['pareto_solutions']
    api_vals, efrf_vals = current_pop[:, 0], current_obj[:, 2]
    feasible_mask = efrf_vals < 0.40
    api_feas, efrf_feas = api_vals[feasible_mask], efrf_vals[feasible_mask]
    sort_idx = np.argsort(api_feas)
    api_sorted, efrf_sorted = api_feas[sort_idx], efrf_feas[sort_idx]
    if len(efrf_sorted) > 0:
        cummax_efrf = np.maximum.accumulate(efrf_sorted)
    else:
        cummax_efrf = efrf_sorted

    # Generate Light Feasible Points
    model = st.session_state.get('_trained_model')
    scaler = st.session_state.get('_trained_scaler')
    feat_api, feat_efrf = generate_feasible_samples(model, scaler)

    fig = go.Figure()
    # Light Points for the feasible region
    if len(feat_api) > 0:
        fig.add_trace(go.Scatter(x=feat_api, y=feat_efrf, mode='markers', name='Feasible Region (Points)', marker=dict(size=3, color='lightblue', opacity=0.4)))
    
    # Transparent Green Area
    fig.add_hrect(y0=0, y1=0.40, x0=API_MIN, x1=API_MAX, fillcolor='rgba(144, 238, 144, 0.1)', line_width=0, layer='below')
    
    # Pareto Front (stops at the boundary naturally via cummax)
    fig.add_trace(go.Scatter(x=api_sorted, y=cummax_efrf, mode='lines+markers', name='Pareto Front', line=dict(color='red', width=2), marker=dict(size=8, color='#a3c4f3', line=dict(width=1, color='#4a6fa5'))))
    
    if golden:
        fig.add_trace(go.Scatter(x=[golden['API (%)']], y=[golden['EFRF']], mode='markers', name='🏆 Golden Solution', marker=dict(size=22, color='gold', symbol='star', line=dict(width=1.5, color='#8a6d00'))))
    
    tested = st.session_state.get('results')
    if tested and 'api' in tested:
        fig.add_trace(go.Scatter(x=[tested['api']], y=[tested['efrf']], mode='markers', name='🔵 Tested', marker=dict(size=14, color='blue', symbol='circle', line=dict(width=1.5, color='white'))))
    
    fig.add_hline(y=0.40, line_dash='dash', line_color='gray', annotation_text='EFRF limit (0.40)', annotation_position='top left')
    fig.add_vline(x=API_MIN, line_dash='dash', line_color='gray', annotation_text=f'API min ({API_MIN}%)', annotation_position='bottom left')
    fig.add_vline(x=API_MAX, line_dash='dash', line_color='gray', annotation_text=f'API max ({API_MAX}%)', annotation_position='bottom right')
    fig.update_layout(title=f'Pareto Front - Generation {gen_slider}', xaxis_title='API (%)', yaxis_title='EFRF', height=500, template='plotly_white', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

def render_golden_solution(golden):
    if not golden: return
    st.markdown("---"); st.markdown("## 🏆 Golden Solution (Balanced Trade-off)")
    st.json(golden)

# ================================================================
# UI INPUT FUNCTIONS
# ================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🧬 Hybrid AI Framework")
        st.markdown("---"); st.markdown(f"**Version:** v32.2-Pro-Full")
        st.markdown(f"**Institution:** Nile Valley University")
        st.markdown("---")
        uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                required_cols = ['API','Binder','PVPP','MgSt','MCC','Moisture','Pressure','Speed','Density','Tensile','EFRF','Disintegration','Dissolution']
                missing = [c for c in required_cols if c not in df.columns]
                if missing: st.error(f"Missing columns: {missing}")
                elif len(df) < 20: st.error(f"Only {len(df)} rows found — at least 20 needed.")
                else:
                    st.session_state.user_data = df
                    st.success(f"✅ Loaded {len(df)} samples")
                    st.session_state.force_retrain = True
            except Exception as e: st.error(f"Error reading file: {e}")
        else:
            st.info("🟢 Using synthetic data (fallback)")

        if st.button("🔄 Force Retrain", use_container_width=True):
            import glob
            for cp in [CHECKPOINT_SYNTHETIC] + glob.glob(os.path.join(tempfile.gettempdir(), 'co_hybai_real_*.pt')):
                if os.path.exists(cp):
                    try: os.remove(cp)
                    except: pass
            train_model.clear()
            for key in ('_trained_model', '_trained_scaler', '_trained_history'): st.session_state.pop(key, None)
            st.session_state.optimization_complete = False
            st.session_state.force_retrain = False
            st.success("Cache cleared.")
        st.markdown("---"); st.caption("© 2024 Nile Valley University · Sudan")

def render_input_panel():
    st.markdown("## 🧪 Formulation Parameters")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.api = st.slider("API Content (%)", API_MIN, API_MAX, st.session_state.api, step=0.5)
        st.session_state.binder = st.slider("Binder (%)", BINDER_MIN, BINDER_MAX, st.session_state.binder, step=0.1)
        st.session_state.pvpp = st.slider("PVPP (%)", PVPP_MIN, PVPP_MAX, st.session_state.pvpp, step=0.1)
        st.session_state.mgst = st.slider("MgSt (%)", MGST_MIN, MGST_MAX, st.session_state.mgst, step=0.05)
    with col2:
        st.session_state.mcc = st.slider("MCC (%)", MCC_MIN, MCC_MAX, st.session_state.mcc, step=0.1)
        st.session_state.moisture = st.slider("Moisture (%)", MOISTURE_MIN, MOISTURE_MAX, st.session_state.moisture, step=0.1)
        st.session_state.pressure = st.slider("Compression Pressure (MPa)", PRESSURE_MIN, PRESSURE_MAX, st.session_state.pressure, step=2.0)
        st.session_state.speed = st.slider("Tableting Speed (rpm)", SPEED_MIN, SPEED_MAX, st.session_state.speed, step=0.5)

# ================================================================
# MAIN APP LOGIC
# ================================================================
def get_model_and_scaler():
    real_df = st.session_state.get('user_data')
    use_real = real_df is not None and len(real_df) > 0
    fingerprint = _data_fingerprint(real_df) if use_real else None
    return train_model(use_real=use_real, _real_df=real_df, data_fingerprint=fingerprint)

def run_real_optimization(progress_callback=None):
    model, scaler, history = get_model_and_scaler()
    optimizer = NSGAIIOptimizer(model, scaler, pop_size=POPULATION_SIZE, generations=NSGA_GENERATIONS, y_train_mean=history.get('y_train_mean'))
    gen_history = []
    final_pop, final_obj = None, None
    for pop, obj, history, gen in optimizer.optimize(n_vars=8):
        final_pop, final_obj = pop, obj
        if history: gen_history = history
        if progress_callback is not None: progress_callback(gen, NSGA_GENERATIONS)
    fronts = optimizer.fast_non_dominated_sort(final_obj)
    pareto_idx = fronts[0]
    pareto_pop = final_pop[pareto_idx]
    pareto_obj = final_obj[pareto_idx]
    preds = model.predict(scaler.transform(pareto_pop))
    solutions = []
    for i, (row, pred) in enumerate(zip(pareto_pop, preds)):
        efrf = float(pareto_obj[i, 2]) # Use shrunk value to ensure Golden matches curve perfectly
        if efrf < 0.40:
            quality = calculate_quality_score(pred[0], pred[1], efrf, api=row[0])
            solutions.append({'Solution': f'S{i+1}', 'API (%)': row[0], 'EFRF': efrf, 'Density': pred[0], 'Tensile (MPa)': pred[1], 'Quality Score': quality['overall']})
    solutions.sort(key=lambda x: x['Quality Score'], reverse=True)
    if not solutions: return [], None, []
    return solutions, solutions[0], gen_history

def main():
    render_sidebar()
    st.markdown("# 🧬 Hybrid AI · v32.2 Pro-Full (PINN+Uncertainty+3D+Radar)")
    render_input_panel()
    run_button = st.button("🚀 Run Hybrid Optimization", type="primary", use_container_width=True)

    if run_button:
        start_time = time.time()
        # Triggering training & progress display
        with st.spinner("Training Physics-Informed Model (PINN)..."):
            model, scaler, history = get_model_and_scaler()
        opt_progress = st.progress(0, text="Running NSGA-II generation 0/%d..." % NSGA_GENERATIONS)
        def _update_opt_progress(gen, total):
            opt_progress.progress((gen + 1) / total, text=f"Running NSGA-II generation {min(gen + 1, total)}/{total}...")
        solutions, golden, gen_history = run_real_optimization(progress_callback=_update_opt_progress)
        opt_progress.empty()
        
        # Update Runtime and Session
        st.session_state.runtime = round(time.time() - start_time, 1)
        st.session_state.optimization_complete = True
        st.session_state.golden_solution = golden
        st.session_state.best_solutions = solutions
        st.session_state.pareto_history = gen_history
        
        # Compute Tested Formulation (Current Sliders) with Uncertainty
        n = normalize_formulation(st.session_state.api, st.session_state.binder, st.session_state.pvpp, st.session_state.mgst, st.session_state.mcc, st.session_state.moisture)
        row = np.array([[n['api'], n['binder'], n['pvpp'], n['mgst'], n['mcc'], n['moisture'], st.session_state.pressure, st.session_state.speed]], dtype=np.float32)
        preds, unc = model.predict_with_uncertainty(torch.tensor(scaler.transform(row), dtype=torch.float32))
        tested_results = {'api': float(n['api']), 'density': float(preds[0][0]), 'tensile': float(preds[0][1]), 'efrf': float(preds[0][2]), 'disintegration': float(preds[0][3]), 'dissolution': float(preds[0][4])}
        st.session_state.results = tested_results

        render_results_summary(tested_results)
        render_pareto_evolution()
        render_golden_solution(golden)
        
        # NEW: 3D and Radar
        st.subheader("🌐 3D Pareto Front & Radar Comparison")
        if solutions:
            final_pop = np.array([list(sol.values())[0:2] for sol in solutions]) # Dummy, just to pass
            # Actually, we need final_pop and final_obj. We can build it from history or pass through.
            # For UI consistency, we use the currently stored best solutions
            col_a, col_b = st.columns([1, 1])
            with col_a:
                render_3d_pareto(np.array([s['API (%)'] for s in solutions]), np.array([s['EFRF'] for s in solutions]), 0)
            with col_b:
                df = pd.DataFrame(solutions)
                selected = st.multiselect("Select for Radar", df['Solution'], default=df['Solution'][:2].tolist())
                render_dynamic_radar(df, selected)
        
        st.success(f"⏱️ Optimization completed in {st.session_state.runtime} seconds!")
        st.balloons()
        st.rerun() # Fix Sidebar Runtime update

if __name__ == "__main__":
    main()
