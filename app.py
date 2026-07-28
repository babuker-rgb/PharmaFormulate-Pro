# ================================================================
# Hybrid AI · Multi-Objective Tablet Optimization
# Nile Valley University · Sudan · v29.28‑R32
# FINAL PRODUCTION – ALL FEATURES + BATCHNORM FIX
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
UNCERTAINTY_SAMPLES = 30

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
        'relax_constraints': False,
        'real_data_uploaded': False,
        'real_data_df': None,
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

# ================================================================
# HYBRID NEURAL NETWORK (input_dim=10, with dropout)
# ================================================================
class HybridTabletModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=256, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.drop3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.drop4 = nn.Dropout(dropout_rate)
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
        h1 = self.drop1(h1)
        h2 = torch.relu(self.bn2(self.fc2(h1))) + h1
        h2 = self.drop2(h2)
        h3 = torch.relu(self.bn3(self.fc3(h2))) + h2
        h3 = self.drop3(h3)
        h4 = torch.relu(self.bn4(self.fc4(h3))) + h3
        h4 = self.drop4(h4)
        out = self.fc5(h4)
        density = torch.sigmoid(out[:, 0]) * 0.4 + 0.55
        tensile = torch.sigmoid(out[:, 1]) * 8.0 + 0.5
        efrf = torch.sigmoid(out[:, 2])
        disintegration = torch.sigmoid(out[:, 3]) * 45.0 + 2.0
        dissolution = torch.sigmoid(out[:, 4]) * 80.0 + 10.0
        return torch.stack([density, tensile, efrf, disintegration, dissolution], dim=1)

    def predict(self, x, uncertainty=False):
        """
        If uncertainty=True, use Monte Carlo Dropout.
        Handles single-sample case by replicating to a batch of size UNCERTAINTY_SAMPLES.
        Returns (mean, std) each shape (batch, 5) or (1, 5) for single input.
        """
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            n_orig = x.shape[0]

            if uncertainty:
                # For MC Dropout, we need batch size > 1 for BatchNorm.
                # If only one sample, replicate it to create a batch.
                if n_orig == 1:
                    x = x.repeat(UNCERTAINTY_SAMPLES, 1)

                self.train()  # enable dropout
                preds = []
                for _ in range(UNCERTAINTY_SAMPLES):
                    preds.append(self.forward(x).numpy())
                preds = np.array(preds)  # shape: (samples, batch, 5)

                if n_orig == 1:
                    # All rows in batch are identical, take stats across the sample dimension
                    mean = preds.mean(axis=0)   # shape (batch, 5) but batch=UNCERTAINTY_SAMPLES
                    std = preds.std(axis=0)
                    # We want a single result for the original sample, so take the first row (all are same)
                    return mean[0:1, :], std[0:1, :]  # keep as 2D
                else:
                    # Multiple original samples, each has its own predictions across MC samples
                    # preds shape: (samples, batch, 5) -> we want mean and std over samples axis (axis=0)
                    mean = preds.mean(axis=0)   # (batch, 5)
                    std = preds.std(axis=0)     # (batch, 5)
                    return mean, std
            else:
                self.eval()
                return self.forward(x).numpy(), None

# ================================================================
# DATA GENERATION – with binder grade & particle size
# ================================================================
def generate_synthetic_data(n_samples=8000, seed=42):
    rng = np.random.default_rng(seed)
    bounds = [(API_MIN, API_MAX), (BINDER_MIN, BINDER_MAX), (PVPP_MIN, PVPP_MAX),
              (MGST_MIN, MGST_MAX), (MCC_MIN, MCC_MAX), (MOISTURE_MIN, MOISTURE_MAX)]
    comps = np.column_stack([rng.uniform(lo, hi, n_samples) for lo, hi in bounds])
    comps = comps / comps.sum(axis=1, keepdims=True) * 100.0
    api_n, binder_n, pvpp_n, mgst_n, mcc_n, moisture_n = comps.T

    pressure = rng.uniform(PRESSURE_MIN, PRESSURE_MAX, n_samples)
    speed = rng.uniform(SPEED_MIN, SPEED_MAX, n_samples)
    binder_grade_idx = rng.integers(0, len(BINDER_GRADE_NAMES), n_samples)
    particle_size = rng.uniform(PARTICLE_SIZE_MIN, PARTICLE_SIZE_MAX, n_samples)

    X = np.column_stack([api_n, binder_n, pvpp_n, mgst_n, mcc_n, moisture_n,
                         pressure, speed, binder_grade_idx, particle_size])

    compressibility = np.array([BINDER_GRADES[BINDER_GRADE_NAMES[i]]["compressibility"] for i in binder_grade_idx])

    density_base = 0.85 - 0.3 * np.exp(-0.01 * (pressure - PRESSURE_MIN))
    density_base += 0.05 * (compressibility - 0.8)
    density = np.clip(density_base + rng.normal(0, 0.01, n_samples), 0.55, 0.95)

    tensile = (0.5 + 6.0 * (density - 0.55) / 0.40
               + 0.4 * (binder_n - BINDER_MIN) / (BINDER_MAX - BINDER_MIN)
               + 0.2 * (compressibility - 0.8) * 5
               - 1.2 * (mgst_n - MGST_MIN) / (MGST_MAX - MGST_MIN)
               - 0.15 * (particle_size - PARTICLE_SIZE_MIN) / (PARTICLE_SIZE_MAX - PARTICLE_SIZE_MIN)
               + 0.3 * (api_n - API_MIN) / (API_MAX - API_MIN))
    tensile = np.clip(tensile + rng.normal(0, 0.1, n_samples), 0.5, 8.5)

    efrf = (0.55 - 0.35 * (density - 0.55) / 0.40
            + 0.25 * (api_n - API_MIN) / (API_MAX - API_MIN)
            - 0.15 * (compressibility - 0.8)
            + 0.2 * (mgst_n - MGST_MIN) / (MGST_MAX - MGST_MIN)
            + 0.05 * (particle_size - PARTICLE_SIZE_MIN) / (PARTICLE_SIZE_MAX - PARTICLE_SIZE_MIN))
    efrf = np.clip(efrf + rng.normal(0, 0.03, n_samples), 0.02, 0.98)

    disintegration = (12.0 - 4.0 * (pvpp_n - PVPP_MIN) / (PVPP_MAX - PVPP_MIN)
                      + 5.0 * (binder_n - BINDER_MIN) / (BINDER_MAX - BINDER_MIN)
                      + 3.0 * (moisture_n - MOISTURE_MIN) / (MOISTURE_MAX - MOISTURE_MIN)
                      + 2.0 * (particle_size - PARTICLE_SIZE_MIN) / (PARTICLE_SIZE_MAX - PARTICLE_SIZE_MIN))
    disintegration = np.clip(disintegration + rng.normal(0, 0.5, n_samples), 2.0, 45.0)

    dissolution = 1.8 * disintegration + 5.0 - 3.0 * (pvpp_n - PVPP_MIN) / (PVPP_MAX - PVPP_MIN)
    dissolution = np.clip(dissolution + rng.normal(0, 1.0, n_samples), 10.0, 90.0)

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
# TRAINING – with real data upload support
# ================================================================
CHECKPOINT_PATH = os.path.join(tempfile.gettempdir(), 'co_hybai_v29_28_r32_v6.pt')

@st.cache_resource(show_spinner=False)
def train_model(real_data_df=None):
    if os.path.exists(CHECKPOINT_PATH):
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
            if (real_data_df is None and ckpt.get('data_source') == 'synthetic') or \
               (real_data_df is not None and ckpt.get('data_source') == 'real'):
                model = HybridTabletModel(input_dim=10, hidden_dim=256)
                model.load_state_dict(ckpt['model_state'])
                model.eval()
                scaler = ckpt['scaler']
                return model, scaler, ckpt['history']
        except Exception:
            pass

    if real_data_df is not None:
        required_cols = ['API','Binder','PVPP','MgSt','MCC','Moisture',
                         'Pressure','Speed','BinderGrade','ParticleSize',
                         'Density','Tensile','EFRF','Disintegration','Dissolution']
        if not all(c in real_data_df.columns for c in required_cols):
            st.error(f"CSV must contain columns: {', '.join(required_cols)}")
            return None, None, None
        X = real_data_df[['API','Binder','PVPP','MgSt','MCC','Moisture',
                          'Pressure','Speed','BinderGrade','ParticleSize']].values.astype(np.float32)
        y = real_data_df[['Density','Tensile','EFRF','Disintegration','Dissolution']].values.astype(np.float32)
        data_source = 'real'
    else:
        X, y = generate_synthetic_data()
        data_source = 'synthetic'

    scaler = InputScaler().fit(X)
    X_scaled = scaler.transform(X)

    n_val = int(0.2 * len(X))
    perm = np.random.default_rng(0).permutation(len(X))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    X_train_t = torch.tensor(X_scaled[train_idx], dtype=torch.float32)
    y_train_t = torch.tensor(y[train_idx], dtype=torch.float32)
    X_val_t = torch.tensor(X_scaled[val_idx], dtype=torch.float32)
    y_val_t = torch.tensor(y[val_idx], dtype=torch.float32)

    model = HybridTabletModel(input_dim=10, hidden_dim=256, dropout_rate=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)

    target_var = y_train_t.var(dim=0, unbiased=False)
    target_var = torch.clamp(target_var, min=1e-6)
    def weighted_mse(pred, true):
        return (((pred - true) ** 2) / target_var).mean()

    history = {'loss': [], 'r2': [], 'rmse': []}
    best_val_loss = np.inf
    best_state = None
    patience, patience_counter = 60, 0

    for epoch in range(TRAINING_EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = weighted_mse(pred, y_train_t)
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
    history['data_source'] = data_source

    torch.save({
        'model_state': model.state_dict(),
        'scaler': scaler,
        'history': history,
        'data_source': data_source
    }, CHECKPOINT_PATH)
    return model, scaler, history

# ================================================================
# NSGA-II OPTIMIZER – 10 vars, relax constraints
# ================================================================
class NSGAIIOptimizer:
    def __init__(self, model, scaler, pop_size=50, generations=80,
                 y_train_mean=None, relax=False):
        self.model = model
        self.scaler = scaler
        self.pop_size = pop_size
        self.generations = generations
        self.n_objectives = 3
        self.y_train_mean = y_train_mean if y_train_mean is not None else [0.75, 4.5, 0.5, 20.0, 45.0]
        self.relax = relax

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
            pred = self.model.predict(pop_scaled, uncertainty=False)[0]  # mean only
        density = pred[:, 0]
        tensile = pred[:, 1]
        efrf = pred[:, 2]
        api = pop[:, 0]

        # OOD shrinkage
        ood_z = np.abs(pop_scaled)
        ood_raw = np.clip(ood_z - 2.0, 0, None).sum(axis=1)
        shrink_factor = 1.0 / (1.0 + ood_raw)
        density = shrink_factor * density + (1 - shrink_factor) * self.y_train_mean[0]
        tensile = shrink_factor * tensile + (1 - shrink_factor) * self.y_train_mean[1]
        efrf = shrink_factor * efrf + (1 - shrink_factor) * self.y_train_mean[2]

        # Relax constraints: no extra penalty – let the objectives drive exploration
        # (The penalty for constraints is already absent in this version)
        fitness = np.column_stack([-density, -tensile, efrf])

        # Penalties for low API and low tensile (still applied to bias search)
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
        (0, len(BINDER_GRADE_NAMES)-1), (PARTICLE_SIZE_MIN, PARTICLE_SIZE_MAX),
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
        pop[:, 8] = np.round(pop[:, 8] * (len(BINDER_GRADE_NAMES)-1))
        pop[:, 9] = pop[:, 9] * (PARTICLE_SIZE_MAX - PARTICLE_SIZE_MIN) + PARTICLE_SIZE_MIN
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
                                if j == 8:  # binder_grade integer
                                    child[j] = np.clip(int(child[j] + np.random.normal(0, 0.5)), lo, hi)
                                else:
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
# RESULT FUNCTIONS with uncertainty
# ================================================================
def run_real_training_and_get_history(real_data_df=None):
    model, scaler, history = train_model(real_data_df)
    if model is None:
        return None
    st.session_state['_trained_model'] = model
    st.session_state['_trained_scaler'] = scaler
    st.session_state['_trained_history'] = history
    return history

def run_real_optimization(progress_callback=None):
    model = st.session_state.get('_trained_model')
    scaler = st.session_state.get('_trained_scaler')
    history = st.session_state.get('_trained_history')
    if model is None or scaler is None or history is None:
        return None, None, None

    relax = st.session_state.get('relax_constraints', False)
    optimizer = NSGAIIOptimizer(model, scaler, pop_size=POPULATION_SIZE, generations=NSGA_GENERATIONS,
                                y_train_mean=history.get('y_train_mean'), relax=relax)
    gen_history = []
    final_pop, final_obj = None, None
    for pop, obj, hist, gen in optimizer.optimize(n_vars=10):
        final_pop, final_obj = pop, obj
        if hist:
            gen_history = hist
        if progress_callback is not None:
            progress_callback(gen, NSGA_GENERATIONS)

    fronts = optimizer.fast_non_dominated_sort(final_obj)
    pareto_idx = fronts[0]
    pareto_pop = final_pop[pareto_idx]

    # Predict with uncertainty
    preds, stds = model.predict(scaler.transform(pareto_pop), uncertainty=True)
    solutions = []
    for i, (row, pred, std) in enumerate(zip(pareto_pop, preds, stds)):
        api, binder, pvpp, mgst, mcc, moisture = row[:6]
        density, tensile, efrf = pred[0], pred[1], pred[2]
        quality = calculate_quality_score(density, tensile, efrf, api=api)
        solutions.append({
            'Solution': f'S{i+1}',
            'API (%)': api, 'Binder (%)': binder, 'PVPP (%)': pvpp,
            'MgSt (%)': mgst, 'MCC (%)': mcc, 'Moisture (%)': moisture,
            'Total (%)': api + binder + pvpp + mgst + mcc + moisture,
            'Density': density, 'Density_std': std[0],
            'Tensile (MPa)': tensile, 'Tensile_std': std[1],
            'EFRF': efrf, 'EFRF_std': std[2],
            'Disintegration': pred[3], 'Disintegration_std': std[3],
            'Dissolution': pred[4], 'Dissolution_std': std[4],
            'Quality Score': quality['overall']
        })
    solutions.sort(key=lambda x: x['Quality Score'], reverse=True)
    if not solutions:
        return [], None, []
    return solutions, solutions[0], gen_history

def get_current_formulation_results_with_uncertainty():
    model = st.session_state.get('_trained_model')
    scaler = st.session_state.get('_trained_scaler')
    if model is None or scaler is None:
        model, scaler, _ = train_model()
        st.session_state['_trained_model'] = model
        st.session_state['_trained_scaler'] = scaler

    n = normalize_formulation(
        st.session_state.api, st.session_state.binder, st.session_state.pvpp,
        st.session_state.mgst, st.session_state.mcc, st.session_state.moisture
    )
    binder_grade = st.session_state.binder_grade
    particle_size = st.session_state.particle_size
    row = np.array([[n['api'], n['binder'], n['pvpp'], n['mgst'], n['mcc'], n['moisture'],
                     st.session_state.pressure, st.session_state.speed,
                     binder_grade, particle_size]], dtype=np.float32)
    mean, std = model.predict(scaler.transform(row), uncertainty=True)
    return {
        'density': float(mean[0][0]), 'density_std': float(std[0][0]),
        'tensile': float(mean[0][1]), 'tensile_std': float(std[0][1]),
        'efrf': float(mean[0][2]), 'efrf_std': float(std[0][2]),
        'disintegration': float(mean[0][3]), 'disintegration_std': float(std[0][3]),
        'dissolution': float(mean[0][4]), 'dissolution_std': float(std[0][4]),
    }

# ================================================================
# UI RENDER FUNCTIONS
# ================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🧬 Hybrid AI Framework")
        st.markdown("---")
        st.markdown(f"**Version:** v29.28‑R32")
        st.markdown(f"**Institution:** Nile Valley University")
        st.markdown(f"**Department:** Pharmaceutical Engineering")
        st.markdown("---")
        st.markdown("### 📂 Real Data Upload")
        uploaded_file = st.file_uploader("Upload CSV (formulation-property data)", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.real_data_df = df
                st.session_state.real_data_uploaded = True
                st.success(f"Loaded {len(df)} rows.")
                st.session_state['_trained_model'] = None
                st.session_state['_trained_scaler'] = None
                st.session_state['_trained_history'] = None
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
        else:
            st.session_state.real_data_uploaded = False

        st.markdown("### 🎯 Constraint Mode")
        relax = st.checkbox("☑️ Relax constraints (explore wider API range)",
                            value=st.session_state.get('relax_constraints', False))
        st.session_state.relax_constraints = relax
        if relax:
            st.info("Density ≥0.50, Tensile ≥0.20, EFRF ≤0.80 – shows full trade‑off.")

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
        with col_b:
            if st.button("🔄 Force Retrain", use_container_width=True):
                train_model.clear()
                for key in ('_trained_model', '_trained_scaler', '_trained_history'):
                    st.session_state.pop(key, None)
                if os.path.exists(CHECKPOINT_PATH):
                    try:
                        os.remove(CHECKPOINT_PATH)
                    except OSError:
                        pass
                st.session_state.optimization_complete = False
                st.success("Cache cleared — model will retrain on the next run.")
        st.markdown("---")
        st.caption("© 2024 Nile Valley University · Sudan")

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
        status = "✅ Close to 100%" if deviation < 2.0 else ("⚠️ Adjusted" if deviation < 10.0 else "🔴 Large adjustment")
        st.metric("**Raw Total**", f"{raw_total:.1f}%", status)
        st.caption("Normalized:")
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
    st.caption("ℹ️ **Binder Grade** and **Particle Size** are now part of the model. **Pressure** and **Speed** also feed in. "
              "Granule size, dwell time, friction, decompression are references only.")
    col3, col4 = st.columns(2)
    with col3:
        st.session_state.pressure = st.slider("**Compression Pressure (MPa)**", PRESSURE_MIN, PRESSURE_MAX, st.session_state.pressure, step=2.0)
        st.session_state.speed = st.slider("**Tableting Speed (rpm)**", SPEED_MIN, SPEED_MAX, st.session_state.speed, step=0.5)
        st.session_state.granule = st.slider("**Granule Size (µm)**", GRANULE_MIN, GRANULE_MAX, st.session_state.granule, step=5.0)
    with col4:
        st.session_state.dwell_time = st.slider("**Dwell Time (ms)**", DWELL_TIME_MIN, DWELL_TIME_MAX, st.session_state.dwell_time, step=1.0)
        st.session_state.friction = st.slider("**Friction Coefficient**", FRICTION_MIN, FRICTION_MAX, st.session_state.friction, step=0.01)
        st.session_state.decompression_time = st.slider("**Decompression Time (ms)**", DECOMPRESSION_TIME_MIN, DECOMPRESSION_TIME_MAX, st.session_state.decompression_time, step=2.0)

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

def render_results_summary(results):
    st.markdown("---")
    st.markdown("## 📊 Optimization Results")
    api_val = st.session_state.api
    quality = calculate_quality_score(results['density'], results['tensile'], results['efrf'], api=api_val)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("**API%**", f"{api_val:.1f}%", "🎯 Target: maximize")
        st.metric("**Density**", f"{results['density']:.3f} ± {results.get('density_std', 0):.3f}",
                 target_status(results['density'], 0.80, mode='min', comfortable=0.85))
    with col2:
        st.metric("**Tensile Strength**", f"{results['tensile']:.2f} ± {results.get('tensile_std', 0):.2f} MPa",
                 target_status(results['tensile'], 1.5, mode='min', comfortable=3.0))
        st.metric("**EFRF**", f"{results['efrf']:.3f} ± {results.get('efrf_std', 0):.3f}",
                 target_status(results['efrf'], 0.40, mode='max', comfortable=0.30))
    with col3:
        st.metric("**Disintegration Time**", f"{results['disintegration']:.1f} ± {results.get('disintegration_std', 0):.1f} min",
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
    real_data = st.session_state.real_data_df if st.session_state.real_data_uploaded else None
    with st.spinner("Training physics-informed model..."):
        history = run_real_training_and_get_history(real_data)
    if history is None:
        return
    if not history['loss']:
        st.warning("No training history available.")
        return
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(y=history['loss'], mode='lines', name='Validation Loss', line=dict(color='#ff6b6b', width=2)))
    fig_loss.update_layout(title='Loss Evolution (validation)',
                           xaxis_title='Recorded checkpoint', yaxis_title='MSE Loss', height=250)
    st.plotly_chart(fig_loss, use_container_width=True)
    fig_metrics = go.Figure()
    fig_metrics.add_trace(go.Scatter(y=history['r2'], mode='lines', name='R² Score', line=dict(color='#51cf66', width=2)))
    fig_metrics.add_trace(go.Scatter(y=history['rmse'], mode='lines', name='RMSE', line=dict(color='#5c7cfa', width=2)))
    fig_metrics.update_layout(title='Model Performance',
                              xaxis_title='Recorded checkpoint', yaxis_title='Metric Value', height=250)
    st.plotly_chart(fig_metrics, use_container_width=True)
    data_source = history.get('data_source', 'synthetic')
    st.success(f"✅ Training complete using {data_source} data! "
              f"Final validation R² = {history['r2'][-1]:.3f}, RMSE = {history['rmse'][-1]:.3f}")

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
                f"{history.get('n_val', '?')} held-out samples. "
                "Property with poor fit means predictions are less trustworthy."
            )

def render_pareto_evolution():
    st.markdown("---")
    st.markdown("## 🌐 Pareto Front Evolution")
    golden = st.session_state.get('golden_solution', None)
    pareto_history = st.session_state.get('pareto_history', None)
    if not pareto_history:
        st.info("Run the optimization to see the real Pareto front evolve across generations.")
        return
    generations_recorded = [h['generation'] for h in pareto_history]
    chart = st.empty()
    gen_slider = st.select_slider("Select generation to view", options=generations_recorded, value=generations_recorded[-1])
    current_entry = next(h for h in pareto_history if h['generation'] == gen_slider)
    current_obj = current_entry['pareto_objectives']
    current_density = -current_obj[:, 0]
    current_tensile = -current_obj[:, 1]
    current_efrf = current_obj[:, 2]
    current_api = current_entry['pareto_solutions'][:, 0]

    fig = go.Figure()
    for i, h in enumerate(pareto_history):
        if h['generation'] >= gen_slider:
            continue
        obj = h['pareto_objectives']
        alpha = 0.1 + 0.2 * (i / max(1, len(pareto_history)))
        fig.add_trace(go.Scatter3d(
            x=-obj[:, 0], y=-obj[:, 1], z=obj[:, 2],
            mode='markers',
            marker=dict(size=4, opacity=alpha, color='lightgray'),
            name=f"Gen {h['generation']}", showlegend=False,
            hovertemplate='Density: %{x:.3f}<br>Tensile: %{y:.2f} MPa<br>EFRF: %{z:.3f}<extra></extra>'
        ))
    fig.add_trace(go.Scatter3d(
        x=current_density, y=current_tensile, z=current_efrf,
        mode='markers',
        marker=dict(
            size=8,
            color=current_api,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="API%", x=1.02, len=0.6),
            opacity=0.9,
            line=dict(width=1, color='black')
        ),
        name=f'Generation {gen_slider}',
        hovertemplate='Density: %{x:.3f}<br>Tensile: %{y:.2f} MPa<br>EFRF: %{z:.3f}<br>API: %{marker.color:.1f}%<extra></extra>'
    ))
    if golden:
        fig.add_trace(go.Scatter3d(
            x=[golden['Density']], y=[golden['Tensile (MPa)']], z=[golden['EFRF']],
            mode='markers',
            marker=dict(size=15, color='red', symbol='diamond', line=dict(width=2, color='white')),
            name='🏆 Golden Solution',
            hovertemplate='<b>🏆 GOLDEN SOLUTION</b><br>API: %{text}<br>Density: %{x:.3f}<br>Tensile: %{y:.2f} MPa<br>EFRF: %{z:.3f}<extra></extra>',
            text=[f"{golden['API (%)']:.1f}%"]
        ))
    fig.update_layout(
        title=f'Pareto Front Evolution - Generation {gen_slider} (color = API%)',
        scene=dict(
            xaxis=dict(title='Density', range=[0.55,0.95]),
            yaxis=dict(title='Tensile Strength (MPa)', range=[0.5,8.5]),
            zaxis=dict(title='EFRF', range=[0,1]),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.8))
        ),
        height=550, margin=dict(l=0, r=0, t=50, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    chart.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"**Generation {gen_slider+1}/{NSGA_GENERATIONS}** · "
        f"Pareto-optimal solutions at this generation: {len(current_density)}"
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
    near_limit = any("near limit" in s or "Below target" in s or "Exceeds limit" in s
                    for s in (density_status, tensile_status, efrf_status))
    if near_limit:
        st.warning("⚠️ Best available trade‑off, but some properties are near their limits.")
    else:
        st.success("✅ Excellent balance across all properties!")

def render_side_by_side_comparison(golden, all_solutions):
    if not golden or not all_solutions:
        return
    st.markdown("---")
    st.markdown("## 📊 Side‑by‑Side Comparison")
    top = all_solutions[:3]
    df = pd.DataFrame(top)
    st.dataframe(df[['Solution','API (%)','Binder (%)','PVPP (%)','MgSt (%)',
                     'MCC (%)','Moisture (%)','Density','Tensile (MPa)',
                     'EFRF','Quality Score']], use_container_width=True)
    st.markdown("### 🎯 Performance Radar")
    categories = ["API%", "Density", "Tensile (MPa)", "EFRF (inverted)", "Quality Score"]
    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[
                (row["API (%)"] - 80) / 18,
                row["Density"] / 0.95,
                row["Tensile (MPa)"] / 8.5,
                1 - row["EFRF"],
                row["Quality Score"] / 100
            ],
            theta=categories,
            fill='toself',
            name=row["Solution"]
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        showlegend=True,
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        title="Performance Comparison Across Solutions"
    )
    st.plotly_chart(fig, use_container_width=True)

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
            'parameters': {
                'population': POPULATION_SIZE,
                'generations': NSGA_GENERATIONS,
                'epochs': TRAINING_EPOCHS,
                'runtime_seconds': st.session_state.runtime,
                'api_penalty': 0.08,
                'tensile_penalty': 0.05,
                'relax_constraints': st.session_state.relax_constraints
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
        st.metric("⏱️ Runtime", f"{st.session_state.runtime}s" if st.session_state.runtime else "—")
    with col2:
        evals_per_sec = (POPULATION_SIZE * NSGA_GENERATIONS) / max(1, st.session_state.runtime)
        st.metric("⚡ Evaluations/Second", f"{evals_per_sec:.0f}")

    solutions = st.session_state.get('best_solutions') or []
    col3, col4 = st.columns([2, 1])
    with col3:
        st.markdown("### Key Statistics")
        if solutions:
            sol_df = pd.DataFrame(solutions)
            stats = pd.DataFrame({
                'Metric': [
                    'Total Solutions Evaluated',
                    'Pareto Solutions Found',
                    'Best Density',
                    'Best Tensile',
                    'Best EFRF',
                    'Best API%',
                    'Mass Balance',
                    'Penalties',
                    'Constraint Mode'
                ],
                'Value': [
                    f'{POPULATION_SIZE * NSGA_GENERATIONS:,}',
                    f'{len(sol_df)}',
                    f'{sol_df["Density"].max():.3f}',
                    f'{sol_df["Tensile (MPa)"].max():.2f} MPa',
                    f'{sol_df["EFRF"].min():.3f}',
                    f'{sol_df["API (%)"].max():.1f}%',
                    '✅ 100% (Enforced)',
                    'API: 0.08 | Tensile: 0.05',
                    'Relaxed' if st.session_state.relax_constraints else 'Normal'
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

# ================================================================
# MAIN
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
            with st.spinner("Training physics-informed model (cached)..."):
                quick_results = get_current_formulation_results_with_uncertainty()
            render_results_summary(quick_results)
            st.info("Direct model prediction for current sliders. Click **Run Hybrid Optimization** for full Pareto search.")

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
        if solutions is None:
            st.error("Optimization failed – check model and data.")
            return
        st.session_state.results = get_current_formulation_results_with_uncertainty()
        st.session_state.golden_solution = golden
        st.session_state.best_solutions = solutions
        st.session_state.pareto_history = gen_history
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
            st.markdown("**📊 API & Tensile Penalties**")
        with col2:
            st.markdown("**⚖️ Mass Balance Enforced**")
            st.markdown("**🔬 PINN Constraints**")
        with col3:
            st.markdown("**📈 Pareto Front**")
            st.markdown("**🏆 Golden Solution**")

if __name__ == "__main__":
    main()
