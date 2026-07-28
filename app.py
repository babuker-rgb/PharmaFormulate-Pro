# ================================================================
# Hybrid AI · Multi-Objective Tablet Optimization
# Nile Valley University · Sudan · v29.28‑R32
# FINAL VERSION – IMPROVED API% & TENSILE (DUAL PENALTY)
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
        'runtime': 0, 'pareto_history': None
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
    """Base quality score (without API) – used for pure quality assessment."""
    density_score = min(100, (density / 0.95) * 100)
    tensile_score = min(100, (tensile / 8.5) * 100)
    efrf_score = max(0, (1 - efrf) * 100)
    weights = {'density': 0.4, 'tensile': 0.3, 'efrf': 0.3}
    overall = (density_score * weights['density'] +
               tensile_score * weights['tensile'] +
               efrf_score * weights['efrf'])
    if api is not None:
        api_score = (api - 80) / 18 * 100
        # Blend: 70% quality, 30% API
        overall = 0.7 * overall + 0.3 * api_score
        return {'overall': overall, 'density_score': density_score,
                'tensile_score': tensile_score, 'efrf_score': efrf_score,
                'api_score': api_score, 'weights': {**weights, 'api': 0.3}}
    else:
        return {'overall': overall, 'density_score': density_score,
                'tensile_score': tensile_score, 'efrf_score': efrf_score,
                'weights': weights}

# ================================================================
# HYBRID NEURAL NETWORK (Physics‑Informed)
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
        # BUGFIX: this previously applied torch.sigmoid(x) directly to raw,
        # unnormalized physical-unit inputs (pressure ~150-250, dwell time
        # ~5-50, etc.). sigmoid saturates to ~1.0 for any input above about
        # +10 and to ~0.0 below about -10, so two formulations with very
        # different pressures (e.g. 150 MPa vs 250 MPa) would both map to
        # sigmoid(x) ≈ 1.0 — indistinguishable to every layer after this
        # one. The network was structurally incapable of learning from most
        # of its own inputs. Callers must now pass pre-scaled input (see
        # `scale_inputs` / the StandardScaler fitted in `train_model`)
        # instead of raw physical values.
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
# REAL SYNTHETIC DATASET + INPUT SCALING
# ================================================================
# NOTE ON WHAT FOLLOWS: none of this existed in the original file. The
# HybridTabletModel and NSGAIIOptimizer classes above were fully defined
# but never instantiated anywhere in the app — every number the UI showed
# (training loss/R², density/tensile/EFRF results, the Pareto front) was
# generated by np.random calls in the "SIMULATION FUNCTIONS" section
# further down, dressed up to look like real model output. This section
# adds an actual physics-based synthetic dataset and an actual training
# loop so the model has something real to learn from.
N_SAMPLES = 8000
BOUNDARY_FRACTION = 0.30  # share of samples deliberately drawn near composition-space corners

def _sample_compositions(n_samples, rng, boundary_fraction=0.0):
    """Sample the 6 raw (pre-mass-balance) composition values. With
    boundary_fraction > 0, a share of rows have a RANDOM SUBSET of
    components pinned near their own min or max bound before the uniform
    draw, so mass-balance normalization actually produces corner-region
    compositions (all-excipients-near-minimum, etc.) instead of relying on
    every one of 6 independent uniform draws happening to land near a
    bound simultaneously — which is so unlikely that plain uniform
    sampling essentially never covers those corners (see BUGFIX note in
    generate_synthetic_data)."""
    bounds = [(API_MIN, API_MAX), (BINDER_MIN, BINDER_MAX), (PVPP_MIN, PVPP_MAX),
              (MGST_MIN, MGST_MAX), (MCC_MIN, MCC_MAX), (MOISTURE_MIN, MOISTURE_MAX)]
    cols = [rng.uniform(lo, hi, n_samples) for lo, hi in bounds]
    comps = np.column_stack(cols)

    n_boundary = int(n_samples * boundary_fraction)
    if n_boundary > 0:
        idx = rng.choice(n_samples, n_boundary, replace=False)
        for row in idx:
            n_pinned = rng.integers(2, 6)  # pin 2-5 of the 6 components near a bound
            pinned_dims = rng.choice(6, n_pinned, replace=False)
            for d in pinned_dims:
                lo, hi = bounds[d]
                span = hi - lo
                near_lo = rng.random() < 0.5
                jitter = rng.uniform(0, 0.08) * span
                comps[row, d] = lo + jitter if near_lo else hi - jitter
    return comps

def generate_synthetic_data(n_samples=N_SAMPLES, seed=42):
    """Physics-motivated synthetic dataset for the 8 decision variables
    (API, binder, PVPP, MgSt, MCC, moisture, pressure, speed) -> 5 targets
    (density, tensile, EFRF, disintegration, dissolution)."""
    rng = np.random.default_rng(seed)

    # BUGFIX: independently drawing each of the 6 composition values
    # uniformly, then rescaling to sum to 100%, essentially never produces
    # "corner" formulations where several excipients are simultaneously
    # near their own minimum — verified empirically: across 8000 samples
    # of the old sampling scheme, ZERO rows had all four excipients
    # (binder, PVPP, MgSt, MCC) simultaneously within 10% of their own
    # minimum. That's exactly the region NSGA-II converged to once the
    # loss-scaling fix let it search freely — because the network had
    # never seen anything there, its prediction was unconstrained
    # extrapolation, and it happened to (incorrectly) predict near-zero
    # EFRF, which the optimizer then exploited relentlessly. Mixing in an
    # explicit boundary-augmented sample set closes that blind spot.
    comps = _sample_compositions(n_samples, rng, boundary_fraction=BOUNDARY_FRACTION)
    comps = comps / comps.sum(axis=1, keepdims=True) * 100.0
    api_n, binder_n, pvpp_n, mgst_n, mcc_n, moisture_n = comps.T

    pressure = rng.uniform(PRESSURE_MIN, PRESSURE_MAX, n_samples)
    speed = rng.uniform(SPEED_MIN, SPEED_MAX, n_samples)

    X = np.column_stack([api_n, binder_n, pvpp_n, mgst_n, mcc_n, moisture_n, pressure, speed])

    # Density: Heckel-style pressure/composition relationship
    porosity0 = 0.45 - 0.001 * (pressure - PRESSURE_MIN) - 0.01 * (binder_n - 3.0)
    density = np.clip(1.0 - porosity0 * np.exp(-0.01 * (pressure - PRESSURE_MIN)), 0.55, 0.95)
    density += rng.normal(0, 0.005, n_samples)
    density = np.clip(density, 0.55, 0.95)

    # Tensile strength: increases with binder & density, decreases with MgSt (lubricant)
    tensile = (0.5 + 6.0 * (density - 0.55) / 0.40 + 0.4 * (binder_n - BINDER_MIN)
               - 1.2 * (mgst_n - MGST_MIN) + 0.3 * (api_n - API_MIN) / (API_MAX - API_MIN))
    tensile += rng.normal(0, 0.1, n_samples)
    tensile = np.clip(tensile, 0.5, 8.5)

    # EFRF (capping risk): rises with API loading and MgSt, falls with binder & density
    efrf = (0.55 - 0.35 * (density - 0.55) / 0.40 + 0.25 * (api_n - API_MIN) / (API_MAX - API_MIN)
            - 0.15 * (binder_n - BINDER_MIN) / (BINDER_MAX - BINDER_MIN) + 0.2 * (mgst_n - MGST_MIN))
    efrf += rng.normal(0, 0.03, n_samples)
    efrf = np.clip(efrf, 0.02, 0.98)

    # Disintegration time: PVPP (disintegrant) speeds it up, binder slows it down
    disintegration = (12.0 - 4.0 * (pvpp_n - PVPP_MIN) / (PVPP_MAX - PVPP_MIN)
                       + 5.0 * (binder_n - BINDER_MIN) / (BINDER_MAX - BINDER_MIN)
                       + 3.0 * (moisture_n - MOISTURE_MIN) / (MOISTURE_MAX - MOISTURE_MIN))
    disintegration += rng.normal(0, 0.5, n_samples)
    disintegration = np.clip(disintegration, 2.0, 45.0)

    # Dissolution time: correlated with disintegration and inversely with PVPP
    dissolution = 1.8 * disintegration + 5.0 - 3.0 * (pvpp_n - PVPP_MIN) / (PVPP_MAX - PVPP_MIN)
    dissolution += rng.normal(0, 1.0, n_samples)
    dissolution = np.clip(dissolution, 10.0, 90.0)

    y = np.column_stack([density, tensile, efrf, disintegration, dissolution])
    return X.astype(np.float32), y.astype(np.float32)


class InputScaler:
    """Minimal StandardScaler-equivalent so we don't add a hard sklearn
    dependency just for this — mean/std computed on the training data,
    applied identically at inference and inside the NSGA-II loop."""
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self
    def transform(self, X):
        return (X - self.mean_) / self.std_


# NOTE: filename bumped (v2) after fixing the loss-scaling bug below — this
# forces a fresh retrain instead of silently reloading a checkpoint that was
# trained under the old, scale-imbalanced loss (which had learned to
# essentially ignore EFRF). If you're iterating further, bump this again
# any time train_model()'s loss/data-generation logic changes.
CHECKPOINT_PATH = os.path.join(tempfile.gettempdir(), 'co_hybai_v29_28_r32_v4.pt')

@st.cache_resource(show_spinner=False)
def train_model():
    """Actually train HybridTabletModel on the synthetic dataset, with
    real backprop, real loss, and a real train/val split — replacing the
    previous simulate_training() fabrication. Cached so it only runs once
    per process instead of on every Streamlit rerun."""
    if os.path.exists(CHECKPOINT_PATH):
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
            model = HybridTabletModel(input_dim=8, hidden_dim=256)
            model.load_state_dict(ckpt['model_state'])
            model.eval()
            scaler = ckpt['scaler']
            return model, scaler, ckpt['history']
        except Exception:
            pass

    X, y = generate_synthetic_data()
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

    # BUGFIX: the 5 outputs live on very different natural scales —
    # density in [0.55, 0.95] (range 0.4), EFRF in [0, 1], tensile in
    # [0.5, 8.5] (range 8), disintegration in [2, 45] (range 43),
    # dissolution in [10, 90] (range 80). A plain nn.MSELoss() over all 5
    # jointly is dominated by whichever outputs have the largest absolute
    # scale: a 5%-of-range error in dissolution contributes ~40,000x more
    # to the raw squared-error sum than the same *relative* error in
    # density. In practice this meant the network barely learned EFRF at
    # all (it collapsed toward a near-constant ~0 prediction regardless of
    # input) while tensile/disintegration/dissolution dominated training —
    # which is exactly why the NSGA-II "golden solution" converged to a
    # single degenerate corner instead of a real trade-off front: the
    # model had no real signal telling it EFRF should trade off against
    # API/tensile. Fixed by dividing each output's squared error by its
    # own variance (computed on the training targets) before averaging,
    # so every output contributes comparably to the loss regardless of
    # its physical units.
    target_var = y_train_t.var(dim=0, unbiased=False)
    target_var = torch.clamp(target_var, min=1e-6)

    def weighted_mse(pred, true):
        return (((pred - true) ** 2) / target_var).mean()

    loss_fn = weighted_mse

    history = {'loss': [], 'r2': [], 'rmse': []}
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
            # BUGFIX: this was previously a single pooled R² across all 5
            # outputs combined (sum of squared residuals / sum of squared
            # deviations over the whole (N,5) tensor at once) — subject to
            # the exact same scale-domination problem as the loss above, so
            # a high reported R² could hide a badly-fit low-variance output
            # like EFRF. Now computed per-output and macro-averaged, so
            # each output counts equally toward the reported metric.
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

    # NEW: per-output R² on the final model, so the UI can show which of
    # the 5 predicted properties are well-fit vs poorly-fit individually —
    # a single averaged R² can look fine while hiding a badly-fit
    # low-variance output (this is exactly what caused EFRF to collapse to
    # a near-constant prediction earlier, invisible in the pooled metric).
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
    # NEW: per-output training-target means, used by NSGAIIOptimizer to
    # shrink predictions for out-of-distribution candidates back toward a
    # plausible value instead of trusting an extrapolated (and often
    # saturated/ceiling) raw prediction. See NSGAIIOptimizer.evaluate().
    history['y_train_mean'] = y_train_t.mean(dim=0).numpy().tolist()

    torch.save({'model_state': model.state_dict(), 'scaler': scaler, 'history': history}, CHECKPOINT_PATH)
    return model, scaler, history



class NSGAIIOptimizer:
    def __init__(self, model, scaler, pop_size=50, generations=80, y_train_mean=None):
        self.model = model
        self.scaler = scaler
        self.pop_size = pop_size
        self.generations = generations
        self.n_objectives = 3  # Density, Tensile, EFRF
        # Training-target means (density, tensile, efrf, disintegration,
        # dissolution) used to shrink out-of-distribution predictions back
        # toward a plausible value in evaluate(). Falls back to reasonable
        # mid-range defaults if not supplied.
        self.y_train_mean = y_train_mean if y_train_mean is not None else [0.75, 4.5, 0.5, 20.0, 45.0]

    def enforce_mass_balance(self, pop):
        # BUGFIX: this previously only rescaled the six formulation
        # components to sum to 100% and clipped the result to a blanket
        # [0, 100] — it never enforced each component's own realistic
        # bound (API 80-98%, binder 1.4-6%, etc.). Combined with the
        # uncapped API reward in evaluate() below, this let NSGA-II drift
        # to degenerate formulations like API=100% / binder=PVPP=MgSt=
        # MCC=moisture=0% — outside every stated bound and not a real
        # tablet formulation (no binder, no disintegrant, no lubricant).
        # Fixed with a two-pass clip-then-renormalize (clip to each
        # component's bound, rescale to 100, clip again, rescale again),
        # the same approach used for the analogous fix in the other two
        # app reviews.
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
        """Fitness: minimize -density, -tensile, efrf, with penalties for low API and low tensile."""
        # BUGFIX: the model now expects scaled input (see the HybridTabletModel
        # fix above) — `pop` holds raw physical-unit values (API%, MPa, rpm,
        # etc.), so it must go through the same scaler used at training time
        # before every prediction.
        pop_scaled = self.scaler.transform(pop)
        with torch.no_grad():
            pred = self.model.predict(pop_scaled)
        density = pred[:, 0]
        tensile = pred[:, 1]
        efrf = pred[:, 2]
        api = pop[:, 0]  # API% (first variable, already normalized)

        # BUGFIX (root-cause fix, replacing an earlier additive-penalty
        # attempt): even after boundary-augmenting the training data
        # twice, NSGA-II kept converging on corners the surrogate model
        # extrapolates in — most recently a point where the model
        # predicted density=0.950 (its sigmoid output's hard ceiling) even
        # though the TRUE synthetic ground-truth formula caps out at 0.865
        # at the maximum possible pressure (verified directly). Patching
        # each individual corner as it's discovered is whack-a-mole. A
        # first attempt added a flat penalty to the fitness objectives,
        # but that has a scale-mismatch problem: density's entire natural
        # objective range is only ~0.4, so a penalty large enough to
        # matter there would completely swamp the ~8-wide tensile
        # objective, or vice versa if kept small. Instead, out-of-
        # distribution PREDICTIONS themselves are shrunk toward the
        # training-target mean before objectives are computed — this
        # automatically scales correctly per-property (a shrunk density
        # stays within density's own natural range, a shrunk tensile
        # within tensile's), and pulls values like the exploited corner's
        # density=0.95 back down close to what the true physics actually
        # supports (verified: shrinks to ~0.83, near the true ~0.865 cap).
        ood_z = np.abs(pop_scaled)
        ood_raw = np.clip(ood_z - 2.0, 0, None).sum(axis=1)  # 0 for well-covered points, grows for extrapolation
        shrink_factor = 1.0 / (1.0 + ood_raw)  # 1.0 = trust the prediction fully; ->0 = trust the training mean instead
        density = shrink_factor * density + (1 - shrink_factor) * self.y_train_mean[0]
        tensile = shrink_factor * tensile + (1 - shrink_factor) * self.y_train_mean[1]
        efrf = shrink_factor * efrf + (1 - shrink_factor) * self.y_train_mean[2]

        # Base objectives (all to be minimized)
        fitness = np.column_stack([
            -density,   # minimize negative density
            -tensile,   # minimize negative tensile
            efrf        # minimize efrf
        ])

        # 🚀 SLIGHT IMPROVEMENT: Penalise low API% AND low Tensile.
        # This gently pushes the optimizer to find solutions with both higher drug load and higher mechanical strength.
        # BUGFIX: api_norm was uncapped — once API exceeded 98% (which
        # enforce_mass_balance now prevents, but this is a defensive
        # second layer) it would push api_norm above 1.0, making
        # penalty_api go NEGATIVE. Since fitness is minimized, a negative
        # penalty acts as an unbounded reward for pushing API arbitrarily
        # high, which is exactly what produced the degenerate 100%-API
        # "golden solution". Clipped to [0, 1] so the reward saturates at
        # the real upper bound instead of running away past it.
        api_norm = np.clip((api - 80) / 18, 0.0, 1.0)               # 0→80%, 1→98%
        tensile_norm = np.clip(tensile / 8.5, 0.0, 1.0)              # Normalize to ~0-1 (max theoretical)

        penalty_api = 0.08 * (1 - api_norm)       # max 0.08 when API=80%
        penalty_tensile = 0.05 * (1 - tensile_norm) # max 0.05 when tensile=0

        # Apply penalties to their respective objectives
        fitness[:, 0] += penalty_api       # penalise low API via density objective
        fitness[:, 1] += penalty_tensile   # penalise low tensile via tensile objective

        return fitness

    def fast_non_dominated_sort(self, obj):
        # BUGFIX (this was the cause of the RuntimeError crash on
        # "optimizer.optimize(n_vars=8)"): this previously appended a
        # SEPARATE single-element front (`fronts.append([i])`) for every
        # individual with dom_count==0, instead of grouping all of them
        # into one shared rank-0 front. Its cascading loop then processed
        # only ONE of those singletons at a time and broke out entirely
        # (`if not next_front: break`) the first time a single individual's
        # dominated set failed to unlock anything — even though dozens of
        # other rank-0 individuals, and every individual they dominate,
        # were still sitting unprocessed. Most individuals in the
        # population were never placed in any front at all. Downstream,
        # `next(i for i, f in enumerate(fronts) if i1 in f)` in optimize()
        # then raised StopIteration for any of those missing individuals —
        # which, because optimize() is a generator (it contains `yield`),
        # Python converts into a RuntimeError (PEP 479). Fixed to build one
        # shared front per rank and only stop once every individual has
        # been assigned.
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
        # BUGFIX: same indexing bug found in the earlier app_4/app_8
        # reviews of this codebase — `dist[i]` must be indexed by each
        # individual's fixed position within `front`, not by its rank
        # position in the per-objective sort (`sorted_front`). Writing to
        # dist[i] using the sort-order index silently mixes contributions
        # between different individuals across objectives.
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

    # BUGFIX: mutation previously clipped every gene to a hardcoded [0,100]
    # range regardless of that gene's actual bounds. Genes 6 (pressure,
    # true range 150-250) and 7 (speed, true range 15-30) would get
    # silently collapsed toward 100 whenever mutation touched them, since
    # np.clip(x, 0, 100) caps anything above 100 — corrupting the process
    # parameters over successive generations. Bounds are now looked up
    # per-gene instead of assumed.
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
        # BUGFIX: there was previously an extra yield here —
        # `yield pop, obj, history, self.generations` — re-emitting the
        # exact same pop/obj/history the loop's final iteration (gen ==
        # self.generations - 1) had already yielded one line above,
        # except labelled with gen = self.generations (80 for the default
        # config) instead of a valid 0-indexed generation number (0..79).
        # That's harmless to callers that only look at the final pop/obj
        # (like run_real_optimization did), but the new live-progress
        # callback computed `(gen + 1) / total` from it — 81/80 — which is
        # outside st.progress()'s valid [0.0, 1.0] range and raised
        # StreamlitAPIException. Since this final yield added no new
        # information beyond what the loop already produced, it's removed
        # rather than patched around.

# ================================================================
# REAL RESULT FUNCTIONS (replace the previous np.random fabrications)
# ================================================================
def run_real_training_and_get_history():
    """Trains (or loads the cached trained) model and returns its real
    validation loss/R²/RMSE history, in the same shape the UI's training
    chart expects — replacing simulate_training()'s fabricated curve."""
    model, scaler, history = train_model()
    st.session_state['_trained_model'] = model
    st.session_state['_trained_scaler'] = scaler
    st.session_state['_trained_history'] = history
    return history

def run_real_optimization(progress_callback=None):
    """Runs the actual NSGA-II optimizer against the actual trained model
    and returns (final_population_df, golden_solution, generation_history)
    — replacing generate_best_solutions_with_mass_balance()'s np.random
    fabrication. If progress_callback is given, it's called as
    progress_callback(gen, total_generations) after every generation so the
    UI can show live progress instead of a single opaque spinner."""
    model = st.session_state.get('_trained_model')
    scaler = st.session_state.get('_trained_scaler')
    history = st.session_state.get('_trained_history')
    if model is None or scaler is None or history is None:
        model, scaler, history = train_model()
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

    # Report the model's own (unpenalised) predictions for display, rather
    # than the NSGA-II fitness values which have the API/tensile penalty
    # terms baked in. Batched into one forward pass instead of one
    # model.predict() call per Pareto solution.
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
            'Quality Score': quality['overall']
        })
    solutions.sort(key=lambda x: x['Quality Score'], reverse=True)
    if not solutions:
        return [], None, []
    return solutions, solutions[0], gen_history

def get_current_formulation_results():
    """Runs the actual trained model on the formulation currently set in
    the sidebar sliders — replacing generate_results()'s np.random
    fabrication."""
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
    row = np.array([[n['api'], n['binder'], n['pvpp'], n['mgst'], n['mcc'], n['moisture'],
                     st.session_state.pressure, st.session_state.speed]], dtype=np.float32)
    pred = model.predict(scaler.transform(row))[0]
    return {
        'density': float(pred[0]), 'tensile': float(pred[1]), 'efrf': float(pred[2]),
        'disintegration': float(pred[3]), 'dissolution': float(pred[4])
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
        # NEW: neither of these existed before — there was no way to
        # restore default slider values or to force a model retrain
        # short of manually clearing Streamlit's cache/temp files.
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("↺ Reset Sliders", use_container_width=True,
                        help="Restore all formulation and process parameters to their default values."):
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
            if st.button("🔄 Force Retrain", use_container_width=True,
                        help="Discard the cached model and retrain from scratch on the next run. "
                             "Use this after changing training/data-generation code."):
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
    # BUGFIX/UX: get_formulation_summary() always rescales to exactly
    # 100% by construction, so the "Total" metric was always a trivial
    # 100.0% no matter what the raw slider values were — it told the user
    # nothing about how much normalization actually changed their inputs.
    # Now shows the true raw (pre-normalization) sum and flags it if it's
    # far enough from 100% that normalization meaningfully changed the
    # formulation the user set.
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
        st.session_state.api = st.slider("**API Content (%)**", API_MIN, API_MAX, st.session_state.api, step=0.5,
                                         help="Active pharmaceutical ingredient loading. Higher API is usually harder to compress well.")
        st.session_state.binder = st.slider("**Binder (%)**", BINDER_MIN, BINDER_MAX, st.session_state.binder, step=0.1,
                                            help="Improves particle-particle bonding and tensile strength.")
        st.session_state.pvpp = st.slider("**PVPP (%)**", PVPP_MIN, PVPP_MAX, st.session_state.pvpp, step=0.1,
                                          help="Disintegrant — helps the tablet break apart in the body.")
        st.session_state.mgst = st.slider("**MgSt (%)**", MGST_MIN, MGST_MAX, st.session_state.mgst, step=0.05,
                                          help="Lubricant — eases tablet ejection but can weaken bonding at high levels.")
    with col2:
        st.session_state.mcc = st.slider("**MCC (%)**", MCC_MIN, MCC_MAX, st.session_state.mcc, step=0.1,
                                         help="Microcrystalline cellulose filler/binder — supports compressibility.")
        st.session_state.moisture = st.slider("**Moisture Content (%)**", MOISTURE_MIN, MOISTURE_MAX, st.session_state.moisture, step=0.1,
                                              help="Residual moisture in the powder blend.")
        grade_idx = st.session_state.get('binder_grade', 0)
        if not isinstance(grade_idx, int) or grade_idx >= len(BINDER_GRADE_NAMES):
            grade_idx = 0
        selected = st.selectbox("**Binder Grade**", BINDER_GRADE_NAMES, index=grade_idx,
                                help="Reference properties shown below for context. Not yet a model input — "
                                     "see the note under Process Parameters.")
        st.session_state.binder_grade = BINDER_GRADE_NAMES.index(selected)
        props = BINDER_GRADES[selected]
        st.caption(f"🔍 **{selected} Properties:**")
        st.caption(f"• Compressibility: {props['compressibility']:.0%}")
        st.caption(f"• Disintegration: {props['disintegration']:.0%}")
        st.caption(f"• Flowability: {props['flow']:.0%}")
        st.session_state.particle_size = st.slider("**Particle Size (µm)**", PARTICLE_SIZE_MIN, PARTICLE_SIZE_MAX, st.session_state.particle_size, step=5.0,
                                                    help="Reference value only — see the note under Process Parameters.")
    render_mass_balance_display(
        st.session_state.api, st.session_state.binder,
        st.session_state.pvpp, st.session_state.mgst,
        st.session_state.mcc, st.session_state.moisture
    )
    st.markdown("---")
    st.markdown("## ⚙️ Process Parameters")
    # NEW: this note didn't exist before. Pressure and speed are the only
    # process parameters the model was actually trained on and predicts
    # from — granule size, particle size, binder grade, dwell time,
    # friction, and decompression time are recorded for context but don't
    # currently feed into any prediction. Making that explicit here avoids
    # the misleading impression that adjusting them changes the results.
    st.caption("ℹ️ Only **Compression Pressure** and **Tableting Speed** currently feed into the model's "
              "predictions. The other process parameters below (granule size, dwell time, friction, "
              "decompression time) and Binder Grade/Particle Size above are recorded for your reference "
              "but are not yet part of the trained model's input space.")
    col3, col4 = st.columns(2)
    with col3:
        st.session_state.pressure = st.slider("**Compression Pressure (MPa)**", PRESSURE_MIN, PRESSURE_MAX, st.session_state.pressure, step=2.0,
                                              help="Higher pressure generally increases density and tensile strength. Used by the model.")
        st.session_state.speed = st.slider("**Tableting Speed (rpm)**", SPEED_MIN, SPEED_MAX, st.session_state.speed, step=0.5,
                                           help="Turret speed. Used by the model.")
        st.session_state.granule = st.slider("**Granule Size (µm)**", GRANULE_MIN, GRANULE_MAX, st.session_state.granule, step=5.0,
                                             help="Reference value only — not yet a model input.")
    with col4:
        st.session_state.dwell_time = st.slider("**Dwell Time (ms)**", DWELL_TIME_MIN, DWELL_TIME_MAX, st.session_state.dwell_time, step=1.0,
                                                help="Reference value only — not yet a model input.")
        st.session_state.friction = st.slider("**Friction Coefficient**", FRICTION_MIN, FRICTION_MAX, st.session_state.friction, step=0.01,
                                              help="Reference value only — not yet a model input.")
        st.session_state.decompression_time = st.slider("**Decompression Time (ms)**", DECOMPRESSION_TIME_MIN, DECOMPRESSION_TIME_MAX, st.session_state.decompression_time, step=2.0,
                                                         help="Reference value only — not yet a model input.")

def target_status(value, threshold, mode='min', comfortable=None):
    """Returns a status label that reflects actual margin from a target,
    not just a flat pass/fail. mode='min' means value should be >=
    threshold (density, tensile — higher is better); mode='max' means
    value should be <= threshold (EFRF, disintegration — lower is
    better). `comfortable` is the margin beyond which the result is
    labelled 'Excellent' rather than just 'Passes' / 'Near limit'.

    NEW: previously several labels (e.g. the golden solution's EFRF and
    Density badges) were hardcoded to "✅ Excellent" regardless of the
    actual value — an EFRF of 0.399 (right at the 0.40 limit) showed the
    same "Excellent" badge as an EFRF of 0.05. This makes the label
    reflect the real margin.
    """
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
    with st.spinner("Training physics-informed model on synthetic formulation data..."):
        history = run_real_training_and_get_history()
    if not history['loss']:
        st.warning("No training history available.")
        return
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

    # NEW: per-output R² breakdown — a pooled/averaged R² can look
    # reasonable while hiding one badly-fit output (this exact blind spot
    # previously let EFRF collapse to a near-constant prediction while the
    # averaged metric still looked fine).
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
                f"{history.get('n_val', '?')} held-out samples. A property with poor fit here means "
                "the optimizer's predictions for it (and any formulation the search converges to that "
                "relies on it) are less trustworthy — treat Pareto-optimal points as candidates for "
                "experimental confirmation, not as guaranteed outcomes."
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
    # NEW: this used to be a flat "excellent tablet quality!" claim
    # regardless of the actual margins above — e.g. an EFRF of 0.399,
    # right at the 0.40 limit, got the same message as an EFRF of 0.05.
    # Now reflects whether any property is only marginally passing.
    near_limit = any("near limit" in s or "Below target" in s or "Exceeds limit" in s
                    for s in (density_status, tensile_status, efrf_status))
    if near_limit:
        st.warning("⚠️ This is the best available trade-off among the Pareto-optimal solutions found, "
                   "but at least one property (see badges above) is close to its limit rather than "
                   "comfortably within it — worth reviewing before committing to this formulation.")
    else:
        st.success("✅ This formulation maximises API% and Tensile while preserving excellent tablet quality!")

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
    # BUGFIX: this previously called generate_best_solutions_with_mass_
    # balance(), a function removed when the fake-data simulation layer
    # was replaced with the real optimizer — calling this would have
    # raised NameError. It was also never invoked from main(), so even
    # before that, the CSV/JSON export buttons below were completely
    # unreachable. Now reads the real results from session state (set by
    # run_real_optimization()) and is called from main().
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
        st.metric("⏱️ Runtime", f"{st.session_state.runtime}s" if st.session_state.runtime else "—")
    with col2:
        evals_per_sec = (POPULATION_SIZE * NSGA_GENERATIONS) / max(1, st.session_state.runtime)
        st.metric("⚡ Evaluations/Second", f"{evals_per_sec:.0f}")

    # BUGFIX: every value in this table was previously fabricated with
    # np.random (e.g. `np.random.randint(8, 15)` for "Pareto Solutions
    # Found"), regardless of what the optimizer actually produced. Now
    # computed from the real best_solutions list stored in session state.
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
                    'Penalties'
                ],
                'Value': [
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

# ================================================================
# MAIN ORCHESTRATION
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
        # NEW: previously the only way to see any prediction at all was to
        # run the full ~10-20s NSGA-II optimization. This lets users get
        # instant feedback on the formulation they currently have dialed
        # in on the sliders, without waiting for a full population/
        # generations search.
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
            with st.spinner("Training physics-informed model (cached after first run)..."):
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

        # BUGFIX: previously called generate_results() and
        # generate_best_solutions_with_mass_balance(), both of which
        # fabricated their output with np.random rather than running any
        # model or optimizer. render_training_progress() now performs real
        # training (cached after the first run), and the block below runs
        # the real NSGA-II optimizer against the real trained model.
        render_training_progress()
        opt_progress = st.progress(0, text="Running NSGA-II generation 0/%d..." % NSGA_GENERATIONS)
        def _update_opt_progress(gen, total):
            # Defensive clamp: st.progress() raises if given anything
            # outside [0.0, 1.0]. The underlying generator is fixed to
            # never yield gen >= total now, but clamping here means this
            # can't crash again even if that changes.
            frac = min(1.0, max(0.0, (gen + 1) / total))
            opt_progress.progress(frac, text=f"Running NSGA-II generation {min(gen + 1, total)}/{total}...")
        solutions, golden, gen_history = run_real_optimization(progress_callback=_update_opt_progress)
        opt_progress.empty()
        st.session_state.results = get_current_formulation_results()
        st.session_state.golden_solution = golden
        st.session_state.best_solutions = solutions
        st.session_state.pareto_history = gen_history
        # BUGFIX: this was previously assigned AFTER render_optimization_summary()
        # was called below, so the summary always displayed the stale runtime
        # from the *previous* run (or 0/"—" on the first run) — the
        # "Evaluations/Second" figure was then dividing by that stale value
        # via max(1, runtime), producing an implausible constant like 4000/1=4000
        # instead of the real ~147/s. Moved before the render calls that
        # actually display it.
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
