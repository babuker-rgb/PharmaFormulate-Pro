# ================================================================
# Hybrid AI v30.2-R32 · 
# Multi-Objective Tablet Optimization
# ================================================================

import streamlit as st
import numpy as np
import pandas as pd  # Important for type checking
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
# PAGE CONFIG & CONSTANTS
# ================================================================
st.set_page_config(page_title="Hybrid AI v30.2", page_icon="🧬", layout="wide")

API_MIN, API_MAX = 80.0, 98.0
BINDER_MIN, BINDER_MAX = 1.4, 6.0
PVPP_MIN, PVPP_MAX = 1.0, 6.0
MGST_MIN, MGST_MAX = 0.10, 1.2
MCC_MIN, MCC_MAX = 1.5, 8.0
MOISTURE_MIN, MOISTURE_MAX = 0.5, 5.0
PRESSURE_MIN, PRESSURE_MAX = 150.0, 250.0
SPEED_MIN, SPEED_MAX = 15.0, 30.0
EFRF_THRESHOLD = 0.40

POPULATION_SIZE = 80
NSGA_GENERATIONS = 80
TRAINING_EPOCHS = 2000
HIDDEN_SIZE = 512
N_SAMPLES = 10000

# ================================================================
# 1. DATA GENERATION (A to Z)
# ================================================================
def generate_synthetic_data(n_samples=N_SAMPLES, seed=42):
    rng = np.random.default_rng(seed)
    api = rng.uniform(API_MIN, API_MAX, n_samples)
    binder = rng.uniform(BINDER_MIN, BINDER_MAX, n_samples)
    pvpp = rng.uniform(PVPP_MIN, PVPP_MAX, n_samples)
    mgst = rng.uniform(MGST_MIN, MGST_MAX, n_samples)
    mcc = rng.uniform(MCC_MIN, MCC_MAX, n_samples)
    moisture = rng.uniform(MOISTURE_MIN, MOISTURE_MAX, n_samples)
    pressure = rng.uniform(PRESSURE_MIN, PRESSURE_MAX, n_samples)
    speed = rng.uniform(SPEED_MIN, SPEED_MAX, n_samples)

    X = np.column_stack([api, binder, pvpp, mgst, mcc, moisture, pressure, speed]).astype(np.float32)

    # Simulate Physics
    density = np.clip(0.6 + 0.3 * ((pressure - 150) / 100) - 0.01 * (binder - 3.0) + rng.normal(0, 0.01, n_samples), 0.55, 0.95)
    tensile = np.clip(1.0 + 6.0 * (density - 0.55) + 0.2 * (api - 80) / 18 - 0.5 * (mgst - 0.1) + rng.normal(0, 0.2, n_samples), 0.5, 8.5)
    efrf = np.clip(0.6 - 0.5 * (density - 0.55) + 0.2 * (mgst - 0.1) + rng.normal(0, 0.05, n_samples), 0.02, 0.98)
    disintegration = np.clip(10.0 - 2.0 * (pvpp - 1.0) / 5.0 + 3.0 * (binder - 1.4) / 4.6 + rng.normal(0, 1.0, n_samples), 2.0, 45.0)
    dissolution = np.clip(2.0 * disintegration + 10.0 + rng.normal(0, 2.0, n_samples), 10.0, 90.0)

    y = np.column_stack([density, tensile, efrf, disintegration, dissolution]).astype(np.float32)
    return X, y

# ================================================================
# 2. SCALER & PINN MODEL
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
    def __init__(self, input_dim=8, hidden_dim=512):
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
        self.dropout = nn.Dropout(0.1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        h1 = torch.relu(self.bn1(self.fc1(x)))
        h1 = self.dropout(h1)
        h2 = torch.relu(self.bn2(self.fc2(h1))) + h1
        h2 = self.dropout(h2)
        h3 = torch.relu(self.bn3(self.fc3(h2))) + h2
        h3 = self.dropout(h3)
        out = self.fc5(h3)
        
        density = torch.sigmoid(out[:, 0]) * 0.4 + 0.55
        tensile = torch.sigmoid(out[:, 1]) * 8.0 + 0.5
        efrf = torch.sigmoid(out[:, 2])
        disintegration = torch.sigmoid(out[:, 3]) * 45.0 + 2.0
        dissolution = torch.sigmoid(out[:, 4]) * 80.0 + 10.0
        return torch.stack([density, tensile, efrf, disintegration, dissolution], dim=1)

    def predict_with_uncertainty(self, x, n_samples=20):
        self.train()
        with torch.no_grad():
            preds = np.array([self.forward(x).numpy() for _ in range(n_samples)])
        self.eval()
        return np.mean(preds, axis=0), np.std(preds, axis=0)

# ================================================================
# 3. ADVANCED OPTIMIZER (WITH FIXES FOR EVALUATION)
# ================================================================
class AdvancedOptimizer:
    def __init__(self, model, scaler, pop_size=80, generations=80, y_train_mean=None):
        self.model = model
        self.scaler = scaler
        self.pop_size = pop_size
        self.generations = generations
        self.y_train_mean = y_train_mean if y_train_mean is not None else [0.75, 4.5, 0.5, 20.0, 45.0]
        self.mutation_rate = 0.1
        self.GENE_BOUNDS = [
            (API_MIN, API_MAX), (BINDER_MIN, BINDER_MAX), (PVPP_MIN, PVPP_MAX),
            (MGST_MIN, MGST_MAX), (MCC_MIN, MCC_MAX), (MOISTURE_MIN, MOISTURE_MAX),
            (PRESSURE_MIN, PRESSURE_MAX), (SPEED_MIN, SPEED_MAX)
        ]

    def enforce_mass_balance(self, pop):
        balanced = pop.copy()
        lo = np.array([b[0] for b in self.GENE_BOUNDS[:6]])
        hi = np.array([b[1] for b in self.GENE_BOUNDS[:6]])
        comps = np.clip(pop[:, :6], lo, hi)
        total = comps.sum(axis=1, keepdims=True)
        total = np.where(total <= 0, 1.0, total)
        balanced[:, :6] = np.clip(comps / total * 100.0, lo, hi)
        return balanced

    def evaluate(self, pop):
        # FIX 1: Ensure input is numpy array and not a pandas object
        pop_scaled = self.scaler.transform(pop)
        if isinstance(pop_scaled, pd.DataFrame):
            pop_scaled = pop_scaled.values
        
        # FIX 2: Set model to eval mode and disable gradient tracking
        self.model.eval()
        with torch.no_grad():
            # Use torch.tensor with explicit float32 for clarity
            pred = self.model(torch.tensor(pop_scaled, dtype=torch.float32)).numpy()
        
        density, tensile, efrf = pred[:, 0], pred[:, 1], pred[:, 2]
        
        # Dynamic Penalty
        ood_z = np.abs(pop_scaled)
        raw_penalty = np.clip(ood_z - 2.5, 0, None).sum(axis=1)
        penalty = 1.0 / (1.0 + raw_penalty)
        
        # Objectives (Density, Tensile, API to maximize; EFRF to minimize)
        fitness = np.column_stack([-density * penalty, -tensile * penalty, -pop[:, 0] * penalty, efrf * penalty])
        
        # Hard Constraint on EFRF (high penalty if violated)
        efrf_violation = np.maximum(0, efrf - EFRF_THRESHOLD) * 100.0
        fitness[:, 3] += efrf_violation
        return fitness

    def adaptive_mutation(self, pop):
        diversity = np.std(pop, axis=0).mean()
        if diversity < 0.05: self.mutation_rate = min(0.2, self.mutation_rate + 0.02)
        elif diversity > 0.2: self.mutation_rate = max(0.02, self.mutation_rate - 0.01)
        return diversity

    def optimize(self):
        pop = np.random.rand(self.pop_size, 8)
        for i, (lo, hi) in enumerate(self.GENE_BOUNDS):
            pop[:, i] = pop[:, i] * (hi - lo) + lo
        pop = self.enforce_mass_balance(pop)
        obj = self.evaluate(pop)
        history = []
        
        for gen in range(self.generations):
            self.adaptive_mutation(pop)
            # Tournament Selection
            selected = []
            for _ in range(self.pop_size):
                idx = np.random.choice(self.pop_size, 2, replace=False)
                selected.append(idx[np.argmin(obj[idx].sum(axis=1))])
            sel_pop = pop[selected]
            
            # Crossover
            offspring = []
            for i in range(0, self.pop_size, 2):
                p1, p2 = sel_pop[i], sel_pop[(i+1)%self.pop_size]
                c1, c2 = p1.copy(), p2.copy()
                for j in range(8):
                    if np.random.rand() < 0.8:
                        beta = 1.0 + 2.0 * np.random.rand()
                        c1[j] = 0.5 * ((1+beta)*p1[j] + (1-beta)*p2[j])
                        c2[j] = 0.5 * ((1-beta)*p1[j] + (1+beta)*p2[j])
                    if np.random.rand() < self.mutation_rate:
                        lo, hi = self.GENE_BOUNDS[j]
                        c1[j] = np.clip(c1[j] + np.random.normal(0, 0.1) * (hi-lo), lo, hi)
                offspring.extend([c1, c2])
            offspring = np.array(offspring[:self.pop_size])
            offspring = self.enforce_mass_balance(offspring)
            
            # Environmental Selection (Elitism)
            combined = np.vstack([pop, offspring])
            combined_obj = np.vstack([obj, self.evaluate(offspring)])
            # Sort by sum of objectives and take top N
            pareto_indices = np.argsort(combined_obj.sum(axis=1))[:self.pop_size]
            pop = combined[pareto_indices]
            obj = combined_obj[pareto_indices]
            
            if gen % 10 == 0 or gen == self.generations - 1:
                history.append({'generation': gen, 'population': pop.copy(), 'objectives': obj.copy()})
            yield pop, obj, history, gen

# ================================================================
# 4. ADVANCED TRAINING LOOP
# ================================================================
CHECKPOINT_PATH = os.path.join(tempfile.gettempdir(), 'hybrid_ai_v30.pt')

@st.cache_resource(show_spinner=False)
def train_model():
    if os.path.exists(CHECKPOINT_PATH):
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
            model = HybridTabletModel(input_dim=8, hidden_dim=HIDDEN_SIZE)
            model.load_state_dict(ckpt['model_state'])
            model.eval()
            return model, ckpt['scaler'], ckpt['history']
        except: pass

    X, y = generate_synthetic_data()
    scaler = InputScaler().fit(X)
    X_scaled = scaler.transform(X)
    
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    model = HybridTabletModel(input_dim=8, hidden_dim=HIDDEN_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    target_var = y_t.var(dim=0, unbiased=False)
    target_var = torch.clamp(target_var, min=1e-6)
    
    def weighted_mse(pred, true):
        return (((pred - true) ** 2) / target_var).mean()
    
    history = {'loss': [], 'r2': []}
    for epoch in range(TRAINING_EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(X_t)
        loss = weighted_mse(pred, y_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 100 == 0:
            with torch.no_grad():
                model.eval()
                val_loss = weighted_mse(model(X_t), y_t).item()
                history['loss'].append(val_loss)
                ss_res = ((y_t - model(X_t)) ** 2).sum(dim=0)
                ss_tot = ((y_t - y_t.mean(dim=0)) ** 2).sum(dim=0)
                history['r2'].append((1 - ss_res / torch.clamp(ss_tot, min=1e-8)).mean().item())
    
    model.eval()
    torch.save({
        'model_state': model.state_dict(),
        'scaler': scaler,
        'history': history
    }, CHECKPOINT_PATH)
    return model, scaler, history

# ================================================================
# 5. ANALYSIS & UI FUNCTIONS
# ================================================================
def perform_sensitivity_analysis(model, scaler, ref_solution):
    try:
        rf = RandomForestRegressor(n_estimators=50)
        X_local = np.random.normal(loc=ref_solution, scale=0.05*np.abs(ref_solution), size=(200, 8))
        X_scaled = scaler.transform(X_local)
        y_local = model(torch.tensor(X_scaled, dtype=torch.float32)).numpy()[:, 0]
        rf.fit(X_scaled, y_local)
        perm_importance = permutation_importance(rf, X_scaled, y_local)
        feature_names = ['API', 'Binder', 'PVPP', 'MgSt', 'MCC', 'Moisture', 'Pressure', 'Speed']
        return dict(zip(feature_names, perm_importance.importances_mean))
    except Exception as e:
        return None

def render_3d_pareto(pop, obj, golden_idx):
    fig = go.Figure(data=[go.Scatter3d(
        x=pop[:, 0], y=obj[:, 3], z=-obj[:, 1],
        mode='markers', marker=dict(size=4, color=pop[:, 0], colorscale='Viridis'),
        name='Pareto Solutions'
    )])
    if golden_idx is not None and golden_idx < len(pop):
        fig.add_trace(go.Scatter3d(
            x=[pop[golden_idx, 0]], y=[obj[golden_idx, 3]], z=[-obj[golden_idx, 1]],
            mode='markers', marker=dict(size=15, color='gold', symbol='diamond'),
            name='Golden Solution'
        ))
    fig.update_layout(scene=dict(xaxis_title='API (%)', yaxis_title='EFRF', zaxis_title='Tensile (MPa)'),
                      height=450, margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig, use_container_width=True)

def render_dynamic_radar(solutions_df, selected_solutions):
    if not selected_solutions: return
    fig = go.Figure()
    for i, row in solutions_df.iterrows():
        if row['Solution'] in selected_solutions:
            fig.add_trace(go.Scatterpolar(
                r=[(row['API (%)']-80)/18, row['Density']/0.95, row['Tensile (MPa)']/8.5, 1-row['EFRF'], row['Quality Score']/100],
                theta=['API%', 'Density', 'Tensile', 'EFRF (Inv)', 'Quality'],
                fill='toself', name=row['Solution']
            ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True, height=380)
    st.plotly_chart(fig, use_container_width=True)

# ================================================================
# 6. MAIN APPLICATION
# ================================================================
def main():
    st.title("🧬 Hybrid AI v30.2-R32 · Complete Framework (Fixed)")
    
    # Load resources
    with st.spinner("Loading Physics-Informed Model..."):
        model, scaler, history = train_model()
    
    # Sidebar Recommender
    st.sidebar.header("⚖️ Custom Recommender")
    w_api = st.sidebar.slider("Weight for API", 0.0, 1.0, 0.4)
    w_quality = st.sidebar.slider("Weight for Quality", 0.0, 1.0, 0.6)
    st.sidebar.info("The system will prioritize solutions matching your preferences.")

    # Main Layout
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⚙️ Formulation & Process")
        api = st.slider("API (%)", API_MIN, API_MAX, 85.0)
        binder = st.slider("Binder (%)", BINDER_MIN, BINDER_MAX, 5.0)
        pvpp = st.slider("PVPP (%)", PVPP_MIN, PVPP_MAX, 2.0)
        mgst = st.slider("MgSt (%)", MGST_MIN, MGST_MAX, 0.5)
    with col2:
        mcc = st.slider("MCC (%)", MCC_MIN, MCC_MAX, 4.0)
        moisture = st.slider("Moisture (%)", MOISTURE_MIN, MOISTURE_MAX, 1.5)
        pressure = st.slider("Pressure (MPa)", PRESSURE_MIN, PRESSURE_MAX, 200.0)
        speed = st.slider("Speed (rpm)", SPEED_MIN, SPEED_MAX, 20.0)

    if st.button("🚀 Run Advanced Optimization"):
        start_time = time.time()
        with st.status("Running Advanced NSGA-II...", expanded=True) as status:
            progress_bar = st.progress(0)
            optimizer = AdvancedOptimizer(model, scaler, pop_size=POPULATION_SIZE, generations=NSGA_GENERATIONS, y_train_mean=history.get('y_train_mean'))
            
            final_pop, final_obj = None, None
            for i, (pop, obj, gen_hist, gen) in enumerate(optimizer.optimize()):
                final_pop, final_obj = pop, obj
                progress_bar.progress((gen+1)/NSGA_GENERATIONS)
                if gen % 10 == 0:
                    status.update(label=f"Generation {gen+1}/{NSGA_GENERATIONS} | Diversity: {np.std(pop, axis=0).mean():.3f}")
            
            status.update(label="Optimization Complete ✅", state="complete")
        
        # Process results
        y_train_mean = history.get('y_train_mean')
        if y_train_mean is None: y_train_mean = [0.75, 4.5, 0.5, 20.0, 45.0]
        
        # Find best based on User Weights
        weights = np.array([w_api, w_quality])
        results = []
        for i in range(len(final_pop)):
            api_val = final_pop[i, 0]
            obj_sum = final_obj[i].sum()
            score = (api_val / 100 * weights[0]) + ((1 - obj_sum / 4) * weights[1])
            results.append(score)
        golden_idx = np.argmax(results)
        best_sol = final_pop[golden_idx]
        
        # Prediction with Uncertainty
        pop_scaled = scaler.transform([best_sol])
        preds, uncertainty = model.predict_with_uncertainty(torch.tensor(pop_scaled, dtype=torch.float32))
        preds, uncertainty = preds[0], uncertainty[0]
        
        st.success(f"🏆 Golden Solution Found!\nAPI: {best_sol[0]:.2f}% | EFRF: {preds[2]:.3f} ± {uncertainty[2]:.3f}")
        st.caption(f"Optimization took {time.time() - start_time:.2f} seconds.")
        
        # 3D Pareto
        st.subheader("🌐 3D Pareto Front (API - EFRF - Tensile)")
        render_3d_pareto(final_pop, final_obj, golden_idx)
        
        # Sensitivity Analysis
        with st.expander("🔬 Sensitivity Analysis (Local)"):
            sens_data = perform_sensitivity_analysis(model, scaler, best_sol)
            if sens_data:
                st.bar_chart(pd.Series(sens_data))
            else:
                st.warning("Could not compute local sensitivity.")
        
        # Top 5 Solutions & Dynamic Radar
        sol_list = []
        sorted_indices = np.argsort([-results[i] for i in range(len(final_pop))])
        for idx in sorted_indices[:10]:
            sol_list.append({
                'Solution': f'S{idx+1}',
                'API (%)': final_pop[idx, 0],
                'Density': -final_obj[idx, 0],
                'Tensile (MPa)': -final_obj[idx, 1],
                'EFRF': final_obj[idx, 3],
                'Quality Score': 100 - (final_obj[idx].sum() * 20)
            })
        sol_df = pd.DataFrame(sol_list)
        
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.subheader("🏆 Top Solutions")
            selected = st.multiselect("Select for Radar", sol_df['Solution'], default=[f'S{str(sorted_indices[0]+1)}', f'S{str(sorted_indices[1]+1)}'])
        with col_b:
            st.subheader("📊 Dynamic Radar Comparison")
            render_dynamic_radar(sol_df, selected)
        
        # Export Report
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'golden_api': best_sol[0], 'golden_efrf': preds[2],
            'golden_tensile': preds[1],
            'top_solutions': sol_df.to_dict('records')
        }
        st.download_button("📥 Download Full Report (JSON)", data=json.dumps(report, indent=2), file_name="optimization_report_v30.json")

if __name__ == "__main__":
    main()
