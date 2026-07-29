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
        try:
            preds = model.predict(scaler.transform(current_pop))
            density_vals = preds[:, 0]
            tensile_vals = preds[:, 1]
            dis_vals = preds[:, 3]
            diss_vals = preds[:, 4]
            # Generate feasible region – catch any error and use empty arrays
            try:
                feat_api, feat_efrf = generate_feasible_samples(model, scaler)
            except Exception:
                feat_api, feat_efrf = np.array([]), np.array([])
        except Exception:
            density_vals = np.full_like(api_vals, np.nan)
            tensile_vals = np.full_like(api_vals, np.nan)
            dis_vals = np.full_like(api_vals, np.nan)
            diss_vals = np.full_like(api_vals, np.nan)
            feat_api, feat_efrf = np.array([]), np.array([])
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
        golden_idx = st.session_state.get('golden_idx')
        if golden_idx is not None and golden_idx < len(current_pop):
            golden_api = current_pop[golden_idx, 0]
            golden_efrf = current_obj[golden_idx, 1]
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
