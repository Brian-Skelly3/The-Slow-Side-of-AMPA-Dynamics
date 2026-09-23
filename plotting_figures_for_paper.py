import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# -------------------------------------------------------------------------
# General plot formatting
# -------------------------------------------------------------------------

# Force serif font and set primary font to Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']

# Match math text (LaTeX-style equations in labels) to Times
plt.rcParams['mathtext.fontset'] = 'stix'

# Ensure fonts are embedded as TrueType (Type 42) in the PDF (required by most journals)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Set font sizes
plt.rcParams['font.size'] = 20        
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['figure.titlesize'] = 24


# Color palette (Monte Carlo vs ODE)
COLOR_1000 = '#2A9D8F'    # Teal (1,000 receptors)
COLOR_50000 = '#E76F51'   # Warm Coral/Amber (50,000 receptors)
COLOR_ODE = '#000000'     # Black (Deterministic ODE)

# Color palette (KR vs reduced)
COLOR_KR = '#7E2F8E'   # MATLAB Purple (Reference KR Model)
COLOR_RED = '#E69F00'  # Okabe-Ito Orange (Reduced Model)
COLOR_ENSEMBLE = '#888888'

# Palette: Light, Medium, Dark tones for levels (Decreased, Normal, Increased)
tarpless_colors = ['#9ecae1', '#3182bd', '#08519c']  # Blue tones (U)
tarped_colors   = ['#fdd0a2', '#e6550d', '#a63603']  # Orange/Red tones (T)


# -------------------------------------------------------------------------
# Figure 3 - Monte Carlo simulations
# -------------------------------------------------------------------------

DATA_DIR = 'Data/MonteCarlo sims'

# Create 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex='col')

# Plot Monte Carlo simulations
for run_idx in range(1, 6):
    # --- TARPless Monte Carlo (Row 0) ---
    path_un_1k = os.path.join(DATA_DIR, f'unTARPed_MonteCarlo1000runs_{run_idx}')
    df_un_1k = pd.read_csv(path_un_1k, sep='\t', skiprows=1, header=None)
    t_un_1k = df_un_1k.iloc[:, 10]*1000
    g_un_1k = df_un_1k.iloc[:, 9] + 2 * df_un_1k.iloc[:, 15]
    d_un_1k = df_un_1k.iloc[:, 11] + df_un_1k.iloc[:, 17]

    path_un_50k = os.path.join(DATA_DIR, f'unTARPed_MonteCarlo50000runs_{run_idx}')
    df_un_50k = pd.read_csv(path_un_50k, sep='\t', skiprows=1, header=None)
    t_un_50k = df_un_50k.iloc[:, 10]*1000
    g_un_50k = df_un_50k.iloc[:, 9] + 2 * df_un_50k.iloc[:, 15]
    d_un_50k = df_un_50k.iloc[:, 11] + df_un_50k.iloc[:, 17]

    axes[0, 0].plot(t_un_1k, g_un_1k, color=COLOR_1000, alpha=0.45, lw=0.9, zorder=2)
    axes[0, 1].plot(t_un_1k, d_un_1k, color=COLOR_1000, alpha=0.45, lw=0.9, zorder=2)
    axes[0, 0].plot(t_un_50k, g_un_50k, color=COLOR_50000, alpha=0.75, lw=1.1, zorder=3)
    axes[0, 1].plot(t_un_50k, d_un_50k, color=COLOR_50000, alpha=0.75, lw=1.1, zorder=3)

    # --- TARPed Monte Carlo (Row 1) ---
    path_tarp_1k = os.path.join(DATA_DIR, f'TARPed_MonteCarlo1000runs_{run_idx}')
    df_tarp_1k = pd.read_csv(path_tarp_1k, sep='\t', skiprows=1, header=None)
    t_tarp_1k = df_tarp_1k.iloc[:, 10]*1000
    g_tarp_1k = df_tarp_1k.iloc[:, 9] + 2 * df_tarp_1k.iloc[:, 15]
    d_tarp_1k = df_tarp_1k.iloc[:, 11] + df_tarp_1k.iloc[:, 17]

    path_tarp_50k = os.path.join(DATA_DIR, f'TARPed_MonteCarlo50000runs_{run_idx}')
    df_tarp_50k = pd.read_csv(path_tarp_50k, sep='\t', skiprows=1, header=None)
    t_tarp_50k = df_tarp_50k.iloc[:, 10]*1000
    g_tarp_50k = df_tarp_50k.iloc[:, 9] + 2 * df_tarp_50k.iloc[:, 15]
    d_tarp_50k = df_tarp_50k.iloc[:, 11] + df_tarp_50k.iloc[:, 17]

    axes[1, 0].plot(t_tarp_1k, g_tarp_1k, color=COLOR_1000, alpha=0.45, lw=0.9, zorder=2)
    axes[1, 1].plot(t_tarp_1k, d_tarp_1k, color=COLOR_1000, alpha=0.45, lw=0.9, zorder=2)
    axes[1, 0].plot(t_tarp_50k, g_tarp_50k, color=COLOR_50000, alpha=0.75, lw=1.1, zorder=3)
    axes[1, 1].plot(t_tarp_50k, d_tarp_50k, color=COLOR_50000, alpha=0.75, lw=1.1, zorder=3)


# Plot Deterministic ODE Solutions

# TARPless ODE
path_t_ode_un = os.path.join(DATA_DIR, 'time_ODE_model_TARPless.csv')
path_g_ode_un = os.path.join(DATA_DIR, 'conductance_ODE_model_TARPless.csv')
path_d_ode_un = os.path.join(DATA_DIR, 'desensitisation_ODE_model_TARPless.csv')

t_ode_un = pd.read_csv(path_t_ode_un).values.squeeze()
g_ode_un = pd.read_csv(path_g_ode_un).values.squeeze()
d_ode_un = pd.read_csv(path_d_ode_un).values.squeeze()

axes[0, 0].plot(t_ode_un, g_ode_un, color=COLOR_ODE, linestyle='--', lw=1.6, zorder=4)
axes[0, 1].plot(t_ode_un, d_ode_un, color=COLOR_ODE, linestyle='--', lw=1.6, zorder=4)

# TARPed ODE
path_t_ode_tarp = os.path.join(DATA_DIR, 'time_ODE_model_TARPed.csv')
path_g_ode_tarp = os.path.join(DATA_DIR, 'conductance_ODE_model_TARPed.csv')
path_d_ode_tarp = os.path.join(DATA_DIR, 'desensitisation_ODE_model_TARPed.csv')

t_ode_tarp = pd.read_csv(path_t_ode_tarp).values.squeeze()
g_ode_tarp = pd.read_csv(path_g_ode_tarp).values.squeeze()
d_ode_tarp = pd.read_csv(path_d_ode_tarp).values.squeeze()

axes[1, 0].plot(t_ode_tarp, g_ode_tarp, color=COLOR_ODE, linestyle='--', lw=1.6, zorder=4)
axes[1, 1].plot(t_ode_tarp, d_ode_tarp, color=COLOR_ODE, linestyle='--', lw=1.6, zorder=4)


# Plot Formatting
# Column Headers (Bold)
axes[0, 0].set_title('Conductance Dynamics', fontweight='bold', pad=12, fontsize=24)
axes[0, 1].set_title('Desensitisation Dynamics', fontweight='bold', pad=12, fontsize=24)

# Y-Axis Row Labels (Bold, integrated context on left column)
axes[0, 0].set_ylabel('TARPless\nConductance', fontweight='bold', fontsize=22)
axes[1, 0].set_ylabel('TARPed\nConductance', fontweight='bold', fontsize=22)
axes[0, 1].set_ylabel('Desensitisation', fontweight='bold', fontsize=22)
axes[1, 1].set_ylabel('Desensitisation', fontweight='bold', fontsize=22)

# X-Axis Labels (Bottom row)
axes[1, 0].set_xlabel('Time (ms)', fontweight='bold', fontsize=22)
axes[1, 1].set_xlabel('Time (ms)', fontweight='bold', fontsize=22)

# Axis limits
axes[0, 0].set_xlim(0, 50)
axes[1, 0].set_xlim(0, 50)
axes[0, 1].set_xlim(0, 500)
axes[1, 1].set_xlim(0, 500)

axes[0, 0].set_ylim(0, 0.9)
axes[1, 0].set_ylim(0, 0.9)
axes[0, 1].set_ylim(0, 0.5)
axes[1, 1].set_ylim(0, 0.5)

# Apply panel styling (Remove top/right spines, light background grid)
for ax in axes.flat:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', color='#e0e0e0', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

legend_elements = [
    plt.Line2D([0], [0], color=COLOR_1000, lw=1.5, label='1,000 Receptors'),
    plt.Line2D([0], [0], color=COLOR_50000, lw=1.5, label='50,000 Receptors'),
    plt.Line2D([0], [0], color=COLOR_ODE, lw=1.5, linestyle='--', label='ODE Model')
]

fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.06),
           ncol=3, frameon=False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

# Save figures
plt.savefig('Plots/fig3_ode_validation_styled.pdf', dpi=300, bbox_inches='tight')
plt.show()


# -------------------------------------------------------------------------
# Figure 4 - Single pulse stimulation
# -------------------------------------------------------------------------

DATA_DIR = 'Data/Single Pulse'
TARPless_PARAM_DIR = os.path.join(DATA_DIR, 'TARPless_fast_slow_A1_A2')
TARPed_PARAM_DIR = os.path.join(DATA_DIR, 'TARPed_fast_slow_A1_A2')

# Create 1x2 subplot grid with shared Y-axis
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)

# Load Data
# TARPless
t_un = pd.read_csv(os.path.join(DATA_DIR, '1pulse_TARPless_t_100ms')).values.squeeze()
kr_un = pd.read_csv(os.path.join(DATA_DIR, '1pulse_TARPless_KR_100ms')).values.squeeze()
red_un = pd.read_csv(os.path.join(DATA_DIR, '1pulse_TARPless_Red_100ms')).values.squeeze()

# TARPed
t_tarp = pd.read_csv(os.path.join(DATA_DIR, '1pulse_TARPed_t_500ms')).values.squeeze()
kr_tarp = pd.read_csv(os.path.join(DATA_DIR, '1pulse_TARPed_KR_500ms')).values.squeeze()
red_tarp = pd.read_csv(os.path.join(DATA_DIR, '1pulse_TARPed_Red_500ms')).values.squeeze()


# Plot Conductance Traces

# Plot for each optimised parameter set
path_t_un_set = os.path.join(TARPless_PARAM_DIR, 'param_sets_TARPless_fsA1A2_t.txt')
path_t_tarp_set = os.path.join(TARPed_PARAM_DIR, 'param_sets_TARPed_fsA1A2_t.txt')
t_un_params = pd.read_csv(path_t_un_set, header=None).values.squeeze()
t_tarp_params = pd.read_csv(path_t_tarp_set, header=None).values.squeeze()
for x in range(30):
    # TARPless ensemble trace
    path_un_set = os.path.join(TARPless_PARAM_DIR, f'param_sets_TARPless_fsA1A2_{x}.txt')
    if os.path.exists(path_un_set):
        trace_un = pd.read_csv(path_un_set, header=None).values.squeeze()
        axes[0].plot(t_un_params, trace_un, color=COLOR_ENSEMBLE, alpha=0.3, lw=0.8, zorder=1)

    # TARPed ensemble trace
    path_tarp_set = os.path.join(TARPed_PARAM_DIR, f'param_sets_TARPed_fsA1A2_{x}.txt')
    if os.path.exists(path_tarp_set):
        trace_tarp = pd.read_csv(path_tarp_set, header=None).values.squeeze()
        axes[1].plot(t_tarp_params, trace_tarp, color=COLOR_ENSEMBLE, alpha=0.3, lw=0.8, zorder=1)

# Plot median data
# Panel 1: TARPless
axes[0].plot(t_un, kr_un, color=COLOR_KR, lw=1.8, linestyle='-', label='KR model', zorder=2)
axes[0].plot(t_un, red_un, color=COLOR_RED, lw=1.5, linestyle='--', label='Reduced model', zorder=3)

# Panel 2: TARPed
axes[1].plot(t_tarp, kr_tarp, color=COLOR_KR, lw=1.8, linestyle='-', zorder=2)
axes[1].plot(t_tarp, red_tarp, color=COLOR_RED, lw=1.5, linestyle='--', zorder=3)


# Plot formatting

axes[0].set_title('TARPless', fontweight='bold', pad=12)
axes[1].set_title('TARPed', fontweight='bold', pad=12)

axes[0].set_xlabel('Time (ms)', fontweight='bold')
axes[1].set_xlabel('Time (ms)', fontweight='bold')
axes[0].set_ylabel('Conductance', fontweight='bold')

# Set specific X-axis windows
axes[0].set_xlim(-1, 50)   # TARPless decay window
axes[1].set_xlim(-10, 500)  # Extended TARPed decay window

# Apply panel styling (remove top/right spines, light background grid)
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', color='#e0e0e0', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

legend_elements = [
    plt.Line2D([0], [0], color=COLOR_KR, lw=1.8, linestyle='-', label='KR model'),
    plt.Line2D([0], [0], color=COLOR_RED, lw=1.5, linestyle='--', label='Reduced model')
]

fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.08),
           ncol=2, frameon=False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.25)

# Save figure
plt.savefig('Plots/single_pulse_comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()


# -------------------------------------------------------------------------
# Figure 5 - Train stimulation
# -------------------------------------------------------------------------

DATA_DIR = 'Data/Train stimulation'
TARPless_PARAM_DIR = os.path.join(DATA_DIR, 'TARPless sweeps')
TARPed_PARAM_DIR = os.path.join(DATA_DIR, 'TARPed sweeps')

# Create 3x2 subplot grid
fig, axes = plt.subplots(3, 2, figsize=(10, 7.5), sharex=True, sharey='row')

# Load Data

# TARPless 5Hz
t_un_5hz = pd.read_csv(os.path.join(DATA_DIR, 'TARPless_t_5Hz_4000ms')).values.squeeze()
kr_un_5hz = pd.read_csv(os.path.join(DATA_DIR, 'TARPless_KR_5Hz_4000ms')).values.squeeze()
red_un_5hz = pd.read_csv(os.path.join(DATA_DIR, 'TARPless_Red_5Hz_4000ms')).values.squeeze()

# TARPless 20Hz
t_un_20hz = pd.read_csv(os.path.join(DATA_DIR, 'TARPless_t_20Hz_4000ms')).values.squeeze()
kr_un_20hz = pd.read_csv(os.path.join(DATA_DIR, 'TARPless_KR_20Hz_4000ms')).values.squeeze()
red_un_20hz = pd.read_csv(os.path.join(DATA_DIR, 'TARPless_Red_20Hz_4000ms')).values.squeeze()

# TARPless Poisson
t_un_pois = pd.read_csv(os.path.join(DATA_DIR, 'Poisson_TARPless_t.txt')).values.squeeze()
kr_un_pois = pd.read_csv(os.path.join(DATA_DIR, 'Poisson_TARPless_KR.txt')).values.squeeze()
red_un_pois = pd.read_csv(os.path.join(DATA_DIR, 'Poisson_TARPless_Red.txt')).values.squeeze()

# TARPed 5Hz
t_tarp_5hz = pd.read_csv(os.path.join(DATA_DIR, 'TARPed_t_5Hz_4000ms')).values.squeeze()
kr_tarp_5hz = pd.read_csv(os.path.join(DATA_DIR, 'TARPed_KR_5Hz_4000ms')).values.squeeze()
red_tarp_5hz = pd.read_csv(os.path.join(DATA_DIR, 'TARPed_Red_5Hz_4000ms')).values.squeeze()

# TARPed 20Hz
t_tarp_20hz = pd.read_csv(os.path.join(DATA_DIR, 'TARPed_t_20Hz_4000ms')).values.squeeze()
kr_tarp_20hz = pd.read_csv(os.path.join(DATA_DIR, 'TARPed_KR_20Hz_4000ms')).values.squeeze()
red_tarp_20hz = pd.read_csv(os.path.join(DATA_DIR, 'TARPed_Red_20Hz_4000ms')).values.squeeze()

# TARPed Poisson
t_tarp_pois = pd.read_csv(os.path.join(DATA_DIR, 'Poisson_TARPed_t.txt')).values.squeeze()
kr_tarp_pois = pd.read_csv(os.path.join(DATA_DIR, 'Poisson_TARPed_KR.txt')).values.squeeze()
red_tarp_pois = pd.read_csv(os.path.join(DATA_DIR, 'Poisson_TARPed_Red.txt')).values.squeeze()


# Plot Conductance Traces

# Plot for each optimised parameter set
path_t_un_set_5Hz = os.path.join(TARPless_PARAM_DIR, 'param_sets_5Hz_TARPless_XUU0_t.txt')
path_t_tarp_set_5Hz = os.path.join(TARPed_PARAM_DIR, 'param_sets_5Hz_TARPed_XUU0_t.txt')
path_t_un_set_20Hz= os.path.join(TARPless_PARAM_DIR, 'param_sets_20Hz_TARPless_XUU0_t.txt')
path_t_tarp_set_20Hz= os.path.join(TARPed_PARAM_DIR, 'param_sets_20Hz_TARPed_XUU0_t.txt')
path_t_un_set_pois = os.path.join(TARPless_PARAM_DIR, 'param_sets_poisson_TARPless_XUU0_t.txt')
path_t_tarp_set_pois = os.path.join(TARPed_PARAM_DIR, 'param_sets_poisson_TARPed_XUU0_t.txt')
t_un_params_5Hz = pd.read_csv(path_t_un_set_5Hz, header=None).values.squeeze()
t_tarp_params_5Hz = pd.read_csv(path_t_tarp_set_5Hz, header=None).values.squeeze()
t_un_params_20Hz = pd.read_csv(path_t_un_set_20Hz, header=None).values.squeeze()
t_tarp_params_20Hz = pd.read_csv(path_t_tarp_set_20Hz, header=None).values.squeeze()
t_un_params_pois = pd.read_csv(path_t_un_set_pois, header=None).values.squeeze()
t_tarp_params_pois = pd.read_csv(path_t_tarp_set_pois, header=None).values.squeeze()
for x in range(30):
    # TARPless ensemble trace
    path_un_set_5Hz = os.path.join(TARPless_PARAM_DIR, f'param_sets_5Hz_TARPless_XUU0_{x}.txt')
    path_un_set_20Hz = os.path.join(TARPless_PARAM_DIR, f'param_sets_20Hz_TARPless_XUU0_{x}.txt')
    path_un_set_pois = os.path.join(TARPless_PARAM_DIR, f'param_sets_poisson_TARPless_XUU0_{x}.txt')
    if os.path.exists(path_un_set):
        trace_un_5Hz = pd.read_csv(path_un_set_5Hz, header=None).values.squeeze()
        trace_un_20Hz = pd.read_csv(path_un_set_20Hz, header=None).values.squeeze()
        trace_un_pois = pd.read_csv(path_un_set_pois, header=None).values.squeeze()
        axes[0,0].plot(t_un_params_5Hz, trace_un_5Hz, color=COLOR_ENSEMBLE, alpha=0.3, lw=0.8, zorder=1)
        axes[1,0].plot(t_un_params_20Hz, trace_un_20Hz, color=COLOR_ENSEMBLE, alpha=0.3, lw=0.8, zorder=1)
        axes[2,0].plot(t_un_params_pois, trace_un_pois, color=COLOR_ENSEMBLE, alpha=0.3, lw=0.8, zorder=1)

    # TARPed ensemble trace
    path_tarp_set_5Hz = os.path.join(TARPed_PARAM_DIR, f'param_sets_5Hz_TARPed_XUU0_{x}.txt')
    path_tarp_set_20Hz = os.path.join(TARPed_PARAM_DIR, f'param_sets_20Hz_TARPed_XUU0_{x}.txt')
    path_tarp_set_pois = os.path.join(TARPed_PARAM_DIR, f'param_sets_poisson_TARPed_XUU0_{x}.txt')
    if os.path.exists(path_un_set):
        trace_tarp_5Hz = pd.read_csv(path_tarp_set_5Hz, header=None).values.squeeze()
        trace_tarp_20Hz = pd.read_csv(path_tarp_set_20Hz, header=None).values.squeeze()
        trace_tarp_pois = pd.read_csv(path_tarp_set_pois, header=None).values.squeeze()
        axes[0,1].plot(t_tarp_params_5Hz, trace_tarp_5Hz, color=COLOR_ENSEMBLE, alpha=0.3, lw=0.8, zorder=1)
        axes[1,1].plot(t_tarp_params_20Hz, trace_tarp_20Hz, color=COLOR_ENSEMBLE, alpha=0.3, lw=0.8, zorder=1)
        axes[2,1].plot(t_tarp_params_pois, trace_tarp_pois, color=COLOR_ENSEMBLE, alpha=0.3, lw=0.8, zorder=1)
        

# Row 0, Col 0: TARPless 5Hz
axes[0, 0].plot(t_un_5hz, kr_un_5hz, color=COLOR_KR, lw=1.6, linestyle='-', zorder=2)
axes[0, 0].plot(t_un_5hz, red_un_5hz, color=COLOR_RED, lw=1.3, linestyle='--', zorder=3)

# Row 1, Col 0: TARPless 20Hz
axes[1, 0].plot(t_un_20hz, kr_un_20hz, color=COLOR_KR, lw=1.6, linestyle='-', zorder=2)
axes[1, 0].plot(t_un_20hz, red_un_20hz, color=COLOR_RED, lw=1.3, linestyle='--', zorder=3)

# Row 2, Col 0: TARPless Poisson
axes[2, 0].plot(t_un_pois, kr_un_pois, color=COLOR_KR, lw=1.6, linestyle='-', zorder=2)
axes[2, 0].plot(t_un_pois, red_un_pois, color=COLOR_RED, lw=1.3, linestyle='--', zorder=3)

# Row 0, Col 1: TARPed 5Hz
axes[0, 1].plot(t_tarp_5hz, kr_tarp_5hz, color=COLOR_KR, lw=1.6, linestyle='-', zorder=2)
axes[0, 1].plot(t_tarp_5hz, red_tarp_5hz, color=COLOR_RED, lw=1.3, linestyle='--', zorder=3)

# Row 1, Col 1: TARPed 20Hz
axes[1, 1].plot(t_tarp_20hz, kr_tarp_20hz, color=COLOR_KR, lw=1.6, linestyle='-', zorder=2)
axes[1, 1].plot(t_tarp_20hz, red_tarp_20hz, color=COLOR_RED, lw=1.3, linestyle='--', zorder=3)

# Row 2, Col 1: TARPedPoisson
axes[2, 1].plot(t_tarp_pois, kr_tarp_pois, color=COLOR_KR, lw=1.6, linestyle='-', zorder=2)
axes[2, 1].plot(t_tarp_pois, red_tarp_pois, color=COLOR_RED, lw=1.3, linestyle='--', zorder=3)

# Plot formatting

# Column Headers
axes[0, 0].set_title('TARPless', fontweight='bold', pad=12)
axes[0, 1].set_title('TARPed', fontweight='bold', pad=12)

# Y-Axis Row Labels
axes[0, 0].set_ylabel('5 Hz\nConductance', fontweight='bold')
axes[1, 0].set_ylabel('20 Hz\nConductance', fontweight='bold')
axes[2, 0].set_ylabel('Poisson\nConductance', fontweight='bold')

# X-Axis Labels (Bottom row)
axes[1, 0].set_xlabel('Time (ms)', fontweight='bold')
axes[1, 1].set_xlabel('Time (ms)', fontweight='bold')

# Axis limits & styling
for ax in axes.flat:
    ax.set_xlim(-100, 4000)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', color='#e0e0e0', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

legend_elements = [
    plt.Line2D([0], [0], color=COLOR_KR, lw=1.8, linestyle='-', label='KR model'),
    plt.Line2D([0], [0], color=COLOR_RED, lw=1.5, linestyle='--', label='Reduced model')
]

fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.06),
           ncol=2, frameon=False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

# Save figure
plt.savefig('Plots/train_comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()


# -------------------------------------------------------------------------
# Figure 6 - Recovery
# -------------------------------------------------------------------------

# --- Path Configuration ---
DATA_DIR = 'Data/Recovery'

# Load Data

isi_tarp = pd.read_csv(os.path.join(DATA_DIR, 'Recovery_ISIs_TARPed')).values.squeeze()
isi_un = pd.read_csv(os.path.join(DATA_DIR, 'Recovery_ISIs_TARPless')).values.squeeze()

kr_tarp = pd.read_csv(os.path.join(DATA_DIR, 'Recovery_KR_TARPed')).values.squeeze()
kr_un = pd.read_csv(os.path.join(DATA_DIR, 'Recovery_KR_TARPless')).values.squeeze()

red_tarp = pd.read_csv(os.path.join(DATA_DIR, 'Recovery_Red_TARPed')).values.squeeze()
red_un = pd.read_csv(os.path.join(DATA_DIR, 'Recovery_Red_TARPless')).values.squeeze()

# Plot Recovery Curves

fig, ax = plt.subplots(figsize=(7, 5))

# TARPless (Solid lines)
ax.plot(isi_un, kr_un, color=COLOR_KR, lw=1.8, linestyle='-', label='KR model, TARPless', zorder=2)
ax.plot(isi_un, red_un, color=COLOR_RED, lw=1.8, linestyle='-', label='Reduced model, TARPless', zorder=3)

# TARPed (Dashed lines)
ax.plot(isi_tarp, kr_tarp, color=COLOR_KR, lw=1.8, linestyle='--', label='KR model, TARPed', zorder=2)
ax.plot(isi_tarp, red_tarp, color=COLOR_RED, lw=1.8, linestyle='--', label='Reduced model, TARPed', zorder=3)

# Plot formatting

ax.set_title('Recovery', fontweight='bold', pad=12)
ax.set_xlabel('Inter-pulse interval (ms)', fontweight='bold')
ax.set_ylabel('Normalised peak recovery', fontweight='bold')

ax.set_xlim(0, 1250)
ax.set_ylim(-0.02, 1.08)

# Apply publication-style border clean up
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, linestyle='--', color='#e0e0e0', alpha=0.7, zorder=0)
ax.set_axisbelow(True)

# Clean legend positioned inside lower-right whitespace
ax.legend(loc='lower right', frameon=False)

plt.tight_layout()

# Save figure
plt.savefig('Plots/recovery_comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()


# -------------------------------------------------------------------------
# Figure 7 - Single neuron response
# -------------------------------------------------------------------------

DATA_DIR = 'Data/Neuron response'

# Frequencies for the two rows
freqs = [5, 20]

# Load x-axis ranges (file names remain the same as your directory structure)
amp_range = np.loadtxt(os.path.join(DATA_DIR, 'increasing_noise_amplitude/amplitude_range.csv'), delimiter=',')
var_range = np.loadtxt(os.path.join(DATA_DIR, 'increasing_noise_variance/variance_range.csv'), delimiter=',')

# Initialize a 2x3 Grid (2 rows for freq, 3 columns for metrics)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 7))

for row_idx, f in enumerate(freqs):
    
    # ---------------------------------------------------------
    # Column 0: Voltage Traces (Baseline/Normal Params)
    # ---------------------------------------------------------
    ax_volt = axes[row_idx, 0]
    
    volt_U = np.load(os.path.join(DATA_DIR, f'1Neur_{f}_Hz_2000_ms_TARPless_baseline_noise.npz'))
    volt_T = np.load(os.path.join(DATA_DIR, f'1Neur_{f}_Hz_2000_ms_TARPed_baseline_noise.npz'))
    
    # Plotting standard traces (using the 'Normal' medium tones)
    ax_volt.plot(volt_U['t'], volt_U['V'], color=tarpless_colors[1], alpha=0.8, linewidth=1.2)
    ax_volt.plot(volt_T['t'], volt_T['V'], color=tarped_colors[1], alpha=0.8, linewidth=1.2)
    
    ax_volt.set_ylabel(f'{f} Hz Stimulation\nVoltage (mV)', fontweight='bold')
    
    # ---------------------------------------------------------
    # Column 1: Amplitude (w) (+/- k Params)
    # ---------------------------------------------------------
    ax_amp = axes[row_idx, 1]
    
    amp_N   = np.loadtxt(os.path.join(DATA_DIR, f'increasing_noise_amplitude/firing_rates_no_input_{f}Hz.csv'), delimiter=',')
    
    amp_U   = np.loadtxt(os.path.join(DATA_DIR, f'increasing_noise_amplitude/firing_rates_TARPless_{f}Hz.csv'), delimiter=',')
    amp_U_D = np.loadtxt(os.path.join(DATA_DIR, f'increasing_noise_amplitude/firing_rates_TARPless_{f}Hz_decreased_k.csv'), delimiter=',')
    amp_U_I = np.loadtxt(os.path.join(DATA_DIR, f'increasing_noise_amplitude/firing_rates_TARPless_{f}Hz_increased_k.csv'), delimiter=',')
    
    amp_T   = np.loadtxt(os.path.join(DATA_DIR, f'increasing_noise_amplitude/firing_rates_TARPed_{f}Hz.csv'), delimiter=',')
    amp_T_D = np.loadtxt(os.path.join(DATA_DIR, f'increasing_noise_amplitude/firing_rates_TARPed_{f}Hz_decreased_k.csv'), delimiter=',')
    amp_T_I = np.loadtxt(os.path.join(DATA_DIR, f'increasing_noise_amplitude/firing_rates_TARPed_{f}Hz_increased_k.csv'), delimiter=',')
    
    # Plot TARPless
    ax_amp.errorbar(amp_range, (amp_U_D - amp_N).mean(axis=1), yerr=(amp_U_D - amp_N).std(axis=1), color=tarpless_colors[0], label=r'TARPless (decreased $k$)', capsize=2, elinewidth=1, alpha=0.9)
    ax_amp.errorbar(amp_range, (amp_U - amp_N).mean(axis=1),   yerr=(amp_U - amp_N).std(axis=1),   color=tarpless_colors[1], label=r'TARPless (default)', capsize=2, elinewidth=1, alpha=0.9)
    ax_amp.errorbar(amp_range, (amp_U_I - amp_N).mean(axis=1), yerr=(amp_U_I - amp_N).std(axis=1), color=tarpless_colors[2], label=r'TARPless (increased $k$)', capsize=2, elinewidth=1, alpha=0.9)
    
    # Plot TARPed
    ax_amp.errorbar(amp_range, (amp_T_D - amp_N).mean(axis=1), yerr=(amp_T_D - amp_N).std(axis=1), color=tarped_colors[0], label=r'TARPed (decreased $k$)', capsize=2, elinewidth=1, alpha=0.9)
    ax_amp.errorbar(amp_range, (amp_T - amp_N).mean(axis=1),   yerr=(amp_T - amp_N).std(axis=1),   color=tarped_colors[1], label=r'TARPed (default)', capsize=2, elinewidth=1, alpha=0.9)
    ax_amp.errorbar(amp_range, (amp_T_I - amp_N).mean(axis=1), yerr=(amp_T_I - amp_N).std(axis=1), color=tarped_colors[2], label=r'TARPed (increased $k$)', capsize=2, elinewidth=1, alpha=0.9)
    
    ax_amp.set_ylabel(r'$\Delta$ Firing Rate', fontweight='bold')
    
    # ---------------------------------------------------------
    # Column 2: Poisson Rate (lambda) (+/- k Params)
    # ---------------------------------------------------------
    ax_var = axes[row_idx, 2]
    
    var_N   = np.loadtxt(os.path.join(DATA_DIR, f'increasing_noise_variance/firing_rates_no_input_{f}Hz.csv'), delimiter=',')
    
    var_U   = np.loadtxt(os.path.join(DATA_DIR, f'increasing_noise_variance/firing_rates_TARPless_{f}Hz.csv'), delimiter=',')
    var_U_D = np.loadtxt(os.path.join(DATA_DIR, f'increasing_noise_variance/firing_rates_TARPless_{f}Hz_decreased_k.csv'), delimiter=',')
    var_U_I = np.loadtxt(os.path.join(DATA_DIR, f'increasing_noise_variance/firing_rates_TARPless_{f}Hz_increased_k.csv'), delimiter=',')
    
    var_T   = np.loadtxt(os.path.join(DATA_DIR, f'increasing_noise_variance/firing_rates_TARPed_{f}Hz.csv'), delimiter=',')
    var_T_D = np.loadtxt(os.path.join(DATA_DIR, f'increasing_noise_variance/firing_rates_TARPed_{f}Hz_decreased_k.csv'), delimiter=',')
    var_T_I = np.loadtxt(os.path.join(DATA_DIR, f'increasing_noise_variance/firing_rates_TARPed_{f}Hz_increased_k.csv'), delimiter=',')
    
    # Plot TARPless
    ax_var.errorbar(var_range, (var_U_D - var_N).mean(axis=1), yerr=(var_U_D - var_N).std(axis=1), color=tarpless_colors[0], capsize=2, elinewidth=1, alpha=0.9)
    ax_var.errorbar(var_range, (var_U - var_N).mean(axis=1),   yerr=(var_U - var_N).std(axis=1),   color=tarpless_colors[1], capsize=2, elinewidth=1, alpha=0.9)
    ax_var.errorbar(var_range, (var_U_I - var_N).mean(axis=1), yerr=(var_U_I - var_N).std(axis=1), color=tarpless_colors[2], capsize=2, elinewidth=1, alpha=0.9)
    
    # Plot TARPed
    ax_var.errorbar(var_range, (var_T_D - var_N).mean(axis=1), yerr=(var_T_D - var_N).std(axis=1), color=tarped_colors[0], capsize=2, elinewidth=1, alpha=0.9)
    ax_var.errorbar(var_range, (var_T - var_N).mean(axis=1),   yerr=(var_T - var_N).std(axis=1),   color=tarped_colors[1], capsize=2, elinewidth=1, alpha=0.9)
    ax_var.errorbar(var_range, (var_T_I - var_N).mean(axis=1), yerr=(var_T_I - var_N).std(axis=1), color=tarped_colors[2], capsize=2, elinewidth=1, alpha=0.9)
    
    # Plot Formatting per row/column
    for col_idx in range(3):
        ax = axes[row_idx, col_idx]
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Set x-labels only on the bottom row (UPDATED)
        if row_idx == 1:
            if col_idx == 0: ax.set_xlabel('Time (ms)', fontweight='bold')
            if col_idx == 1: ax.set_xlabel(r'Amplitude ($w$)', fontweight='bold')
            if col_idx == 2: ax.set_xlabel(r'Poisson Rate ($\lambda$)', fontweight='bold')
            
        # Set titles only on the top row (UPDATED)
        if row_idx == 0:
            if col_idx == 0: ax.set_title('Voltage Traces', fontweight='bold', pad=12)
            if col_idx == 1: ax.set_title(r'Amplitude ($w$)', fontweight='bold', pad=12)
            if col_idx == 2: ax.set_title(r'Poisson Rate ($\lambda$)', fontweight='bold', pad=12)


# Plot Formatting 
ylim_5Hz = axes[0, 0].get_ylim()
ylim_10Hz = axes[1, 0].get_ylim()

global_y_min = min(ylim_5Hz[0], ylim_10Hz[0])
global_y_max = max(ylim_5Hz[1], ylim_10Hz[1])

axes[0, 0].set_ylim(global_y_min, global_y_max)
axes[1, 0].set_ylim(global_y_min, global_y_max)


handles, labels = axes[0, 1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.25)

# Save figure
plt.savefig('Plots/multi_panel_voltage_noise_w_lambda.pdf', dpi=300, bbox_inches='tight')
plt.show()


# -------------------------------------------------------------------------
# Figure B1 - Varying glutmate concentration
# -------------------------------------------------------------------------

# --- Path Configuration ---
BASE_DIR = 'Data/Glutamate concentration'


concentrations = ['1mM', '5mM', '10mM']
conc_labels = ['0.1 mM', '0.5 mM', '1 mM']

dir_un = os.path.join(BASE_DIR, 'TARPless')
dir_tarp = os.path.join(BASE_DIR, 'TARPed')

fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex='col', sharey=True)

t_un = pd.read_csv(os.path.join(dir_un, '1ms_glu_TARPless_t.txt'), header=None).values.squeeze()
t_tarp = pd.read_csv(os.path.join(dir_tarp, '1ms_glu_TARPed_t.txt'), header=None).values.squeeze()

# Load and Plot Data across Concentrations (Cols) and Receptors (Rows)
for row_idx, conc in enumerate(concentrations):
    # --- Col 0: TARPless ---
    kr_un = pd.read_csv(os.path.join(dir_un, f'1ms_{conc}_glu_TARPless_KR.txt'), header=None).values.squeeze()
    red_un = pd.read_csv(os.path.join(dir_un, f'1ms_{conc}_glu_TARPless_red.txt'), header=None).values.squeeze()

    axes[row_idx, 0].plot(t_un, kr_un, color=COLOR_KR, lw=1.8, linestyle='-', zorder=2)
    axes[row_idx, 0].plot(t_un, red_un, color=COLOR_RED, lw=1.5, linestyle='--', zorder=3)

    # --- Col 1: TARPed ---
    kr_tarp = pd.read_csv(os.path.join(dir_tarp, f'1ms_{conc}_glu_TARPed_KR.txt'), header=None).values.squeeze()
    red_tarp = pd.read_csv(os.path.join(dir_tarp, f'1ms_{conc}_glu_TARPed_red.txt'), header=None).values.squeeze()

    axes[row_idx, 1].plot(t_tarp, kr_tarp, color=COLOR_KR, lw=1.8, linestyle='-', zorder=2)
    axes[row_idx, 1].plot(t_tarp, red_tarp, color=COLOR_RED, lw=1.5, linestyle='--', zorder=3)

    # Row Y-Axis Labels
    axes[row_idx, 0].set_ylabel(f'{conc_labels[row_idx]}\nConductance', fontweight='bold')

# Formatting
# Column Headers (Top row)
axes[0, 0].set_title('TARPless', fontweight='bold', pad=12)
axes[0, 1].set_title('TARPed', fontweight='bold', pad=12)

# Bottom Row X-Axis Labels
axes[2, 0].set_xlabel('Time (ms)', fontweight='bold')
axes[2, 1].set_xlabel('Time (ms)', fontweight='bold')

# Axis limits
axes[2, 0].set_xlim(-1, 50)
axes[2, 1].set_xlim(-10, 500)

axes[0, 0].set_ylim(0, 0.9)

# Panel styling
for ax in axes.flat:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', color='#e0e0e0', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

legend_elements = [
    plt.Line2D([0], [0], color=COLOR_KR, lw=1.8, linestyle='-', label='KR model'),
    plt.Line2D([0], [0], color=COLOR_RED, lw=1.6, linestyle='--', label='Reduced model')
]

fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.04),
           ncol=2, frameon=False)

plt.subplots_adjust(bottom=0.10)

plt.savefig('Plots/glutamate_concentration.pdf', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------------------------------------------------
# Figure B2 - Long application of glutamate
# -------------------------------------------------------------------------

# --- Path Configuration ---
DATA_DIR = 'Data/Long glutamate'

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)


# Load Data
# TARPless
t_un = pd.read_csv(os.path.join(DATA_DIR, '100ms_1mM_glu_TARPless_t.txt'), header=None).values.squeeze()
kr_un = pd.read_csv(os.path.join(DATA_DIR, '100ms_1mM_glu_TARPless_KR.txt'), header=None).values.squeeze()
red_un = pd.read_csv(os.path.join(DATA_DIR, '100ms_1mM_glu_TARPless_Red.txt'), header=None).values.squeeze()

# TARPed
t_tarp = pd.read_csv(os.path.join(DATA_DIR, '100ms_1mM_glu_TARPed_t.txt'), header=None).values.squeeze()
kr_tarp = pd.read_csv(os.path.join(DATA_DIR, '100ms_1mM_glu_TARPed_KR.txt'), header=None).values.squeeze()
red_tarp = pd.read_csv(os.path.join(DATA_DIR, '100ms_1mM_glu_TARPed_Red.txt'), header=None).values.squeeze()

# Plot Conductance Traces

# Panel 1: TARPless
axes[0].plot(t_un, kr_un, color=COLOR_KR, lw=1.8, linestyle='-', label='KR model', zorder=2)
axes[0].plot(t_un, red_un, color=COLOR_RED, lw=1.5, linestyle='--', label='Reduced model', zorder=3)

# Panel 2: TARPed
axes[1].plot(t_tarp, kr_tarp, color=COLOR_KR, lw=1.8, linestyle='-', zorder=2)
axes[1].plot(t_tarp, red_tarp, color=COLOR_RED, lw=1.52, linestyle='--', zorder=3)

# Formatting

axes[0].set_title('TARPless', fontweight='bold', pad=12)
axes[1].set_title('TARPed', fontweight='bold', pad=12)

axes[0].set_xlabel('Time (ms)', fontweight='bold')
axes[1].set_xlabel('Time (ms)', fontweight='bold')
axes[0].set_ylabel('Conductance', fontweight='bold')

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', color='#e0e0e0', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

legend_elements = [
    plt.Line2D([0], [0], color=COLOR_KR, lw=1.8, linestyle='-', label='KR model'),
    plt.Line2D([0], [0], color=COLOR_RED, lw=1.5, linestyle='--', label='Reduced model')
]

fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.08),
           ncol=2, frameon=False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.25)

plt.savefig('Plots/long_glutamate_comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------------------------------------------------
# Figure C1 and C2 - Paramater senstitivity
# -------------------------------------------------------------------------

DATA_DIR = 'Data/Neuron response'

# Configuration settings for both noise amplitude and noise variance figures
plot_configs = [
    {
        'folder': os.path.join(DATA_DIR, 'increasing_noise_amplitude'),
        'range_file': os.path.join(DATA_DIR, 'increasing_noise_amplitude/amplitude_range.csv'),
        'xlabel': 'Noise Amplitude',
        'output_file': 'Plots/multi_panel_amplitude.pdf'
    },
    {
        'folder': os.path.join(DATA_DIR, 'increasing_noise_variance'),
        'range_file': os.path.join(DATA_DIR, 'increasing_noise_variance/variance_range.csv'),
        'xlabel': 'Noise Variance',
        'output_file': 'Plots/multi_panel_variance.pdf'
    }
]

# Parameter mapping to LaTeX formatting
params = ['tauX', 'tauU', 'U0', 'k']
param_latex = {
    'tauX': r'$\tau_X$',
    'tauU': r'$\tau_U$',
    'U0': r'$U_0$',
    'k': r'$k$'
}

freq = [5, 10, 20]

for cfg in plot_configs:
    x_range = np.loadtxt(cfg['range_file'], delimiter=',')
    folder = cfg['folder']
    
    # Initialize 4x3 Grid (4 rows for params, 3 columns for frequencies)
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10), sharex=True, sharey=True)

    for col_idx, p in enumerate(params[:3]):
        p_label = param_latex[p]
        
        for row_idx, f in enumerate(freq):
            ax = axes[row_idx, col_idx]
            
            # File loading for current parameter and frequency
            data_N = np.loadtxt(f'{folder}/firing_rates_no_input_{f}Hz.csv', delimiter=',')
            
            data_U   = np.loadtxt(f'{folder}/firing_rates_TARPless_{f}Hz.csv', delimiter=',')
            data_U_D = np.loadtxt(f'{folder}/firing_rates_TARPless_{f}Hz_decreased_{p}.csv', delimiter=',')
            data_U_I = np.loadtxt(f'{folder}/firing_rates_TARPless_{f}Hz_increased_{p}.csv', delimiter=',')
            
            data_T   = np.loadtxt(f'{folder}/firing_rates_TARPed_{f}Hz.csv', delimiter=',')
            data_T_D = np.loadtxt(f'{folder}/firing_rates_TARPed_{f}Hz_decreased_{p}.csv', delimiter=',')
            data_T_I = np.loadtxt(f'{folder}/firing_rates_TARPed_{f}Hz_increased_{p}.csv', delimiter=',')
            
            # Calculate means and standard deviations
            u_d_m, u_d_s = (data_U_D - data_N).mean(axis=1), (data_U_D - data_N).std(axis=1)
            u_n_m, u_n_s = (data_U - data_N).mean(axis=1), (data_U - data_N).std(axis=1)
            u_i_m, u_i_s = (data_U_I - data_N).mean(axis=1), (data_U_I - data_N).std(axis=1)
            
            t_d_m, t_d_s = (data_T_D - data_N).mean(axis=1), (data_T_D - data_N).std(axis=1)
            t_n_m, t_n_s = (data_T - data_N).mean(axis=1), (data_T - data_N).std(axis=1)
            t_i_m, t_i_s = (data_T_I - data_N).mean(axis=1), (data_T_I - data_N).std(axis=1)
            
            # Plot TARPless curves (Blues)
            ax.errorbar(x_range, u_d_m, yerr=u_d_s, color=tarpless_colors[0], label='TARPless (decreased)', capsize=2, elinewidth=0.8, alpha=0.9)
            ax.errorbar(x_range, u_n_m, yerr=u_n_s, color=tarpless_colors[1], label='TARPless (default)', capsize=2, elinewidth=0.8, alpha=0.9)
            ax.errorbar(x_range, u_i_m, yerr=u_i_s, color=tarpless_colors[2], label='TARPless (increased)', capsize=2, elinewidth=0.8, alpha=0.9)
            
            # Plot TARPed curves (Oranges)
            ax.errorbar(x_range, t_d_m, yerr=t_d_s, color=tarped_colors[0], label='TARPed (decreased)', capsize=2, elinewidth=0.8, alpha=0.9)
            ax.errorbar(x_range, t_n_m, yerr=t_n_s, color=tarped_colors[1], label='TARPed (default)', capsize=2, elinewidth=0.8, alpha=0.9)
            ax.errorbar(x_range, t_i_m, yerr=t_i_s, color=tarped_colors[2], label='TARPed (increased)', capsize=2, elinewidth=0.8, alpha=0.9)
            
            # Panel Styling & Font Adjustments
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.tick_params(axis='both', which='major')
            
         
                
            # Row labels with LaTeX formatting (leftmost column only)
            if row_idx == 0:
                ax.set_title(f'{p_label}', fontweight='bold')
                
            # Column titles (top row only)
            if col_idx == 0:
                ax.set_ylabel(f'{f} Hz\n$\Delta$ Firing Rate', fontweight='bold')
                
            # X-axis labels (bottom row only)
            if row_idx == 2:
                ax.set_xlabel(cfg['xlabel'], fontweight='bold')

    # Shared bottom legend across 3 columns
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)
    plt.savefig(cfg['output_file'], dpi=300, bbox_inches='tight')
    plt.show()
    
 
# -------------------------------------------------------------------------
# Figure D1 - Network simulations
# -------------------------------------------------------------------------

# --- Path Configuration ---
DATA_DIR = 'Data/Network simulations'

# Load datasets
x = np.loadtxt(os.path.join(DATA_DIR, 'HH_network_reduced_model_run_times.csv') , delimiter=',')
y = np.loadtxt(os.path.join(DATA_DIR, 'HH_network_KR_model_run_times.csv'), delimiter=',')
x2 = np.loadtxt(os.path.join(DATA_DIR, 'HH_network_reduced_model_run_times_500.csv'), delimiter=',')
y2 = np.loadtxt(os.path.join(DATA_DIR, 'HH_network_KR_model_run_times_500.csv'), delimiter=',')


# Calculate mean and standard deviation per dataset first, then concatenate
reduced_mean = np.append(np.mean(x, axis=1), np.mean(x2))
reduced_std = np.append(np.std(x, axis=1), np.std(x2))

kr_mean = np.append(np.mean(y, axis=1), np.mean(y2))
kr_std = np.append(np.std(y, axis=1), np.std(y2))

x_vals = [10, 20, 50, 100, 200, 500]

# Generate plot
fig, ax = plt.subplots(figsize=(7, 5))

ax.errorbar(x_vals, reduced_mean, yerr=reduced_std, 
            fmt='-o', color='#E69F00', label='Reduced Model',
            capsize=4, capthick=1.5, elinewidth=1.5, linewidth=2, markersize=6)

ax.errorbar(x_vals, kr_mean, yerr=kr_std, 
            fmt='-s', color='#7E2F8E', label='KR Model',
            capsize=4, capthick=1.5, elinewidth=1.5, linewidth=2, markersize=6)


# Apply publication-style border clean up
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, linestyle='--', color='#e0e0e0', alpha=0.7, zorder=0)
ax.set_axisbelow(True)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Network Size', fontweight='bold')
ax.set_ylabel('Run Time (s)', fontweight='bold')
ax.set_xticks(x_vals)
ax.set_xticklabels([str(v) for v in x_vals])
ax.legend(frameon=False)
ax.grid(True, linestyle='--', alpha=0.4)


ax.legend(loc='upper left', frameon=False)

plt.tight_layout()

# Save figures
plt.savefig('Plots/network_run_times_log.pdf', format='pdf', dpi=300)
plt.show()

