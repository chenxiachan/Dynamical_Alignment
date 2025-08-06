import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import f as f_dist
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import json
import os
import glob
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, FancyArrowPatch



# Set publication-quality style
plt.style.use('default')
# plt.rcParams['font.family'] = 'Arial'
plt.rcParams.update({
    'font.size': 8,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'legend.frameon': False,
    'legend.fontsize': 7,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300
})

# Improved color palette, optimized for print
colors = {
    'dissipative': '#b3cde3',
    'chaotic': '#f7b8b8',
    'edge': '#1b7b1b',            
    'points': '#2c5d8c',          
    'fit_line': '#b22222',        
    'text': '#333333',
    'grid': '#e0e0e0',
}

# Curve fitting functions
def sigmoid(x, a, b, c, d):
    """Sigmoid function for S-curve fitting"""
    # Limit input to prevent overflow
    x_safe = np.clip(x, -100, 100)
    result = a / (1 + np.exp(-b * (x_safe - c))) + d
    return result

def generalized_logistic(x, a, b, c, d, q):
    """Generalized logistic/Richards function - more flexible S-curve"""
    # Limit input to prevent overflow
    x_safe = np.clip(x, -100, 100)
    # Use np.maximum to avoid negative or zero values
    exp_term = np.maximum(1e-10, 1 + q * np.exp(-c * (x_safe - d)))
    return a + (b - a) / (exp_term ** (1/q))

def inverse_sigmoid(x, a, b, c, d):
    """Inverse S-curve, for decreasing metrics"""
    # Limit input to prevent overflow
    x_safe = np.clip(x, -100, 100)
    result = a - a / (1 + np.exp(-b * (x_safe - c))) + d
    return result

def power_law(x, a, b, c=0):
    """Power law function: y = a * |x|^b + c"""
    return a * np.abs(x)**b + c

def piecewise_function(x, a1, b1, c1, a2, b2, c2, d2, breakpoint=-5):
    result = np.zeros_like(x)
    mask1 = x < breakpoint
    mask2 = ~mask1

    result[mask1] = a1 * x[mask1]**2 + b1 * x[mask1] + c1


    result[mask2] = a2 / (1 + np.exp(-b2 * (x[mask2] - c2))) + d2

    return result

def combined_sigmoid_quadratic(x, a, b, c, d, e, f):
    """
    y = a/(1 + np.exp(-b * (x - c))) + d + e*x + f*x^2"""
    return a/(1 + np.exp(-b * (x - c))) + d + e*x + f*x**2

# Function to load detailed run data
def load_detailed_run_data(detailed_runs_dir="detailed_runs"):
    """
    Load detailed run data for all experiments
    Returns a DataFrame with original data points
    """
    all_data = []

    # Find all run files
    run_files = glob.glob(os.path.join(detailed_runs_dir, "*_runs.json"))

    for run_file in run_files:
        with open(run_file, 'r') as f:
            experiment_runs = json.load(f)

        experiment_name = os.path.basename(run_file).replace("_runs.json", "")

        # Process each run
        for run in experiment_runs:
            # Load corresponding epoch history
            epoch_file = os.path.join(detailed_runs_dir, f"{experiment_name}_run{run['run']}_epochs.json")

            try:
                with open(epoch_file, 'r') as f:
                    epochs = json.load(f)

                # Get data at convergence epoch
                conv_epoch = min(run['convergence_epoch'], len(epochs)-1)  # Prevent out of bounds
                epoch_data = epochs[conv_epoch]

                # Add to global dataset
                all_data.append({
                    'Experiment': run['experiment'],
                    'Run': run['run'],
                    'Accuracy': run['best_accuracy'],
                    'Convergence': run['convergence_epoch'],
                    'Spikes': run['spikes_at_convergence'],
                    'Lyapunov Sum': run['lyapunov_sum'],
                    'Largest Lyapunov': run['largest_lyapunov'],
                    'delta': run['delta'],
                    'beta': run['beta'],
                    'alpha': run['alpha'],
                    'gamma': run['gamma'],
                })
            except Exception as e:
                print(f"Error loading epoch data for {epoch_file}: {e}")

    # Convert to DataFrame
    runs_df = pd.DataFrame(all_data)
    runs_df.to_csv('detailed_run_data_fig5.csv')

    # Calculate "Accuracy per Million Spikes" (efficiency metric)
    if 'Spikes' in runs_df.columns and runs_df['Spikes'].sum() > 0:
        runs_df['Energy_Efficiency'] = runs_df['Accuracy'] / (runs_df['Spikes'] / 1000000)

    # Validate data
    if len(runs_df) == 0:
        print("Warning: No detailed run data found! Using summary statistics file.")
        return None

    print(f"Successfully loaded {len(runs_df)} detailed runs")
    return runs_df

# Main function: Calculate fit statistics
def calculate_fit_statistics(x, y, func, initial_params=None):
    """
    Calculate fit statistics (R² and F-test p-value)

    Parameters:
    x, y: Input data arrays
    func: Fitting function
    initial_params: Initial parameters (optional)

    Returns:
    r2: R-squared value
    p_value: p-value from F-test
    popt: Fitted parameters
    """
    try:
        # Convert to numpy arrays
        x_np = np.array(x, dtype=float)
        y_np = np.array(y, dtype=float)

        # Sort by x value
        sort_idx = np.argsort(x_np)
        x_sorted = x_np[sort_idx]
        y_sorted = y_np[sort_idx]

        # Fit curve
        if initial_params is None:
            popt, _ = curve_fit(func, x_sorted, y_sorted, maxfev=10000)
        else:
            popt, _ = curve_fit(func, x_sorted, y_sorted, p0=initial_params, maxfev=10000)

        # Calculate predicted values
        y_pred = func(x_sorted, *popt)

        # Calculate R²
        ss_tot = np.sum((y_sorted - np.mean(y_sorted))**2)
        ss_res = np.sum((y_sorted - y_pred)**2)
        r2 = 1 - (ss_res / ss_tot)

        # Calculate F-statistic and p-value
        n = len(x_sorted)  # Sample size
        p_count = len(popt)  # Number of parameters

        # Calculate F-statistic
        if ss_res < 1e-10:  # Prevent division by zero
            f_stat = float('inf')
        else:
            f_stat = ((ss_tot - ss_res)/p_count) / (ss_res/(n-p_count-1))

        # Calculate p-value (F distribution)
        try:
            p_value = 1 - f_dist.cdf(f_stat, p_count, n-p_count-1)
        except:
            p_value = 0 if f_stat > 0 else 1

        return r2, p_value, popt

    except Exception as e:
        print(f"Statistics calculation failed: {e}")
        return 0, 1, None

def load_information_dynamics_data(filename="information_dynamics_results.csv"):
    try:
        try:
            info_data = pd.read_csv(filename)
        except:
            info_data = pd.read_csv(filename, delim_whitespace=True)

        print(f"Successfully loaded information dynamics data with {len(info_data)} rows")
        return info_data
    except Exception as e:
        print(f"Failed to load information dynamics data: {e}")
        return None

# Function to add fit curve
def add_fit_curve(ax, x, y, r2, p_value, func=None, color=colors['fit_line'],
                 linestyle='-', linewidth=2.5, alpha=1.0, initial_params=None, x_range=None, popt=None, zorder=10):
    """Add fitted curve to the plot, using pre-calculated statistics"""
    try:
        # Convert to numpy arrays
        x_np = np.array(x, dtype=float)
        y_np = np.array(y, dtype=float)

        # Sort by x value
        sort_idx = np.argsort(x_np)
        x_sorted = x_np[sort_idx]
        y_sorted = y_np[sort_idx]

        # Determine line style based on p-value
        is_significant = p_value < 0.05
        if not is_significant:
            linestyle = ':'
            alpha = alpha * 0.7  # Reduce opacity

        # If no function provided, use cubic polynomial
        if func is None:
            if popt is None:
                z = np.polyfit(x_sorted, y_sorted, 3)
            else:
                z = popt

            p = np.poly1d(z)

            # Create smooth curve for plotting
            if x_range is None:
                x_smooth = np.linspace(min(x_sorted), max(x_sorted), 100)
            else:
                x_smooth = np.linspace(x_range[0], x_range[1], 100)

            y_smooth = p(x_smooth)

            # Plot fitted curve with zorder
            ax.plot(x_smooth, y_smooth, color=color, linestyle=linestyle,
                   linewidth=linewidth, alpha=alpha, label='Polynomial Fit', zorder=zorder)
        else:
            # Fit using provided function
            if popt is None:
                if initial_params is None:
                    popt, _ = curve_fit(func, x_sorted, y_sorted, maxfev=10000)
                else:
                    popt, _ = curve_fit(func, x_sorted, y_sorted, p0=initial_params, maxfev=10000)

            # Create smooth curve for plotting
            if x_range is None:
                x_smooth = np.linspace(min(x_sorted) - 0.2, max(x_sorted) + 0.2, 1000)
            else:
                x_smooth = np.linspace(x_range[0], x_range[1], 1000)

            y_smooth = func(x_smooth, *popt)

            # Filter points within range, avoid extrapolation issues
            if x_range is not None:
                valid_points = (x_smooth >= x_range[0]) & (x_smooth <= x_range[1])
                x_smooth = x_smooth[valid_points]
                y_smooth = y_smooth[valid_points]

            # Plot fitted curve with zorder
            ax.plot(x_smooth, y_smooth, color=color, linestyle=linestyle,
                   linewidth=linewidth, alpha=alpha, label='Fitted Curve', zorder=zorder)

    except Exception as e:
        print(f"Failed to plot fit curve: {e}")

# Configure Lyapunov axis
def setup_largest_lyapunov_axis(ax, x_values):
    """Configure axis for Largest Lyapunov plot"""
    min_x = min(x_values)
    max_x = max(x_values)
    margin = (max_x - min_x) * 0.1  # 10% margin

    # Ensure range includes -1
    if min_x > -1:
        min_x = -1

    ax.set_xlim(min_x - margin, max_x + margin)

    # Set specific tick positions for cleaner look
    ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
    ax.set_xticks(ticks)

    # Ensure all tick labels are displayed and properly formatted
    ax.set_xticklabels([f'{t:.1f}' for t in ticks])

    # Apply grid
    ax.grid(True, linestyle='--', alpha=0.3, color=colors['grid'])

    return ax

# Broken axis plot function
def create_broken_axis_plot(fig, gs, row, col, data, x_col, y_col,
                           x_label, y_label, title, r2, p_value, popt=None, fit_func=None,
                           initial_params=None, panel_label=None,
                           highlight_critical=True, critical_value=0,
                           linestyle='-', linewidth=2.5, alpha=1.0,
                           y_limits=None, text_on_left=False, text_position="top-left",
                           text_align="left", y_scale=1.0,
                           highlight_ordered_chaos=False,
                           beta_pos=None, beta_neg=None):  # Added beta parameters
    """Create broken axis plot using raw data points rather than means and std devs"""
    # Create two sub-grids with different widths
    gs_sub = GridSpecFromSubplotSpec(1, 6, subplot_spec=gs[row, col], wspace=0.05)

    # Create two axes with unequal widths (1:5 ratio)
    ax_left = fig.add_subplot(gs_sub[0, 0:2])
    ax_right = fig.add_subplot(gs_sub[0, 2:])

    # Create a hidden spanning axis for common title and x-label
    ax_span = fig.add_subplot(gs_sub[:], frameon=False)
    ax_span.set_xticks([])
    ax_span.set_yticks([])

    # Get X and Y data
    x_values = data[x_col].values
    y_values = data[y_col].values

    # Calculate means and stds for each experiment
    experiments = data['Experiment'].unique()
    exp_means = []
    exp_stds = []
    exp_x = []

    for exp in experiments:
        exp_data = data[data['Experiment'] == exp]
        exp_x.append(exp_data[x_col].iloc[0])
        # Apply scaling to mean and std values
        exp_means.append(exp_data[y_col].mean() / y_scale)
        exp_stds.append(exp_data[y_col].std() / y_scale)

    # Convert to numpy arrays
    exp_x = np.array(exp_x)
    exp_means = np.array(exp_means)
    exp_stds = np.array(exp_stds)

    # Split data into regions
    left_mask = exp_x < -4
    right_mask = exp_x >= -4

    # Left plot data
    x_left = exp_x[left_mask]
    y_left = exp_means[left_mask]
    yerr_left = exp_stds[left_mask]

    # Right plot data
    x_right = exp_x[right_mask]
    y_right = exp_means[right_mask]
    yerr_right = exp_stds[right_mask]

    # Plot error bars for each region
    ax_left.errorbar(x_left, y_left, yerr=yerr_left,
                   fmt='o', color=colors['points'], ecolor=colors['points'],
                   capsize=4, elinewidth=1.5, markeredgewidth=1, markersize=7, alpha=0.7)

    ax_right.errorbar(x_right, y_right, yerr=yerr_right,
                    fmt='o', color=colors['points'], ecolor=colors['points'],
                    capsize=4, elinewidth=1.5, markeredgewidth=1, markersize=7, alpha=0.7)

    # Add fit curves to both plots, with specific ranges
    if fit_func is not None:
        # Use pre-calculated statistics to determine line style and opacity
        add_fit_curve(ax_left, x_values, y_values, r2, p_value,
                     func=fit_func, initial_params=initial_params,
                     x_range=[-22, -4], linestyle=linestyle,
                     linewidth=linewidth, alpha=alpha, popt=popt)

        # Use same fit parameters for right side
        add_fit_curve(ax_right, x_values, y_values, r2, p_value,
                    func=fit_func, initial_params=initial_params,
                    x_range=[-4, 3.5], linestyle=linestyle,
                    linewidth=linewidth, alpha=alpha, popt=popt)

    # Format p-value for display
    if p_value < 0.001:
        p_text = "p < 0.001"
    elif p_value < 0.05:
        p_text = f"p = {p_value:.3f}*"  # Add asterisk for significant p-values
    else:
        p_text = f"p = {p_value:.3f} (n.s.)"  # Mark as not significant

    # Determine which plot to add text to
    text_ax = ax_left if text_on_left else ax_right

    # Determine text position and alignment based on text_position parameter
    if text_position == "bottom-left":
        y_pos = 0.05
        vert_align = 'bottom'
    else:  # Default to top-left
        y_pos = 0.95
        vert_align = 'top'

    # Determine horizontal position and alignment based on text_align parameter
    if text_align == "right":
        x_pos = 0.95
        horiz_align = 'right'
    else:  # Default to left
        x_pos = 0.05
        horiz_align = 'left'

    # Add R² and p-value text to specified plot with specified position and alignment
    text_ax.text(
        x_pos, y_pos,
        f"R² = {r2:.3f}\n{p_text}",
        transform=text_ax.transAxes,
        fontsize=9,
        verticalalignment=vert_align,
        horizontalalignment=horiz_align,
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
    )

    # Add panel label if provided
    if panel_label:
        ax_left.text(
            -0.15, 1.05,
            panel_label,
            transform=ax_left.transAxes,
            fontsize=13,
            fontweight='bold'
        )

    # Add vertical line at critical value and shaded regions if requested
    if highlight_critical:
        # Only add critical line to right plot (if it's in that range)
        if critical_value >= -4:
            ax_right.axvline(x=critical_value, color=colors['edge'], linestyle='--', linewidth=1.5,
                           alpha=0.8, label=f'Critical Point (λ = 0)')
            ax_right.axvspan(-4, critical_value, alpha=0.15, color=colors['dissipative'],
                           label='Dissipative Region')
            ax_right.axvspan(critical_value, 3.5, alpha=0.15, color=colors['chaotic'],
                           label='Chaotic Region')

        # Add shading to left plot (all dissipative)
        ax_left.axvspan(-22, -4, alpha=0.15, color=colors['dissipative'])

        # Highlight "Ordered Chaos" window (if requested)
        if highlight_ordered_chaos and x_col == 'Lyapunov Sum':
            ax_right.axvspan(-0.15, 0.15, alpha=0.35, color=colors['ordered_chaos'],
                            label='Ordered Chaos Window')
        elif highlight_ordered_chaos and x_col == 'Largest Lyapunov':
            ax_right.axvspan(0, 0.3, alpha=0.35, color=colors['ordered_chaos'],
                            label='Ordered Chaos Window')

    # Set x limits for each axis
    ax_left.set_xlim(-22, -4)
    ax_right.set_xlim(-4, 3.5)

    # Set y limits - custom or auto-calculated
    if y_limits is not None:
        # Use custom y limits if provided
        y_min, y_max = y_limits
        ax_left.set_ylim(y_min, y_max)
        ax_right.set_ylim(y_min, y_max)
    else:
        # Otherwise calculate from data
        all_y = np.concatenate([y_left, y_right])
        all_err = np.concatenate([yerr_left, yerr_right])
        y_min = min(all_y - all_err) * 0.95
        y_max = max(all_y + all_err) * 1.05

        ax_left.set_ylim(y_min, y_max)
        ax_right.set_ylim(y_min, y_max)

    # Set tick positions for each axis
    ax_left.set_xticks([-20, -10])
    ax_right.set_xticks([-4, -2, 0, 2])

    # Remove y tick labels from right plot to avoid duplication
    ax_right.set_yticklabels([])

    # Add break symbols between plots
    d = .015  # Size of break symbol
    kwargs = dict(transform=ax_left.transAxes, color='gray', clip_on=False, lw=1.5)
    ax_left.plot((1-d, 1+d), (-d, +d), **kwargs)  # Bottom right diagonal
    ax_left.plot((1-d, 1+d), (1-d, 1+d), **kwargs)  # Top right diagonal

    kwargs.update(transform=ax_right.transAxes)
    ax_right.plot((-d, +d), (-d, +d), **kwargs)  # Bottom left diagonal
    ax_right.plot((-d, +d), (1-d, 1+d), **kwargs)  # Top left diagonal

    # Set labels and title
    # Only add y label to left plot
    ax_left.set_ylabel(y_label, fontsize=10, fontweight='medium')

    # Use spanning axis to add common x label
    ax_span.set_xlabel(x_label, fontsize=10, fontweight='medium', labelpad=15)

    # Remove individual x labels from subplots
    ax_left.set_xlabel('')
    ax_right.set_xlabel('')

    # Add grid to both plots
    ax_left.grid(True, linestyle='--', alpha=0.3, color=colors['grid'])
    ax_right.grid(True, linestyle='--', alpha=0.3, color=colors['grid'])

    # Add power law exponent annotations if provided
    if beta_pos is not None and beta_neg is not None:
        # Position the beta+ annotation to avoid overlap
        if y_col == 'Spikes':
            # Chaotic region power law exponent (beta+)
            ax_right.text(0.7, 0.8, r"$\beta_{+} = " + f"{beta_pos}" + r"$", transform=ax_right.transAxes,
                         fontsize=9, fontweight='bold', color='darkred',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            # Dissipative region power law exponent (beta-)
            ax_right.text(0.2, 0.8, r"$\beta_{-} = " + f"{beta_neg}" + r"$", transform=ax_right.transAxes,
                         fontsize=9, fontweight='bold', color='darkblue',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        elif y_col == 'Energy_Efficiency':
            # Position differently for Energy Efficiency plot
            ax_right.text(0.7, 0.2, r"$\beta_{+} = " + f"{beta_pos}" + r"$", transform=ax_right.transAxes,
                         fontsize=9, fontweight='bold', color='darkred',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            ax_right.text(0.2, 0.8, r"$\beta_{-} = " + f"{beta_neg}" + r"$", transform=ax_right.transAxes,
                         fontsize=9, fontweight='bold', color='darkblue',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        elif y_col == 'Accuracy':
            # Position differently for Accuracy plot
            ax_right.text(0.7, 0.8, r"$\beta_{+} = " + f"{beta_pos}" + r"$", transform=ax_right.transAxes,
                         fontsize=9, fontweight='bold', color='darkred',
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            ax_right.text(0.2, 0.2, r"$\beta_{-} = " + f"{beta_neg}" + r"$", transform=ax_right.transAxes,
                         fontsize=9, fontweight='bold', color='darkblue',
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    return ax_left, ax_right

# Single axis plot function
def create_single_axis_plot(fig, gs, row, col, data, x_col, y_col,
                          x_label, y_label, title, r2, p_value, popt=None, fit_func=None,
                          initial_params=None, panel_label=None,
                          highlight_critical=True, critical_value=0,
                          linestyle='-', linewidth=2.5, alpha=1.0,
                          y_limits=None, text_position="top-left", y_scale=1.0,
                          highlight_ordered_chaos=False,
                          beta_pos=None, beta_neg=None,
                          color_by=None):  # 添加参数用于指定颜色映射列
    """Create single axis plot using raw data points"""
    ax = fig.add_subplot(gs[row, col])

    # Get X and Y data
    x_values = data[x_col].values
    y_values = data[y_col].values

    # Calculate means and stds for each experiment for error bars
    experiments = data['Experiment'].unique()
    exp_means = []
    exp_stds = []
    exp_x = []
    color_values = []  # 用于存储颜色值

    for exp in experiments:
        exp_data = data[data['Experiment'] == exp]
        exp_x.append(exp_data[x_col].iloc[0])
        # Apply scaling to means and stds
        exp_means.append(exp_data[y_col].mean() / y_scale)
        exp_stds.append(exp_data[y_col].std() / y_scale)

        # 如果指定了color_by参数，并且该列存在于数据中，则获取该值
        if color_by is not None and color_by in exp_data.columns:
            color_values.append(exp_data[color_by].iloc[0])
        else:
            color_values.append(0)  # 默认值

    # Convert to numpy arrays
    exp_x = np.array(exp_x)
    exp_means = np.array(exp_means)
    exp_stds = np.array(exp_stds)
    color_values = np.array(color_values)

    # 如果有颜色值，则使用散点图，颜色由color_by列确定
    if color_by is not None and color_by in data.columns:
        # 创建散点图
        sc = ax.scatter(exp_x, exp_means, c=color_values, cmap='viridis',
                      s=55, alpha=0.6, zorder=10)  # 提高zorder使点在线上方

        # 添加误差线
        for i in range(len(exp_x)):
            ax.errorbar(exp_x[i], exp_means[i], yerr=exp_stds[i],
                       fmt='none', ecolor='gray', capsize=4,
                       elinewidth=1.0, alpha=0.5, zorder=5)

        # 添加简化的颜色条（减少刻度数量）
        cbar = fig.colorbar(sc, ax=ax, pad=0.025, aspect=30)

        # 设置更少的刻度数量（只显示最小值、中间值和最大值）
        min_val = min(color_values)
        max_val = max(color_values)
        mid_val = (min_val + max_val) / 2
        cbar.set_ticks([min_val, max_val])
        cbar.set_ticklabels([f'{min_val:.0f}', f'{max_val:.0f}'])

        # cbar.set_label('Average Active Information Storage', fontsize=8)
        cbar.set_label('Average Active Information\nStorage (AIS)', fontsize=9, labelpad=-1)
        cbar.ax.tick_params(labelsize=8)
    else:
        # 如果没有指定颜色列，则使用原始的errorbar方法
        ax.errorbar(exp_x, exp_means, yerr=exp_stds,
                   fmt='o', color=colors['points'], ecolor=colors['points'],
                   capsize=4, elinewidth=1.5, markeredgewidth=1, markersize=7, alpha=0.6,
                   zorder=3)

    # Add fit curve
    if fit_func is not None:
        # Use pre-calculated statistics to determine line style and opacity
        add_fit_curve(ax, x_values, y_values / y_scale, r2, p_value,  # Scale y values
                     func=fit_func, initial_params=initial_params,
                     linestyle=linestyle, linewidth=linewidth, alpha=alpha, popt=popt,
                     zorder=5)  # 设置zorder使曲线在点下方

    # Format p-value for display
    if p_value < 0.001:
        p_text = "p < 0.001"
    elif p_value < 0.05:
        p_text = f"p = {p_value:.3f}*"  # Add asterisk for significant p-values
    else:
        p_text = f"p = {p_value:.3f} (n.s.)"  # Mark as not significant

    # Determine text position and alignment based on text_position parameter
    if "bottom" in text_position:
        y_pos = 0.05
        vert_align = 'bottom'
    else:  # Default to top
        y_pos = 0.95
        vert_align = 'top'

    # Determine horizontal position and alignment
    if "right" in text_position:
        x_pos = 0.95
        horiz_align = 'right'
    else:  # Default to left
        x_pos = 0.05
        horiz_align = 'left'

    # Add R² and p-value text with specified position
    ax.text(
        x_pos, y_pos,
        f"R² = {r2:.3f}\n{p_text}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment=vert_align,
        horizontalalignment=horiz_align,
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
    )

    # Add panel label if provided
    if panel_label:
        ax.text(
            -0.15, 1.05,
            panel_label,
            transform=ax.transAxes,
            fontsize=13,
            fontweight='bold'
        )

    # Add vertical line at critical value and shaded regions if requested
    if highlight_critical:
        ax.axvline(x=critical_value, color=colors['edge'], linestyle='--', linewidth=1.5,
                  alpha=0.8, label=f'Critical Point (λ = 0)')
        ax.axvspan(min(exp_x) - 1, critical_value, alpha=0.15, color=colors['dissipative'],
                  label='Dissipative Region')
        ax.axvspan(critical_value, max(exp_x) + 1, alpha=0.15, color=colors['chaotic'],
                  label='Chaotic Region')

        # Highlight "Ordered Chaos" window (if requested)
        if highlight_ordered_chaos and x_col == 'Largest Lyapunov':
            ax.axvspan(0, 0.3, alpha=0.35, color=colors['ordered_chaos'],
                      label='Ordered Chaos Window')

    # Configure axis layout based on which type of plot it is
    if x_col == 'Largest Lyapunov':
        # Use the existing function for Largest Lyapunov plots
        setup_largest_lyapunov_axis(ax, exp_x)
    else:
        # For Lyapunov Sum plots, use a custom configuration
        min_x = min(exp_x)
        max_x = max(exp_x)
        margin = (max_x - min_x) * 0.05  # 5% margin

        ax.set_xlim(min_x - margin, max_x + margin)

        # Set appropriate ticks for Lyapunov Sum plots
        if min_x < -15:  # For wide range plots (like in broken axis case)
            ticks = [-20, -15, -10, -5, 0, 1, 2, 3]
        else:  # For narrower range plots
            ticks = [-4, -2, 0, 2]

        ax.set_xticks(ticks)
        ax.grid(True, linestyle='--', alpha=0.3, color=colors['grid'])

    # Set y limits if provided
    if y_limits is not None:
        y_min, y_max = y_limits
        ax.set_ylim(y_min, y_max)

    if y_col == 'Convergence':
        ax.set_yticks([5, 10, 15, 20])
        ax.set_yticklabels(['5', '10', '15', '20'])  
    elif y_col == 'Spikes':
        ax.set_yticks([4, 8, 12, 16, 20])
        ax.set_yticklabels(['4', '8', '12', '16','20'])  

    # Add labels and title
    ax.set_xlabel(x_label, fontsize=10, fontweight='medium')
    ax.set_ylabel(y_label, fontsize=10, fontweight='medium')

    # Add power law exponent annotations if requested
    if beta_pos is not None and beta_neg is not None and y_col == 'Spikes' and panel_label == 'A':
        # Chaotic region power law exponent
        ax.text(0.7, 0.8, r"$\beta_{+} = " + f"{beta_pos}" + r"$", transform=ax.transAxes,
               fontsize=9, fontweight='bold', color='darkred',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        # Dissipative region power law exponent
        ax.text(0.2, 0.8, r"$\beta_{-} = " + f"{beta_neg}" + r"$", transform=ax.transAxes,
               fontsize=9, fontweight='bold', color='darkblue',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    return ax

# Main visualization function
def create_visualization(data):
    """Create main visualization using detailed run data"""
    # Create publication-quality sized figure
    fig = plt.figure(figsize=(9, 5))  # Wider figure for 2x3 layout

    # Create 2x3 GridSpec layout instead of 4x2
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.2, 1, 1], wspace=0.35, hspace=0.37)

    # Verify data is different
    print("\nData validation:")

    lyapunov_sum = data['Lyapunov Sum'].unique()
    largest_lyapunov = data['Largest Lyapunov'].unique()

    print(f"Unique Lyapunov Sum values: {len(lyapunov_sum)}")
    print(f"Unique Largest Lyapunov values: {len(largest_lyapunov)}")

    is_identical = np.array_equal(sorted(lyapunov_sum), sorted(largest_lyapunov))
    print(f"Data completely identical: {is_identical}")

    # Pre-calculate all required statistics - using raw data points
    print("\nCalculating statistics:")

    # 1. Lyapunov Sum - Spike Activity
    r2_sum_spikes, p_sum_spikes, popt_sum_spikes = calculate_fit_statistics(
        data['Lyapunov Sum'], data['Spikes'] / 1000000,
        generalized_logistic, [
            (data['Spikes'] / 1000000).min(),
            (data['Spikes'] / 1000000).max() - (data['Spikes'] / 1000000).min(),
            5, 0.28, 1
        ]
    )

    # 2. Lyapunov Sum - Accuracy
    # r2_sum_acc, p_sum_acc, popt_sum_acc = calculate_fit_statistics(
    #     data['Lyapunov Sum'], data['Accuracy'],
    #     sigmoid, [
    #         data['Accuracy'].max() - data['Accuracy'].min(),
    #         5, 0, data['Accuracy'].min()
    #     ]
    # )
    r2_sum_acc, p_sum_acc, popt_sum_acc = calculate_fit_statistics(
        data['Lyapunov Sum'], data['Accuracy'],
        combined_sigmoid_quadratic, [5, 5, 0, 90, -0.1, 0.01]
    )

    # 3. Lyapunov Sum - Convergence
    r2_sum_conv, p_sum_conv, popt_sum_conv = calculate_fit_statistics(
        data['Lyapunov Sum'], data['Convergence'],
        inverse_sigmoid, [
            data['Convergence'].max(),
            5, 0, data['Convergence'].min()
        ]
    )

    # 4. Largest Lyapunov - Spike Activity
    r2_largest_spikes, p_largest_spikes, popt_largest_spikes = calculate_fit_statistics(
        data['Largest Lyapunov'], data['Spikes'] / 1000000,
        generalized_logistic, [
            (data['Spikes'] / 1000000).min(),
            (data['Spikes'] / 1000000).max() - (data['Spikes'] / 1000000).min(),
            8, 0.3, 1
        ]
    )

    # 5. Largest Lyapunov - Accuracy
    r2_largest_acc, p_largest_acc, popt_largest_acc = calculate_fit_statistics(
        data['Largest Lyapunov'], data['Accuracy'],
        sigmoid, [
            data['Accuracy'].max() - data['Accuracy'].min(),
            8, 0.3, data['Accuracy'].min()
        ]
    )

    # 6. Largest Lyapunov - Convergence
    r2_largest_conv, p_largest_conv, popt_largest_conv = calculate_fit_statistics(
        data['Largest Lyapunov'], data['Convergence'],
        inverse_sigmoid, [
            data['Convergence'].max(),
            8, 0.3, data['Convergence'].min()
        ]
    )

    # Print calculated statistics
    print("\nCalculated Statistics:")
    print(f"Lyapunov Sum vs Spike Activity: R² = {r2_sum_spikes:.3f}, p = {p_sum_spikes:.6f}")
    print(f"Lyapunov Sum vs Accuracy: R² = {r2_sum_acc:.3f}, p = {p_sum_acc:.6f}")
    print(f"Lyapunov Sum vs Convergence: R² = {r2_sum_conv:.3f}, p = {p_sum_conv:.6f}")
    print(f"Largest Lyapunov vs Spike Activity: R² = {r2_largest_spikes:.3f}, p = {p_largest_spikes:.6f}")
    print(f"Largest Lyapunov vs Accuracy: R² = {r2_largest_acc:.3f}, p = {p_largest_acc:.6f}")
    print(f"Largest Lyapunov vs Convergence: R² = {r2_largest_conv:.3f}, p = {p_largest_conv:.6f}")

    plot_configs = [
    {
        'row': 0, 'col': 0, 'is_broken': False,
        'x_col': 'Lyapunov Sum',
        'y_col': 'Accuracy',
        'x_label': 'Lyapunov Sum (Σλᵢ)',
        'y_label': 'Accuracy (%)',
        'title': 'Classification Accuracy',
        'fit_func': combined_sigmoid_quadratic,  
        'initial_params': [
            5, 5, 0, 90, -0.1, 0.01  
        ],
        'panel_label': 'A',
        'linestyle': '-',
        'linewidth': 2.5,
        'alpha': 1.0,
        'y_limits': (90, 95.9),
        'text_position': "top-left",
        'r2': r2_sum_acc,
        'p_value': p_sum_acc,
        'popt': popt_sum_acc,
        'highlight_ordered_chaos': False,
        'color_by': 'avg_ais',  
    },
    {
        'row': 0, 'col': 2, 'is_broken': False,  
        'x_col': 'Lyapunov Sum',
        'y_col': 'Convergence',
        'x_label': 'Lyapunov Sum (Σλᵢ)',
        'y_label': 'Convergence Epochs',  
        'title': 'Learning Convergence',
        'fit_func': inverse_sigmoid,
        'initial_params': [
            data['Convergence'].max(),
            5, 0, data['Convergence'].min()
        ],
        'panel_label': 'C',
        'linestyle': '-',
        'linewidth': 2.5,
        'alpha': 1.0,
        'y_limits': None,
        'text_position': "bottom-left",
        'r2': r2_sum_conv,
        'p_value': p_sum_conv,
        'popt': popt_sum_conv,
        'highlight_ordered_chaos': False
    },
    {
        'row': 0, 'col': 1, 'is_broken': False,  
        'x_col': 'Lyapunov Sum',
        'y_col': 'Spikes',
        'y_scale': 1000000,  # Convert to millions
        'x_label': 'Lyapunov Sum (Σλᵢ)',
        'y_label': 'Spike Count (millions)',
        'title': 'Spike Activity',
        'fit_func': generalized_logistic,
        'initial_params': [
            (data['Spikes'] / 1000000).min(),
            (data['Spikes'] / 1000000).max() - (data['Spikes'] / 1000000).min(),
            5, 0.28, 1
        ],
        'panel_label': 'B',
        'linestyle': '-',
        'linewidth': 2.5,
        'alpha': 1.0,
        'y_limits': (2, 21),
        'text_position': "top-left",  
        'r2': r2_sum_spikes,
        'p_value': p_sum_spikes,
        'popt': popt_sum_spikes,
        'highlight_ordered_chaos': False,
    },
    # Row 2: Largest Lyapunov plots - Rearranged: E, F, D
    {
        'row': 1, 'col': 0, 'is_broken': False,
        'x_col': 'Largest Lyapunov',
        'y_col': 'Accuracy',
        # 'x_label': 'Lyapunov Max ($\\mathbf{λ_{max}}$)',
        'x_label': 'Lyapunov Max ($λ_{max}$)',
        'y_label': 'Accuracy (%)',
        'title': 'Classification Accuracy',
        'fit_func': sigmoid,
        'initial_params': [
            data['Accuracy'].max() - data['Accuracy'].min(),
            8, 0.3, data['Accuracy'].min()
        ],
        'panel_label': 'D',
        'linestyle': '-',
        'linewidth': 2.5,
        'alpha': 1.0,
        'y_limits': (90, 95.9),
        'text_position': "top-left",
        'r2': r2_largest_acc,
        'p_value': p_largest_acc,
        'popt': popt_largest_acc,
        'highlight_ordered_chaos': False,
        'color_by': 'avg_ais', 
    },
    {
        'row': 1, 'col': 2, 'is_broken': False,
        'x_col': 'Largest Lyapunov',
        'y_col': 'Convergence',
        # 'x_label': 'Lyapunov Max ($\\mathbf{λ_{max}}$)',
        'x_label': 'Lyapunov Max ($λ_{max}$)',
        'y_label': 'Convergence Epochs',
        'title': 'Learning Convergence',
        'fit_func': inverse_sigmoid,
        'initial_params': [
            data['Convergence'].max(),
            8, 0.3, data['Convergence'].min()
        ],
        'panel_label': 'F',
        'linestyle': '-',
        'linewidth': 2.5,
        'alpha': 1.0,
        'y_limits': None,
        'text_position': "top-right",
        'r2': r2_largest_conv,
        'p_value': p_largest_conv,
        'popt': popt_largest_conv,
        'highlight_ordered_chaos': False
    },
    {
        'row': 1, 'col': 1, 'is_broken': False,
        'x_col': 'Largest Lyapunov',
        'y_col': 'Spikes',
        'y_scale': 1000000,  # Convert to millions
        # 'x_label': 'Lyapunov Max ($\\mathbf{λ_{max}}$)',
        'x_label': 'Lyapunov Max ($λ_{max}$)',
        'y_label': 'Spike Count (millions)',
        'title': 'Spike Activity',
        'fit_func': generalized_logistic,
        'initial_params': [
            (data['Spikes'] / 1000000).min(),
            (data['Spikes'] / 1000000).max() - (data['Spikes'] / 1000000).min(),
            8, 0.3, 1
        ],
        'panel_label': 'E',
        'linestyle': '-',
        'linewidth': 2.5,
        'alpha': 1.0,
        'y_limits': (2, 21),
        'text_position': "top-left",
        'r2': r2_largest_spikes,
        'p_value': p_largest_spikes,
        'popt': popt_largest_spikes,
        'highlight_ordered_chaos': False,
    }
]

    # Create each plot
    for config in plot_configs:
        if config['is_broken']:
            # Create broken axis plot
            create_broken_axis_plot(
                fig, gs,
                config['row'], config['col'],
                data, config['x_col'], config['y_col'],
                config['x_label'], config['y_label'], config['title'],
                config['r2'], config['p_value'], config['popt'],
                fit_func=config['fit_func'],
                initial_params=config['initial_params'],
                panel_label=config['panel_label'],
                highlight_critical=True,
                critical_value=0,
                linestyle=config['linestyle'],
                linewidth=config['linewidth'],
                alpha=config['alpha'],
                y_limits=config['y_limits'],
                text_on_left=config.get('text_on_left', False),
                text_position=config.get('text_position', "top-left"),
                text_align=config.get('text_align', "left"),
                y_scale=config.get('y_scale', 1.0),
                highlight_ordered_chaos=config.get('highlight_ordered_chaos', False),
                beta_pos=config.get('beta_pos', None),
                beta_neg=config.get('beta_neg', None)
            )
        else:
            # Create single axis plot
            create_single_axis_plot(
                fig, gs,
                config['row'], config['col'],
                data, config['x_col'], config['y_col'],
                config['x_label'], config['y_label'], config['title'],
                config['r2'], config['p_value'], config['popt'],
                fit_func=config['fit_func'],
                initial_params=config['initial_params'],
                panel_label=config['panel_label'],
                highlight_critical=True,
                critical_value=0,
                linestyle=config['linestyle'],
                linewidth=config['linewidth'],
                alpha=config['alpha'],
                y_limits=config['y_limits'],
                text_position=config.get('text_position', "top-left"),
                y_scale=config.get('y_scale', 1.0),
                highlight_ordered_chaos=config.get('highlight_ordered_chaos', False),
                beta_pos=config.get('beta_pos', None),
                beta_neg=config.get('beta_neg', None),
                color_by=config.get('color_by', None)
            )

    # Create legend items
    handles = [
        mpatches.Patch(color=colors['dissipative'], alpha=0.3, label='Dissipative Region (λ < 0)'),
        mpatches.Patch(color=colors['chaotic'], alpha=0.3, label='Expansive Region (λ > 0)'),
        plt.Line2D([0], [0], color=colors['edge'], linestyle='--', linewidth=1.5, label='Critical Point (λ = 0)'),
        plt.Line2D([0], [0], color=colors['fit_line'], linestyle='-', linewidth=2.0, label='Fitted Curve')
    ]

    labels = [
        'Dissipative Region (λ < 0)',
        'Expansive Region (λ > 0)',
        'Critical Point (λ = 0)',
        'Fitted Curve'
    ]

    # Place legend at the bottom
    plt.figlegend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02),
                 fontsize=9, frameon=False, columnspacing=1.0)

    # Adjust layout
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.17, top=0.95)

    # Print all panel statistics
    print("\nPanel Statistics:")
    for config in plot_configs:
        panel = config['panel_label']
        r2 = config['r2']
        p_value = config['p_value']
        x_var = config['x_col']
        y_var = config['y_label'].replace('\n', ' ')

        # Format p-value for display
        if p_value < 0.001:
            p_text = "p < 0.001"
        elif p_value < 0.05:
            p_text = f"p = {p_value:.3f}*"
        else:
            p_text = f"p = {p_value:.3f} (n.s.)"

        print(f"Panel {panel} ({x_var} vs {y_var}): R² = {r2:.3f}, {p_text}")

    # Return figure for saving
    return fig

# Main execution function
def main():
    # Try to load detailed run data
    detailed_data = load_detailed_run_data()
    info_data = load_information_dynamics_data()

    # If no detailed data, fall back to using summary data
    if detailed_data is None:
        print(f"Failed to load summary data.")
        return
    else:
        print(f"Using real detailed run data with {len(detailed_data)} rows.")

    if info_data is not None:
        ais_map = {}
        for idx, row in info_data.iterrows():
            exp_name = row['experiment']
            ais_value = row['avg_ais']
            ais_map[exp_name] = ais_value

        detailed_data['avg_ais'] = detailed_data['Experiment'].apply(
            lambda x: next((ais_map[k] for k in ais_map if k in x), np.nan)
        )

        print(f"Added avg_ais to {detailed_data['avg_ais'].notna().sum()} out of {len(detailed_data)} rows")

    # Create and save visualization
    fig = create_visualization(detailed_data)
    fig.savefig('ordered_chaos_visualization.png', dpi=400, bbox_inches='tight')
    fig.savefig('ordered_chaos_visualization.pdf', bbox_inches='tight')
    print("Optimized visualization saved.")

    print("All visualizations completed!")

if __name__ == "__main__":
    main()