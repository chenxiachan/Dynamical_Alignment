import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import stats
import os
import glob
import json

# Log-log Appendix D2

plt.style.use('default')
# plt.rcParams['font.family'] = 'Arial'
# plt.rcParams['font.size'] = 9
# plt.rcParams['axes.labelsize'] = 9
# plt.rcParams['axes.titlesize'] = 10
# plt.rcParams['xtick.labelsize'] = 8
# plt.rcParams['ytick.labelsize'] = 8
# plt.rcParams['legend.fontsize'] = 8
# plt.rcParams['figure.titlesize'] = 12

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
# 定义颜色方案
colors = {
    'dissipative': '#b3cde3',  
    'chaotic': '#f7b8b8',  
    'edge': '#1b7b1b',  
    'points': '#2c5d8c',
    'fit_line': '#b22222',  
    'grid': '#e0e0e0',  
    'text': '#333333'  
}


def remove_outliers(df, x_col='log_distance', y_col='log_y', method='iqr', iqr_factor=2.5):
    """
    Removes outlier data points.

    Args:
        df: The data DataFrame.
        x_col: The name of the column to check for outliers (x-axis).
        y_col: The name of the column to check for outliers (y-axis).
        method: The method for outlier removal, options: 'iqr', 'zscore', 'manual'.
        iqr_factor: The multiplier factor for the IQR method.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    if len(df) == 0:
        return df

    df_clean = df.copy()

    if method == 'iqr':
        # Remove outliers using the IQR method
        for col in [y_col]:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_factor * IQR
            upper_bound = Q3 + iqr_factor * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

    elif method == 'manual':
        # Manually remove obvious outliers (based on observed visual patterns)
        # Remove data points with unusually low y values near the critical point
        outlier_condition = (df_clean[x_col] > -1.0) & (df_clean[y_col] < -5.0)
        df_clean = df_clean[~outlier_condition]

        # Remove extreme cases with unusually low y values
        df_clean = df_clean[df_clean[y_col] > -6.5]

    elif method == 'zscore':
        # Use the Z-score method
        from scipy import stats
        z_scores = np.abs(stats.zscore(df_clean[y_col]))
        df_clean = df_clean[z_scores < 3]

    print(f"  Outlier Removal: {len(df)} -> {len(df_clean)} data points")
    return df_clean


def analyze_critical_behavior(data, x_col, y_col, critical_value=0, remove_outlier=True):
    """
    Analyzes critical behavior and calculates critical exponents.

    Args:
        data (pd.DataFrame): The input DataFrame.
        x_col (str): The column name for the x-axis (Lyapunov exponent).
        y_col (str): The column name for the y-axis (performance metric).
        critical_value (int, optional): The value of the critical point. Defaults to 0.
        remove_outlier (bool, optional): Whether to remove outliers. Defaults to True.

    Returns:
        dict: A dictionary containing the analysis results.
    """
    # Copy data to prevent modification of the original DataFrame
    df = data.copy()

    # Calculate distance to the critical point
    df['distance'] = df[x_col] - critical_value

    # Separate data for positive and negative regions
    positive_region = df[df['distance'] > 0].copy()  # Expansive region
    negative_region = df[df['distance'] < 0].copy()  # Dissipative region

    # Prepare data for power-law fitting, taking the log of the absolute distance
    positive_region['log_distance'] = np.log10(np.abs(positive_region['distance']))
    negative_region['log_distance'] = np.log10(np.abs(negative_region['distance']))

    # Calculate normalized y-values (starting from a baseline)
    if 'Convergence' in y_col:
        # For convergence epochs, use the maximum value as the baseline (fewer epochs are better)
        baseline = df[y_col].max()
        # Calculate the deviation from the baseline (reduction in convergence epochs)
        positive_region['normalized_y'] = baseline - positive_region[y_col]
        negative_region['normalized_y'] = baseline - negative_region[y_col]
    elif 'Spikes' in y_col:
        baseline = df[y_col].min()
        positive_region['normalized_y'] = positive_region[y_col] - baseline
        negative_region['normalized_y'] = negative_region[y_col] - baseline
    else:
        # For accuracy, find a stable region far from the critical point as the baseline
        far_negative = df[(df['distance'] < -0.5)]
        baseline = far_negative[y_col].mean() if len(far_negative) > 0 else df[y_col].min()
        positive_region['normalized_y'] = positive_region[y_col] - baseline
        negative_region['normalized_y'] = negative_region[y_col] - baseline

    # Handle possible zero or negative values
    epsilon = 1e-8
    pos_min = positive_region['normalized_y'].min() if not positive_region.empty else 0
    neg_min = negative_region['normalized_y'].min() if not negative_region.empty else 0
    offset = max(0, -min(pos_min, neg_min)) + 1e-6  # Ensure all values are positive

    positive_region['normalized_y'] = positive_region['normalized_y'] + offset
    negative_region['normalized_y'] = negative_region['normalized_y'] + offset

    # Take the logarithm of the y-values
    positive_region['log_y'] = np.log10(positive_region['normalized_y'] + epsilon)
    negative_region['log_y'] = np.log10(negative_region['normalized_y'] + epsilon)

    # Remove outliers (if enabled)
    if remove_outlier:
        print(f"Removing outliers for {y_col}...")
        positive_region = remove_outliers(positive_region, method='manual')
        negative_region = remove_outliers(negative_region, method='manual')

    # Perform power-law fitting for the positive region
    positive_critical_exponent = None
    positive_fit_stats = None
    if len(positive_region) >= 3:
        try:
            # Use log-log linear regression to calculate the power-law exponent
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                positive_region['log_distance'],
                positive_region['log_y']
            )
            positive_critical_exponent = slope
            positive_fit_stats = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'std_err': std_err
            }
        except Exception as e:
            print(f"Failed to fit expansive region: {e}")
            pass

    # Perform power-law fitting for the negative region
    negative_critical_exponent = None
    negative_fit_stats = None
    if len(negative_region) >= 3:
        try:
            # Use log-log linear regression to calculate the power-law exponent
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                negative_region['log_distance'],
                negative_region['log_y']
            )
            negative_critical_exponent = slope
            negative_fit_stats = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'std_err': std_err
            }
        except Exception as e:
            print(f"Failed to fit dissipative region: {e}")
            pass

    return {
        'positive_region': positive_region,
        'negative_region': negative_region,
        'positive_critical_exponent': positive_critical_exponent,
        'negative_critical_exponent': negative_critical_exponent,
        'positive_fit_stats': positive_fit_stats,
        'negative_fit_stats': negative_fit_stats,
        'baseline': baseline
    }


# Convert p-value to star notation
def p_value_to_stars(p_value):
    """
    Converts p-value to star notation.

    Args:
    p_value: The p-value.

    Returns:
    A string with star notation.
    """
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "n.s."  # not significant


def create_loglog_plot(analysis_results, x_col, y_col, metric_name, ax=None, panel_label=None):
    """
    Create a log-log plot to verify the power-law relationship.

    Args:
        analysis_results: Results from the analyze_critical_behavior function.
        x_col (str): Column name for the x-axis.
        y_col (str): Column name for the y-axis.
        metric_name (str): Name of the metric (for the title).
        ax: Optional matplotlib axes object.
        panel_label (str): Subplot label, e.g., 'A', 'B', 'C'.

    Returns:
        matplotlib axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.7, 2.4))

    # Get analysis results
    positive_region = analysis_results['positive_region']
    negative_region = analysis_results['negative_region']

    # Plot data points
    ax.scatter(positive_region['log_distance'], positive_region['log_y'],
               color=colors['chaotic'], s=40, alpha=0.8, edgecolor='k', linewidth=0.5)
    ax.scatter(negative_region['log_distance'], negative_region['log_y'],
               color=colors['dissipative'], s=40, alpha=0.8, edgecolor='k', linewidth=0.5)

    # Add fit lines and create subplot legends
    legend_elements = []

    if analysis_results['positive_fit_stats'] is not None:
        stats_pos = analysis_results['positive_fit_stats']
        x_range = np.array([min(positive_region['log_distance']), max(positive_region['log_distance'])])
        y_fit = stats_pos['slope'] * x_range + stats_pos['intercept']
        ax.plot(x_range, y_fit, 'r-', linewidth=2, alpha=0.8)

        # Create red line legend element
        p_stars = p_value_to_stars(stats_pos['p_value'])
        legend_elements.append(mlines.Line2D([0], [0], color='red', linewidth=2,
                                             label=f'β = {stats_pos["slope"]:.2f} (R² = {stats_pos["r_squared"]:.3f} {p_stars})'))

    if analysis_results['negative_fit_stats'] is not None:
        stats_neg = analysis_results['negative_fit_stats']
        x_range = np.array([min(negative_region['log_distance']), max(negative_region['log_distance'])])
        y_fit = stats_neg['slope'] * x_range + stats_neg['intercept']
        ax.plot(x_range, y_fit, 'b-', linewidth=2, alpha=0.8)

        # Create blue line legend element
        p_stars = p_value_to_stars(stats_neg['p_value'])
        legend_elements.append(mlines.Line2D([0], [0], color='blue', linewidth=2,
                                             label=f'β = {stats_neg["slope"]:.2f} (R² = {stats_neg["r_squared"]:.3f} {p_stars})'))

    # Add a small legend to the bottom-left corner
    if legend_elements:
        ax.legend(handles=legend_elements, loc='lower left', fontsize=7, frameon=True,
                  fancybox=True, framealpha=0.9, edgecolor='gray')

    # Add panel label if provided
    if panel_label:
        ax.text(-0.2, 1.05, panel_label, transform=ax.transAxes, fontsize=12, fontweight='bold')

    # Set x-axis label
    if (x_col == 'Lyapunov Sum'):
        ax.set_xlabel('log|Σλᵢ - λc|', fontsize=9)
    else:
        ax.set_xlabel('log|$λ_{max}$ - λc|', fontsize=9)

    # Set y-axis label based on the metric
    if 'Convergence' in metric_name:
        ax.set_ylabel(f'log(Δ{metric_name})', fontsize=9)
    else:
        ax.set_ylabel(f'log(Δ{metric_name})', fontsize=9)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3, color=colors['grid'])

    return ax


# Create log-log plot analysis containing both Lyapunov indicators
def create_dual_lyapunov_analysis(data, remove_outliers=True):
    """
    Create a two-row, three-column chart to compare the Lyapunov exponent sum and largest Lyapunov exponent.

    Args:
    data: DataFrame containing experimental data

    Returns:
    matplotlib figure object and analysis results
    """
    # Define metrics for analysis - first row for sum, second row for max
    # Order adjusted to align with Figure 5: A-Accuracy, B-Spike Count, C-Convergence Epochs
    metrics = [
        # First row: Lyapunov sum
        ('Lyapunov Sum', 'Accuracy', 'Accuracy'),
        ('Lyapunov Sum', 'Spikes', 'Spike Count'),
        ('Lyapunov Sum', 'Convergence', 'Convergence Epochs'),
        # Second row: Largest Lyapunov exponent
        ('Largest Lyapunov', 'Accuracy', 'Accuracy'),
        ('Largest Lyapunov', 'Spikes', 'Spike Count'),
        ('Largest Lyapunov', 'Convergence', 'Convergence Epochs')
    ]

    # Create figure with two rows and three columns, leaving more space for the bottom legend
    fig, axes = plt.subplots(2, 3, figsize=(8.1, 5.6))

    # Store analysis results
    all_results = []

    # Panel labels - corresponding to the new order: A-Accuracy, B-Spike Count, C-Convergence Epochs
    panel_labels = ['A', 'B', 'C', 'D', 'E', 'F']

    # Analyze each metric one by one
    for i, (x_col, y_col, name) in enumerate(metrics):
        # Calculate row and column indices
        row = i // 3
        col = i % 3

        # Analyze critical behavior
        results = analyze_critical_behavior(data, x_col, y_col, remove_outlier=remove_outliers)
        all_results.append((x_col, results, name))

        # Create log-log plot
        create_loglog_plot(results, x_col, y_col, name, axes[row, col], panel_labels[i])

    # Adjust subplot spacing, leaving space for the bottom legend
    plt.tight_layout(rect=[0.02, 0.07, 1, 1])

    # Create a unified legend at the bottom (simplified version, showing only data point types)
    # Create legend elements
    legend_elements = [
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['chaotic'],
                      markersize=8, markeredgecolor='k', markeredgewidth=0.5,
                      label='Expansive Region (λ > 0)', linestyle='None'),
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['dissipative'],
                      markersize=8, markeredgecolor='k', markeredgewidth=0.5,
                      label='Dissipative Region (λ < 0)', linestyle='None'),
        mlines.Line2D([0], [0], color='red', linewidth=2,
                      label='Power-law fit (Expansive)', alpha=0.8),
        mlines.Line2D([0], [0], color='blue', linewidth=2,
                      label='Power-law fit (Dissipative)', alpha=0.8)
    ]

    # Create unified legend
    fig.legend(handles=legend_elements,
               loc='lower center',
               bbox_to_anchor=(0.5, 0.02),
               ncol=4,
               fontsize=8,
               frameon=False, columnspacing=1.0)

    # Save figure
    suffix = "_no_outliers" if remove_outliers else ""
    fig.savefig(f'critical_dual_lyapunov_analysis_convergence{suffix}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'critical_dual_lyapunov_analysis_convergence{suffix}.pdf', bbox_inches='tight')

    # Print analysis results
    print("\nCritical Phase Transition Analysis Results (Σλᵢ vs λₘₐₓ):")
    for x_col, results, name in all_results:
        indicator = "Σλᵢ" if x_col == 'Lyapunov Sum' else "λₘₐₓ"
        print(f"\nMetric: {name} ({indicator})")
        print(f"  Expansive Region Critical Exponent: {results['positive_critical_exponent']:.4f}" if results[
            'positive_critical_exponent'] else "  Expansive Region Critical Exponent: Undeterminable")
        print(f"  Dissipative Region Critical Exponent: {results['negative_critical_exponent']:.4f}" if results[
            'negative_critical_exponent'] else "  Dissipative Region Critical Exponent: Undeterminable")

        if results['positive_fit_stats']:
            print(
                f"  Expansive Region log-log: R² = {results['positive_fit_stats']['r_squared']:.4f}, p-value: {results['positive_fit_stats']['p_value']:.4f}")
        if results['negative_fit_stats']:
            print(
                f"  Dissipative Region log-log: R² = {results['negative_fit_stats']['r_squared']:.4f}, p-value: {results['negative_fit_stats']['p_value']:.4f}")

    return fig, all_results


def load_detailed_run_data(detailed_runs_dir="detailed_runs"):
    """
    Loads detailed run data for all experiments.
    Returns a DataFrame containing the raw data points.
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
            # Load the corresponding epoch history
            epoch_file = os.path.join(detailed_runs_dir, f"{experiment_name}_run{run['run']}_epochs.json")

            try:
                with open(epoch_file, 'r') as f:
                    epochs = json.load(f)

                # Get epoch data at convergence
                conv_epoch = min(run['convergence_epoch'], len(epochs) - 1)
                epoch_data = epochs[conv_epoch]

                # Add to the global dataset
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

    # Validate data
    if len(runs_df) == 0:
        print("Warning: No detailed run data found! Using summary statistics file instead.")
        return None

    print(f"Successfully loaded detailed data for {len(runs_df)} runs")
    return runs_df


def main():
    """Main function"""
    try:
        # Attempt to load real data
        print("Attempting to load detailed run data...")
        detailed_data = load_detailed_run_data()
        if detailed_data is not None and len(detailed_data) > 0:
            print(f"Successfully loaded detailed run data with {len(detailed_data)} rows.")
        else:
            raise ValueError("Data is empty")
    except Exception as e:
        # If no real data, create synthetic data
        print(f"Could not load detailed run data: {str(e)}")

    # Create dual Lyapunov indicator analysis plots
    try:
        fig_with_outliers, results_with_outliers = create_dual_lyapunov_analysis(detailed_data, remove_outliers=False)
        fig_clean, results_clean = create_dual_lyapunov_analysis(detailed_data, remove_outliers=True)

    except Exception as e:
        import traceback
        print(f"An error occurred during analysis: {str(e)}")
        traceback.print_exc()
        print("Please check data format and structure.")


if __name__ == "__main__":
    main()