import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import pandas as pd
from scipy import stats 

plt.style.use('default')
# ===== 配色方案 =====
# ANN 颜色
ann_color = '#7D7D7D'      # 深灰色 for ANN

# SNN 颜色序列 (按 delta 10.0 -> -1.5 顺序)
diss_color_dark = '#4a8bbd'  # 深蓝色变体 (for delta=10.0)
diss_color_base = '#b3cde3'  # 中蓝色       (for delta=2.0)
exp_color_base = '#f7b8b8'   # 中粉色       (for delta=0.0)
exp_color_dark = '#d46161'   # 深粉色变体 (for delta=-1.5)


snn_violin_colors = [diss_color_dark, diss_color_base, exp_color_base, exp_color_dark]

# ===== 数据文件和参数 =====
delta_values = [10.00, 2.00, 0.00, -1.50]
labels = ['Dissipative Region\n($\delta=10.0$)',
          'Tough Region\n($\delta=2.0$)',
          'Critical Point\n($\delta=0.0$)',
          'Expansive Region\n($\delta=-1.5$)']


# --- 确保文件路径正确 ---
file_dir = 'detailed_runs/' # 定义文件夹路径
ann_files = {
    10.00: file_dir + 'MixedOsc_d10.00_ANN_runs.json',
    2.00: file_dir + 'MixedOsc_d2.00_ANN_runs.json',
    0.00: file_dir + 'MixedOsc_d0.00_ANN_runs.json',
    -1.50: file_dir + 'MixedOsc_d-1.50_ANN_runs.json',
}
snn_files = {
    10.00: file_dir + 'MixedOsc_d10.00_runs.json',
    2.00: file_dir + 'MixedOsc_d2.00_runs.json',
    0.00: file_dir + 'MixedOsc_d0.00_runs.json',
    -1.50: file_dir + 'MixedOsc_d-1.50_runs.json',
}
# --- ---

# ===== 加载并准备数据 =====
data_list = []

for i, delta in enumerate(delta_values):
    try: # ANN Data
        with open(ann_files[delta], 'r') as f: ann_data = json.load(f)
        for run_data in ann_data:
            acc = run_data.get('best_accuracy')
            if acc is not None and not np.isnan(acc): data_list.append({'delta': delta, 'label': labels[i], 'model_type': 'ANN', 'accuracy': float(acc)})
            else: print(f"警告: ANN 文件 {ann_files[delta]} 跳过无效准确率: {acc}")
    except Exception as e: print(f"警告: 读取 ANN 文件 {ann_files[delta]} 出错: {e}")
    try: # SNN Data
        with open(snn_files[delta], 'r') as f: snn_data = json.load(f)
        for run_data in snn_data:
            color = snn_violin_colors[i]; acc = run_data.get('best_accuracy')
            if acc is not None and not np.isnan(acc): data_list.append({'delta': delta, 'label': labels[i], 'model_type': 'SNN', 'accuracy': float(acc), 'color': color})
            else: print(f"警告: SNN 文件 {snn_files[delta]} 跳过无效准确率: {acc}")
    except Exception as e: print(f"警告: 读取 SNN 文件 {snn_files[delta]} 出错: {e}")


df = pd.DataFrame(data_list)

# 检查DataFrame是否为空
if df.empty:
    print("错误：未能加载任何有效数据，无法生成图表。请检查文件路径和JSON内容。")
else:
    # ===== 创建图表 =====
    fig, ax = plt.subplots(figsize=(6, 2.1)) 

    # ===== 绘图参数 =====
    n_groups = len(delta_values)
    x_positions = np.arange(n_groups)
    total_width = 0.7
    box_width = total_width / 2 * 0.85
    point_size = 4 
    point_alpha = 0.5 
    jitter_amount = 0.1 
    zorder_box = 10
    zorder_points = 5
    zorder_median = 11
    max_y_for_annotation = -np.inf 

    # ===== 循环绘制每个分组并添加统计检验 =====
    for i, delta in enumerate(delta_values):
        current_label = labels[i]
        ann_acc = df[(df['model_type'] == 'ANN') & (df['delta'] == delta)]['accuracy'].dropna()
        snn_acc = df[(df['model_type'] == 'SNN') & (df['delta'] == delta)]['accuracy'].dropna()
        snn_color = snn_violin_colors[i]
        pos_ann = x_positions[i] - total_width / 4
        pos_snn = x_positions[i] + total_width / 4

        # --- 绘制 ANN 数据 ---
        if not ann_acc.empty:
            bp_ann = ax.boxplot(ann_acc, positions=[pos_ann], widths=box_width, patch_artist=True, showfliers=False,
                                medianprops={'color': 'white', 'linewidth': 1.5, 'zorder': zorder_median},
                                boxprops={'edgecolor': 'black', 'linewidth': 0.7, 'zorder': zorder_box},
                                whiskerprops={'color': 'black', 'linewidth': 0.7, 'linestyle': '-', 'zorder': zorder_box},
                                capprops={'color': 'black', 'linewidth': 0.7, 'zorder': zorder_box})
            bp_ann['boxes'][0].set_facecolor(ann_color)
            bp_ann['boxes'][0].set_alpha(0.9)
#             x_jitter_ann = np.random.normal(pos_ann, jitter_amount, size=len(ann_acc))
#             ax.plot(x_jitter_ann, ann_acc, 'o', color=ann_color, markersize=point_size, alpha=point_alpha,
#                     markeredgecolor='black', markeredgewidth=0.3, zorder=zorder_points, label='_nolegend_') # Prevent points from creating labels
            max_y_for_annotation = max(max_y_for_annotation, ann_acc.max())

        # --- 绘制 SNN 数据 ---
        if not snn_acc.empty:
            bp_snn = ax.boxplot(snn_acc, positions=[pos_snn], widths=box_width, patch_artist=True, showfliers=False,
                                medianprops={'color': 'white', 'linewidth': 1.5, 'zorder': zorder_median},
                                boxprops={'edgecolor': 'black', 'linewidth': 0.7, 'zorder': zorder_box},
                                whiskerprops={'color': 'black', 'linewidth': 0.7, 'linestyle': '-', 'zorder': zorder_box},
                                capprops={'color': 'black', 'linewidth': 0.7, 'zorder': zorder_box})
            bp_snn['boxes'][0].set_facecolor(snn_color)
            bp_snn['boxes'][0].set_alpha(0.9)
#             x_jitter_snn = np.random.normal(pos_snn, jitter_amount, size=len(snn_acc))
#             ax.plot(x_jitter_snn, snn_acc, 'o', color=snn_color, markersize=point_size, alpha=point_alpha,
#                     markeredgecolor='black', markeredgewidth=0.3, zorder=zorder_points, label='_nolegend_')
            max_y_for_annotation = max(max_y_for_annotation, snn_acc.max())

        # --- 添加统计显著性标记 ---
        # 检查是否有足够的数据进行测试 (例如，每组至少需要几个点)
        if len(ann_acc) >= 3 and len(snn_acc) >= 3:
            try:
                # 执行 Mann-Whitney U 检验
                stat, p_value = stats.mannwhitneyu(ann_acc, snn_acc, alternative='two-sided')

                # 确定显著性标记符号
                if p_value < 0.001: sig_symbol = '* * *'
                elif p_value < 0.01: sig_symbol = '* *'
                elif p_value < 0.05: sig_symbol = '*'
                    
                else: sig_symbol = 'n.s.' # Non-significant

                # --- 绘制标记 ---
                # 只绘制显著的标记（或 'ns' 如果需要）
                show_ns = True # 改为 False 则不显示 'ns'
                if sig_symbol != 'ns' or show_ns:
                    # 计算标记的Y坐标：在当前组最高点之上留出一些空间
                    y_start = max(ann_acc.max() if not ann_acc.empty else -np.inf,
                                  snn_acc.max() if not snn_acc.empty else -np.inf)
                    # 使用相对高度来确定标记位置，避免绝对值问题
                    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                    y_offset_bracket = y_range * 0.04 # 标记线距离最高点的距离
                    y_bracket_height = y_range * 0.02 # 标记线垂直部分的高度
                    y_text_offset = y_range * 0.01    # 文本距离标记线的距离

                    bracket_y = y_start + y_offset_bracket
                    text_y = bracket_y + y_bracket_height + y_text_offset

                    # 绘制连接线 (横线 + 两边竖线)
                    ax.plot([pos_ann, pos_ann, pos_snn, pos_snn],
                            [bracket_y, bracket_y + y_bracket_height, bracket_y + y_bracket_height, bracket_y],
                            lw=1.0, c='black')

                    # 添加显著性文本
                    ax.text((pos_ann + pos_snn) / 2, text_y, sig_symbol,
                            ha='center', va='bottom', color='black', fontsize=10) # 减小字体

                    # 更新全局最高Y坐标，以便后续调整ylim
                    max_y_for_annotation = max(max_y_for_annotation, text_y)

            except ValueError as e:
                # 处理检验错误 (例如，数据完全相同或数据太少)
                print(f"跳过 delta={delta} 的统计检验，错误: {e}")
        else:
            print(f"跳过 delta={delta} 的统计检验，样本量不足 (ANN: {len(ann_acc)}, SNN: {len(snn_acc)})")


    # ===== 图表定制 =====
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=9) # 减小标签字体以防重叠
    ax.set_ylabel('Best Accuracy (%)', fontweight='bold', fontsize=10)
    ax.set_xlabel('Region (Delta Value)', fontweight='bold', fontsize=10)
    # ax.text(-0.1, 1.05, 'A', transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom') # 标题A可能需要调整或移除

    # 动态调整Y轴范围 (在所有绘图完成后)
    min_data_y = df['accuracy'].min()
    # 确保Y轴上限足够高以容纳最高的注释
    y_top_final = max_y_for_annotation + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05 # 在最高注释上再加5%边距
    y_bottom_final = min_data_y - (y_top_final - min_data_y) * 0.12 # 在最低点下加5%边距

    ax.set_ylim(y_bottom_final, y_top_final)

    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=9) 


    text_y_pos = ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.04 + 1.3 
    ax.text(x_positions[0] - total_width / 4, text_y_pos, 'MLP', ha='center', va='top', color=ann_color, fontsize=9)
    ax.text(x_positions[0] + total_width / 4, text_y_pos, 'SNN', ha='center', va='top', color='black', fontsize=9) # 

    
    ax.text(x_positions[1] - total_width / 4, text_y_pos, 'MLP', ha='center', va='top', color=ann_color, fontsize=9)
    ax.text(x_positions[1] + total_width / 4, text_y_pos, 'SNN', ha='center', va='top', color='black', fontsize=9) # 

    
    ax.text(x_positions[2] - total_width / 4, text_y_pos, 'MLP', ha='center', va='top', color=ann_color, fontsize=9)
    ax.text(x_positions[2] + total_width / 4, text_y_pos, 'SNN', ha='center', va='top', color='black', fontsize=9) # 

    
    ax.text(x_positions[3] - total_width / 4, text_y_pos, 'MLP', ha='center', va='top', color=ann_color, fontsize=9)
    ax.text(x_positions[3] + total_width / 4, text_y_pos, 'SNN', ha='center', va='top', color='black', fontsize=9) # 


    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.98]) 


    # ===== 保存或显示 =====
    plt.savefig('A_performance_comparison_boxplot_stripplot_stats.png', dpi=300)
    plt.savefig('A_performance_comparison_boxplot_stripplot_stats.pdf')
    plt.show()