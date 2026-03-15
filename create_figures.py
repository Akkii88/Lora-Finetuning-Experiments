"""
Generate all academic figures for PEFT/LoRA research paper
IEEE Journal Style - 300 DPI
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon
from matplotlib.lines import Line2D

# Set IEEE style defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': 'grey'
})

def save_fig(fig, filename):
    """Save figure with IEEE quality settings"""
    fig.savefig(f'/Users/ankit/Desktop/SabinaResearchPaper/LLM-Parameter-Efficient-Fine-Tuning-with-LoRA/figures/{filename}', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)

def create_figure1_memory_storage():
    """Figure 1: Two-panel chart - Training Memory and Disk Storage"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor('white')
    
    # Left panel: Training Memory (log scale)
    models = ['DistilBERT\n66M', 'BERT-Large\n340M', 'LLaMA-7B', 'GPT-3\n175B']
    full_ft = [4, 25, 160, 2800]  # GB
    lora = [0.8, 3.2, 28, 420]    # GB
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, full_ft, width, label='Full Fine-Tuning', 
                    color='#FF6B35', edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, lora, width, label='PEFT LoRA r=8', 
                    color='#4682B4', edgecolor='black', linewidth=0.5)
    
    ax1.set_yscale('log')
    ax1.set_ylabel('Training Memory (GB)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylim(0.5, 5000)
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.set_title('(a) Training Memory Comparison', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, color='grey')
    
    # Right panel: Disk Storage vs Task Adaptations
    tasks = np.arange(1, 11)
    full_ft_storage = tasks * 4  # 4GB per full model copy
    lora_storage = 4 + (tasks * 0.08)  # 4GB base + 80MB per adapter
    
    ax2.plot(tasks, full_ft_storage, 'o-', color='#FF6B35', linewidth=2, 
             markersize=6, label='Full Fine-Tuning')
    ax2.plot(tasks, lora_storage, 's-', color='#4682B4', linewidth=2, 
             markersize=6, label='PEFT LoRA')
    
    # Green shaded region
    ax2.fill_between(tasks, lora_storage, full_ft_storage, 
                      color='#90EE90', alpha=0.5, label='Storage Saved')
    
    ax2.set_xlabel('Number of Task Adaptations', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Disk Storage (GB)', fontsize=11, fontweight='bold')
    ax2.set_xlim(0.5, 10.5)
    ax2.set_ylim(0, 50)
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.set_title('(b) Disk Storage Requirements', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, color='grey')
    
    plt.tight_layout()
    save_fig(fig, 'fig1_memory_storage.png')


def create_figure2_lora_architecture():
    """Figure 2: Technical Architecture Diagram - LoRA Adaptation"""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Input
    input_box = FancyBboxPatch((0.3, 3.5), 1.2, 1, boxstyle="round,pad=0.05",
                                 facecolor='#000080', edgecolor='black', linewidth=1)
    ax.add_patch(input_box)
    ax.text(0.9, 4, 'x input', ha='center', va='center', fontsize=10, 
            color='white', fontweight='bold', fontfamily='serif')
    
    # W0 frozen path
    w0_box = FancyBboxPatch((3.5, 3.8), 1.8, 0.8, boxstyle="round,pad=0.1",
                            facecolor='#D3D3D3', edgecolor='black', linewidth=1)
    ax.add_patch(w0_box)
    ax.text(4.4, 4.2, 'W$_0$ frozen', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(4.4, 3.95, 'd×k weights', ha='center', va='center', fontsize=8, style='italic')
    
    # Arrow from input to W0
    ax.annotate('', xy=(3.5, 4.2), xytext=(1.5, 4.2),
                arrowprops=dict(arrowstyle='->', color='grey', lw=1.5))
    
    # LoRA path - A matrix
    lora_a = FancyBboxPatch((3.5, 1.8), 1.4, 0.7, boxstyle="round,pad=0.1",
                            facecolor='#4682B4', edgecolor='black', linewidth=1)
    ax.add_patch(lora_a)
    ax.text(4.2, 2.2, 'A', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.text(4.2, 1.95, 'r×k trainable', ha='center', va='center', fontsize=7)
    
    # Arrow from input to A
    ax.annotate('', xy=(3.5, 2.15), xytext=(1.5, 2.5),
                arrowprops=dict(arrowstyle='->', color='#4682B4', lw=1.5))
    
    # LoRA path - B matrix
    lora_b = FancyBboxPatch((5.3, 1.8), 1.4, 0.7, boxstyle="round,pad=0.1",
                            facecolor='#4682B4', edgecolor='black', linewidth=1)
    ax.add_patch(lora_b)
    ax.text(6.0, 2.2, 'B', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.text(6.0, 1.95, 'd×r trainable', ha='center', va='center', fontsize=7)
    
    # Arrow from A to B
    ax.annotate('', xy=(5.3, 2.15), xytext=(4.9, 2.15),
                arrowprops=dict(arrowstyle='->', color='#4682B4', lw=1.5))
    
    # Scale factor circle
    scale_circle = Circle((7.2, 2.15), 0.25, facecolor='#ADD8E6', edgecolor='black', linewidth=1)
    ax.add_patch(scale_circle)
    ax.text(7.2, 2.15, 'α/r', ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Arrow from B to scale
    ax.annotate('', xy=(7.0, 2.15), xytext=(6.7, 2.15),
                arrowprops=dict(arrowstyle='->', color='#4682B4', lw=1.5))
    
    # Plus circle
    plus_circle = Circle((7.7, 4.2), 0.35, facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(plus_circle)
    ax.text(7.7, 4.2, '+', ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Arrow from W0 to plus
    ax.annotate('', xy=(7.35, 4.2), xytext=(5.3, 4.2),
                arrowprops=dict(arrowstyle='->', color='grey', lw=1.5))
    
    # Arrow from scale to plus
    ax.annotate('', xy=(7.35, 4.2), xytext=(7.45, 2.4),
                arrowprops=dict(arrowstyle='->', color='#4682B4', lw=1.5))
    
    # Output
    output_box = FancyBboxPatch((8.3, 3.5), 1.2, 1, boxstyle="round,pad=0.05",
                                facecolor='#000080', edgecolor='black', linewidth=1)
    ax.add_patch(output_box)
    ax.text(8.9, 4, 'h output', ha='center', va='center', fontsize=10, 
            color='white', fontweight='bold', fontfamily='serif')
    
    # Arrow from plus to output
    ax.annotate('', xy=(8.3, 4.2), xytext=(8.05, 4.2),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Info box at bottom
    info_box = FancyBboxPatch((1.5, 0.2), 7, 1.2, boxstyle="round,pad=0.1",
                               facecolor='#E6F3FF', edgecolor='#4682B4', linewidth=1)
    ax.add_patch(info_box)
    ax.text(5, 1.1, 'h = W$_0$x + (α/r)BAx', ha='center', va='center', 
            fontsize=11, fontweight='bold', fontfamily='serif')
    ax.text(5, 0.65, 'Trainable params: r(d+k) instead of d×k', ha='center', 
            va='center', fontsize=9, style='italic')
    
    # Labels
    ax.text(2.5, 5.5, 'Frozen path W$_0$x', ha='center', va='center', 
            fontsize=9, color='grey', style='italic')
    ax.text(5, 0.95, 'LoRA adaptation: only A and B trained', ha='center', 
            va='center', fontsize=9, color='#4682B4', style='italic')
    
    ax.set_title('Fig. 2 — LoRA Architecture in Transformer Block', 
                 fontsize=12, fontweight='bold', pad=20)
    
    save_fig(fig, 'fig2_lora_architecture.png')


def create_figure3_pipeline_flowchart():
    """Figure 3: Horizontal Flowchart - NLP Pipeline with PEFT"""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    def add_box(x, y, w, h, color, text, text_color='white'):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
                fontsize=8, color=text_color, fontweight='bold', fontfamily='serif')
        return x + w
    
    def add_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    def add_branch_arrow(x, y1, y2, label, color='black'):
        ax.annotate('', xy=(x, y2), xytext=(x, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        ax.text(x + 0.1, (y1 + y2)/2, label, fontsize=7, va='center')
    
    # Top row
    y_top = 4.5
    x = 0.5
    x = add_box(x, y_top, 1.2, 0.8, '#333333', 'Raw Text')
    add_arrow(x, y_top + 0.4, x + 0.2, y_top + 0.4)
    x = add_box(x + 0.3, y_top, 1.6, 0.8, '#333333', 'Tokenizer\nWordPiece')
    add_arrow(x, y_top + 0.4, x + 0.2, y_top + 0.4)
    x = add_box(x + 0.3, y_top, 2, 0.8, '#333333', 'Token IDs +\nAttention Mask')
    add_arrow(x, y_top + 0.4, x + 0.2, y_top + 0.4)
    x = add_box(x + 0.3, y_top, 2.2, 0.8, '#808080', 'DistilBERT\nFrozen 66M')
    
    # Branch to LoRA
    add_branch_arrow(x + 1.1, y_top, y_top - 1.5, 'Yes')
    add_box(x + 0.6, y_top - 2.3, 1.8, 0.8, '#4682B4', 'LoRA Adapter\n739,586 params')
    
    # Branch to Prompt Tuning
    add_branch_arrow(x + 1.1, y_top, y_top - 3.5, 'No', '#808080')
    add_box(x + 0.6, y_top - 4.3, 1.8, 0.8, '#CC5500', 'Prompt Tuning\n15,360 params')
    
    # Both go to loss
    add_arrow(x + 1.5, y_top - 2.3, x + 1.5, y_top - 4.3)
    add_arrow(x + 1.5, y_top - 4.3, x + 1.5, y_top - 5.3)
    ax.annotate('', xy=(x + 1.5, 1.8), xytext=(x + 1.5, y_top - 4.7),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    add_box(x + 0.6, 1, 1.8, 0.8, '#228B22', 'Linear Head\n→ Loss')
    ax.text(x + 1.5, 0.7, '(gradients restricted)', fontsize=7, style='italic', ha='center')
    
    # Legend
    legend_x = 8
    ax.add_patch(FancyBboxPatch((legend_x, 5.5), 0.4, 0.4, boxstyle="round,pad=0.02",
                                facecolor='#808080', edgecolor='black'))
    ax.text(legend_x + 0.5, 5.7, 'Frozen (no gradients)', fontsize=8, va='center')
    
    ax.add_patch(FancyBboxPatch((legend_x, 5), 0.4, 0.4, boxstyle="round,pad=0.02",
                                facecolor='#4682B4', edgecolor='black'))
    ax.text(legend_x + 0.5, 5.2, 'Trainable (gradients computed)', fontsize=8, va='center')
    
    ax.set_title('Fig. 3 — NLP Pipeline with PEFT Methods', 
                 fontsize=12, fontweight='bold', pad=20)
    
    save_fig(fig, 'fig3_pipeline_flowchart.png')


def create_figure4_rank_accuracy():
    """Figure 4: Dual-axis chart - LoRA Rank vs Accuracy and Parameters"""
    fig, ax1 = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('white')
    
    ranks = ['r=4', 'r=8', 'r=16']
    x = np.arange(len(ranks))
    accuracy = [85.3, 88.7, 88.9]
    params = [370, 740, 1480]  # in thousands
    
    # Bars - left axis
    bars = ax1.bar(x, accuracy, width=0.6, color='#4682B4', edgecolor='black', 
                   linewidth=0.5, label='Accuracy')
    ax1.set_ylabel('Accuracy (%)', color='#4682B4', fontsize=11, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#4682B4')
    ax1.set_ylim(80, 94)
    ax1.set_xlabel('LoRA Rank', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(ranks)
    
    # Value labels on bars
    for i, (bar, val) in enumerate(zip(bars, accuracy)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Line - right axis
    ax2 = ax1.twinx()
    ax2.plot(x, params, 'o-', color='#CC5500', linewidth=2, markersize=8,
             markerfacecolor='#CC5500', markeredgecolor='black', label='Trainable Parameters')
    ax2.set_ylabel('Trainable Parameters (×10³)', color='#CC5500', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#CC5500')
    ax2.set_ylim(0, 2000)
    
    # Parameter labels
    for i, val in enumerate(params):
        ax2.annotate(f'{val}K', xy=(i, val), xytext=(5, 5),
                    textcoords='offset points', fontsize=8, color='#CC5500')
    
    # Vertical dashed line at r=8
    ax1.axvline(x=1, color='grey', linestyle='--', linewidth=1.5)
    ax1.text(1.05, 81.5, 'recommended\noperating point', fontsize=8, 
             style='italic', color='grey')
    
    # Combined legend
    lines1 = Line2D([0], [0], color='#4682B4', linewidth=2, marker='s', markersize=8)
    lines2 = Line2D([0], [0], color='#CC5500', linewidth=2, marker='o', markersize=8)
    ax1.legend([bars, lines2], ['Accuracy', 'Trainable Parameters'], 
               loc='lower right', framealpha=0.9)
    
    ax1.set_title('Fig. 4 — LoRA Rank: Accuracy vs. Parameter Count', 
                  fontsize=12, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_fig(fig, 'fig4_rank_accuracy.png')


def create_figure5_learning_curve():
    """Figure 5: Learning Curve - Data Efficiency"""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('white')
    
    samples = [100, 250, 500, 750, 1000]
    accuracy = [70.0, 82.4, 87.6, 88.2, 88.7]
    
    # Fill under curve
    ax.fill_between(samples, 0, accuracy, alpha=0.3, color='#ADD8E6')
    
    # Line with markers
    ax.plot(samples, accuracy, 'o-', color='#000080', linewidth=2, 
            markersize=10, markerfacecolor='white', markeredgecolor='#000080',
            markeredgewidth=2, label='LoRA Accuracy')
    
    # Value labels
    for s, a in zip(samples, accuracy):
        ax.annotate(f'{a}%', xy=(s, a), xytext=(0, 10),
                   textcoords='offset points', ha='center', fontsize=9, 
                   fontweight='bold', color='#000080')
    
    # Horizontal ceiling line
    ax.axhline(y=88.7, color='blue', linestyle='--', linewidth=1.5)
    ax.text(150, 89.2, '1000-sample ceiling', fontsize=8, style='italic', color='blue')
    
    # Diminishing returns zone
    ax.axvspan(500, 1000, alpha=0.2, color='#90EE90')
    ax.text(750, 83, 'Diminishing\nreturns zone', fontsize=8, 
            color='green', ha='center', style='italic')
    
    # Arrow annotation
    ax.annotate('', xy=(500, 87.6), xytext=(100, 70),
                arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
    ax.text(300, 79, '+17.6 pp\n(100→500)', fontsize=8, 
            color='green', ha='center', fontweight='bold')
    
    ax.set_xlabel('Training Samples', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 1100)
    ax.set_ylim(60, 95)
    ax.set_xticks([100, 250, 500, 750, 1000])
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    ax.set_title('Fig. 5 — Data Efficiency: LoRA Accuracy vs. Training Set Size', 
                 fontsize=12, fontweight='bold', pad=15)
    
    save_fig(fig, 'fig5_learning_curve.png')


def create_figure6_domain_shift():
    """Figure 6: Grouped Bar Chart - In-Domain vs Cross-Domain"""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('white')
    
    groups = ['Base\nUntrained', 'Prompt\nTuning', 'LoRA\nr=8']
    x = np.arange(len(groups))
    width = 0.35
    
    imdb_acc = [52.3, 87.2, 89.1]
    yelp_acc = [48.7, 85.3, 87.2]
    
    # Bars
    bars1 = ax.bar(x - width/2, imdb_acc, width, label='IMDB in-domain', 
                   color='#4682B4', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, yelp_acc, width, label='Yelp cross-domain', 
                   color='#CC5500', edgecolor='black', linewidth=0.5,
                   hatch='///')
    
    # Value labels
    for bar, val in zip(bars1, imdb_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{val}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar, val in zip(bars2, yelp_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{val}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Delta annotations
    ax.text(1, 91, 'Δ = +1.9 pp', fontsize=9, color='green', 
            fontweight='bold', ha='center')
    ax.text(2, 91, 'Δ = -1.9 pp', fontsize=9, color='red', 
            fontweight='bold', ha='center')
    
    # Domain shift note
    ax.text(1.5, 44, 'domain shift applied to PEFT models only', 
            fontsize=8, style='italic', ha='center')
    
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Method', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(40, 95)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    
    ax.set_title('Fig. 6 — In-Domain IMDB vs. Cross-Domain Yelp Accuracy', 
                 fontsize=12, fontweight='bold', pad=15)
    
    save_fig(fig, 'fig6_domain_shift.png')


def create_figure7_decision_flowchart():
    """Figure 7: Decision Flowchart - PEFT Method Selection"""
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    def add_decision_box(x, y, w, h, text):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                            facecolor='#505050', edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
                fontsize=9, color='white', fontweight='bold', fontfamily='serif')
    
    def add_action_box(x, y, w, h, color, text):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
                fontsize=9, color='white', fontweight='bold', fontfamily='serif')
    
    def add_terminal(x, y, w, h, text):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                            facecolor='#4682B4', edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
                fontsize=9, color='white', fontweight='bold', fontfamily='serif')
    
    def add_arrow(x1, y1, x2, y2, label='', label_pos='mid'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x + 0.3, mid_y, label, fontsize=8, va='center')
    
    # Start
    add_action_box(3.5, 9, 3, 0.7, '#000080', 'START — Choose PEFT Method')
    add_arrow(5, 9, 5, 8.5)
    
    # Q1: Model > 10B?
    add_decision_box(3, 7.5, 4, 0.8, 'Is base model > 10B\nparameters?')
    add_arrow(5, 7.5, 7.5, 7.5, 'Yes')
    add_action_box(7.8, 7, 2, 0.6, '#FF6B35', 'Prompt Tuning\ncompetitive')
    add_arrow(5, 7.5, 5, 6.8, 'No')
    
    # Q2: Storage constrained?
    add_decision_box(3, 5.8, 4, 0.8, 'Is storage/memory\nseverely constrained?')
    add_arrow(5, 5.8, 7.5, 5.8, 'Yes')
    add_action_box(7.8, 5.3, 2, 0.6, '#FF6B35', 'Prompt Tuning\n15,360 params')
    add_arrow(5, 5.8, 5, 5.1, 'No')
    
    # Q3: Domain shift?
    add_decision_box(3, 4.1, 4, 0.8, 'Do training/deployment\ndomains differ?')
    add_arrow(5, 4.1, 7.5, 4.1, 'Yes')
    add_action_box(7.8, 3.6, 2, 0.6, '#800080', 'Lean Prompt\nTuning')
    add_arrow(5, 4.1, 5, 3.4, 'No')
    
    # Q4: Small data?
    add_decision_box(3, 2.4, 4, 0.8, 'Is labeled data\n< 200 samples?')
    add_arrow(5, 2.4, 7.5, 2.4, 'Yes')
    add_action_box(7.8, 1.9, 2, 0.6, '#808080', 'LoRA degrades\nless')
    add_arrow(5, 2.4, 5, 1.7, 'No')
    
    # Terminal
    add_terminal(2.5, 0.5, 5, 0.8, 'Use LoRA r=8 — Default Recommendation')
    
    ax.set_title('Fig. 7 — PEFT Method Selection Decision Tree', 
                 fontsize=12, fontweight='bold', pad=20)
    
    save_fig(fig, 'fig7_decision_flowchart.png')


def main():
    """Generate all figures"""
    import os
    
    # Create figures directory
    fig_dir = '/Users/ankit/Desktop/SabinaResearchPaper/LLM-Parameter-Efficient-Fine-Tuning-with-LoRA/figures'
    os.makedirs(fig_dir, exist_ok=True)
    
    print("Generating Figure 1: Memory and Storage Comparison...")
    create_figure1_memory_storage()
    
    print("Generating Figure 2: LoRA Architecture Diagram...")
    create_figure2_lora_architecture()
    
    print("Generating Figure 3: Pipeline Flowchart...")
    create_figure3_pipeline_flowchart()
    
    print("Generating Figure 4: Rank vs Accuracy...")
    create_figure4_rank_accuracy()
    
    print("Generating Figure 5: Learning Curve...")
    create_figure5_learning_curve()
    
    print("Generating Figure 6: Domain Shift Chart...")
    create_figure6_domain_shift()
    
    print("Generating Figure 7: Decision Flowchart...")
    create_figure7_decision_flowchart()
    
    print(f"\nAll figures saved to: {fig_dir}")
    print("Figures generated successfully!")


if __name__ == "__main__":
    main()
