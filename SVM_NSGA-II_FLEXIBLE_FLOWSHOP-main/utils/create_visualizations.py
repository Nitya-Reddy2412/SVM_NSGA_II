"""
Simple Visualization of FJSP NSGA-II Results
Shows the improvements achieved by the algorithm
"""

import matplotlib.pyplot as plt
import numpy as np

def create_improvement_visualization():
    """Create a comprehensive visualization of the results"""
    
    # Data from the quick test
    objectives = ['Makespan', 'Waiting Time', 'Weighted Time']
    initial_values = [7.00, 3.00, 28.80]
    final_values = [4.00, 0.00, 8.60]
    improvements = [42.86, 100.00, 70.14]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Bar chart comparing initial vs final
    ax1 = plt.subplot(2, 3, 1)
    x = np.arange(len(objectives))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, initial_values, width, label='Initial', 
                    color='coral', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width/2, final_values, width, label='Final', 
                    color='lightgreen', alpha=0.8, edgecolor='black')
    
    ax1.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax1.set_title('Initial vs Final Values', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(objectives)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)
    
    # 2. Improvement percentages
    ax2 = plt.subplot(2, 3, 2)
    colors_imp = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    bars = ax2.bar(objectives, improvements, color=colors_imp, 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Percentage Improvements', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 110)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 3. Convergence simulation (based on actual results)
    ax3 = plt.subplot(2, 3, 3)
    generations = [1, 10, 20, 30]
    makespan_values = [7.0, 5.0, 4.0, 4.0]
    
    ax3.plot(generations, makespan_values, 'b-o', linewidth=2, 
             markersize=8, label='Best Makespan')
    ax3.fill_between(generations, makespan_values, alpha=0.3)
    ax3.set_xlabel('Generation', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Makespan', fontsize=11, fontweight='bold')
    ax3.set_title('Makespan Convergence', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    
    # 4. Reduction visualization
    ax4 = plt.subplot(2, 3, 4)
    reductions = [initial - final for initial, final in zip(initial_values, final_values)]
    colors_red = ['#E74C3C', '#3498DB', '#F39C12']
    
    bars = ax4.barh(objectives, reductions, color=colors_red, 
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Reduction Amount', fontsize=12, fontweight='bold')
    ax4.set_title('Absolute Reductions', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax4.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.2f}',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    # 5. Pareto front size over generations
    ax5 = plt.subplot(2, 3, 5)
    pareto_sizes = [5, 14, 30, 29]
    
    ax5.plot(generations, pareto_sizes, 'g-s', linewidth=2, 
             markersize=8, label='Pareto Front Size')
    ax5.fill_between(generations, pareto_sizes, alpha=0.3, color='green')
    ax5.set_xlabel('Generation', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Number of Solutions', fontsize=11, fontweight='bold')
    ax5.set_title('Pareto Front Growth', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=10)
    
    # 6. Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"""
    NSGA-II PERFORMANCE SUMMARY
    {'='*40}
    
    Instance: Kacem 4x5
    - Jobs: 4
    - Machines: 5
    - Operations: 9
    
    Execution Details:
    - Population: 30
    - Generations: 30
    - Time: 0.24 seconds
    
    Final Results:
    - Pareto Solutions: 29
    - Makespan: {final_values[0]:.2f} (↓{improvements[0]:.1f}%)
    - Waiting Time: {final_values[1]:.2f} (↓{improvements[1]:.1f}%)
    - Weighted Time: {final_values[2]:.2f} (↓{improvements[2]:.1f}%)
    
    Status: ✅ ALL OBJECTIVES IMPROVED
    
    Average Improvement: {np.mean(improvements):.1f}%
    """
    
    ax6.text(0.1, 0.5, summary_text, fontsize=10, 
             verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Overall title
    fig.suptitle('NSGA-II for Flexible Job Shop Scheduling - Performance Report', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Save figure
    output_path = 'd:\\nithya\\results_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    plt.show()


def create_algorithm_flowchart():
    """Create a simple flowchart showing the algorithm process"""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')
    
    # Define boxes
    boxes = [
        ("START", 0.5, 0.95, 'lightblue'),
        ("Initialize Population\n(Random Schedules)", 0.5, 0.85, 'lightgreen'),
        ("Evaluate Objectives\n(Makespan, Waiting, Weighted)", 0.5, 0.75, 'lightyellow'),
        ("Fast Non-Dominated Sort", 0.5, 0.65, 'lightcoral'),
        ("Calculate Crowding Distance", 0.5, 0.55, 'lightcoral'),
        ("Selection (Tournament)", 0.5, 0.45, 'lightblue'),
        ("Crossover (POX)", 0.5, 0.35, 'lightgreen'),
        ("Mutation 1 & 2\n(with Decay)", 0.5, 0.25, 'lightgreen'),
        ("Create New Population", 0.5, 0.15, 'lightyellow'),
        ("Max Generations?", 0.5, 0.05, 'orange'),
    ]
    
    # Draw boxes
    for label, x, y, color in boxes:
        if label == "Max Generations?":
            # Diamond shape for decision
            ax.add_patch(plt.Polygon([(x, y+0.04), (x+0.08, y), 
                                      (x, y-0.04), (x-0.08, y)],
                                     facecolor=color, edgecolor='black', linewidth=2))
            ax.text(x, y, label, ha='center', va='center', 
                   fontsize=9, fontweight='bold')
        else:
            bbox = dict(boxstyle='round,pad=0.5', facecolor=color, 
                       edgecolor='black', linewidth=2)
            ax.text(x, y, label, ha='center', va='center', 
                   fontsize=10, fontweight='bold', bbox=bbox)
    
    # Draw arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    
    for i in range(len(boxes)-1):
        _, x1, y1, _ = boxes[i]
        _, x2, y2, _ = boxes[i+1]
        ax.annotate('', xy=(x2, y2+0.04), xytext=(x1, y1-0.04),
                   arrowprops=arrow_props)
    
    # Loop back arrow
    ax.annotate('No', xy=(0.5, 0.65), xytext=(0.42, 0.05),
               arrowprops=dict(arrowstyle='->', lw=2, color='red',
                             connectionstyle="arc3,rad=.5"))
    
    # End arrow
    ax.annotate('Yes', xy=(0.7, -0.02), xytext=(0.58, 0.05),
               arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.text(0.75, -0.02, 'END\n(Return Pareto Front)', 
           ha='center', va='center', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen',
                    edgecolor='black', linewidth=2))
    
    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(-0.1, 1.0)
    
    plt.title('NSGA-II Algorithm Flow for FJSP', 
             fontsize=14, fontweight='bold', pad=20)
    
    output_path = 'd:\\nithya\\algorithm_flowchart.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Flowchart saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Generating visualizations...")
    print("\n1. Creating performance report...")
    create_improvement_visualization()
    
    print("\n2. Creating algorithm flowchart...")
    create_algorithm_flowchart()
    
    print("\n" + "="*80)
    print("VISUALIZATIONS COMPLETED!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. results_visualization.png - Performance report")
    print("  2. algorithm_flowchart.png - Algorithm flow diagram")
    print("\nThese visualizations show:")
    print("  ✓ Initial vs Final values comparison")
    print("  ✓ Percentage improvements")
    print("  ✓ Convergence behavior")
    print("  ✓ Pareto front growth")
    print("  ✓ Algorithm flowchart")
    print("="*80)
