import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Define the file to read
CSV_FILE = "evaluation_results.csv"
OUTPUT_FILE_V2_DN121 = "DenseNet121_Comparison_v2.png"
OUTPUT_FILE_V2_DN201 = "DenseNet201_Comparison_v2.png"

def create_comprehensive_comparison_plots():
    """
    Loads the evaluation results and generates two comprehensive 2x2 comparison plots:
    1. DenseNet121: Time, Accuracy, F1, Precision/Recall
    2. DenseNet201: Time, Accuracy, F1, Precision/Recall
    """
    
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found. Please run the evaluation first.")
        return

    print(f"Loading data from {CSV_FILE}")
    df = pd.read_csv(CSV_FILE)
    
    # --- 1. Data Preparation ---
    
    try:
        # Get baseline stats for both models
        dn121_baseline = df.loc[df['model_name'] == 'DenseNet121_TF_FP32'].iloc[0]
        dn201_baseline = df.loc[df['model_name'] == 'DenseNet201_TF_FP32'].iloc[0]
        
        dn121_baseline_time = dn121_baseline['avg_time_ms']
        dn121_baseline_acc = dn121_baseline['accuracy']
        dn121_baseline_f1 = dn121_baseline['f1_score']
        
        dn201_baseline_time = dn201_baseline['avg_time_ms']
        dn201_baseline_acc = dn201_baseline['accuracy']
        dn201_baseline_f1 = dn201_baseline['f1_score']

        # Function to apply speedup calculation
        def get_speedup(row):
            if row['base_model'] == 'DenseNet121':
                return dn121_baseline_time / row['avg_time_ms']
            elif row['base_model'] == 'DenseNet201':
                return dn201_baseline_time / row['avg_time_ms']
            return 1.0

        df['speedup_vs_tf'] = df.apply(get_speedup, axis=1)
        
    except IndexError:
        print("Error: Could not find 'DenseNet121_TF_FP32' or 'DenseNet201_TF_FP32' in the CSV.")
        return
    except Exception as e:
        print(f"An error occurred during data preparation: {e}")
        return

    # Define a logical order for the x-axis
    quant_order = [
        'TF_FP32', 'ONNX_FP32', 'ONNX_FP16', 
        'TRT_FP32', 'TRT_FP16', 'TRT_INT8'
    ]
    
    # Filter the data for each base model
    df_dn121 = df[df['base_model'] == 'DenseNet121'].copy()
    df_dn201 = df[df['base_model'] == 'DenseNet201'].copy()

    # --- 2. Plotting Function ---
    
    def plot_model_comparison_v2(df_model, baselines, model_title, filename):
        """
        Generates a 2x2 figure with four subplots: Time, Accuracy, F1, Precision/Recall.
        """
        print(f"Generating comprehensive plot: {filename}")
        
        # Set up the figure and axes (2x2)
        fig, axes = plt.subplots(2, 2, figsize=(24, 18))
        
        # --- Plot 0,0: Average Inference Time (ms) ---
        ax = axes[0, 0]
        sns.barplot(
            x='quant_type', 
            y='avg_time_ms', 
            data=df_model, 
            order=quant_order, 
            ax=ax,
            palette='viridis'
        )
        ax.set_title("Average Inference Time (ms) - Lower is Better", fontsize=16)
        ax.set_xlabel(None) # Remove x-label for top row
        ax.set_ylabel("Time (ms)", fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        
        # Add speedup labels on top of the bars
        data_to_plot = df_model.set_index('quant_type').reindex(quant_order).reset_index()
        for i, (speedup, time) in enumerate(zip(data_to_plot['speedup_vs_tf'], data_to_plot['avg_time_ms'])):
            ax.text(
                i, time, 
                f"{speedup:.1f}x Speedup", 
                ha='center', 
                va='bottom', 
                fontsize=12,
                fontweight='bold'
            )

        # --- Plot 0,1: Accuracy ---
        ax = axes[0, 1]
        sns.barplot(
            x='quant_type', 
            y='accuracy', 
            data=df_model, 
            order=quant_order, 
            ax=ax,
            palette='plasma'
        )
        ax.set_title("Model Accuracy - Higher is Better", fontsize=16)
        ax.set_xlabel(None) # Remove x-label for top row
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        
        # Add baseline accuracy line
        baseline_acc = baselines['acc']
        ax.axhline(
            y=baseline_acc, 
            color='red', 
            linestyle='--', 
            linewidth=2,
            label=f"Baseline Accuracy ({baseline_acc:.4f})"
        )
        ax.legend()
        # Set Y-axis limit to zoom in
        min_acc = df_model['accuracy'].min()
        ax.set_ylim(min(min_acc - 0.05, 0.5), 1.0)
        
        # --- Plot 1,0: F1-Score (Weighted) ---
        ax = axes[1, 0]
        sns.barplot(
            x='quant_type', 
            y='f1_score', 
            data=df_model, 
            order=quant_order, 
            ax=ax,
            palette='cividis'
        )
        ax.set_title("F1-Score (Weighted) - Higher is Better", fontsize=16)
        ax.set_xlabel("Quantization Type", fontsize=12)
        ax.set_ylabel("F1-Score", fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        
        # Add baseline F1 line
        baseline_f1 = baselines['f1']
        ax.axhline(
            y=baseline_f1, 
            color='red', 
            linestyle='--', 
            linewidth=2,
            label=f"Baseline F1-Score ({baseline_f1:.4f})"
        )
        ax.legend()
        min_f1 = df_model['f1_score'].min()
        ax.set_ylim(min(min_f1 - 0.05, 0.5), 1.0)

        # --- Plot 1,1: Precision & Recall (Weighted) ---
        ax = axes[1, 1]
        # We need to "melt" the dataframe to plot grouped bars
        df_pr = df_model.melt(
            id_vars=['quant_type'], 
            value_vars=['precision', 'recall'],
            var_name='Metric', 
            value_name='Score'
        )
        
        sns.barplot(
            x='quant_type', 
            y='Score', 
            hue='Metric',
            data=df_pr, 
            order=quant_order, 
            ax=ax,
            palette='coolwarm'
        )
        ax.set_title("Precision & Recall (Weighted)", fontsize=16)
        ax.set_xlabel("Quantization Type", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        
        # Set Y-axis limit
        min_pr = min(df_model['precision'].min(), df_model['recall'].min())
        ax.set_ylim(min(min_pr - 0.05, 0.5), 1.0)
        ax.legend(title='Metric')

        # --- Final Figure ---
        fig.suptitle(model_title, fontsize=24, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
        
        # Save the figure
        plt.savefig(filename)
        print(f"Plot saved to {filename}")

    # --- 3. Generate Both Plots ---
    
    # DenseNet121
    plot_model_comparison_v2(
        df_dn121, 
        {'acc': dn121_baseline_acc, 'f1': dn121_baseline_f1}, 
        "DenseNet121: Comprehensive Performance Analysis",
        OUTPUT_FILE_V2_DN121
    )
    
    # DenseNet201
    plot_model_comparison_v2(
        df_dn201, 
        {'acc': dn201_baseline_acc, 'f1': dn201_baseline_f1}, 
        "DenseNet201: Comprehensive Performance Analysis",
        OUTPUT_FILE_V2_DN201
    )

if __name__ == "__main__":
    create_comprehensive_comparison_plots()