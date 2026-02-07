import glob
import json
import pandas as pd
import numpy as np

def aggregate_strategy(strategy_suffix):
    print(f"Aggregating results for {strategy_suffix}...")
    files = glob.glob(f"log/metrics_*_{strategy_suffix}_seed*.json")
    
    data = []
    
    for f in files:

        base = f.replace("metrics_", "").replace(f"_{strategy_suffix}", "")
        parts = base.split("_seed")
        method = parts[0]
        seed = parts[1].replace(".json", "")
        
        with open(f, 'r') as fd:
            metrics = json.load(fd)
            # Flatten
            row = {
                'Method': method,
                'Seed': int(seed),
                'F1_Macro': metrics.get('f1_macro', 0),
                'F1_Micro': metrics.get('f1_micro', 0),
                'AUC_Macro': metrics.get('auc_macro', 0),
                'F1_Minority': metrics.get('minority_f1_macro', 0)
            }
            # Add per-class F1 if available
            for k, v in metrics.items():
                # Capture f1_, prec_, rec_, auc_ for all classes
                if any(k.startswith(prefix) for prefix in ["f1_", "prec_", "rec_", "auc_"]):
                     if k not in ['f1_macro', 'f1_micro', 'auc_macro', 'prec_macro', 'rec_macro']:
                        row[k] = v
            data.append(row)
            
    if not data:
        print("No data found.")
        return None
        
    df = pd.DataFrame(data)
    
    # Aggregation
    # Group by Method
    # Calc Mean and Std for numeric cols
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Seed' in numeric_cols: numeric_cols.remove('Seed')
    
    agg_funcs = {col: ['mean', 'std'] for col in numeric_cols}
    report = df.groupby('Method').agg(agg_funcs)
    
    # Flatten columns
    report.columns = ['_'.join(col).strip() for col in report.columns.values]
    report = report.reset_index()
    
    # Sort by F1_Macro_mean
    report = report.sort_values(by='F1_Macro_mean', ascending=False)
    
    return report

def format_table(df):
    # Select key columns for cleaner display
    display_df = pd.DataFrame()
    display_df['Method'] = df['Method']
    
    def fmt(m, s):
        if pd.isna(s) or s == 0:
            return f"{m:.4f}"
        return f"{m:.4f} ± {s:.4f}"
    
    display_df['Macro F1'] = df.apply(lambda x: fmt(x['F1_Macro_mean'], x['F1_Macro_std']), axis=1)
    if 'F1_Minority_mean' in df.columns:
        display_df['Minority F1'] = df.apply(lambda x: fmt(x['F1_Minority_mean'], x['F1_Minority_std']), axis=1)
    display_df['Micro F1'] = df.apply(lambda x: fmt(x['F1_Micro_mean'], x['F1_Micro_std']), axis=1)
    
    # Add Minority Class Metrics
    minority_classes = ['Falls', 'Fractures', 'GI_Bleed', 'Stroke']
    
    for cls in minority_classes:
        # F1
        f1_col_mean = f"f1_{cls}_mean"
        f1_col_std = f"f1_{cls}_std"
        if f1_col_mean in df.columns:
            display_df[f"{cls} F1"] = df.apply(lambda x: fmt(x[f1_col_mean], x[f1_col_std]), axis=1)
            
        # Percision
        prec_col_mean = f"prec_{cls}_mean"
        prec_col_std = f"prec_{cls}_std"
        if prec_col_mean in df.columns:
            display_df[f"{cls} Prec"] = df.apply(lambda x: fmt(x[prec_col_mean], x[prec_col_std]), axis=1)
            
        # Recall
        rec_col_mean = f"rec_{cls}_mean"
        rec_col_std = f"rec_{cls}_std"
        if rec_col_mean in df.columns:
            display_df[f"{cls} Rec"] = df.apply(lambda x: fmt(x[rec_col_mean], x[rec_col_std]), axis=1)

    return display_df

def main():
    # 1X
    df_1x = aggregate_strategy("Target_Multiplier_1X")
    if df_1x is not None:
        print("\n=== 1X Strategy Results ===")
        print(format_table(df_1x).to_markdown(index=False))
        df_1x.to_csv("benchmark_results_1x_full.csv", index=False)
        format_table(df_1x).to_csv("benchmark_results_1x_summary.csv", index=False)

    # 3X
    df_3x = aggregate_strategy("Target_Multiplier_3X")
    if df_3x is not None:
        print("\n=== 3X Strategy Results ===")
        print(format_table(df_3x).to_markdown(index=False))
        df_3x.to_csv("benchmark_results_3x_full.csv", index=False)
        format_table(df_3x).to_csv("benchmark_results_3x_summary.csv", index=False)

if __name__ == "__main__":
    main()
