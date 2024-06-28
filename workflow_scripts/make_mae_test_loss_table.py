import os
import json
import argparse
import pandas as pd
import numpy as np
from icecream import ic
def format_values(df):
    """Format all numeric values in the DataFrame to two significant figures."""
    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: f"{x:.6f}" if isinstance(x, (int, float)) else x
        )
    return df


def highlight_best(df, stat):
    """Apply LaTeX textbf to the best strategy for each simulation_dim."""
    # Ascertain whether lower or higher values are better based on the statistic
    for index, row in df.iterrows():
        best_val = float(row.min())
        # Apply bold formatting to the best value in the row
        df.loc[index] = row.apply(
            lambda x: f"\\textbf{{{x}}}" if x == f"{best_val:.4f}" else x
        )
    return df


def create_latex_table(df, stat):
    """Create a fully formatted LaTeX table block."""
    latex = "\\begin{table}[h]\n\\centering\n"
    latex += "\\begin{tabular}{l" + "c" * (df.shape[1]) + "}\n\\hline\n"
    # Column headers
    column_labels = " & ".join(df.columns)
    latex += f"Sim & {column_labels} \\\\\n\\hline\n"
    # Rows of the table
    for index, row in df.iterrows():
        row_data = " & ".join(row)
        latex += f"{index} & {row_data} \\\\\n"
    latex += "\\hline\n\\end{tabular}\n"
    latex += (
        "\\caption{Performance statistics for "
        + stat.replace("_", " ").title()
        + "}\n"
    )
    latex += "\\label{tab:" + stat + "}\n"
    latex += "\\end{table}\n"
    return latex


def main(results_dir, output_dir, save_tex):
    desired_order = [
        "Charge-2",
        "Charge-3",
        "r$^{-1}$-2",
        "r$^{-1}$-3",
        "r$^{-2}$-2",
        "r$^{-2}$-3",
        "Spring-2",
        "Spring-3",
    ]
    data = {
        "Mean": {},
        "Standard Deviation": {},
        "Median": {},
        "Min": {},
        "Max": {},
        "Upper Quartile": {},
        "Lower Quartile": {},
    }

    formatted_str_dict = {
        "standard": "Standard",
        "bottleneck": "Bottleneck",
        "l1": "L$_{1}$",
        "kl": "KL",
        "charge": "Charge",
        "r1": "r$^{-1}$",
        "r2": "r$^{-2}$",
        "spring": "Spring",
    }

    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith("test_results.json"):
                sim, dim = root.split("/")[-2].split("_")
                sim = formatted_str_dict.get(sim, sim)
                strat = root.split("/")[-1]
                strat = formatted_str_dict.get(strat, strat)
                sim_dim = f"{sim}-{dim[:-1]}"
                with open(os.path.join(root, file), "r") as f:
                    test_results = json.load(f)
                    for stat in data.keys():
                        if sim_dim not in data[stat]:
                            data[stat][sim_dim] = {}
                        data[stat][sim_dim][strat] = test_results[stat]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for stat, table in data.items():
        ic(stat, table)
        df = pd.DataFrame(table).transpose()
        ic(df)
        df = df.reindex(desired_order)
        ic(df)
        formatted_df = format_values(df.copy())
        ic(formatted_df)
        highlighted_df = highlight_best(formatted_df, stat)
        csv_path = os.path.join(
            output_dir, f"{stat.replace(' ', '_').lower()}_table.csv"
        )
        highlighted_df.to_csv(csv_path, index_label="Sim.")
        print(f"Saved {stat} table to {csv_path}")

        if save_tex:
            tex_content = create_latex_table(highlighted_df, stat)
            tex_path = os.path.join(
                output_dir, f"{stat.replace(' ', '_').lower()}_table.tex"
            )
            with open(tex_path, "w") as tex_file:
                tex_file.write(tex_content)
            print(f"Saved {stat} LaTeX table to {tex_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make a table of test loss for MAE"
    )
    parser.add_argument(
        "results_dir", type=str, help="Directory containing results"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save output CSV and LaTeX files",
    )
    parser.add_argument(
        "--save_tex",
        action="store_true",
        help="Save tables in LaTeX format as well",
    )
    args = parser.parse_args()

    main(args.results_dir, args.output_dir, args.save_tex)
    print("[SUCCESS]")
