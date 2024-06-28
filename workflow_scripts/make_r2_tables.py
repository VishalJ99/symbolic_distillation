import os
import json
import argparse
import pandas as pd
import numpy as np
from icecream import ic


def read_r2_values(file_path):
    """Read and return the average R2 values from the text file."""
    with open(file_path, 'r') as file:
        data = eval(file.read())
        return np.mean(data)


def format_values(df):
    """Format all numeric values in the DataFrame to two significant figures."""
    for col in df.columns:
        df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
    return df


def create_latex_table(df, label):
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
    latex += "\\caption{Average R2 Values for different strategies}\n"
    latex += f"\\label{{tab:{label}}}\n"
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
    data = {}

    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith("R2_stats.txt"):
                sim, dim = root.split("/")[-3].split("_")
                strat = root.split("/")[-2]
                sim_dim = f"{sim}-{dim[:-1]}"
                file_path = os.path.join(root, file)
                r2_value = read_r2_values(file_path)
                if sim_dim not in data:
                    data[sim_dim] = {}
                data[sim_dim][strat] = r2_value
    

    os.makedirs(output_dir, exist_ok=True)

    for sim_dim, values in data.items():
        df = pd.DataFrame(values, index=[0]).transpose()
        ic(df)
        df.columns = ['R2']
        df = df.reindex(desired_order)
        ic(df)
        exit(1)
        formatted_df = format_values(df)
        csv_path = os.path.join(output_dir, f"{sim_dim.lower()}_r2_table.csv")
        formatted_df.to_csv(csv_path, index_label="Strategy")
        print(f"Saved R2 table for {sim_dim} to {csv_path}")

        if save_tex:
            tex_content = create_latex_table(formatted_df, sim_dim.lower())
            tex_path = os.path.join(output_dir, f"{sim_dim.lower()}_r2_table.tex")
            with open(tex_path, "w") as tex_file:
                tex_file.write(tex_content)
            print(f"Saved R2 LaTeX table for {sim_dim} to {tex_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and tabulate R2 values")
    parser.add_argument("results_dir", type=str, help="Directory containing results")
    parser.add_argument("output_dir", type=str, help="Directory to save output CSV and LaTeX files")
    parser.add_argument("--save_tex", action="store_true", help="Save tables in LaTeX format as well")
    args = parser.parse_args()

    main(args.results_dir, args.output_dir, args.save_tex)
    print("[SUCCESS]")
