import pickle as pkl
import argparse
import sympy as sp
import re

def format_var_names(var_list):
    formatted_list = []
    pattern = re.compile(r'^([a-zA-Z])(\d+)$')  # Pattern to match a letter followed by numbers

    for var in var_list:
        match = pattern.match(var)
        if match:
            # If it matches, format it as 'letter_{number}'
            formatted_var = sp.symbols(f"{match.group(1)}_{match.group(2)}")
        else:
            # If it does not match the pattern, use it directly as a symbol
            formatted_var = sp.symbols(var)
        formatted_list.append(formatted_var)
    return formatted_list


def display_equations(pkl_path):
    # Load the model data from the specified path
    with open(pkl_path, "rb") as f:
        symbolic_model = pkl.load(f)
        model = symbolic_model["model"]
        var_names = format_var_names(symbolic_model["var_names"])

        # Assuming 'model.sympy()' gives us sympy expressions
        equations = model.sympy()
        for i, eq in enumerate(equations):
            # Substituting x_i variables with actual variable names
            for j, name in enumerate(var_names):
                eq = eq.subs(sp.Symbol(f'x{j}'), name)
            print(f"Equation {i + 1}:")
            sp.pprint(eq, use_unicode=True)


def main():
    parser = argparse.ArgumentParser(description="Display formatted equations from a symbolic model pickle file.")
    parser.add_argument("pkl_path", help="Path to the symbolic model pickle file.")

    args = parser.parse_args()
    display_equations(args.pkl_path)


if __name__ == "__main__":
    main()
