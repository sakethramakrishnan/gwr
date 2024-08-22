import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.stats.multicomp as mc

# Step 1: Load the CSV file
file_path = 'al_sn_liquid_1/model_performance_.8.csv'  # Replace with your actual file path
results_df = pd.read_csv(file_path)

# Set the number of decimal places for rounding
decimal_places = 10

# Step 2: Perform ANOVA on the Testing MSE
anova_result = stats.f_oneway(
    *(results_df[results_df['Model'] == model]['Testing MSE'] for model in results_df['Model'].unique())
)

# Print the ANOVA p-value, rounded to the specified number of decimal places
print(f"ANOVA p-value: {anova_result.pvalue:.{decimal_places}f}")

# Initialize a list to store result strings for saving to a file
results_text = []

# Append ANOVA result to the list, rounded to the specified number of decimal places
results_text.append(f"ANOVA p-value: {anova_result.pvalue:.{decimal_places}f}\n")

# Step 3: Post-hoc analysis if ANOVA is significant
if anova_result.pvalue < 0.05:
    print("Significant differences found, proceeding with post-hoc tests.")
    
    # Perform pairwise comparisons using Tukey's HSD
    comp = mc.MultiComparison(results_df['Testing MSE'], results_df['Model'])
    post_hoc_res = comp.tukeyhsd()
    
    # Print and store the post-hoc test results
    print(post_hoc_res.summary())
    results_text.append("Significant differences found. Post-hoc pairwise comparisons (Tukey's HSD):\n")
    results_text.append(post_hoc_res.summary().as_text())
    
    # Step 4: Determine the best model
    means = results_df.groupby('Model')['Testing MSE'].mean()
    best_model_name = means.idxmin()
    best_model_mse = means.min()

    # Check if the best model is statistically different from the others
    significant = post_hoc_res.reject[post_hoc_res.meandiffs.argmin()]

    if significant:
        results_text.append(f"\nBest Model: {best_model_name} with Testing MSE = {best_model_mse:.{decimal_places}f} (statistically different from others)\n")
        print(f"Best Model: {best_model_name} with Testing MSE = {best_model_mse:.{decimal_places}f} (statistically different from others)")
    else:
        results_text.append(f"\nBest Model: {best_model_name} with Testing MSE = {best_model_mse:.{decimal_places}f} (not statistically different from others)\n")
        print(f"Best Model: {best_model_name} with Testing MSE = {best_model_mse:.{decimal_places}f} (not statistically different from others)")
else:
    print("No significant differences found between models.")
    results_text.append("No significant differences found between models.\n")
    
    # Select the model with the lowest average MSE as the best model
    means = results_df.groupby('Model')['Testing MSE'].mean()
    best_model_name = means.idxmin()
    best_model_mse = means.min()

    results_text.append(f"\nBest Model: {best_model_name} with Testing MSE = {best_model_mse:.{decimal_places}f} (based on lowest MSE)\n")
    print(f"Best Model: {best_model_name} with Testing MSE = {best_model_mse:.{decimal_places}f} (based on lowest MSE)")

# Step 5: Write the results to a file
output_file = "anova_results.txt"
with open(output_file, "w") as f:
    f.writelines(results_text)

print(f"ANOVA and post-hoc results saved to '{output_file}'.")
