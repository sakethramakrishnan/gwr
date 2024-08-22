import pandas as pd
import matplotlib.pyplot as plt

def plot_scatter_plots(file_path, plot_corr_matrix=True):

    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.txt'):
        df = pd.read_csv(file_path, delim_whitespace=True)
    else:
        print("Unsupported file format. Please use .txt or .xlsx files.")
        return
    
    # Step 2: Identify the dependent and independent variables
    dependent_var = None
    independent_vars = []
    
    for col in df.columns:
        if 'gibbs' in col.lower():
            dependent_var = col
        else:
            independent_vars.append(col)
    
    # Check if we found the dependent variable
    if not dependent_var:
        print("No dependent variable containing 'gibbs' found.")
        return

    if len(independent_vars) < 2:
        print("Not enough independent variables for 3D plotting.")

    elif len(independent_vars) == 2:
        # Step 4: Create a 3D scatterplot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Plotting the data
        ax.scatter(df[independent_vars[0]], df[independent_vars[1]], df[dependent_var], c='blue', marker='o')

        # Setting labels and title
        ax.set_xlabel(independent_vars[0])
        ax.set_ylabel(independent_vars[1])
        ax.set_zlabel(dependent_var)
        ax.set_title(f'3D Scatterplot: {independent_vars[0]}, {independent_vars[1]} vs. {dependent_var}')

        ax.view_init(elev=20, azim=160)  # Adjust elevation and azimuth

        # Show the 3D plot
        plt.show()
    else:
        print("The dataset has more than 3 columns. Proceeding to create 2D scatter plots.")

    # Step 3: Create 2D scatterplots for each independent variable against the dependent variable
    for x_col in independent_vars:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plotting the data
        ax.scatter(df[x_col], df[dependent_var], c='blue', marker='o')
        
        # Setting labels and title
        ax.set_xlabel(x_col)
        ax.set_ylabel(dependent_var)
        ax.set_title(f'Scatterplot: {x_col} vs. {dependent_var}')
        
        # Save the 2D scatterplot
        file_name = f'scatter_{x_col}_vs_{dependent_var}.png'
        #plt.savefig(file_name)
        print(f"2D scatterplot saved as '{file_name}'")
        
        # Show the 2D plot
        plt.show()

    # Step 4: Filter numeric columns and plot the correlation matrix if flag is True
    if plot_corr_matrix:
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[float, int])
        
        # Compute the correlation matrix
        corr_mat = numeric_df.corr(method='pearson')
        
        # Plot the heatmap
        plt.figure(figsize=(20, 10))
        sns.heatmap(corr_mat, vmax=1, square=True, annot=True, cmap='icefire')
        
        # Save and show the correlation matrix
        corr_file_name = 'correlation_matrix.png'
        #plt.savefig(corr_file_name)
        print(f"Correlation matrix saved as '{corr_file_name}'")
        
        plt.show()

# Example usage:
file_path = 'al_sn_liquid_1_and_2.xlsx'  # Replace with your actual file path
plot_scatter_plots(file_path)


# al_sn_liquid_gibbs_mole_frac.txt DOES HAVE A CORRELATION BTWN MOLE FRACTION SN AND GIBBS

