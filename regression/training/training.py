import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import (LinearRegression, Ridge, RidgeCV, SGDRegressor, ElasticNet, 
                                  ElasticNetCV, Lars, LarsCV, Lasso, LassoCV, LassoLars, LassoLarsCV, 
                                  OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV, 
                                  ARDRegression, BayesianRidge, MultiTaskElasticNet, MultiTaskElasticNetCV,
                                  MultiTaskLasso, MultiTaskLassoCV, HuberRegressor, QuantileRegressor,
                                  RANSACRegressor, TheilSenRegressor, GammaRegressor, PoissonRegressor,
                                  TweedieRegressor, PassiveAggressiveRegressor)

system='al_sn_liquid_1/al_sn_liquid_1'
f = pd.read_csv(f"{system}.txt", delimiter='\t')
#f = pd.read_csv("nb_temp_gibbs.txt", delimiter=' ')

# Parameters
prop_samples_for_train = 0.75
n_splits = 10  # Number of different train-test splits

# Define the number of decimal places for formatting
decimal_places = 10

data=DataFrame(f)
# all columns
# 'Gibbs_energy','Mass_fraction_Nb','Mass_fraction_Al','Mass_fraction_Ti','Mass_fraction_Mo','Mass_fraction_Fe','Mass_fraction_Ni','Mass_fraction_Cr','Temperature_[K]','temp_class'
# our_cols=['Gibbs_energy','Mass_fraction_Nb','Mass_fraction_Al','Mass_fraction_Ti','Mass_fraction_Mo','Mass_fraction_Fe','Mass_fraction_Ni','Mass_fraction_Cr','Temperature_[K]']
# data=data.reindex(columns=our_cols)


X_data=data.dtypes[data.dtypes!='object'].index
X_train=data[X_data]
#X_train = normalize(X_train)
#X_train_pd = pd.DataFrame(X_train, columns=X_data.)



# Filling all Null values
X_train=X_train.fillna(0)
columns=X_train.columns.tolist()
y=X_train['Gibbs_energy']
#X_train.drop(['Gibbs_energy_1','Gibbs_energy_2'],axis=1,inplace=True)
X_train.drop(['Gibbs_energy',],axis=1,inplace=True)


X_Train=X_train.values
X_Train=np.asarray(X_Train)
# Finding normalised array of X_Train
#X_std=StandardScaler().fit_transform(X_Train)
# X_std=MinMaxScaler().fit_transform(X_Train)
y_2d = np.reshape(y,(y.shape[0],1))
#y_std=StandardScaler().fit_transform(y_2d)
# y_std=MinMaxScaler().fit_transform(y_2d)

X_std = X_Train
y_std = y_2d


# Function to plot predictions vs. true values and print MSE for training and testing
def plot_predictions_and_print_mse(model, x_train, y_train, x_test, y_test, model_name, decimal_places, plot=True, print_mse=True):
    # Predict the target values
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    # Calculate MSE for training and testing sets
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    
    # Print MSE for training and testing sets if print_mse is True
    if print_mse:
        print(f"MSE for {model_name} - Training Set: {mse_train:.{decimal_places}f}")
        print(f"MSE for {model_name} - Testing Set: {mse_test:.{decimal_places}f}")
    
    # Plot predictions vs. true values if plot is True
    if plot:
        # Create a DataFrame for plotting
        preds = pd.DataFrame({"preds": y_test_pred.squeeze(), "true": y_test.squeeze()})
        preds["residuals"] = preds["true"] - preds["preds"]

        # Set the plot size
        plt.figure(figsize=(6.0, 6.0))
        
        # Plot predictions vs. true values
        ax = preds.plot(x="true", y="preds", kind="scatter", color='blue', label='Prediction')
        preds.plot(x="true", y="true", color="red", kind="scatter", ax=ax, label='Ground Truth')
        
        # Add title and legend
        plt.title(f"Prediction vs. True Values ({model_name})")
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.legend()
        
        # Show plot
        plt.show()

    return mse_train, mse_test

# List of models to test
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "RidgeCV": RidgeCV(),
    "SGDRegressor": SGDRegressor(),
    "ElasticNet": ElasticNet(alpha=1.0),
    "ElasticNetCV": ElasticNetCV(),
    "Lars": Lars(),
    "LarsCV": LarsCV(),
    "Lasso": Lasso(alpha=1.0),
    "LassoCV": LassoCV(),
    "LassoLars": LassoLars(),
    "LassoLarsCV": LassoLarsCV(),
    "OrthogonalMatchingPursuit": OrthogonalMatchingPursuit(),
    "OrthogonalMatchingPursuitCV": OrthogonalMatchingPursuitCV(),
    "ARDRegression": ARDRegression(),
    "BayesianRidge": BayesianRidge(),
    "MultiTaskElasticNet": MultiTaskElasticNet(),
    "MultiTaskElasticNetCV": MultiTaskElasticNetCV(),
    "MultiTaskLasso": MultiTaskLasso(),
    "MultiTaskLassoCV": MultiTaskLassoCV(),
    "HuberRegressor": HuberRegressor(),
    "QuantileRegressor": QuantileRegressor(),
    "RANSACRegressor": RANSACRegressor(),
    "TheilSenRegressor": TheilSenRegressor(),
    "GammaRegressor": GammaRegressor(),
    "PoissonRegressor": PoissonRegressor(),
    "TweedieRegressor": TweedieRegressor(),
    "PassiveAggressiveRegressor": PassiveAggressiveRegressor(),
    # "K-Nearest Neighbors Regressor": KNeighborsRegressor(n_neighbors=5),
    # "Decision Tree Regressor": DecisionTreeRegressor(max_depth=None),
    # "Kernel Ridge Regression": KernelRidge(alpha=1.0, kernel='rbf'), takes too long and is high loss
}

# Dictionary to store MSE values
mse_results = {}


# Loop through models and perform n_splits cross-validation
for name, model in models.items():
    if hasattr(model, 'fit'):
        try:
            mse_train_splits = []
            mse_test_splits = []
            
            for i in range(n_splits):
                # Split the data into training and test sets
                x_train, x_test, y_train, y_test = train_test_split(
                    X_std, 
                    y_std, 
                    test_size=1 - prop_samples_for_train,  
                    random_state=i  # Seed for reproducibility, changes each split
                )
                
                model.fit(x_train, y_train)
                mse_train, mse_test = plot_predictions_and_print_mse(model, x_train, y_train, x_test, y_test, name, decimal_places, plot=False, print_mse=False)
                
                mse_train_splits.append(mse_train)
                mse_test_splits.append(mse_test)

            mse_results[name] = {
                "MSE Train": mse_train_splits,
                "MSE Test": mse_test_splits
            }
        
        except Exception as e:
            print(f"Error with {name}: {e}")

# Sort models by average test MSE and get the top n models
top_n = 5  # Adjust this number to the desired number of top models
sorted_models = sorted(mse_results.items(), key=lambda item: np.mean(item[1]["MSE Test"]))
top_models = sorted_models[:top_n]

print(f"Top {top_n} models with lowest average test MSE:")
for model_name, mse in top_models:
    print(f"{model_name}: Testing MSE = {np.mean(mse['MSE Test']):.{decimal_places}f}")

# Save results to a text file
with open("model_performance_no_std.txt", "w") as f:
    for model_name, mse in mse_results.items():
        f.write(f"{model_name}:\n")
        for i in range(n_splits):
            f.write(f"  Split {i+1} - Training MSE: {mse['MSE Train'][i]:.{decimal_places}f}, Testing MSE: {mse['MSE Test'][i]:.{decimal_places}f}\n")
    
    f.write("\nTop Models with Lowest Average Test MSE:\n")
    for model_name, mse in top_models:
        f.write(f"{model_name}: Testing MSE = {np.mean(mse['MSE Test']):.{decimal_places}f}\n")

print(f"Results saved to 'model_performance.txt'.")

# Save results to a CSV file
# Convert the results into a DataFrame where each row corresponds to one split
all_results = []
for model_name, mse in mse_results.items():
    for i in range(n_splits):
        all_results.append({
            "Model": model_name,
            "Split": i + 1,
            "Training MSE": mse['MSE Train'][i],
            "Testing MSE": mse['MSE Test'][i]
        })

results_df = pd.DataFrame(all_results)
results_df.to_csv("model_performance_no_std.csv", index=False)

print(f"Results saved to 'model_performance.csv'.")
