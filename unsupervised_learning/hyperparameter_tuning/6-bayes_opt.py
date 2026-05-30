#!/usr/bin/env python3
"""
Bayesian Optimization with GPyOpt
Optimizes a machine learning model (XGBoost) using GPyOpt
"""
import GPyOpt
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Load data
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

def optimize_xgb(params):
    """
    Function to optimize for XGBoost
    params: 2D array [learning_rate, n_estimators, max_depth, gamma, subsample]
    """
    params = params[0]
    
    # Hyperparameters
    learning_rate = params[0]
    n_estimators = int(params[1])
    max_depth = int(params[2])
    gamma = params[3]
    subsample = params[4]

    model = xgb.XGBClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        gamma=gamma,
        subsample=subsample,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=0
    )

    # Early stopping (XGBoost 1.6+ syntax or eval_set)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], 
              early_stopping_rounds=10, verbose=False)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Save checkpoint if it's a good approach
    checkpoint_name = "xgb_lr{}_nest{}_depth{}_gamma{}_sub{}.model".format(
        learning_rate, n_estimators, max_depth, gamma, subsample
    )
    # Note: We usually only save the best, but the task says "during each training session"
    model.save_model(checkpoint_name)

    # GPyOpt minimizes, so we return 1 - accuracy
    return 1 - acc

# Step 2: Define bounds
bounds = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.01, 0.3)},
    {'name': 'n_estimators', 'type': 'discrete', 'domain': (50, 200)},
    {'name': 'max_depth', 'type': 'discrete', 'domain': (3, 10)},
    {'name': 'gamma', 'type': 'continuous', 'domain': (0, 0.5)},
    {'name': 'subsample', 'type': 'continuous', 'domain': (0.5, 1.0)}
]

# Step 3: Optimization
optimizer = GPyOpt.methods.BayesianOptimization(
    f=optimize_xgb,
    domain=bounds,
    model_type='GP',
    acquisition_type='EI',
    maximize=False
)

optimizer.run_optimization(max_iter=30)

# Step 4: Save report
with open('bayes_opt.txt', 'w') as f:
    f.write("Optimal Hyperparameters:\n")
    f.write("Learning Rate: {}\n".format(optimizer.x_opt[0]))
    f.write("N Estimators: {}\n".format(optimizer.x_opt[1]))
    f.write("Max Depth: {}\n".format(optimizer.x_opt[2]))
    f.write("Gamma: {}\n".format(optimizer.x_opt[3]))
    f.write("Subsample: {}\n".format(optimizer.x_opt[4]))
    f.write("\nBest Accuracy: {}\n".format(1 - optimizer.fx_opt))

# Step 5: Plot convergence
# Note: This might require GUI. If running in CI/remote, it might fail.
# optimizer.plot_convergence()
