import numpy as np 
import pandas as pd 
from UQpy.surrogates.polynomial_chaos.polynomials.TotalDegreeBasis import TotalDegreeBasis
from UQpy.surrogates.polynomial_chaos.regressions.LeastSquareRegression import LeastSquareRegression
from UQpy.surrogates.polynomial_chaos.regressions.LassoRegression import LassoRegression
from UQpy.surrogates.polynomial_chaos.PolynomialChaosExpansion import PolynomialChaosExpansion
from UQpy.distributions import Uniform, JointIndependent
from UQpy.surrogates import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os 

base_path = r"C:\Users\User\Downloads\research project\heart data 3"

for a in range(1,11) : 
    # inputs as a numpy array
    x_path = os.path.join(base_path, f"case{a:02}", "X.txt")
    input_data = pd.read_csv(x_path, sep=r"\s+", header=None).values

    # outputs as a numpy array
    y_path = os.path.join(base_path, f"case{a:02}", "Y.txt")
    output_data = pd.read_csv(y_path, sep=r"\s+", header=None).values

    # split data into train and test 
    x = input_data
    y = output_data

    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123)

    pce_models = [] # initialises array

    # create a marginal uniform distribution for each input (not too sure about this)
    marginals = [Uniform(loc=X_train[:, i].min(), scale=X_train[:, i].max() - X_train[:, i].min())
                for i in range(X_train.shape[1])]
    joint = JointIndependent(marginals=marginals)

    for i in range(y_train.shape[1]):  # loop over each output
        
        # create polynomial basis 
        poly_basis = TotalDegreeBasis(distributions=joint, max_degree=2)

        # regression method
        reg = LassoRegression()  # can change this 
        
        # create PCE object
        pce = PolynomialChaosExpansion(polynomial_basis = poly_basis, regression_method=reg)
        
        # fit PCE to the i-th output
        pce.fit(X_train, y_train[:, i])
        pce_models.append(pce) # add to array
        
    y_pred = np.column_stack([pce.predict(X_test).flatten() for pce in pce_models]) # predicts outputs for test inputs

    # calculates r^2 for each output 
    r2_scores = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])]
    for i, r2 in enumerate(r2_scores):
        print(f"R^2 for output {i+1}: {r2}")

    r2_df = pd.DataFrame(r2_scores, columns=["R2"])
    output_path = os.path.join(base_path, f"case{a:02}", "r2vals.csv")
    r2_df.to_csv(output_path, index=False)
