"""
Author: Philip Ciunkiewicz

Custom functions for the I4H data science workshop.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.regressor import ResidualsPlot
from yellowbrick.regressor import PredictionError
from yellowbrick.features import RFECV
from yellowbrick.features.importances import FeatureImportances


def load_data(dataset, simplify=True, missing=0):
    """ Load either the sample classification
    or regression dataset provided in the
    SKLearn library.
    
    Parameters:
    -----------
    dataset : str
        One of "classification" or "regression".
    simplify : bool
        If True removes some features for simplicity.
    missing : float
        Fraction of values to remove at random [0, 1).
        
    Returns:
    --------
    features : pd.DataFrame
        DataFrame containing the data features 
        (independent variables).
    target : pd.DataFrame
        DataFrame containing the data target
        (dependent variable).
    description : str
        Descriptive text providing a summary of the data.
    """
    if dataset == 'classification':
        data = load_breast_cancer()
    if dataset == 'regression':
        data = fetch_california_housing()
        
    features = pd.DataFrame(data['data'], columns=data['feature_names'])
    target = pd.DataFrame(data['target'], columns=['target'])
    description = data['DESCR']
    
    if simplify:
        features = _simplify_features(features)

    if missing:
        nulls = np.random.random(features.shape) < missing
        features = features.mask(nulls)
    
    return features, target, description


def _simplify_features(df):
    """ Remove 'standard error' columns from
    the breast cancer dataset for simplicity
    and replace spaces with underscores.
    """
    simple = [col for col in df.columns if 'error' not in col]
    simple_df = df[simple]
    simple_df.columns = [col.replace(' ', '_') for col in simple]
    
    return simple_df

def region_grid(X, n_pixels):
    """ Compute a square grid of n_pixels
    by n_pixels in the provided data range.
    """
    X_range = np.ptp(X.values, axis=0)
    X_max = X.max().values + (0.05 * X_range)
    X_min = X.min().values - (0.05 * X_range)
    
    res = X_range / n_pixels
    xx, yy = np.meshgrid(
        np.arange(X_min[0], X_max[0], res[0]),
        np.arange(X_min[1], X_max[1], res[1]))
    
    return xx, yy
    

def draw_decisions(model, X, y, features):
    """ Plots the decision boundaries in
    two dimensions for a given classifier.
    
    Parameters:
    -----------
    model : sklearn.model
        SKLearn classification model.
    X : pd.DataFrame
        Input features for classification.
    y : pd.DataFrame
        Target labels for classification.
    features : array_like
        Names of features for analysis.
    """
    assert len(features) == 2, 'Requires exactly two features.'
    
    X, y = X[features], y.target
    model.fit(X, y)
    
    # Predict value probabilities across the full axis
    xx, yy = region_grid(X, 1000)
    if hasattr(model, "decision_function"):
        Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    zz = Z.reshape(xx.shape)
    
    # Put the result into a color plot
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set(
        xlabel=features[0],
        ylabel=features[1],
        title=model.__class__.__name__)
    
    CS = ax.contourf(
        xx, yy, zz, 
        cmap='RdBu', 
        extend='both', 
        levels=np.linspace(0, 1, 6))

    ax.text(
        0.95, 0.95, 
        f'Acc={model.score(X, y)}', 
        horizontalalignment='right',
        bbox={'facecolor': 'white', 'pad':5}, 
        transform=ax.transAxes)

    ax.scatter(*X[y == 0].values.T, alpha=0.75, c=[[1, 0, 0]])
    ax.scatter(*X[y == 1].values.T, alpha=0.75, c=[[0, 0, 1]])
    plt.colorbar(CS)
    plt.show()


def draw_confusion_matrix(model, X, y, classnames=None):
    """ Renders the confusion matrix in terms
    of % accuracy for a given classifier.
    
    Parameters:
    -----------
    model : sklearn.model
        SKLearn classification model.
    X : pd.DataFrame
        Input features for classification.
    y : pd.DataFrame
        Target labels for classification.
    classnames : array_like
        Name mapping of class labels in
        ascending order (len = #classes).
    """
    split = train_test_split(X, y, random_state=123)
    X_train, X_test, y_train, y_test = split

    visualizer = ConfusionMatrix(model, percent=True, cmap='Greens')
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    
    if classnames is not None:
        visualizer.classes_ = classnames
        
    visualizer.draw()
    visualizer.poof()


def draw_feature_importances(model, X, y):
    """ Displays relative feature importances
    for models with coeff or feature attributes.
    
    Parameters:
    -----------
    model : sklearn.model
        SKLearn classification/regression model.
    X : pd.DataFrame
        Input features for classification/regression.
    y : pd.DataFrame
        Target data for classification/regression.
    """
    visualizer = FeatureImportances(model)
    visualizer.fit(X, y['target'])
    visualizer.poof()


def draw_RFECV(model, X, y):
    """ Displays recursive feature elimination
    cross-validation for most models.
    
    Parameters:
    -----------
    model : sklearn.model
        SKLearn classification/regression model.
    X : pd.DataFrame
        Input features for classification/regression.
    y : pd.DataFrame
        Target data for classification/regression.
    """
    visualizer = RFECV(model, cv=2)
    visualizer.fit(X, y['target'])
    visualizer.poof()


def draw_residuals(model, X, y):
    """ Displays regression model residuals.
    
    Parameters:
    -----------
    model : sklearn.model
        SKLearn regression model.
    X : pd.DataFrame
        Unscaled input features for regression.
    y : pd.DataFrame
        Target data for regression.
    """
    split = train_test_split(X, y, random_state=123)
    X_train, X_test, y_train, y_test = split

    scaler = StandardScaler()
    X_train_rs = scaler.fit_transform(X_train)
    X_test_rs = scaler.transform(X_test)

    visualizer = ResidualsPlot(model, alpha=0.15)
    visualizer.fit(X_train_rs, y_train['target'])
    visualizer.score(X_test_rs, y_test['target'])
    visualizer.poof()


def draw_prediction_error(model, X, y):
    """ Displays regression model prediction error.
    
    Parameters:
    -----------
    model : sklearn.model
        SKLearn regression model.
    X : pd.DataFrame
        Unscaled input features for regression.
    y : pd.DataFrame
        Target data for regression.
    """
    split = train_test_split(X, y, random_state=123)
    X_train, X_test, y_train, y_test = split

    scaler = StandardScaler()
    X_train_rs = scaler.fit_transform(X_train)
    X_test_rs = scaler.transform(X_test)

    visualizer = PredictionError(model, alpha=0.25)
    visualizer.fit(X_train_rs, y_train['target'])
    visualizer.score(X_test_rs, y_test['target'])
    visualizer.poof()
