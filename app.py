import pandas as pd
import numpy as np
import streamlit as st
import joblib
import shap
import lime

from sklean.compose import ColumnTransformer, make_column_selector
from sklearn.preprocess import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from lime.lime_tabular import LimeTabularExplainer

import os
import config

# Preprocessing pipeline
def create_preprocessing_pipeline():
    
    # Select numeric and categorical columns
    num_cols = make_column_selector(dtype_include='number')
    cat_cols = make_column_selector(dtype_include='object')
    
    # Instantiate the transformers
    scaler = StandardScaler()
    encoder = OneHotEncoder()
    knn_imputer = KNNImputer(n_neighbors=2, weights='uniform')
    
    # Create pipeline
    num_pipe = Pipeline([
        ('scaler', scaler),
        ('imputer', knn_imputer)
    ])
     
    cat_pipe = Pipeline([
        ('encoder', encoder)
    ])
    
    preprocessor = ColumnTransformer([
        ('numeric', num_pipe, num_cols),
        ('categorical', cat_pipe, cat_cols),
    ], remainder='drop')
    
    return preprocessor


