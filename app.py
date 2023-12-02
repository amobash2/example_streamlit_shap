import pandas as pd
import numpy as np
import streamlit as st
from streamlit_shap import st_shap
import shap
from sklearn.model_selection import train_test_split
import xgboost
from sklearn.metrics import accuracy_score

threshold = 0.5

@st.cache_data
def load_data(target=["is_canceled"], features=["adr", "lead_time", "arrival_date_day_of_month", "stays_in_weekend_nights", "stays_in_week_nights"]):
    hotel_booking_data = pd.read_csv("./data/hotel_booking.csv")
    hotel_booking_data[target] = hotel_booking_data[target].astype(int)
    X = hotel_booking_data[features]
    y = hotel_booking_data[target]
    return X, y

@st.cache_data
def fit_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=1234)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=1234)
    d_train = xgboost.DMatrix(X_train, label=y_train)
    d_val = xgboost.DMatrix(X_val, label=y_val)
    d_test = xgboost.DMatrix(X_test, label=y_test)
    params = {
        "objective": "binary:logistic",
        "base_score": np.mean(y_train),
        "eval_metric": "logloss",
    }
    model = xgboost.train(params, d_train, 100, evals = [(d_val, "val")], verbose_eval=10, early_stopping_rounds=20)
    preds = [1 if p > threshold else 0 for p in model.predict(d_test)]
    model_accuracy = round(accuracy_score(y_test, preds), 3)

    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)
    tree_explainer = shap.TreeExplainer(model, X_test)
    tree_shap_values = tree_explainer.shap_values(X_test)
    return shap_values, explainer, tree_shap_values, tree_explainer, X_test, y_test, model_accuracy

st.title("Data explorations...")

# Load data with given set of features
# Assuming a binary classification
X,y = load_data()

# Extracting SHAP values
shap_values, explainer, tree_shap_values, tree_explainer, X_test, y_test, model_accuracy = fit_model(X, y)

# Concatenate test features and target data for presentation
concat_data = pd.concat([X_test, y_test], axis = 1, names = list(X_test) + list(y_test))

# Create the side bar to allow user select an observation number
# From the list of possible test observations
st.sidebar.header("Test observation...")
with st.sidebar:
    obs_number = st.selectbox(
        'Select observation number:',
        options = list(range(1, len(X_test)+1))
    )

# Populate current model accuracy if populated
if model_accuracy > 0:
    st.write("\n")
    st.write(f"Current model accuracy : **{model_accuracy}**")
    st.markdown("***")

# Populating different SHAP vizualizations
st.write("\n")
st.write("**SHAP values for all test observations:**")
st_shap(shap.plots.beeswarm(shap_values), height=300)

st.markdown("***")
st.write("\n")

st.write("**SHAP values of maximum 1000 observations:**")
st_shap(shap.force_plot(tree_explainer.expected_value, tree_shap_values[:1000,:], X_test.iloc[:1000,:]), height=400, width=800)

st.markdown("***")
st.write("\n")
st.write(f"**SHAP values for observation # {obs_number}:**")
st_shap(shap.plots.waterfall(shap_values[obs_number-1]), height=300)

st.markdown("***")
st.write("\n")
st.write(f"**Observed values for observation # {obs_number}:**")
cur_obs = concat_data.iloc[obs_number-1]
cur_obs_df = pd.DataFrame(cur_obs).T
st.markdown(cur_obs_df.style.hide(axis="index").to_html(), unsafe_allow_html=True)

st.markdown("***")
st.write("\n")
st.write(f"**Feature contributions per observation # {obs_number}:**")
st_shap(shap.force_plot(tree_explainer.expected_value, tree_shap_values[obs_number-1,:], X_test.iloc[obs_number-1,:]), height=200, width=800)
