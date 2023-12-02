This folder contains a very simple example to create a streamlit application to present SHAP values from a XGBoost classification model. The model is not optimized and selected features are selected for illustration purposes. To run this streamlit application:
- Install all required packages with a Python version >= 3.11 by running `python -m pip install -r requirements.txt`
- In the same folder run `streamlit run app.py`
The second command, should initiate a streamlit application on your local machine browser.You can use this application to visualize some of the SHAP plots for different test observations by selecting an observation number from a given list of possible observations.

Example snapshots of the application once rendered:
![top_view](./data/example_snapshot.jpg)

![middle_view](./data/example_snapshot_1.jpg)