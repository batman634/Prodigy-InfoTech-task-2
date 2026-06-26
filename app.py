import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Load trained KMeans model
# -------------------------------
with open("kmeans_customer_model.pkl", "rb") as file:
    model = pickle.load(file)

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(page_title="Customer Segmentation App", page_icon="🛍️", layout="centered")

# -------------------------------
# App Title
# -------------------------------
st.title("🛍️ Customer Segmentation using K-Means")
st.write("Predict the **customer cluster** based on **Annual Income** and **Spending Score**.")

st.markdown("---")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Enter Customer Details")

annual_income = st.sidebar.number_input(
    "Annual Income (k$)",
    min_value=0,
    max_value=200,
    value=60,
    step=1
)

spending_score = st.sidebar.number_input(
    "Spending Score (1-100)",
    min_value=1,
    max_value=100,
    value=55,
    step=1
)

# -------------------------------
# Display Input
# -------------------------------
st.subheader("Entered Customer Details")
st.write(f"**Annual Income (k$):** {annual_income}")
st.write(f"**Spending Score (1-100):** {spending_score}")

# -------------------------------
# Predict Cluster
# -------------------------------
if st.button("Predict Customer Cluster"):
    input_data = pd.DataFrame({
        'Annual Income (k$)': [annual_income],
        'Spending Score (1-100)': [spending_score]
    })

    cluster = model.predict(input_data)[0]

    st.success(f"Predicted Customer Cluster: **Cluster {cluster}**")

    # Optional interpretation
    if cluster == 0:
        st.info("This customer belongs to Segment 0.")
    elif cluster == 1:
        st.info("This customer belongs to Segment 1.")
    elif cluster == 2:
        st.info("This customer belongs to Segment 2.")
    elif cluster == 3:
        st.info("This customer belongs to Segment 3.")
    elif cluster == 4:
        st.info("This customer belongs to Segment 4.")

st.markdown("---")
st.caption("Built with Streamlit + K-Means Clustering + Pickle Model")