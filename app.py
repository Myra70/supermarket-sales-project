import streamlit as st
import pandas as pd
import joblib


# =====================================================
# PAGE SETTINGS
# =====================================================

st.set_page_config(
    page_title="Supermarket Sales Analysis",
    page_icon="🛒",
    layout="wide"
)


# =====================================================
# LOAD CSV FILE
# =====================================================

df = pd.read_csv("/Users/apple/streamlit_project/SuperMarket Analysis - SuperMarket Analysis.csv")


# =====================================================
# DATE AND TIME FEATURES
# =====================================================

df["Date"] = pd.to_datetime(df["Date"])

df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

df["Hour"] = pd.to_datetime(
    df["Time"],
    format="%I:%M:%S %p"
).dt.hour


# =====================================================
# LOAD TRAINED MODEL
# =====================================================

model = joblib.load("model.pkl")


# =====================================================
# TITLE
# =====================================================

st.title("🛒 Supermarket Sales Analysis")

st.write(
    "Machine Learning Based Sales Analysis and Prediction"
)

st.divider()


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("📌 Menu")

page = st.sidebar.selectbox(
    "Select Page",
    [
        "Dashboard",
        "Data Analysis",
        "Visualization",
        "Model Comparison",
        "Sales Prediction"
    ]
)


# =====================================================
# 1. DASHBOARD
# =====================================================

if page == "Dashboard":

    st.header("📊 Dashboard")

    # Metrics

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Sales",
            f"${df['Sales'].sum():,.2f}"
        )

    with col2:
        st.metric(
            "Total Transactions",
            df.shape[0]
        )

    with col3:
        st.metric(
            "Average Sales",
            f"${df['Sales'].mean():,.2f}"
        )

    with col4:
        st.metric(
            "Average Rating",
            f"{df['Rating'].mean():.2f}"
        )

    st.divider()

    # Dataset preview

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# =====================================================
# 2. DATA ANALYSIS
# =====================================================

elif page == "Data Analysis":

    st.header("📋 Data Analysis")

    # Dataset information

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            f"Rows: {df.shape[0]}"
        )

    with col2:
        st.info(
            f"Columns: {df.shape[1]}"
        )

    # Dataset

    st.subheader("Complete Dataset")

    st.dataframe(
        df,
        use_container_width=True
    )

    # Statistics

    st.subheader("📊 Statistical Summary")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    # Missing values

    st.subheader("🔍 Missing Values")

    missing_values = df.isnull().sum()

    missing_data = pd.DataFrame({
        "Column": missing_values.index,
        "Missing Values": missing_values.values
    })

    st.dataframe(
        missing_data,
        use_container_width=True
    )


# =====================================================
# 3. VISUALIZATION
# =====================================================

elif page == "Visualization":

    st.header("📈 Sales Visualization")


    # ---------------------------------------------
    # Sales by Branch
    # ---------------------------------------------

    st.subheader("Sales by Branch")

    branch_sales = (
        df.groupby("Branch")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(branch_sales)


    # ---------------------------------------------
    # Sales by Product Line
    # ---------------------------------------------

    st.subheader("Sales by Product Line")

    product_sales = (
        df.groupby("Product line")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(product_sales)


    # ---------------------------------------------
    # Sales by City
    # ---------------------------------------------

    st.subheader("Sales by City")

    city_sales = (
        df.groupby("City")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(city_sales)


    # ---------------------------------------------
    # Customer Gender
    # ---------------------------------------------

    st.subheader("Customer Gender")

    gender_count = df["Gender"].value_counts()

    st.bar_chart(gender_count)


    # ---------------------------------------------
    # Payment Method
    # ---------------------------------------------

    st.subheader("Payment Method")

    payment_count = df["Payment"].value_counts()

    st.bar_chart(payment_count)


    # ---------------------------------------------
    # Monthly Sales
    # ---------------------------------------------

    st.subheader("Monthly Sales")

    monthly_sales = (
        df.groupby("Month")["Sales"]
        .sum()
    )

    st.line_chart(monthly_sales)


# =====================================================
# 4. MODEL COMPARISON
# =====================================================

elif page == "Model Comparison":

    st.header("🤖 Machine Learning Model Comparison")

    st.write(
        "Three regression algorithms were used "
        "to predict supermarket sales."
    )


    # Load model results

    try:

        results = pd.read_csv(
            "model_results.csv"
        )

        st.subheader("Model Performance")

        st.dataframe(
            results,
            use_container_width=True
        )


        # R2 chart

        st.subheader("R² Score Comparison")

        r2_data = results.set_index(
            "Model"
        )["R2 Score"]

        st.bar_chart(r2_data)


        # Find best model

        best_index = results["R2 Score"].idxmax()

        best_model = results.loc[
            best_index
        ]


        st.success(
            f"🏆 Best Model: {best_model['Model']}"
        )

        st.info(
            f"R² Score: "
            f"{best_model['R2 Score']:.4f}"
        )


    except FileNotFoundError:

        st.error(
            "model_results.csv not found. "
            "Please run train.py first."
        )


# =====================================================
# 5. SALES PREDICTION
# =====================================================

elif page == "Sales Prediction":

    st.header("🔮 Sales Prediction")

    st.write(
        "Enter the transaction details "
        "to predict sales."
    )

    st.divider()


    # ---------------------------------------------
    # INPUTS
    # ---------------------------------------------

    col1, col2 = st.columns(2)


    # LEFT SIDE

    with col1:

        branch = st.selectbox(
            "Branch",
            sorted(df["Branch"].unique())
        )

        city = st.selectbox(
            "City",
            sorted(df["City"].unique())
        )

        customer_type = st.selectbox(
            "Customer Type",
            sorted(df["Customer type"].unique())
        )

        gender = st.selectbox(
            "Gender",
            sorted(df["Gender"].unique())
        )

        product_line = st.selectbox(
            "Product Line",
            sorted(df["Product line"].unique())
        )


    # RIGHT SIDE

    with col2:

        payment = st.selectbox(
            "Payment",
            sorted(df["Payment"].unique())
        )

        quantity = st.number_input(
            "Quantity",
            min_value=1,
            max_value=20,
            value=5
        )

        month = st.number_input(
            "Month",
            min_value=1,
            max_value=12,
            value=1
        )

        day = st.number_input(
            "Day",
            min_value=1,
            max_value=31,
            value=15
        )

        hour = st.number_input(
            "Hour",
            min_value=0,
            max_value=23,
            value=12
        )


    st.divider()


    # ---------------------------------------------
    # PREDICT BUTTON
    # ---------------------------------------------

    if st.button(
        "🔮 Predict Sales",
        use_container_width=True
    ):


        # Create input DataFrame

        input_data = pd.DataFrame({

            "Branch": [branch],

            "City": [city],

            "Customer type": [customer_type],

            "Gender": [gender],

            "Product line": [product_line],

            "Payment": [payment],

            "Quantity": [quantity],

            "Month": [month],

            "Day": [day],

            "Hour": [hour]
        })


        # Prediction

        prediction = model.predict(
            input_data
        )


        sales = prediction[0]


        # Display result

        st.success(
            f"💰 Predicted Sales: ${sales:,.2f}"
        )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Supermarket Sales Analysis | "
    "BCA Data Science Minor Project"
)