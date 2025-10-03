import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sqlite3
from io import StringIO
import plotly.graph_objects as go
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🎯 Customer Segmentation Dashboard")
st.markdown("""
This application performs KMeans clustering on customer data to identify distinct segments based on Age, Income, and Spending Score.
Upload your data or connect to a database to get started!
""")

# Sidebar for data source selection
st.sidebar.header("📥 Data Source")
data_source = st.sidebar.radio("Select data source:", ["Upload CSV", "SQL Database"])

# Initialize session state for data and results
if 'data' not in st.session_state:
    st.session_state.data = None
if 'clustered_data' not in st.session_state:
    st.session_state.clustered_data = None
if 'cluster_centers' not in st.session_state:
    st.session_state.cluster_centers = None

def load_csv_data(uploaded_file):
    """Load data from uploaded CSV file"""
    try:
        if uploaded_file is not None:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("CSV file uploaded successfully!")
            return df
        return None
    except Exception as e:
        st.sidebar.error(f"Error reading CSV file: {e}")
        return None

def connect_sqlite():
    """Connect to SQLite database and load data"""
    try:
        # For demo purposes, we'll create a sample SQLite database
        conn = sqlite3.connect(':memory:')
        
        # Create sample data (in real scenario, this would be your actual database)
        sample_data = pd.DataFrame({
            'Age': np.random.randint(18, 70, 200),
            'Annual Income (k$)': np.random.randint(15, 150, 200),
            'Spending Score (1-100)': np.random.randint(1, 100, 200)
        })
        
        sample_data.to_sql('customers', conn, index=False, if_exists='replace')
        
        # Query the data
        df = pd.read_sql('SELECT * FROM customers', conn)
        conn.close()
        
        st.sidebar.success("Connected to SQLite database successfully!")
        return df
    except Exception as e:
        st.sidebar.error(f"Error connecting to database: {e}")
        return None

def perform_clustering(df):
    """Perform KMeans clustering on the data"""
    try:
        # Select relevant features
        X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform KMeans clustering with 5 clusters
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to the dataframe
        df['Cluster'] = clusters
        
        # Get cluster centers (in original scale)
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        
        return df, cluster_centers
    except Exception as e:
        st.error(f"Error during clustering: {e}")
        return None, None

def name_cluster(age, income, spending):
    """Name clusters based on characteristics"""
    if spending > 60:
        if income > 80:
            return "High Rollers"
        else:
            return "Budget Spenders"
    else:
        if income > 80:
            return "Wealthy Savers"
        elif age > 50:
            return "Older Conservative"
        else:
            return "Young Savers"

def create_3d_plot(df, cluster_centers):
    """Create interactive 3D scatter plot"""
    fig = go.Figure()
    
    # Add data points
    for cluster in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster]
        fig.add_trace(go.Scatter3d(
            x=cluster_data['Age'],
            y=cluster_data['Annual Income (k$)'],
            z=cluster_data['Spending Score (1-100)'],
            mode='markers',
            marker=dict(size=5, opacity=0.8),
            name=f'Cluster {cluster}',
            hovertemplate='<b>Age:</b> %{x}<br>' +
                         '<b>Income:</b> %{y}k<br>' +
                         '<b>Spending:</b> %{z}<br>' +
                         '<b>Cluster:</b> %{text}<extra></extra>',
            text=[f'Cluster {cluster}'] * len(cluster_data)
        ))
    
    # Add cluster centers
    fig.add_trace(go.Scatter3d(
        x=cluster_centers[:, 0],
        y=cluster_centers[:, 1],
        z=cluster_centers[:, 2],
        mode='markers',
        marker=dict(
            size=10,
            color='yellow',
            symbol='x',
            line=dict(width=2, color='black')
        ),
        name='Cluster Centers',
        hovertemplate='<b>Center Age:</b> %{x:.1f}<br>' +
                     '<b>Center Income:</b> %{y:.1f}k<br>' +
                     '<b>Center Spending:</b> %{z:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Age',
            yaxis_title='Annual Income (k$)',
            zaxis_title='Spending Score (1-100)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=1000,
        height=800,
        title='3D Customer Segmentation',
        showlegend=True
    )
    
    return fig

# Main application logic
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file is not None:
        df = load_csv_data(uploaded_file)
        if df is not None:
            st.session_state.data = df

elif data_source == "SQL Database":
    if st.sidebar.button("Connect to SQLite Database"):
        df = connect_sqlite()
        if df is not None:
            st.session_state.data = df

# Display uploaded data
if st.session_state.data is not None:
    st.subheader("📋 Data Preview")
    st.dataframe(st.session_state.data.head())
    
    # Check if required columns exist
    required_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    if all(col in st.session_state.data.columns for col in required_columns):
        if st.sidebar.button("🚀 Run Clustering Analysis"):
            with st.spinner("Performing clustering analysis..."):
                clustered_df, cluster_centers = perform_clustering(st.session_state.data)
                
                if clustered_df is not None and cluster_centers is not None:
                    st.session_state.clustered_data = clustered_df
                    st.session_state.cluster_centers = cluster_centers
                    
                    # Display results
                    st.success("Clustering completed successfully!")
                    
                    # Show 3D plot
                    st.subheader("📊 3D Cluster Visualization")
                    fig = create_3d_plot(clustered_df, cluster_centers)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Cluster insights
                    st.subheader("🔍 Cluster Insights")
                    
                    insights = []
                    for i, center in enumerate(cluster_centers):
                        age, income, spending = center
                        cluster_name = name_cluster(age, income, spending)
                        
                        cluster_data = clustered_df[clustered_df['Cluster'] == i]
                        avg_age = cluster_data['Age'].mean()
                        avg_income = cluster_data['Annual Income (k$)'].mean()
                        avg_spending = cluster_data['Spending Score (1-100)'].mean()
                        count = len(cluster_data)
                        
                        insights.append({
                            'Cluster': i,
                            'Name': cluster_name,
                            'Avg Age': f"{avg_age:.1f}",
                            'Avg Income': f"{avg_income:.1f}k",
                            'Avg Spending': f"{avg_spending:.1f}",
                            'Count': count
                        })
                    
                    # Display insights as a table
                    insights_df = pd.DataFrame(insights)
                    st.dataframe(insights_df)
                    
                    # Show detailed statistics for each cluster
                    st.subheader("📈 Detailed Cluster Statistics")
                    
                    for i, center in enumerate(cluster_centers):
                        age, income, spending = center
                        cluster_name = name_cluster(age, income, spending)
                        cluster_data = clustered_df[clustered_df['Cluster'] == i]
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(f"Cluster {i} - {cluster_name}", f"{len(cluster_data)} customers")
                        with col2:
                            st.metric("Average Age", f"{cluster_data['Age'].mean():.1f}")
                        with col3:
                            st.metric("Average Income", f"{cluster_data['Annual Income (k$)'].mean():.1f}k")
                        with col4:
                            st.metric("Average Spending", f"{cluster_data['Spending Score (1-100)'].mean():.1f}")
                    
                    # Download button
                    st.subheader("💾 Download Results")
                    csv = clustered_df.to_csv(index=False)
                    st.download_button(
                        label="Download segmented data as CSV",
                        data=csv,
                        file_name="segmented_customers.csv",
                        mime="text/csv"
                    )
                    
    else:
        st.error(f"CSV file must contain these columns: {required_columns}")

else:
    st.info("👈 Please upload a CSV file or connect to a database to get started.")

# Footer
st.markdown("---")
st.markdown("""
<style>
.footer {
    text-align: center;
    color: gray;
    padding: 10px;
}
</style>
<div class="footer">
    Customer Segmentation Dashboard • Built with Streamlit
</div>
""", unsafe_allow_html=True)
