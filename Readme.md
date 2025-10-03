# 📊 Customer Segmentation Dashboard

A **Streamlit-based interactive dashboard** for performing **KMeans clustering** on customer data.
It segments customers based on **Age, Income, and Spending Score** and provides **3D visualizations, insights, and downloadable results**.

---

## 🚀 Features

* Upload customer data via **CSV file** or connect to a **SQLite database**.
* Automatically **standardizes features** and performs **KMeans clustering**.
* Interactive **3D scatter plot** of customer clusters using **Plotly**.
* Provides **cluster insights**: average age, income, spending score, and customer counts.
* Downloadable **segmented customer data** in CSV format.
* Custom **cluster naming logic** for easy interpretation.

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Mayank-Ai7/Customer-Segmentation

```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate     
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run the Streamlit app:

```bash
streamlit run app.py
```

### Options inside the dashboard:

1. **Upload CSV** – Upload a dataset containing at least these columns:

   * `Age`
   * `Annual Income (k$)`
   * `Spending Score (1-100)`
2. **SQL Database** – Use a demo SQLite database (generates sample data).
3. **Run Clustering** – Segments customers into **5 groups**.
4. **Visualize Results** – Interactive 3D scatter plot of clusters.
5. **Download** – Export results as a CSV file.

---

## 📊 Example Output

* **3D Visualization** of clusters (interactive Plotly graph).
* **Cluster Insights Table** with names like *High Rollers, Budget Spenders, Wealthy Savers*.
* **Downloadable CSV** containing the segmented data.

---

## 📂 Project Structure

```
customer-segmentation-dashboard/
│── app.py                 # Main Streamlit application
│── requirements.txt       # Python dependencies
│── README.md              # Project documentation
```

---

## 🛠 Requirements

See `requirements.txt` below.

```txt
streamlit
pandas
numpy
matplotlib
scikit-learn
sqlite3-binary
plotly
```

---

## ⚡ Future Improvements

* Support for **external SQL connections** (PostgreSQL, MySQL).
* Option to choose **number of clusters dynamically**.
* More **visualizations** (2D projections, heatmaps, etc.).
* Enhanced **cluster naming logic** using domain knowledge.

---

## 👨‍💻 Author

Developed with ❤️ using **Streamlit, Scikit-learn, and Plotly**.
