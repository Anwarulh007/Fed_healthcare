import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
from PIL import Image
import os
import seaborn as sns

# Page configuration
st.set_page_config(page_title="Federated Learning - Healthcare", layout="wide", page_icon="🧬")

# Title
st.title("🏥 Privacy-Preserving Federated Learning for Healthcare")

# Sidebar Navigation
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to", [
    "Introduction", "Architecture", "Simulation", "Encryption", 
    "Model Demo", "Insights", "About"
])

# Page: Introduction
if page == "Introduction":
    st.header("📌 Project Overview")
    st.markdown("""
    This project demonstrates a privacy-preserving, federated learning (FL) setup for predicting patient readmission risks in a healthcare setting.

    **Goals:**
    - Avoid centralizing sensitive patient data
    - Train collaboratively using Federated Learning
    - Preserve data privacy using Homomorphic Encryption
    - Use Dockerized FL setup with TensorFlow and Flower
    """)
    st.success("Built using: TensorFlow • Flower • Docker • Streamlit • Python")

# Page: Architecture
elif page == "Architecture":
    st.header("📊 System Architecture")
    st.markdown("""


This architecture showcases how **Federated Learning (FL)** is implemented using **TensorFlow** and **Flower** in a **Dockerized environment**,  
with a special focus on **Privacy-Preserving Training** using **Homomorphic Encryption**.

---

### 🔑 Key Components

- **Clients (Hospitals)**:  
  Each hospital holds private patient data and runs local model training inside isolated Docker containers.

- **Server**:  
  Coordinates global model aggregation.  
  It **never sees raw data**, only receives **encrypted model weights**.

- **Encryption Layer**:  
  Ensures secure communication by encrypting model weights at the client-side before sending them to the server.

- **Docker**:  
  Provides an isolated and reproducible environment for client-server simulation across distributed systems.

""")

    st.image("assets/Picture1.jpg", caption="Federated Learning with Encrypted Docker Clients", use_column_width=True)
    st.info("No data leaves any client. Only encrypted gradients/weights are transmitted to the central server.")

# Page: Simulation
elif page == "Simulation":
    st.header("🔁 FL Simulation File Overview")

    option = st.radio("Select Input Method", ["Upload CSV Files", "Use Folder Path"])

    uploaded_filenames = []

    if option == "Upload CSV Files":
        uploaded_files = st.file_uploader("Upload CSV log files from clients", type=["csv"], accept_multiple_files=True)
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully")
            uploaded_filenames = [file.name for file in uploaded_files]
    else:
        folder_path = st.text_input("Enter folder path containing .csv files", value="logs/")
        if os.path.isdir(folder_path):
            for filename in sorted(os.listdir(folder_path)):
                if filename.endswith(".csv"):
                    uploaded_filenames.append(filename)
            if uploaded_filenames:
                st.success(f"✅ Found {len(uploaded_filenames)} CSV files in folder.")
            else:
                st.warning("No CSV files found in the specified folder.")
        else:
            st.error("❌ Folder not found. Please check the path.")

    if uploaded_filenames:
        st.markdown("### 📂 Files Detected")
        for fname in uploaded_filenames:
            st.markdown(f"- `{fname}`")
    else:
        st.info("📭 No files uploaded or found.")

# Page: Encryption
elif page == "Encryption":
    st.header("🔐 Encryption Workflow")
    st.markdown("""
    In our FL setup, we simulate Homomorphic Encryption to protect weight sharing:

    1. Clients locally encrypt model weights
    2. Server aggregates encrypted weights (cannot decrypt them)
    3. Decryption occurs only at the client side after aggregation

    🔐 **No raw data or unencrypted weights ever leave a client!**
    """)
    st.image("assets/encr.jpg", caption="Encryption workflow", use_column_width=True)

# Page: Model Demo
elif page == "Model Demo":
    st.header("🧪 Model Demo")

    uploaded_files = st.file_uploader("Upload client CSV files", type="csv", accept_multiple_files=True)

    if uploaded_files and len(uploaded_files) == 4:
        st.success("✅ All 4 CSV files uploaded successfully.")

        st.markdown("### 📂 Uploaded Datasets")
        for file in uploaded_files:
            st.write(f"📂 {file.name}")

        view_option = st.radio("Choose output format:", ["View Log Output", "View Image Output"])

        if view_option == "View Log Output":
            log_path = r"D:\project\federated_fl_project\Fed_Health\simulation_output.txt"
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    log_data = f.read()
                clean_log = re.sub(r"\\x1b\\[[0-9;]*[a-zA-Z]", "", log_data)
                clean_log = re.sub(r"\n{3,}", "\n\n", clean_log)
                clean_log = re.sub(r" +", " ", clean_log)

                with st.expander("📜 Click to view full training log", expanded=True):
                    st.code(clean_log.strip(), language="none")
            else:
                st.warning("⚠️ No simulation log file found at './shared/simulation.txt'.")

        elif view_option == "View Image Output":
            image_folder = "assets/docker_logs"
            if os.path.exists(image_folder):
                image_files = [img for img in os.listdir(image_folder) if img.endswith((".jpg", ".png"))]
                if image_files:
                    for img_name in sorted(image_files):
                        img_path = os.path.join(image_folder, img_name)
                        st.image(img_path, caption=img_name,)
                else:
                    st.warning("No Docker output images found in 'assets/docker_logs'.")
            else:
                st.warning("Image folder 'assets/docker_logs' not found.")
    else:
        st.info("📌 Please upload exactly 4 CSV files to view the model demo.")

# Page: Insights
elif page == "Insights":
    st.header("💡 Project Insights")
    st.markdown("""
    ### 📌 Key Findings:
    - Engineered a **federated learning simulation** for privacy-aware healthcare analytics.
    - Integrated **client-side encryption** to protect weight transmission across nodes.
    - Delivered **zero data leakage** across the pipeline using Docker-isolated clients.
                
    ### 💡 Lessons Learned:
    - Docker & Flower simplify federated setups.
    - Homomorphic encryption is crucial for secure FL.
    - Real-world FL requires careful design to balance privacy and performance.

    ### 📊 Impact:
    - ✅ Realistic privacy-preserving FL design
    - ✅ Encourages secure inter-institutional collaboration
    - ✅ Foundation for future real-world FL systems
    """)

# Page: About
elif page == "About":
    st.header("👨‍💻 About & Credits")
    st.markdown("""
    **Team Members:**
    - ANWARUL HAQUE(TEAM LEAD)
    - FAISAL SHAMIM
    - TAHSEEN ATIQUE ALI
    - MD MUJTABA

    

    **Tools Used:** TensorFlow, Flower, Streamlit, Docker, Python

    For collaboration or queries, feel free to reach out!
    """)
    st.write("CREATED WITH ❤️ BY THE TEAM: TECHMIND INNOVATORS")
