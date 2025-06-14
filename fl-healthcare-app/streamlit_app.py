import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
from PIL import Image
import os
import seaborn as sns

# Page configuration
st.set_page_config(page_title="FedHealth - Healthcare", layout="wide", page_icon="🧬")

# Custom CSS to beautify Streamlit app with tabs, hover, and theme
st.markdown("""
    <style>
        /* Main Page Background */
        [data-testid="stAppViewContainer"] {
            background-color: #001f3f; /* dark blue */
        }

        /* Main Content Block (where everything sits) */
        [data-testid="stVerticalBlock"] {
            background-color: rgba(255, 255, 255, 1); /* white with transparency */
            padding: 20px;
            border-radius: 15px;
        }

        /* Box around the Tab Navigation Bar with background image */
        .stTabs {
            background-color:#d0f0f9;
            background-size: cover;
            background-position: center;
            padding: 8px;
            border-radius: 12px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }

      /* Individual Tabs Styling - light blue with white text */
        .stTabs [data-baseweb="tab"] {
            background-color: #001f3f  !important; /* Light blue tabs */
            color: white !important;              /* White text */
            border-radius: 8px;
            margin-right: 8px;
            padding: 10px 18px;
            font-size: 16px;
            transition: all 0.4s ease-in-out;
            border: 1px solid #b0e0f0;
        }

        /* Hover effect on Tabs */
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #09B19A !important;
            color: white !important;
            font-size: 18px;
            font-weight: bold;
        }

        /* Active/Selected Tab Styling */
        .stTabs [aria-selected="true"] {
            background-color: #09B19A !important;
            color: white !important;
            font-weight: bold;
            font-size: 18px;
        }

        /* Global Header Styling (affects st.title, st.header, st.subheader) */
        h1, h2, h3, h4, h5, h6 {
            color: #001f3f;
            text-align: center;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Button Styling */
        .stButton button {
            background-color: #005f73;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 16px;
            transition: all 0.4s ease-in-out;
        }

        .stButton button:hover {
            background-color: #0a9396;
        }
        @media screen and (max-width: 768px) {
        h1 {
            font-size: 22px;
        }

        h2 {
            font-size: 20px;
        }

        h3 {
            font-size: 18px;
            text-align: center; /* Center h3 on small devices */
        }

        .stTabs {
            flex-direction: column; /* Stack tabs vertically */
            align-items: center;
        }

        .stTabs [data-baseweb="tab"] {
            flex-grow: unset; /* Remove grow on small screens */
            width: 90%; /* Tab buttons take full width on mobile */
            margin: 5px 0;
        }

        .stButton button {
            padding: 8px 16px;
            font-size: 14px;
        }

        [data-testid="stVerticalBlock"] {
            padding: 10px;
            margin: 10px;
        }

        .stImage img {
            max-width: 100%;
            height: auto;
        }
    }
    </style>
""", unsafe_allow_html=True)



st.markdown("""
<h1 style='text-align: center; color: #09B19A; margin-bottom: 0px;'>
    <img src='https://raw.githubusercontent.com/Anwarulh007/Fed_healthcare/main/fl-healthcare-app/assets/fh%2Blogo%20updated.jpg' width='60' style='vertical-align: middle;'> FedHealthcare 
</h1>
<h1 style='text-align: center; margin-top: 0px;'>
    Privacy-Preserving Federated Learning for Healthcare Analysis
</h1>
""", unsafe_allow_html=True)
st.markdown("""<h2>Secure, Collaborate, Build Trust!</h2>""", unsafe_allow_html=True)
st.markdown("""<h5>Empowering healthcare with trust, privacy, and innovation.</h5>""", unsafe_allow_html=True)
st.markdown("""<h3 style='text-align: left;'>Welcome to the Future of Healthcare — Where Your Data Stays Yours, Always!</h3>""", unsafe_allow_html=True)
st.markdown("""
In a world where healthcare insights could save lives, one critical barrier stands tall: **privacy**.

This project breaks that barrier — using **Federated Learning (FL)** and **Homomorphic Encryption (HE)** to collaboratively train machine learning models without ever sharing sensitive patient data. We designed a **Dockerized Federated Learning** powered by **Flower** and **TensorFlow**, tailored for preserving privacy.

Security, modularity, and industry-grade scalability lie at the heart of this architecture.
""")

# Custom Tab-style Section (all displayed below Project Overview)
tabs = st.tabs(["Home", "Architecture", "Simulation", "Encryption", "Model Demo", "Insights", "About"])


with tabs[0]:
    st.markdown("""<h3 style='text-align: left;'>Inside FedHealthcare: Our Mission and Methodology</h3>""", unsafe_allow_html=True)
    st.markdown("""
    FedHealthcare is a privacy-preserving healthcare analytics platform that leverages Federated Learning (FL) and Homomorphic Encryption (HE) to securely train machine learning models on distributed healthcare data. Unlike traditional approaches, FedHealth enables multiple hospitals and medical institutions to collaboratively improve prediction models—such as hospital readmission risk—without sharing sensitive patient data. This ensures compliance with regulations like HIPAA and GDPR, while enabling high-performance, trustworthy, and privacy-protected healthcare AI solutions. The entire framework is containerized using Docker and orchestrated with Flower and TensorFlow, making it scalable and production-ready.
    """)
    st.image("assets/FlFinal.jpg", caption="Federated Learning with Homomorphic encrypted Docker Clients",use_container_width=True)
    st.markdown("""
    <h3 style='text-align: left;'>Why FedHealthcare?</h3>

    Federated Learning emerges as the revolution:
    - **Collaborate without centralizing data**: Multiple hospitals and institutions can jointly train models while keeping their sensitive patient data securely within their local systems.

    - **Improve models globally while keeping sensitive data locally**: The global model benefits from diverse, real-world datasets without any raw data ever leaving its source.

    - **Empower healthcare without breaking trust**: By preserving data privacy through encryption and secure aggregation, FedHealthcare builds trust among stakeholders and ensures compliance with strict healthcare regulations.
    """, unsafe_allow_html=True)
    
    st.markdown("""<h3 style='text-align: left;'>Our Privacy Strategy</h3>""", unsafe_allow_html=True)

    st.markdown("""
    We do not just federate — **we encrypt**.

    - **Federated Learning**: Clients train models locally and only share model updates (never raw data) with the server, ensuring complete data residency at the client side.
    - **Homomorphic Encryption**: Model updates are encrypted before transmission, guaranteeing that even the coordinating server cannot access or interpret individual updates.
    - **Secure Aggregation**: Encrypted updates from all clients are securely aggregated before decryption, maintaining the anonymity and integrity of each participant’s contribution.
    - **Client-Side Verification**: After receiving the aggregated global model, each client verifies privacy preservation by comparing the initial weights with the received weights. If they match, the integrity and privacy of the data are ensured.
    """)

    # Heading: "🛠 Tech Stack" 
    st.markdown("""<h3 style='text-align: left;'>🛠 Tools & Frameworks</h3>""", unsafe_allow_html=True)

    # Content: Tech Stack table
    st.markdown("""
    | Tech | Usage |
    |------|-------|
    |  **Flower Framework** | Federated Learning Implementation |
    |  **TensorFlow** | Deep Learning model training |
    |  **PySEAL** | Homomorphic encryption (weight-level) |
    |  **Docker** | Containerization of clients and server |
    |  **Python** | Backend programming |
    """)


    # Content (left-aligned)
    st.markdown("""
    🌟 "**Train together, stay private forever.**"
    """)


with tabs[1]:
    st.header("Architecture")
    st.markdown("""<h3 style='text-align: left;'>Components of the System Architecture</h3>""", unsafe_allow_html=True)
    st.markdown("""
    - **Clients (Hospitals)**: Local model training on private data inside Docker containers.
    - **Server**: Aggregates encrypted model weights.
    - **Encryption Layer**: Ensures secure weight transmission.
    - **Docker**: Isolated and reproducible training environment.
    """)
    st.image("assets/Picture1.jpg", caption="System Architecture", use_container_width=True)
    # Heading: "How It Works"
    st.markdown("""<h3 style='text-align: left;'>Working Mechanism</h3>""", unsafe_allow_html=True)

    # Content (left-aligned by default)
    st.markdown("""
    - **Local Model Training (Client-Side)**:
    Each participating hospital (client) trains the machine learning model on its own private patient data without sharing raw data with others.

    - **Encryption of Model Updates**:
    After local training, the model updates (weights) are encrypted using Homomorphic Encryption (HE) to ensure that no sensitive information is exposed during communication.

    - **Transmission to Central Server**:
    These encrypted model updates are securely transmitted to a central aggregation server.

    - **Secure Aggregation at Server**:
    The server aggregates all the encrypted updates without decrypting them, maintaining the privacy of individual client data throughout the process.

    - **Distribution of Global Model**:
    The aggregated and updated global model is then sent back to all the participating clients.

    - **Client-Side Verification**:
    Upon receiving the global model, each client performs verification by comparing the initial weights with the received weights to ensure that privacy has been preserved and the model integrity remains intact.

    - **Model Improvement**:
    This process is repeated across multiple rounds (iterations), resulting in a globally improved, privacy-preserving machine learning model.
    """)
    st.image("assets/FlowchartFinal.jpg", caption="Workflow Mechanism", use_container_width=True)

with tabs[2]:
    st.header("Simulation")
    option = st.radio("Select Input Method", ["Upload CSV Files", "Use Folder Path"], horizontal=True)
    uploaded_filenames = []
    if option == "Upload CSV Files":
        uploaded_files = st.file_uploader("Upload CSV log files from clients", type=["csv"], accept_multiple_files=True)
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully.")
            uploaded_filenames = [file.name for file in uploaded_files]
    else:
        folder_path = st.text_input("Enter folder path containing .csv files", value="logs/")
        if os.path.isdir(folder_path):
            for filename in sorted(os.listdir(folder_path)):
                if filename.endswith(".csv"):
                    uploaded_filenames.append(filename)
            if uploaded_filenames:
                st.success(f"✅ Found {len(uploaded_filenames)} CSV files.")
            else:
                st.warning("No CSV files found in the specified folder.")
        else:
            st.error("❌ Folder not found.")
    if uploaded_filenames:
        st.markdown("### 📂 Files Detected:")
        st.write(uploaded_filenames)
    else:
        st.info("📭 No files uploaded or found.")

with tabs[3]:
    st.header("Encryption")
    st.markdown("""
    1. Clients encrypt model weights.
    2. Server aggregates encrypted weights.
    3. Decryption only on the client-side post-aggregation.
    """)
    st.image("assets/encr.jpg", caption="Encryption Workflow", use_container_width=True)

with tabs[4]:
    st.header("Model Demo")
    uploaded_files = st.file_uploader("Upload 4 CSV files from clients", type="csv", accept_multiple_files=True)
    if uploaded_files and len(uploaded_files) == 4:
        st.success("✅ 4 CSV files uploaded successfully.")
        view_option = st.radio("Choose output: ", ["Log Output", "Image Output"], horizontal=True)
        if view_option == "Log Output":
            log_path = "assets/simulation_output.log"
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    log_data = f.read()
                st.code(log_data)
            else:
                st.warning("No simulation log file found.")
        elif view_option == "Image Output":
            image_folder = "assets/docker_logs"
            if os.path.exists(image_folder):
                image_files = [img for img in os.listdir(image_folder) if img.endswith((".jpg", ".png"))]
                for img_name in sorted(image_files):
                    st.image(os.path.join(image_folder, img_name), caption=img_name)
            else:
                st.warning("Docker output images folder not found.")
    else:
        st.info("📌 Please upload exactly 4 CSV files.")

with tabs[5]:
    st.header("Insights")
    st.markdown("""
    - **Secure FL Simulation** designed for privacy-aware healthcare.
    - Client-side encryption ensures **data confidentiality**.
    - Docker isolation maintains environment reproducibility.
    """)

with tabs[6]:
    st.header("About")
    st.markdown("""
    **Team:** Techmind Innovators  
    **Mentor** - **Dr. Tapasi Bhattacharjee, Professor, TINT(Techno International New Town)**
    **Team Members:**
    - **Anwarul Haque** (Team Lead)
    - **Faisal Shamim**
    - **Tahseen Atique Ali**
    - **Md Mujtaba** 

    **Technologies Used:** TensorFlow, Flower, Docker, Streamlit, Python
    """)
    st.write("Made with ❤️ by Techmind Innovators")
