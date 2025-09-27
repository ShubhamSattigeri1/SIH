#!/usr/bin/env python3
"""
Streamlit Demo for LandCover Carbon Credits Estimation
Interactive web interface for uploading images and calculating carbon credits
"""

import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os

# -----------------------------
# Carbon Credits Calculator
# -----------------------------
def calculate_carbon_credits(area_hectares, rate):
    """Simple carbon credits calculation"""
    return area_hectares * rate

# -----------------------------
# Mock Segmentation Model
# -----------------------------
class MockSegmentationModel:
    """Simulate a segmentation model for demonstration"""
    def predict_single_image(self, image_path, meters_per_pixel):
        # Load image
        image = np.array(Image.open(image_path).convert("RGB")) / 255.0
        height, width, _ = image.shape
        vegetation_mask = np.random.randint(0, 2, size=(height, width))  # Random vegetation mask
        vegetation_pixels = vegetation_mask.sum()
        area_m2 = vegetation_pixels * (meters_per_pixel ** 2)
        area_hectares = area_m2 / 10000
        estimated_plant_count = int(area_hectares * 100)  # mock count
        plant_density_per_ha = estimated_plant_count / area_hectares if area_hectares > 0 else 0
        
        return {
            "original_image": image,
            "vegetation_mask": vegetation_mask,
            "vegetation_pixels": vegetation_pixels,
            "area_m2": area_m2,
            "area_hectares": area_hectares,
            "estimated_plant_count": estimated_plant_count,
            "plant_density_per_ha": plant_density_per_ha,
            "carbon_credits_per_year": 0  # placeholder
        }

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Carbon Credits Estimator", page_icon="🌱", layout="wide")

# Load model (mock for now)
@st.cache_resource
def load_model():
    return MockSegmentationModel()

# Load IoT data (dummy)
@st.cache_data
def load_iot_data():
    try:
        df = pd.read_csv("dummy_iot_data.csv")
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    except:
        return pd.DataFrame()

def create_overlay_visualization(original_image, vegetation_mask, alpha=0.6):
    overlay = np.zeros_like(original_image)
    overlay[:, :, 1] = vegetation_mask * 255
    blended = cv2.addWeighted(
        (original_image * 255).astype(np.uint8),
        1 - alpha,
        overlay.astype(np.uint8),
        alpha,
        0
    )
    return blended

def create_results_chart(results):
    metrics = ['Area (hectares)', 'Plant Count', 'Plant Density (per ha)', 'Carbon Credits (per year)']
    values = [
        results['area_hectares'],
        results['estimated_plant_count'],
        results['plant_density_per_ha'],
        results['carbon_credits_per_year']
    ]
    fig = go.Figure(go.Bar(
        x=metrics,
        y=values,
        text=[f"{v:.2f}" for v in values],
        textposition='auto',
        marker_color=['#2E8B57', '#228B22', '#32CD32', '#90EE90']
    ))
    fig.update_layout(title="Carbon Credits Estimation Results", showlegend=False, height=400)
    return fig

# -----------------------------
# Main App
# -----------------------------
def main():
    st.title("🌱 Carbon Credits Estimation from Aerial Imagery")
    st.markdown("Upload an aerial image to estimate vegetation area, plant density, and carbon credit potential")
    
    # Load model and IoT data
    model = load_model()
    iot_data = load_iot_data()
    
    # Sidebar
    st.sidebar.header("Configuration")
    carbon_rate = st.sidebar.slider("Carbon Credits Rate (per hectare per year)", 2.0, 6.0, 4.0, 0.1)
    meters_per_pixel = st.sidebar.number_input("Image Resolution (meters per pixel)", 0.1, 2.0, 0.25, 0.05)
    
    if not iot_data.empty:
        plot_ids = iot_data['plot_id'].unique()
        selected_plot = st.sidebar.selectbox("Select Plot ID for IoT Data", options=['None'] + list(plot_ids))
    else:
        selected_plot = 'None'
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Image Upload")
        uploaded_file = st.file_uploader("Choose an aerial image...", type=['png','jpg','jpeg'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            temp_path = "temp_uploaded_image.png"
            image.save(temp_path)
            
            if st.button("🔍 Analyze Image"):
                with st.spinner("Analyzing image..."):
                    results = model.predict_single_image(temp_path, meters_per_pixel)
                    results['carbon_credits_per_year'] = calculate_carbon_credits(
                        results['area_hectares'], carbon_rate
                    )
                    st.session_state['results'] = results
                    st.session_state['temp_path'] = temp_path
                    
                if os.path.exists(temp_path):
                    os.remove(temp_path)
    
    with col2:
        st.header("📊 Results")
        if 'results' in st.session_state:
            results = st.session_state['results']
            st.metric("Vegetation Area", f"{results['area_m2']:.1f} m²", f"{results['area_hectares']:.4f} ha")
            st.metric("Estimated Plant Count", f"{results['estimated_plant_count']:,}")
            st.metric("Plant Density", f"{results['plant_density_per_ha']:.1f} plants/ha")
            st.metric("Carbon Credits/Year", f"{results['carbon_credits_per_year']:.2f} credits")
            
            overlay_image = create_overlay_visualization(results['original_image'], results['vegetation_mask'])
            viz_col1, viz_col2 = st.columns(2)
            with viz_col1: st.image(results['original_image'], caption="Original Image", use_column_width=True)
            with viz_col2: st.image(overlay_image, caption="Vegetation Overlay", use_column_width=True)
            
            fig = create_results_chart(results)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👆 Upload an image and click 'Analyze Image' to see results")
    
    st.markdown("---")
    st.markdown("ℹ This is a demo app. Replace MockSegmentationModel with your trained U-Net model for real results.")

if _name_ == "_main_":
    main()
#Streamlit one