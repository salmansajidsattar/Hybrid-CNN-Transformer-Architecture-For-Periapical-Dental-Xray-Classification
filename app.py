import streamlit as st
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.config import Config
from src.model import create_model
from src.dataset import get_transforms
from src.utils import load_checkpoint

# Page configuration
st.set_page_config(
    page_title="Dental X-ray Classifier",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .healthy {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .disease {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = create_model(num_classes=Config.NUM_CLASSES)
        model = model.to(Config.DEVICE)
        
        checkpoint_path = Config.CHECKPOINT_DIR / 'best_model.pth'
        if checkpoint_path.exists():
            optimizer = torch.optim.AdamW(model.parameters())
            load_checkpoint(model, optimizer, checkpoint_path)
            model.eval()
            return model, True
        else:
            st.warning("⚠️ No trained model found. Please train the model first.")
            return model, False
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, False

def predict_image(model, image):
    """Make prediction on uploaded image"""
    try:
        transform = get_transforms(augment=False)
        image_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        return predicted.item(), confidence.item(), probs[0].cpu().numpy()
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None

def plot_prediction_bars(probs, class_names):
    """Create bar plot for prediction probabilities"""
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#28a745', '#dc3545']
    bars = ax.barh(class_names, probs * 100, color=colors, alpha=0.8)
    
    ax.set_xlabel('Confidence (%)', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Confidence', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height()/2,
                f'{prob*100:.2f}%',
                ha='left', va='center', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🦷 Dental X-ray Classification System</h1>', 
                unsafe_allow_html=True)
    st.markdown("### Hybrid CNN + Transformer Architecture")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/tooth.png", width=100)
        st.title("⚙️ Settings")
        
        st.markdown("### 📊 Model Information")
        st.info(f"""
        **Architecture:** Hybrid CNN + Transformer
        
        **Classes:** {', '.join(Config.CLASS_NAMES)}
        
        **Image Size:** {Config.IMG_SIZE}×{Config.IMG_SIZE}
        
        **Device:** {Config.DEVICE}
        
        **Embed Dim:** {Config.EMBED_DIM}
        
        **Transformer Layers:** {Config.NUM_TRANSFORMER_LAYERS}
        
        **Attention Heads:** {Config.NUM_HEADS}
        """)
        
        # Load metrics if available
        metrics_path = Config.RESULTS_DIR / 'test_metrics.json'
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            st.markdown("### 📈 Model Performance")
            col1, col2 = st.columns(2)
            with col1:
                pass
                # st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
                # st.metric("Precision", f"{metrics['precision']*100:.2f}%")
            with col2:
                pass
                # st.metric("Recall", f"{metrics['recall']*100:.2f}%")
                # st.metric("F1-Score", f"{metrics['f1_score']*100:.2f}%")
        
        st.markdown("### ℹ️ About")
        st.write("""
        This application uses a hybrid CNN-Transformer architecture 
        to classify periapical dental X-ray images.
        
        **How to use:**
        1. Upload an X-ray image
        2. Click "Analyze Image"
        3. View prediction results
        """)
        
        st.markdown("---")
        st.markdown("### 🔬 Model Architecture")
        st.write("""
        1. **CNN Feature Extraction**
           - 4 convolutional blocks
           - Spatial feature extraction
        
        2. **Patch Embedding**
           - Convert features to patches
           - Positional encoding
        
        3. **Transformer Encoder**
           - Multi-head self-attention
           - Feed-forward networks
        
        4. **Classification Head**
           - Global pooling
           - Dense layers
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload X-ray Image")
        uploaded_file = st.file_uploader(
            "Choose a dental X-ray image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a periapical dental X-ray image for classification"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded X-ray', use_container_width=True)
            
            # Image info
            st.markdown("#### 📋 Image Details")
            st.write(f"- **Filename:** {uploaded_file.name}")
            st.write(f"- **Size:** {image.size[0]}×{image.size[1]} pixels")
            st.write(f"- **Format:** {image.format}")
            st.write(f"- **Mode:** {image.mode}")
            
            # Analyze button
            st.markdown("---")
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("🔄 Analyzing image..."):
                    model, model_loaded = load_model()
                    
                    if model_loaded and model is not None:
                        predicted_class, confidence, probs = predict_image(model, image)
                        
                        if predicted_class is not None:
                            # Store results in session state
                            st.session_state['prediction'] = predicted_class
                            st.session_state['confidence'] = confidence
                            st.session_state['probs'] = probs
                            st.success("✅ Analysis complete!")
                            st.rerun()
                    else:
                        st.error("❌ Model not loaded. Please train the model first.")
    
    with col2:
        st.markdown("### 📊 Prediction Results")
        
        if 'prediction' in st.session_state:
            predicted_class = st.session_state['prediction']
            confidence = st.session_state['confidence']
            probs = st.session_state['probs']
            
            class_name = Config.CLASS_NAMES[predicted_class]
            
            # Prediction box
            box_class = 'healthy' if predicted_class == 0 else 'disease'
            st.markdown(f"""
            <div class="prediction-box {box_class}">
                <h2 style="margin: 0; color: #333;">🎯 Diagnosis: {class_name}</h2>
                <h3 style="margin: 0.5rem 0 0 0; color: #666;">
                    Confidence: {confidence*100:.2f}%
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence bars
            st.markdown("#### 📊 Confidence Distribution")
            fig = plot_prediction_bars(probs, Config.CLASS_NAMES)
            st.pyplot(fig)
            
            # Detailed probabilities
            st.markdown("#### 📋 Detailed Probabilities")
            for i, (class_name, prob) in enumerate(zip(Config.CLASS_NAMES, probs)):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"**{class_name}**")
                with col_b:
                    st.write(f"{prob*100:.2f}%")
                st.progress(float(prob))
            
            # Additional information
            st.markdown("---")
            if predicted_class == 1:  # Disease
                st.warning("""
                ⚠️ **Disease Detected**
                
                The model has detected signs of disease in this X-ray. 
                
                **Recommended Actions:**
                - Consult with a dental professional immediately
                - Schedule a thorough examination
                - Discuss treatment options
                
                **Note:** This is an AI-assisted preliminary analysis.
                """)
            else:  # Healthy
                st.success("""
                ✅ **Healthy Classification**
                
                The model has classified this X-ray as healthy.
                
                **Recommendations:**
                - Continue regular dental check-ups
                - Maintain good oral hygiene
                - Schedule routine cleanings
                
                **Note:** Regular professional check-ups are still important.
                """)
            
            # Disclaimer
            st.markdown("---")
            st.info("""
            **⚕️ Medical Disclaimer**
            
            This is an AI-assisted diagnostic tool and should **NOT** replace 
            professional medical diagnosis. Always consult with qualified 
            healthcare professionals for medical advice, diagnosis, and treatment.
            
            The predictions are based on machine learning models and may not 
            be 100% accurate. Use this tool as a supplementary aid only.
            """)
            
        else:
            st.info("👆 Upload an image and click 'Analyze Image' to see results")
            
            # Show sample workflow
            st.markdown("#### 🎬 How It Works")
            st.markdown("""
            1. **Upload:** Select an X-ray image from your device
            2. **Process:** Image is preprocessed and normalized
            3. **CNN Analysis:** Features extracted through convolutional layers
            4. **Transformer:** Global context captured via attention mechanisms
            5. **Prediction:** Final classification with confidence scores
            """)
            
            # Show example
            st.markdown("#### 💡 Expected Input")
            st.write("""
            - **Format:** JPG, JPEG, or PNG
            - **Type:** Periapical dental X-ray
            - **Quality:** Clear, well-exposed radiograph
            - **Size:** Any size (will be resized to 224×224)
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p><strong>Dental X-ray Classification System</strong></p>
        <p>Hybrid CNN + Transformer Architecture | Powered by PyTorch & Streamlit</p>
        <p style='font-size: 0.9em;'>
            📧 For research and educational purposes only<br>
            🔬 Always validate results with medical professionals
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()