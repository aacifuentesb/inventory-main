import streamlit as st
from app_utils import get_base64_of_bin_file

st.set_page_config(
    page_title="Supply Chain Management",
    page_icon="📦",
    layout="wide"
)

# Customize the sidebar
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            padding: 0.5rem 0;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            font-size: 1rem;
            color: #1f2937;
            padding: 0.3rem 0;
        }
        
        /* Navigation buttons styling */
        .stButton button {
            width: 100%;
            border: none;
            padding: 0.5rem 1rem;
            background: transparent;
            text-align: left;
            color: #1f2937;
            font-size: 1rem;
            margin: 0.2rem 0;
            border-radius: 0.5rem;
            transition: all 0.2s ease;
        }
        .stButton button:hover {
            background: #e2e8f0;
            color: #1a56db;
        }
        
        /* Section headers */
        .sidebar-header {
            color: #1f2937;
            font-size: 1.2rem;
            font-weight: 600;
            margin: 1.5rem 0 0.5rem 0;
            padding-bottom: 0.3rem;
            border-bottom: 2px solid #e2e8f0;
        }
        
        /* About section */
        .about-section {
            background: #f1f5f9;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .about-section p {
            font-size: 0.9rem !important;
            color: #4b5563 !important;
        }
        
        /* Help section */
        .help-section {
            font-size: 0.9rem;
            color: #6b7280;
            padding: 0.5rem;
        }
        .help-section a {
            color: #2563eb;
            text-decoration: none;
        }
        .help-section a:hover {
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)

def display_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-header">🔍 Navigation</div>', unsafe_allow_html=True)
        
        # Main navigation buttons
        if st.button("🏠 Home", use_container_width=True):
            st.switch_page("Home.py")
        if st.button("📈 Single SKU Analysis", use_container_width=True):
            st.switch_page("pages/1_single_sku_analysis.py")
        if st.button("📊 Multiple SKU Analysis", use_container_width=True):
            st.switch_page("pages/2_multiple_sku_analysis.py")
        
        # About section
        st.markdown('<div class="sidebar-header">ℹ️ About</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="about-section">
        <p>Supply Chain Management System v1.0</p>
        <p>Features:</p>
        <ul>
            <li>Demand Forecasting</li>
            <li>Inventory Optimization</li>
            <li>Supply Planning</li>
            <li>Cost Analysis</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Help section
        st.markdown('<div class="sidebar-header">💡 Help</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="help-section">
        <p>📧 <a href="mailto:support@example.com">support@example.com</a></p>
        <p>📚 <a href="https://docs.example.com" target="_blank">Documentation</a></p>
        </div>
        """, unsafe_allow_html=True)

def display_logo_and_title():
    logo_base64 = get_base64_of_bin_file("logo.svg")
    st.markdown(
    f"""
    <style>
    .logo-title {{
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 2rem 0;
        position: relative;
        background: linear-gradient(135deg, #f6f8fa 0%, #ffffff 100%);
        border-radius: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    .logo {{
        height: 5rem;
        position: absolute;
        left: 2rem;
        transition: transform 0.3s ease;
    }}
    .logo:hover {{
        transform: scale(1.1);
    }}
    .title {{
        color: #1f2937;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
        flex: 1;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }}
    
    /* Card styles */
    .feature-card {{
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }}
    .feature-card:hover {{
        transform: translateY(-5px);
    }}
    .feature-title {{
        color: #1f2937;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }}
    .feature-list {{
        color: #4b5563;
        margin: 0;
        padding-left: 1.5rem;
    }}
    .feature-list li {{
        margin-bottom: 0.5rem;
    }}
    
    /* Section styles */
    .section-title {{
        color: #1f2937;
        font-size: 2rem;
        font-weight: 700;
        margin: 2rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3b82f6;
    }}
    </style>
    <div class="logo-title">
        <img src="data:image/svg+xml;base64,{logo_base64}" alt="Company Logo" class="logo">
        <h1 class="title">📦 Supply Chain Management</h1>
    </div>
    """,
    unsafe_allow_html=True
    )

def main():
    display_logo_and_title()
    
    # Welcome section
    st.markdown("""
    <div style='text-align: center; max-width: 800px; margin: 0 auto; padding: 2rem;'>
        <p style='font-size: 1.2rem; color: #4b5563; line-height: 1.6;'>
            Welcome to our advanced Supply Chain Management System. This platform provides powerful tools for inventory optimization,
            demand forecasting, and supply chain analytics, helping you make data-driven decisions for your business.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Analysis Options section
    st.markdown("<h2 class='section-title'>🔍 Choose Your Analysis</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='feature-card'>
            <h3 class='feature-title'>📊 Single SKU Analysis</h3>
            <ul class='feature-list'>
                <li>Detailed analysis of individual products</li>
                <li>Multiple forecasting models</li>
                <li>Inventory optimization</li>
                <li>Monte Carlo simulations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Single SKU Analysis", key="single_sku"):
            st.switch_page("pages/single_sku_analysis.py")
    
    with col2:
        st.markdown("""
        <div class='feature-card'>
            <h3 class='feature-title'>📈 Multiple SKU Analysis</h3>
            <ul class='feature-list'>
                <li>Batch processing of multiple products</li>
                <li>Automated parameter optimization</li>
                <li>Global inventory insights</li>
                <li>Aggregated reports</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Multiple SKU Analysis", key="multiple_sku"):
            st.switch_page("pages/multiple_sku_analysis.py")
    
    # Features Overview section
    st.markdown("<h2 class='section-title'>🎯 Key Features</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='feature-card'>
            <h3 class='feature-title'>📈 Analysis Types</h3>
            <ul class='feature-list'>
                <li>Historical Analysis</li>
                <li>Demand Forecasting</li>
                <li>Inventory Optimization</li>
                <li>Supply Planning</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-card'>
            <h3 class='feature-title'>🛠️ Tools & Models</h3>
            <ul class='feature-list'>
                <li>Multiple Forecasting Models</li>
                <li>Inventory Control Policies</li>
                <li>Monte Carlo Simulations</li>
                <li>Parameter Optimization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Getting Started section
    st.markdown("<h2 class='section-title'>🚀 Getting Started</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='feature-card'>
        <ol style='color: #4b5563; margin: 0; padding-left: 1.5rem;'>
            <li style='margin-bottom: 1rem;'>Choose your analysis type:
                <ul style='margin-top: 0.5rem;'>
                    <li>Single SKU Analysis for detailed individual product analysis</li>
                    <li>Multiple SKU Analysis for batch processing</li>
                </ul>
            </li>
            <li style='margin-bottom: 1rem;'>Prepare your data:
                <ul style='margin-top: 0.5rem;'>
                    <li>Excel file with columns: SKU, Date, QTY</li>
                    <li>Optional: SKU master data for additional insights</li>
                </ul>
            </li>
            <li style='margin-bottom: 1rem;'>Configure parameters and run the analysis</li>
            <li style='margin-bottom: 1rem;'>Explore results and download reports</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 