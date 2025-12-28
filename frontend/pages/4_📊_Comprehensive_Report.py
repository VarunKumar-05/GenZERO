import streamlit as st
import os
import sys
import google.generativeai as genai
from datetime import datetime

# Add parent directory to path and import backend
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend_integration import get_gemini_api_key

st.set_page_config(
    page_title="GenZERO - Comprehensive Report",
    page_icon="📊",
    layout="wide"
)

st.title("📊 GenZERO Comprehensive Clinical Report")
st.markdown("Integrates findings from Speech Analysis, Child Mind, and Psychosis Detection models using **Gemini 2.5 Flash**.")

# ============================================================================
# RESULTS AGGREGATION
# ============================================================================

st.divider()

col1, col2, col3 = st.columns(3)

# 1. Speech Analysis Results
speech_results = st.session_state.get('speech_results', {})
with col1:
    st.subheader("🎤 Speech Analysis")
    if speech_results:
        states = speech_results.get('detected_states', [])
        if states:
            for s in states:
                st.error(f"⚠️ {s}")
        else:
            st.success("✅ Stable")
        
        # Word analysis summary
        wa = speech_results.get('word_analysis', {})
        if wa:
            st.metric("Risk Words", len(wa.get('top_determining_words', [])))
    else:
        st.info("No data available. Run Speech Analysis (Page 1) first.")

# 2. Child Mind Results
child_results = st.session_state.get('child_mind_results', {})
with col2:
    st.subheader("👶 Child Mind")
    if child_results:
        sii_label = child_results.get('sii_label', 'Unknown')
        sii_score = child_results.get('sii_score', 0)
        
        color = "green" if sii_score == 0 else "orange" if sii_score == 1 else "red"
        st.markdown(f"**SII:** :{color}[{sii_label}]")
        st.metric("PCIAT Score", child_results.get('pciat_score', 0))
    else:
        st.info("No data available. Run Child Mind (Page 2) first.")

# 3. Psychosis Results
psych_results = st.session_state.get('psychosis_results', {})
with col3:
    st.subheader("🧬 Psychosis Detection")
    if psych_results:
        pred = psych_results.get('prediction', 'Unknown')
        conf = psych_results.get('confidence', 0)
        
        st.markdown(f"**Pred:** {pred}")
        st.metric("Confidence", f"{conf:.1f}%")
        
        probs = psych_results.get('probs', {})
        st.progress(probs.get('SZ', 0)/100, text=f"SZ Risk: {probs.get('SZ', 0):.1f}%")
    else:
        st.info("No data available. Run Psychosis Detection (Page 3) first.")

#Check if we have enough data to generate a report
has_data = speech_results or child_results or psych_results

# ============================================================================
# GEMINI REPORT GENERATION
# ============================================================================

st.divider()
st.subheader("🤖 AI Clinical Synthesis")

api_key = get_gemini_api_key()

if not api_key:
    st.error("❌ Gemini API Key not found. Please set GEMINI_API_KEY in .env file.")
else:
    if st.button("🚀 Generate Comprehensive Report", disabled=not has_data, type="primary"):
        if not has_data:
            st.warning("⚠️ Please run at least one model analysis first.")
        else:
            with st.spinner("Consulting Gemini 2.5 Flash Clinical Model..."):
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.5-flash') # Updated to 2.5 as requested
                    
                    # Construct Prompt
                    prompt = f"""
                    You are GenZERO, an advanced AI clinical assistant specializing in neurological and psychiatric assessment.
                    Generate a comprehensive clinical report based on the following multi-modal analysis results.

                    ### PATIENT DATA & MODEL OUTPUTS:

                    1. SPEECH ANALYSIS (Bio-Digital Hybrid SNN):
                    {speech_results}

                    2. CHILD MIND (Screentime/Addiction Analysis):
                    {child_results}

                    3. PSYCHOSIS DETECTION (fMRI BDHNet):
                    {psych_results}

                    ### REPORT INSTRUCTIONS:
                    1. **Executive Summary**: Synthesize the findings into a coherent clinical narrative.
                    2. **Risk Assessment**: Identify high-risk correlations between speech patterns, addiction markers (SII), and psychosis indicators.
                    3. **Neurological Interpretation**: Interpret the neural statistics (sparsity, activation) provided in the context of brain stability.
                    4. **Recommendations**: Suggest actionable clinical next steps or interventions.
                    5. **Format**: Use professional medical report formatting with Markdown.
                    """
                    
                    response = model.generate_content(prompt)
                    report_content = response.text
                    
                    # Display Report
                    st.markdown("### 📄 GenZERO Clinical Report")
                    st.markdown(report_content)
                    
                    # Download Button
                    st.download_button(
                        label="📥 Download Clinical Report",
                        data=report_content,
                        file_name=f"genzero_clinical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating report: {e}")
