"""
Clinical Analysis Report Generator

A professional Python application that generates clinical behavioral analysis reports 
from DAIC (Distress Analysis Interview Corpus) data using Google's Gemini API.

Features:
- Environment variable configuration
- Comprehensive logging
- Professional PDF report generation
- Error handling and validation
"""

import os
import logging
from pathlib import Path
import google.genai as genai
import markdown
from xhtml2pdf import pisa
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "clinical_report.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class ClinicalReportGenerator:
    """Main class for generating clinical analysis reports."""
    
    def __init__(self):
        """Initialize the report generator with configuration from environment variables."""
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("MODEL_NAME", "gemini-2.0-flash-exp")
        self.input_file_path = Path(os.getenv("INPUT_FILE_PATH", "data/daic_analysis_report.txt"))
        self.output_pdf_path = Path(os.getenv("OUTPUT_PDF_PATH", "output/Clinical_Analysis_Report.pdf"))
        
        # Validate API key
        if not self.api_key or self.api_key == "your_api_key_here":
            logger.error("GEMINI_API_KEY not found or not set in .env file")
            raise ValueError("Please set GEMINI_API_KEY in your .env file")
        
        logger.info(f"Initialized ClinicalReportGenerator with model: {self.model_name}")
    
    def read_daic_analysis_report(self):
        """
        Read the DAIC analysis report text file and return its contents.
        
        Returns:
            str: Contents of the file or error message
        """
        logger.info(f"Reading input file: {self.input_file_path}")
        
        # Check if file exists
        if not self.input_file_path.exists():
            logger.warning(f"File not found at {self.input_file_path}")
            logger.info("Creating sample data for testing...")
            
            # Create sample data
            dummy_content = """DAIC Analysis Report
====================

Global State-Determining Indicators:

State: Tremor Detected
  Top Determining Words: just(2748), so(2443), it's(2095), like(1873), um(1068)
  
State: Speech Disfluency Detected
  Top Determining Words: just(2748), so(2443), it's(2095), like(1873), um(1068)

====================
Session Reports:

- Session 300_P
  Frames: 3243
  Hidden Norm Mean: 346.8597
  Anomalies Detected: {'Tremor Detected': 3243, 'Speech Disfluency Detected': 3243}
  State Indicators: um(224), uh(172), like(96), going(93), school(78)

- Session 301_P
  Frames: 4120
  Hidden Norm Mean: 714.1008
  Anomalies Detected: {'Tremor Detected': 4120, 'Speech Disfluency Detected': 4120}
  State Indicators: just(2239), it's(1744), think(1520), like(1323), so(1270)
"""
            
            # Ensure data directory exists
            self.input_file_path.parent.mkdir(exist_ok=True)
            
            try:
                with open(self.input_file_path, "w", encoding="utf-8") as f:
                    f.write(dummy_content)
                logger.info(f"Created sample data file at {self.input_file_path}")
                return dummy_content
            except Exception as e:
                logger.error(f"Error creating sample file: {e}")
                return f"Error creating sample file: {e}"
        
        try:
            with open(self.input_file_path, "r", encoding="utf-8") as f:
                content = f.read()
                logger.info(f"Successfully read {len(content)} characters from input file")
                return content
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return f"Error reading file: {e}"
    
    def generate_clinical_analysis_report(self, input_data):
        """
        Generate a clinical behavioral analysis report using Gemini API.
        
        Args:
            input_data (str): Raw DAIC analysis report text
            
        Returns:
            str: Markdown-formatted clinical analysis report
        """
        logger.info(f"Sending data to Gemini ({self.model_name})...")
        
        try:
            client = genai.Client(api_key=self.api_key)
            
            prompt1 = f"""### SYSTEM ROLE
You are an Expert Clinical Data Scientist and Behavioral Analyst specializing in the Distress Analysis Interview Corpus (DAIC). Your task is to interpret raw automated analysis logs and synthesize them into a professional, human-readable clinical insight report.

### CONTEXT & TASK
I will provide you with a raw text file titled "DAIC Analysis Report". This file contains:
1. **Global State-Determining Indicators**: Aggregated words and sentences associated with specific states.
2. **Session Reports**: Data for individual sessions including frame counts, hidden norm means, and anomalies.

### INSTRUCTIONS (Step-by-Step)
Please follow this Chain-of-Thought process:

1. **Executive Summary**: Summarize the data's purpose. Explicitly mention how many distinct sessions are analyzed.
2. **Global Indicator Analysis**:
    * Analyze the "Global State-Determining Indicators".
    * **CRITICAL SANITY CHECK**: Compare word counts for "Tremor" vs. "Speech Disfluency". If identical, flag as "System Artifact" or "High Feature Overlap".
3. **Session-by-Session Deep Dive**:
    * Process EVERY session found in the input.
    * **Isolation Rule**: Treat each session as a data silo. Never mix metrics between sessions.
    * **Psycholinguistic Analysis**: Categorize words (e.g., "Narrative Fillers" like 'um/uh' vs. "Cognitive Hedges" like 'think/know').
4. **Comparative Insights**:
    * Compare "Hidden Norm Means" and discuss magnitude of difference.
    * Contrast vocabulary patterns.

### OUTPUT FORMAT (Markdown)
# Clinical Behavioral Analysis Report

## 1. Executive Summary
[Summary including count of sessions analyzed]

## 2. Global Behavioral Markers
* **Tremor Indicators**: [Analysis with specific word counts]
* **Disfluency Indicators**: [Analysis - include note on overlap if found]

## 3. Session Analysis
### Session [ID]
* **Metrics**: Frames: [X] | Hidden Norm Mean: [Y]
* **Psycholinguistic Profile**: [Analysis of specific words]

[Repeat for ALL sessions]

## 4. Comparative Insights & Critical Conclusion
* **Intensity Check:** Compare "Hidden Norm Means"
* **Vocabulary Contrast:** Contrast patterns
* **Medical Consequence:** Flag any system artifacts

---
### INPUT DATA
{input_data}
"""
            
            prompt=prompt = f"""### SYSTEM ROLE
You are an Expert Clinical Data Scientist and Behavioral Analyst with specialized expertise in the Distress Analysis Interview Corpus (DAIC). Your analysis directly informs clinical decision-making.

### CRITICAL CONTEXT
You will analyze a DAIC Analysis Report containing:
1. **Global State-Determining Indicators**: Aggregated linguistic markers across sessions
2. **Session-Specific Data**: Individual session metrics (frames, hidden norm means, anomalies, word patterns)

**Data Integrity Note**: This system may conflate motor tremors with cognitive disfluency. Your analysis must identify these artifacts.

### RESPONSE REQUIREMENTS
- **Length:** 800-1200 words (comprehensive but concise)
- **Tone:** Professional, clinical, objective
- **Precision:** Use exact numbers from input data
- **Avoid:** Speculation beyond data, personal opinions, unsupported claims
- **Include:** All quantitative metrics, percentage calculations, clinical interpretations

### ANALYSIS PROTOCOL (Execute in Order)

#### Step 1: Executive Summary
- Count and list all distinct sessions (e.g., "3 sessions: 300_P, 301_P, 304_P")
- State the analysis timeframe and data sources
- Provide a one-sentence clinical significance statement

#### Step 2: Global Behavioral Markers - WITH CRITICAL VALIDATION
**For each indicator (Tremor, Speech Disfluency):**
- List top 5 determining words with exact counts
- Calculate overlap percentage: (identical words / total unique words) × 100

**MANDATORY SANITY CHECK:**
IF overlap > 90% THEN:
  - Flag as "⚠️ SYSTEM ARTIFACT DETECTED"
  - State: "The {{overlap}}% overlap suggests the model is conflating distinct phenomena"
  - Note clinical implication: "Findings should not be interpreted as independent biological markers"

#### Step 3: Session-by-Session Analysis (Isolation Protocol)
**For EACH session, provide:**

**Quantitative Metrics:**
- Frames: [exact number]
- Hidden Norm Mean: [value] → Interpret: [Low <400 / Medium 400-600 / High >600]

**Psycholinguistic Profile:**
Categorize words into:
- **Narrative Fillers** (um, uh, like): [count] - indicates speech planning difficulty
- **Cognitive Hedges** (think, know, maybe): [count] - suggests uncertainty
- **Temporal Markers** (when, going): [count] - contextual anchoring
- **Affective Terms**: [if present, list with interpretation]

**Clinical Interpretation:**
Based on Hidden Norm Mean and vocabulary, assess:
- Cognitive load (low/medium/high)
- Speech fluency pattern
- Potential distress markers

#### Step 4: Comparative Clinical Insights

**A. Intensity Gradient Analysis:**
- Rank sessions by Hidden Norm Mean (lowest to highest)
- Calculate the ratio: (Highest HNM / Lowest HNM)
- Interpret: >2.0 ratio suggests significant inter-session variability

**B. Vocabulary Pattern Contrast:**
Compare dominant word categories across sessions:
- Session with highest filler ratio: [ID] - [ratio]% fillers
- Session with highest hedge ratio: [ID] - [ratio]% hedges
- Clinical significance: [interpretation]

**C. System Artifact Impact:**
IF system artifact detected in Step 2:
  - Reassess: Can Tremor and Disfluency findings be trusted as independent?
  - Recommend: "Clinicians should interpret motor and speech findings with caution"

### VALIDATION STEP (Complete Before Finalizing)
Before submitting your analysis, verify:
- [ ] All sessions from input data are analyzed (no sessions skipped)
- [ ] No metrics are mixed between different sessions
- [ ] Overlap percentage is calculated and reported
- [ ] Each session has quantitative metrics AND clinical interpretation
- [ ] Comparative insights include ratio calculations
- [ ] System artifact warning included if overlap >90%
- [ ] Word counts match input data exactly
- [ ] Response length is within 800-1200 words
- [ ] All numbers are sourced from input data (no fabrication)
- [ ] Professional clinical tone maintained throughout

### OUTPUT FORMAT

# Clinical Behavioral Analysis Report
**Analysis Date:** [Current Date]
**Sessions Analyzed:** [N sessions]
**Report Status:** Validated ✓

## 1. Executive Summary
[3-4 sentences covering: session count, date range, key finding, clinical relevance]

## 2. Global Behavioral Markers

### Tremor Indicators
- **Top Markers:** [word1(count), word2(count), word3(count), word4(count), word5(count)]
- **Total Unique Words:** [N]
- **Sample Sentences:** [1-2 representative examples]

### Speech Disfluency Indicators  
- **Top Markers:** [word1(count), word2(count), word3(count), word4(count), word5(count)]
- **Total Unique Words:** [N]
- **Sample Sentences:** [1-2 representative examples]

### ⚠️ Data Integrity Assessment
- **Overlap Analysis:** [X]% of indicators are identical
- **Calculation:** [Y identical words / Z total unique words × 100]
- **Artifact Status:** [CLEAN / ARTIFACT DETECTED]
- **Clinical Implication:** [specific statement about interpretation validity]

## 3. Individual Session Analysis

### Session 300_P
**Quantitative Metrics:**
- Frames: 3,243
- Hidden Norm Mean: 346.86 (Low intensity - below 400 threshold)
- Anomalies: Tremor Detected (3,243 frames), Speech Disfluency Detected (3,243 frames)

**Psycholinguistic Profile:**
- **Narrative Fillers:** um(224), uh(172), like(96) = 492 total (15.2% of vocabulary)
- **Cognitive Hedges:** going(93), would(57) = 150 total (4.6% of vocabulary)
- **Contextual Markers:** school(78), right(62), high(54) = 194 total (6.0% of vocabulary)
- **Dominant Pattern:** High filler usage with educational context markers

**Clinical Interpretation:**
Lower Hidden Norm Mean (346.86) combined with elevated filler ratio (15.2%) suggests active speech processing challenges, possibly anxiety-related. Educational context words indicate discussion of academic stressors.

### Session 301_P
**Quantitative Metrics:**
- Frames: 4,120
- Hidden Norm Mean: 714.10 (High intensity - above 600 threshold)
- Anomalies: Tremor Detected (4,120 frames), Speech Disfluency Detected (4,120 frames)

**Psycholinguistic Profile:**
- **Narrative Fillers:** just(2,239), like(1,323) = 3,562 total (86.5% of vocabulary)
- **Cognitive Hedges:** think(1,520), it's(1,744), don't(902), know(853) = 5,019 total (121.8% - indicates word repetition)
- **Work-Related Terms:** work(436), well(574) = 1,010 total (24.5% of vocabulary)
- **Dominant Pattern:** Extremely high hedge usage with work-related stress markers

**Clinical Interpretation:**
Significantly elevated Hidden Norm Mean (714.10) with pervasive cognitive hedging (think, don't, know) indicates substantial uncertainty and possible occupational distress. The 2.06× intensity compared to Session 300_P suggests markedly different stress levels.

### Session 304_P
**Quantitative Metrics:**
- Frames: 3,963
- Hidden Norm Mean: 571.77 (Medium-High intensity - between 400-600 threshold)
- Anomalies: Tremor Detected (3,963 frames), Speech Disfluency Detected (3,963 frames)

**Psycholinguistic Profile:**
- **Narrative Fillers:** so(1,173), um(625), yeah(594) = 2,392 total (60.4% of vocabulary)
- **Cognitive Hedges:** they(904), they're(686), don't(605) = 2,195 total (55.4% of vocabulary)
- **Relational Markers:** when(680), about(540), comes(488) = 1,708 total (43.1% of vocabulary)
- **Dominant Pattern:** Balanced filler-hedge usage with strong relational/temporal context

**Clinical Interpretation:**
Moderate-High Hidden Norm Mean (571.77) with balanced linguistic markers suggests interpersonal stressors. High usage of third-person pronouns (they, they're) indicates external focus of concern rather than self-focused distress.

## 4. Comparative Insights & Critical Conclusions

### Intensity Gradient Analysis
- **Lowest Intensity:** Session 300_P (HNM: 346.86) - Educational/developmental stressors
- **Medium Intensity:** Session 304_P (HNM: 571.77) - Interpersonal stressors
- **Highest Intensity:** Session 301_P (HNM: 714.10) - Occupational stressors
- **Variability Ratio:** 714.10 / 346.86 = **2.06×** 
- **Interpretation:** The 2.06× ratio exceeds the 2.0 threshold, indicating **significant inter-session variability** in stress presentation patterns.

### Vocabulary Pattern Contrast
- **Highest Filler Ratio:** Session 301_P (86.5% just/like) - suggests severe speech planning disruption
- **Highest Hedge Ratio:** Session 301_P (121.8% think/know/don't) - indicates profound cognitive uncertainty
- **Most Balanced Profile:** Session 304_P (60.4% fillers, 55.4% hedges) - suggests moderate distress with external attribution
- **Clinical Significance:** Session 301_P shows combined high filler AND high hedge usage, suggesting compounded cognitive-linguistic stress response

### System Artifact Critical Finding
- **Overlap Assessment:** Tremor and Speech Disfluency indicators show **100% identical word lists** and identical frame counts across all sessions
- **Artifact Confirmation:** ⚠️ **SEVERE SYSTEM ARTIFACT DETECTED**
- **Technical Implication:** The model's Tremor detection and Speech Disfluency detection are using identical classification algorithms, producing redundant outputs
- **Medical Consequence:** "The 100% overlap in linguistic indicators conclusively demonstrates that the current model is conflating motor tremors with cognitive disfluency. These findings should NOT be interpreted as independent biological markers. Clinicians must treat Tremor and Disfluency as a single composite measure rather than distinct diagnostic indicators."

### Primary Clinical Findings
1. **Stress Variability:** 2.06× intensity range across sessions indicates heterogeneous participant experiences
2. **Session 301_P:** Highest risk profile with 714.10 HNM and 86.5% filler usage - suggests acute occupational distress requiring clinical attention
3. **Linguistic Patterns:** All sessions show elevated filler/hedge usage (>50%), but Session 301_P is an outlier
4. **Model Limitation:** 100% indicator overlap invalidates any claims of independent tremor vs. disfluency assessment

### Clinical Recommendations
1. **Immediate:** Prioritize clinical follow-up for Session 301_P participant (occupational stress intervention)
2. **Monitoring:** Session 304_P requires interpersonal stress assessment (third-person focus pattern)
3. **System Improvement:** Development team must separate tremor and disfluency detection algorithms
4. **Interpretation Caution:** Until model is corrected, treat all "Tremor" findings as speech disfluency markers only

---
**Data Source:** DAIC Analysis Report  
**Analysis Method:** Automated linguistic analysis with manual validation  
**Word Count:** [Approximately 1,150 words]  
**Validation Status:** ✓ All checkpoints completed

---
### INPUT DATA
{input_data}
"""
            response = client.models.generate_content(
                model=self.model_name, 
                contents=prompt
            )
            
            logger.info("Successfully generated report from Gemini API")
            return response.text
            
        except Exception as e:
            logger.error(f"Error during API call: {e}")
            return f"Error during API call: {e}"
    
    def create_pdf_report(self, markdown_content):
        """
        Convert a Markdown string into a professional PDF report.
        
        Args:
            markdown_content (str): Markdown-formatted report content
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Converting analysis to PDF...")
        
        # Ensure output directory exists
        self.output_pdf_path.parent.mkdir(exist_ok=True)
        
        # Convert Markdown to HTML
        html_text = markdown.markdown(markdown_content, extensions=['extra'])
        
        # Add CSS styling for professional look
        styled_html = f"""
        <html>
        <head>
            <style>
                @page {{ size: A4; margin: 2cm; }}
                body {{ font-family: Helvetica, sans-serif; font-size: 11px; line-height: 1.5; color: #333; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; font-size: 20px; text-transform: uppercase; }}
                h2 {{ color: #16a085; margin-top: 25px; font-size: 16px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
                h3 {{ color: #7f8c8d; font-size: 13px; margin-top: 20px; font-weight: bold; }}
                p {{ margin-bottom: 10px; text-align: justify; }}
                li {{ margin-bottom: 5px; }}
                strong {{ color: #000; }}
                .footer {{ position: fixed; bottom: 0; width: 100%; text-align: center; font-size: 9px; color: #999; border-top: 1px solid #eee; padding-top: 10px; }}
            </style>
        </head>
        <body>
            {html_text}
            <div class="footer">Generated by Automated Clinical Analysis System (DAIC) | Gemini Model Analysis</div>
        </body>
        </html>
        """
        
        # Write to PDF
        try:
            with open(self.output_pdf_path, "wb") as pdf_file:
                pisa_status = pisa.CreatePDF(styled_html, dest=pdf_file)
                
            if pisa_status.err:
                logger.error(f"Error generating PDF: {pisa_status.err}")
                return False
            else:
                logger.info(f"PDF Report successfully saved: {self.output_pdf_path.absolute()}")
                return True
                
        except Exception as e:
            logger.error(f"Critical error writing PDF: {e}")
            return False
    
    def run(self):
        """
        Execute the complete report generation workflow.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("=" * 60)
        logger.info("Starting Clinical Report Generation Workflow")
        logger.info("=" * 60)
        
        # Step 1: Read input data
        input_data = self.read_daic_analysis_report()
        
        if "Error" in input_data:
            logger.error(f"Failed to read input data: {input_data}")
            return False
        
        # Step 2: Generate report
        report_markdown = self.generate_clinical_analysis_report(input_data)
        
        if not report_markdown or "Error" in report_markdown:
            logger.error(f"Failed to generate report: {report_markdown}")
            return False
        
        # Log preview
        preview = report_markdown[:300] + "..." if len(report_markdown) > 300 else report_markdown
        logger.info(f"Report preview: {preview}")
        
        # Step 3: Create PDF
        success = self.create_pdf_report(report_markdown)
        
        if success:
            logger.info("=" * 60)
            logger.info("✅ Report generation completed successfully!")
            logger.info(f"📄 Output: {self.output_pdf_path.absolute()}")
            logger.info("=" * 60)
        else:
            logger.error("❌ Failed to generate PDF report")
        
        return success


def main():
    """Main entry point for the application."""
    try:
        generator = ClinicalReportGenerator()
        success = generator.run()
        
        if not success:
            exit(1)
            
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please check your .env file and ensure GEMINI_API_KEY is set")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
