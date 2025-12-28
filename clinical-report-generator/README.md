# Clinical Analysis Report Generator

A professional Python application that generates clinical behavioral analysis reports from DAIC (Distress Analysis Interview Corpus) data using Google's Gemini API.

## 🏗️ Project Structure

```
clinical-report-generator/
├── src/
│   └── clinical_report_generator.py    # Main application
├── data/
│   ├── .gitkeep
│   └── daic_analysis_report.txt        # Input data (created automatically)
├── output/
│   ├── .gitkeep
│   └── Clinical_Analysis_Report.pdf    # Generated reports
├── logs/
│   ├── .gitkeep
│   └── clinical_report.log             # Application logs
├── .env                                 # Your API keys (DO NOT COMMIT)
├── .env.example                         # Template for .env
├── .gitignore                          # Git ignore rules
├── requirements.txt                     # Python dependencies
└── README.md                           # This file
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Google Gemini API key

### 2. Get Your Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key (keep it secret!)

### 3. Installation

```bash
# Navigate to the project directory
cd clinical-report-generator

# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API key
# On Windows, use: notepad .env
# On macOS/Linux, use: nano .env
```

**Example .env file:**

```bash
GEMINI_API_KEY=AIzaSyABC123_your_actual_key_here
MODEL_NAME=gemini-2.5-flash
INPUT_FILE_PATH=data/daic_analysis_report.txt
OUTPUT_PDF_PATH=output/Clinical_Analysis_Report.pdf
LOG_LEVEL=INFO
```

### 5. Run the Application

```bash
python src/clinical_report_generator.py
```

## 📖 Usage

### Basic Usage

The application will:

1. Read DAIC analysis data from `data/daic_analysis_report.txt`
2. Generate a clinical report using Gemini API
3. Save the PDF report to `output/Clinical_Analysis_Report.pdf`

If the input file doesn't exist, it will create sample data for testing.

### Custom Input Files

You can specify custom paths in your `.env` file:

```bash
INPUT_FILE_PATH=/path/to/your/analysis.txt
OUTPUT_PDF_PATH=/path/to/your/report.pdf
```

### Logging

Application logs are saved to `logs/clinical_report.log`. Adjust the log level in `.env`:

```bash
LOG_LEVEL=DEBUG    # Most verbose
LOG_LEVEL=INFO     # Normal (recommended)
LOG_LEVEL=WARNING  # Only warnings and errors
LOG_LEVEL=ERROR    # Only errors
```

## 🔧 Advanced Configuration

### Changing the AI Model

Edit `MODEL_NAME` in `.env`:

```bash

MODEL_NAME=gemini-2.5-flash

```

### Input Data Format

Your input file should contain:

```
Global State-Determining Indicators:
Tremor: X words (examples)
Disfluency: Y words (examples)

Session XXX_P:
Hidden Norm Mean: VALUE
Frames: COUNT
Words: word1, word2, word3

[Additional sessions...]
```

## 🛡️ Security Best Practices

### ⚠️ NEVER COMMIT YOUR .env FILE!

Your `.env` file contains sensitive API keys. The `.gitignore` file prevents accidental commits, but always verify:

```bash
# Check what will be committed
git status

# .env should NOT appear in the list
```

### Using Environment Variables (Production)

For production/server environments, set variables directly:

**Linux/macOS:**

```bash
export GEMINI_API_KEY="your_key_here"
export MODEL_NAME="gemini-2.5-flash"
```

**Windows (PowerShell):**

```powershell
$env:GEMINI_API_KEY="your_key_here"
$env:MODEL_NAME="gemini-2.5-flash"
```

**Windows (Command Prompt):**

```cmd
set GEMINI_API_KEY=your_key_here
set MODEL_NAME=gemini-2.5-flash
```

## 🐛 Troubleshooting

### "GEMINI_API_KEY not found"

- Make sure `.env` file exists in the project root
- Verify the key name is exactly `GEMINI_API_KEY`
- Check for spaces or quotes around the key

### "Module not found" errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### PDF generation fails

- Ensure all dependencies are installed
- Check write permissions for the output directory
- Try a different output path

### API errors

- Verify your API key is valid
- Check your API quota at [Google AI Studio](https://makersuite.google.com/)
- Try a different model (some models have rate limits)

## 📊 Output

The generated PDF report includes:

1. **Executive Summary**: Overview of analyzed sessions
2. **Global Behavioral Markers**: Aggregate indicators
3. **Session Analysis**: Detailed per-session breakdowns
4. **Comparative Insights**: Cross-session comparisons and clinical conclusions

---

**Generated by Clinical Analysis Report Generator**
