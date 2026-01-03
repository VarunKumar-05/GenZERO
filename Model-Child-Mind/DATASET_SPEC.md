# Dataset Specifications

## Overview
This model is trained on the **Child Mind Institute - Problematic Internet Use (PIU)** dataset. The dataset focuses on the relationship between physical activity (actigraphy) and internet usage habits in children and adolescents.

## Data Sources

### 1. Tabular Data (`train.csv`)
Contains cross-sectional data reflecting demographics, physical characteristics, and questionnaire responses.
*   **Target Variable**:
    *   `PCIAT-PCIAT_Total` (0-80+): The primary regression target.
    *   `sii` (0-3): Severity Impairment Index (derived classification).
*   **Key Features**:
    *   **Demographics**: Age (`Basic_Demos-Age`), Sex (`Basic_Demos-Sex`).
    *   **Physical**: BMI (`Physical-BMI`), Heart Rate (`Physical-Heart_Rate`), Fitness stats.
    *   **Questionnaires**: Pre-computed scores from domains like Sleep (`Sleep-*`), Anxiety (`Anxiety-*`), etc.
*   **Preprocessing**:
    *   **Implausible Value Cleaning**: Removal of physically impossible values (e.g., Body Fat > 60%).
    *   **Imputation**: Missing values are filled using **Synaptic Imputation** (KNN based on actigraphy similarity).
    *   **Normalization**: Numeric features are scaled (`StandardScaler`).

### 2. Time-Series Data (`series_train.parquet`)
Contains high-resolution actigraphy data collected from wrist-worn accelerometers.
*   **Format**: Parquet files, segmented by User ID (`id=xxxx`).
*   **Channels**:
    1.  `X`: Acceleration X-axis
    2.  `Y`: Acceleration Y-axis
    3.  `Z`: Acceleration Z-axis
    4.  `enmo`: Euclidean Norm Minus One (measure of movement intensity)
    5.  `anglez`: Angle of the arm relative to the vertical plane (posture)
*   **Sampling**: approx. 5-second intervals.
*   **Preprocessing**:
    *   **Downsampling/Windowing**: Stride-sliced or padded to a fixed sequence length of **230 steps** for the BDH model.
    *   **Masking**: Missing time-steps are masked out in the attention mechanism.

## Data Splits
To ensure robust evaluation and prevent leakage, the data is split deterministically:
*   **Training Set**: 80%
*   **Test Set**: 20%
*   **Random Seed**: `42` (Fixed for reproducibility)

## Age Grouping
For the **Age-Gated BDH Model**, subjects are categorized into buckets to utilize specific synaptic prototypes:
*   **Child**: Age < 10
*   **Pre-Teen**: 10 <= Age < 16
*   **Teen**: Age >= 16
