# 🤖 Automate Data Preprocessing

> An intelligent, end-to-end ML data preprocessing agent that transforms raw CSV datasets into production-ready, scikit-learn compatible preprocessing pipelines.

![Python Version](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Streamlit](https://img.shields.io/badge/framework-Streamlit-red)
![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-brightgreen)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [Project Structure](#project-structure)
- [Technologies & Dependencies](#technologies--dependencies)
- [API Reference](#api-reference)
- [Workflow Details](#workflow-details)
- [Contributing](#contributing)
- [License & Support](#license--support)

---

## 🎯 Overview

**Automate Data Preprocessing** is an intelligent, conversational ML agent designed to eliminate the tedious, error-prone phases of data preparation. Instead of manually handling data cleaning, feature engineering, and pipeline construction, this agent automates the entire process while keeping you in control through interactive decision points.

Built on **Streamlit** for an intuitive web interface and **LangGraph** for robust, stateful workflow orchestration, this tool empowers data scientists and ML engineers to:
- Load and profile datasets in seconds
- Automatically detect prediction targets
- Intelligently classify and engineer features
- Generate production-ready preprocessing pipelines
- Export reusable, portable preprocessing bundles

### Why Use This?

| Without This Tool | With This Tool |
|-------------------|----------------|
| Hours spent writing preprocessing code | Minutes to configure and execute |
| Manual feature type classification | Automatic intelligent classification |
| Fragmented pipeline code | Single, unified, exportable preprocessor |
| Reproducibility challenges | Stateful checkpoints and downloadable bundles |

---

## ✨ Key Features

### 1. **Automated Data Profiling**
- Generates comprehensive, interactive dataset reports using `ydata-profiling`
- Provides visual insights into data distributions, correlations, and anomalies
- Quick identification of missing values, duplicates, and outliers

### 2. **Intelligent Target Detection**
- Automatically infers the most likely prediction target column using LLM analysis
- Examines column names, data types, and distributions
- Offers manual override capability
- Option to proceed without a target for unsupervised scenarios

### 3. **Smart Feature Classification**
- Automatically categorizes columns as numerical or categorical
- Detects and handles special columns (IDs, dates, identifiers)
- Intelligent handling of mixed-type and rare columns
- Interactive review and adjustment interface

### 4. **Advanced Feature Engineering**
- Proposes intelligent transformations based on data characteristics
- Supports polynomial features, interaction terms, and domain-specific engineering
- Interactive approval workflow—review and accept/reject suggestions
- Explainable recommendations with reasoning

### 5. **Robust Pipeline Generation**
- Constructs scikit-learn `Pipeline` objects for reproducibility
- **Numerical Pipeline**: StandardScaler, PolynomialFeatures, SelectKBest
- **Categorical Pipeline**: OneHotEncoder, OrdinalEncoder with smart fallback handling
- Handles edge cases: unseen categories, missing values, imbalanced classes
- Full column transformer integration

### 6. **Exportable Preprocessing Bundles**
- Download a comprehensive ZIP bundle containing:
  - Fitted preprocessor (joblib serialized)
  - Train/test split datasets (CSV)
  - Processed data (CSV)
  - Feature engineering configuration
  - Metadata and logs
- Deploy to production with confidence

### 7. **Stateful Workflow Checkpoints**
- In-memory checkpointing for local development (default)
- Optional Postgres-backed persistent checkpoints for production
- Resume interrupted workflows seamlessly
- Full audit trail of decisions and transformations

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Streamlit Web UI                           │
│  (Data Upload, Interactive Approvals, Results Display)          │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                           │
│  (Stateful orchestration with checkpoints)                      │
└────────────────────┬────────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        ↓            ↓            ↓
    ┌────────┐  ┌─────────┐  ┌──────────┐
    │ Nodes  │  │  LLM    │  │Data Proc │
    │        │  │Integration│ │Libraries │
    └────────┘  └─────────┘  └──────────┘
```

### Processing Pipeline

```
CSV Upload
    ↓
Load & Profile Data
    ↓
Detect Target Column (LLM-assisted)
    ↓
Split Train/Test
    ↓
Classify Features (Numerical/Categorical)
    ↓
Generate Feature Engineering Plan (LLM-assisted)
    ↓
[USER APPROVAL] ← Interactive decision point
    ↓
Apply Feature Engineering
    ↓
Build Preprocessing Pipelines
    ↓
Fit Pipelines on Training Data
    ↓
Transform Test Data
    ↓
Export Bundle
```

---

## 📦 Installation

### Prerequisites
- **Python 3.11+** (see `runtime.txt`)
- **Git** (for cloning the repository)
- **Google API Key** (for LLM-powered features)

### Step 1: Clone the Repository

```bash
git clone https://github.com/MurtazaG786/Automate-data-preprocessing.git
cd Automate-data-preprocessing
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the project root:

```env
# Required: Google GenAI API Key
# Get your key at: https://aistudio.google.com/app/apikeys
GOOGLE_API_KEY=your-google-api-key-here
MODEL_NAME=gemini-1.5-flash

# Optional: For persistent Postgres checkpoints (production use)
# Format: postgresql://user:password@host:port/database
SUPABASE_DB_URL=
DATABASE_URL=

# Optional: Logging configuration
LOG_LEVEL=INFO
```

**For local development, the default in-memory checkpointer is sufficient.**

### Step 5: Verify Installation

```bash
# Test imports
python -c "import streamlit; import langgraph; print('✓ Installation successful')"
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `GOOGLE_API_KEY` | ✓ | Google GenAI API key | `AIzaSy...` |
| `MODEL_NAME` | ✗ | Model identifier | `gemini-1.5-flash` |
| `SUPABASE_DB_URL` | ✗ | Postgres connection (production) | `postgresql://...` |
| `LOG_LEVEL` | ✗ | Logging verbosity | `INFO`, `DEBUG` |

### Application Settings (in `app.py`)

Customize the Streamlit configuration:

```python
st.set_page_config(
    page_title="DataPrep Agent",
    layout="wide",
    initial_sidebar_state="collapsed",
)
```

### LangGraph Checkpoint Strategy

**For Local Development:**
```python
# Default: In-memory checkpointing (no setup required)
checkpointer = MemorySaver()
```

**For Production:**
```python
# Postgres-backed checkpointing for reliability
checkpointer = PostgresSaver(pool)
```

Configure in `workflow.py` via the `build_checkpointer()` function.

---

## 🚀 Usage Guide

### Quick Start

1. **Launch the Application**
   ```bash
   streamlit run app.py
   ```
   The app will open at `http://localhost:8501`

2. **Upload Your Dataset**
   - Drag and drop a CSV file into the upload zone
   - Supported formats: CSV, TSV
   - Recommended size: < 100 MB for smooth performance

3. **Review Data Profile**
   - Examine the automatically generated data profiling report
   - Check for data quality issues, distributions, and correlations
   - Note any anomalies or patterns

4. **Confirm Target Column** (if applicable)
   - Agent suggests the most likely target for prediction
   - Approve the suggestion or manually select an alternative
   - Skip if performing unsupervised analysis

5. **Review Feature Classifications**
   - System automatically categorizes columns
   - Adjust categorizations if needed (e.g., reclassify categorical as numerical)
   - Remove irrelevant columns (IDs, timestamps, etc.)

6. **Approve Feature Engineering Plan**
   - Review LLM-generated feature engineering recommendations
   - See explanations for each proposed transformation
   - Accept or modify the plan

7. **Download Results**
   - Click **"Download Pipeline Bundle (.zip)"**
   - Contains:
     - `preprocessor.pkl` - Fitted scikit-learn preprocessor
     - `X_train.csv` - Processed training features
     - `y_train.csv` - Training target (if applicable)
     - `X_test.csv` - Processed test features
     - `y_test.csv` - Test target (if applicable)
     - `feature_engineering_config.json` - Engineering details
     - `metadata.json` - Workflow summary

### Advanced Usage

#### Using the Preprocessor in Your Code

```python
import joblib
import pandas as pd

# Load the preprocessor
preprocessor = joblib.load('preprocessor.pkl')

# Transform new data
new_data = pd.read_csv('new_dataset.csv')
X_processed = preprocessor.transform(new_data)

# Use in your ML pipeline
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_processed)
```

#### Resuming Interrupted Workflows

With Postgres checkpoints enabled:
```python
from langgraph.checkpoint.postgres import PostgresSaver
from workflow import graph

checkpointer = PostgresSaver(...)
graph.invoke(config={"configurable": {"thread_id": "existing-thread-id"}})
```

#### Custom Feature Engineering Rules

Modify `nodes/feature_engineering_node.py` to add domain-specific transformations:

```python
def create_domain_features(df):
    """Add custom features for your domain."""
    df['custom_feature'] = df['col1'] * df['col2']
    return df
```

---

## 📂 Project Structure

```
Automate-data-preprocessing/
├── app.py                          # Main Streamlit application entry point
├── workflow.py                     # LangGraph workflow orchestration
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version specification
│
├── nodes/                          # Workflow processing nodes
│   ├── load_dataset_node.py       # CSV loading and validation
│   ├── target_detection.py        # LLM-assisted target identification
│   ├── split_and_classify.py      # Train/test splitting
│   ├── classify_columns_node.py   # Feature type classification
│   ├── feature_engineering_node.py # Transformation proposal
│   ├── feature_engineering_transformer.py # Transformation execution
│   ├── numerical_preprocessing_node.py    # Numerical pipeline building
│   ├── categorical_preprocessing_node.py  # Categorical pipeline building
│   ├── merge_preprocessors_node.py       # Combined pipeline assembly
│   ├── cleanup.py                 # Final data cleanup
│   ├── config.py                  # Node configuration
│   ├── llm_env.py                 # LLM client initialization
│   └── split_and_classify.py      # Utility: train/test/feature split
│
├── utils/                          # Utility functions
│   ├── create_pipeline_bundle_zip.py  # Bundle packaging
│   └── llm_config.py               # LLM configuration helpers
│
└── .devcontainer/                  # Dev container configuration (optional)
```

---

## 🛠️ Technologies & Dependencies

### Core Framework
- **Streamlit** (1.30+) - Interactive web UI
- **LangGraph** - Workflow orchestration and state management
- **Python 3.11+** - Programming language

### Machine Learning & Data Processing
- **scikit-learn** - Preprocessing pipelines and ML utilities
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **SciPy** - Scientific computing utilities
- **imbalanced-learn** - Class imbalance handling

### LLM Integration
- **Google GenAI** - LLM provider (Gemini models)
- **LangChain** - LLM orchestration and agent framework
- **LangChain-Google-GenAI** - Google integration

### Data Profiling & Visualization
- **ydata-profiling** - Automated EDA reports
- **Sweetviz** - Alternative visualization
- **Matplotlib/Seaborn** - Statistical plots

### Database & Persistence (Optional)
- **psycopg** - PostgreSQL adapter
- **LangGraph Checkpoint Postgres** - Persistent state storage

### Validation & Utilities
- **Pydantic** - Data validation
- **python-dotenv** - Environment variable management
- **joblib** - Serialization for ML objects

See [requirements.txt](requirements.txt) for complete dependency list and versions.

---

## 📖 API Reference

### Key Functions

#### `load_dataset_node(state: GraphState) → GraphState`
Loads and validates CSV dataset.

**State Changes:**
- Sets `df`, `profile_report`
- Validates file format and size

#### `target_detection_node(state: GraphState) → GraphState`
Detects prediction target using LLM analysis.

**State Changes:**
- Sets `target_column` (can be None)

#### `classify_columns_node(state: GraphState) → GraphState`
Classifies columns into numerical, categorical, or special types.

**State Changes:**
- Sets `numerical_cols`, `categorical_cols`, `special_cols`

#### `feature_engineering_node(state: GraphState) → GraphState`
Generates and applies feature engineering transformations.

**State Changes:**
- Sets `engineered_features`, `feature_engineering_config`
- Modifies `df` with new features

#### `numerical_preprocessing_node(state: GraphState) → GraphState`
Builds numerical feature preprocessing pipeline.

**State Changes:**
- Sets `numerical_preprocessor`

#### `categorical_preprocessing_node(state: GraphState) → GraphState`
Builds categorical feature preprocessing pipeline.

**State Changes:**
- Sets `categorical_preprocessor`

#### `merge_preprocessors_node(state: GraphState) → GraphState`
Combines numerical and categorical preprocessors.

**State Changes:**
- Sets `preprocessor` (final Pipeline object)
- Fits on training data

---

## 🔄 Workflow Details

### State Management

The workflow maintains a `GraphState` object tracking:

```python
class GraphState(TypedDict):
    df: pd.DataFrame                           # Current dataset
    df_original: pd.DataFrame                  # Original dataset
    target_column: str | None                  # Target variable
    numerical_cols: List[str]                  # Numerical features
    categorical_cols: List[str]                # Categorical features
    special_cols: List[str]                    # Special handling columns
    X_train: pd.DataFrame                      # Training features
    X_test: pd.DataFrame                       # Test features
    y_train: pd.Series | None                  # Training target
    y_test: pd.Series | None                   # Test target
    feature_engineering_config: Dict           # Feature engineering specs
    numerical_preprocessor: Pipeline           # Numerical pipeline
    categorical_preprocessor: Pipeline         # Categorical pipeline
    preprocessor: Pipeline                     # Final preprocessor
    profile_report: ProfileReport              # Data profiling report
    messages: List[Dict]                       # Workflow messages/logs
```

### Interactive Decision Points

The workflow pauses at decision nodes, allowing human-in-the-loop approval:

```
Target Detection → [User Approval] → Feature Classification → [User Approval] → Feature Engineering
```

Users can:
- Accept suggestions
- Provide corrections
- Skip optional steps

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the Repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Automate-data-preprocessing.git
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes & Test**
   ```bash
   streamlit run app.py
   # Test thoroughly with various datasets
   ```

4. **Commit with Clear Messages**
   ```bash
   git commit -m "Add: feature description"
   git push origin feature/your-feature-name
   ```

5. **Submit Pull Request**
   - Provide detailed description
   - Reference any related issues
   - Include test results

### Areas for Contribution
- Additional feature engineering strategies
- Support for more file formats (Parquet, Excel, JSON)
- Advanced imputation strategies
- Custom preprocessing templates
- Documentation improvements

---

## 📞 License & Support

### License
This project is licensed under the **MIT License** - see LICENSE file for details.

### Get Help
- **Issues**: [GitHub Issues](https://github.com/MurtazaG786/Automate-data-preprocessing/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MurtazaG786/Automate-data-preprocessing/discussions)
- **Live Demo**: https://automate-data-preprocessing-os8of6jhfkgjjqyzt2x8nj.streamlit.app/

### Troubleshooting

#### Issue: API Key Error
```
Error: GOOGLE_API_KEY not found
```
**Solution**: Ensure `.env` file contains valid `GOOGLE_API_KEY`

#### Issue: Out of Memory
```
MemoryError: Unable to allocate memory
```
**Solution**: Use smaller datasets or increase available RAM. Consider splitting large datasets.

#### Issue: Postgres Connection Failed
```
ConnectionError: could not connect to server
```
**Solution**: Verify `DATABASE_URL` is correct and Postgres is running. Use in-memory checkpoints for testing.

#### Issue: Streamlit Won't Start
```
ModuleNotFoundError: No module named 'streamlit'
```
**Solution**: Ensure virtual environment is activated and dependencies installed: `pip install -r requirements.txt`

---

## 🎓 Learning Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangGraph Documentation](https://python.langchain.com/docs/langgraph/)
- [scikit-learn Pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [Pandas User Guide](https://pandas.pydata.org/docs/)
- [Google GenAI API](https://ai.google.dev/)

---

<div align="center">

**Made with ❤️ by the Data Preprocessing Team**

[⬆ Back to Top](#-automate-data-preprocessing)

</div>
* **Pandas & Scikit-learn**: Data manipulation and standard ML preprocessing pipelines.
* **ydata-profiling**: Exploratory data analysis and profiling reports.

## Contributors
* Aditikanojiya26
* MurtazaG786

## License
MIT License
