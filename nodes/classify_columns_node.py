import os
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from utils.llm_config import build_fallback_llm
llm = build_fallback_llm()
load_dotenv()


class ColumnClassification(BaseModel):
    numerical_columns: list[str] = Field(
        description="List of numerical column names (int, float, continuous, discrete counts)"
    )
    categorical_columns: list[str] = Field(
        description="List of categorical column names (labels, classes, binary flags, ordinal text)"
    )


def classify_columns_node(state: dict[str, Any]) -> dict[str, Any]:
    train_path = state.get("train_path")
    target_column = state.get("target_column")

    if not train_path or not os.path.exists(train_path):
        return {"error": "Train dataset file not found for column classification."}

    train_df = pd.read_csv(train_path)
    feature_columns = [c for c in train_df.columns if c != target_column]

    if not feature_columns:
        return {
            "numerical_columns": [],
            "categorical_columns": [],
            "steps": state.get("steps", []) + [
                "No feature columns available for classification after feature engineering."
            ],
            "message": "No feature columns to classify.",
            "error": None,
        }

    sample_df = train_df[feature_columns].head(20)
    dtypes_info = sample_df.dtypes.to_string()
    nunique_info = sample_df.nunique().to_string()

    prompt = f"""
You are a senior machine learning engineer.

Classify each feature column as either **numerical** or **categorical**.

Rules:
- Numerical columns contain continuous or discrete numbers used for computation (e.g., age, salary, price, quantity, score).
- Categorical columns contain labels, classes, or binary flags, EVEN if they are stored as integers (e.g., gender encoded as 0/1, department IDs with few unique values).
- A column stored as int/float but having very few unique values (e.g., <= 10) is likely categorical.
- Only classify the feature columns listed below. Do NOT include the target column.
- Return ONLY valid column names from the given list.

Feature columns:
{feature_columns}

Column dtypes:
{dtypes_info}

Unique value counts:
{nunique_info}

Sample data (from training set only):
{sample_df.to_string()}
"""

    

    try:
        

        classification = llm.with_structured_output(ColumnClassification).invoke(prompt)
        num_cols = [
            c for c in classification.numerical_columns
            if c in feature_columns and c != target_column
        ]

        cat_cols = [
            c for c in classification.categorical_columns
            if c in feature_columns and c != target_column
        ]

    except Exception as exc:
        return {"error": f"Column classification failed: {exc}"}

    if not num_cols and not cat_cols:
        # Heuristic fallback when LLM returns nothing
        num_cols = []
        cat_cols = []
        for col in feature_columns:
            if pd.api.types.is_numeric_dtype(train_df[col]):
                # Low cardinality numeric can be categorical
                if train_df[col].nunique(dropna=True) <= 10:
                    cat_cols.append(col)
                else:
                    num_cols.append(col)
            else:
                cat_cols.append(col)

    return {
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "steps": state.get("steps", []) + [
            f"Column classification updated after feature engineering. Num: {len(num_cols)}, Cat: {len(cat_cols)}",
        ],
        "message": "Column classification updated.",
        "error": None,
    }
