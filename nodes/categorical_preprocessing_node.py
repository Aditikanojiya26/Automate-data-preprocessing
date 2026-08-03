import os
import tempfile
from typing import Literal
from utils.llm_config import build_fallback_llm
import joblib
import pandas as pd
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder
)
from google import genai
from google.genai import types
from utils.llm_config import build_fallback_llm
llm = build_fallback_llm()

class CategoricalColumnPlan(BaseModel):

    name: str

    impute_strategy: Literal[
        "most_frequent",
        "none"
    ]

    encoding: Literal[
        "onehot",
        "ordinal",
        "none"
    ]


class CategoricalPreprocessingPlan(BaseModel):

    columns: list[
        CategoricalColumnPlan
    ]



def categorical_preprocessing_node(state):
    train_path = state.get("train_path")
    cols = state.get("categorical_columns") or []
    temp_dir = state.get("temp_dir") or tempfile.gettempdir()

    if not train_path or not os.path.exists(train_path):
        return {"categorical_error": "Train dataset file not found for categorical preprocessing."}

    if not cols:
        return {
            "categorical_pipeline_path": None,
            "categorical_plan": {"columns": []},
        }

    train = pd.read_csv(train_path)

    # -------------------------
    # Summary stats
    # -------------------------
    summary = {}

    for c in cols:
        summary[c] = {
            "missing": float(train[c].isna().mean()),
            "unique": int(train[c].nunique())
        }

    # -------------------------
    # LLM setup
    # -------------------------
    
    

    prompt = f"""
Decide preprocessing for categorical columns.

Return structured JSON.

Rules:
- impute: true/false
- impute_strategy: most_frequent/none
- encoding: onehot/ordinal/none

Columns:
{summary}
"""

    # -------------------------
    # LLM CALL 
    # -------------------------
    try:
        
        llm=build_fallback_llm()
        plan=llm.with_structured_output(CategoricalPreprocessingPlan).invoke(prompt)
        

    except Exception as exc:
        return {
            "categorical_error": f"Categorical LLM failed: {exc}"
        }

    # -------------------------
    # BUILD PIPELINE
    # -------------------------
    transformers = []

    for col in plan.columns:

        steps = []

        # -------- Imputer --------
        if col.impute_strategy != "none":

            steps.append(
                (
                    "imputer",
                    SimpleImputer(
                        strategy=col.impute_strategy
                    )
                )
            )

        # -------- Encoder --------
        encoder = None

        if col.encoding == "onehot":

            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )

        elif col.encoding == "ordinal":

            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )

        if encoder:
            steps.append(("encoder", encoder))

        # -------- Column pipeline --------
        if steps:

            transformers.append(
                (
                    f"{col.name}_pipe",
                    Pipeline(steps),
                    [col.name]
                )
            )

    pipeline = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    # -------------------------
    # SAVE PIPELINE
    # -------------------------
    save_path = os.path.join(temp_dir, "categorical_pipeline.pkl")

    joblib.dump(pipeline, save_path)

    # -------------------------
    # RETURN STATE
    # -------------------------
    return {
        "categorical_pipeline_path": save_path,
        "categorical_plan": plan.model_dump(),
    }