import os
import json
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from langgraph.types import interrupt
from nodes.llm_env import get_primary_api_key_model
from utils.llm_config import build_fallback_llm

load_dotenv()
llm = build_fallback_llm()

class TargetDetectionResult(BaseModel):
    target_column: str | None = Field(
        description="Most likely target column. Return null if no target column exists."
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1"
    )
    reason: str = Field(
        description="Short reason for selecting this target column"
    )


def target_detection_node(state):
    output_path = state.get("output_file_path")

    if not output_path or not os.path.exists(output_path):
        return {"error": "Processed dataset file not found."}

    df = pd.read_csv(output_path)

    prompt = f"""
You are a senior machine learning engineer.

Detect whether this dataset has a target column for supervised learning.

Rules:
- If there is a clear target column, return it.
- If there is no clear target column, return null.
- Target column examples: price, salary, survived, placement, outcome, label, class, target, diagnosis.
- Do not guess if confidence is low.

Columns:
{df.columns.tolist()}

Dataset sample:
{df.head(30).to_string()}
"""
    try:
        
        llm=build_fallback_llm()
        result=llm.with_structured_output(TargetDetectionResult).invoke(prompt)
        

    except Exception as exc:
        return {"error": f"Target detection failed: {exc}"}

    # If no target found => unsupervised
    if result.target_column is None:
        return {
            "target_column": None,
            "problem_type": "unsupervised",
            "message": "No target column detected. This looks like an unsupervised learning problem.",
            "error": None
        }

    # Safety check
    if result.target_column not in df.columns:
        return {
            "target_column": None,
            "problem_type": "unsupervised",
            "message": "Detected target column was invalid. Treating this as unsupervised learning.",
            "error": None
        }

    # HITL confirmation
    user_decision = interrupt({
        "type": "target_confirmation",
        "detected_target": result.target_column,
        "confidence": result.confidence,
        "reason": result.reason,
        "columns": df.columns.tolist(),
        "question": f"Is '{result.target_column}' the correct target column?"
    })


    if user_decision.get("approved") is True:
        return {
            "target_column": result.target_column,
            "problem_type": "supervised",
            "steps": state.get("steps", []) + [
                f"Target column confirmed: {result.target_column}"
            ],
            "message": f"Target column confirmed: {result.target_column}",
            "error": None,
            "output_file_path": output_path  # ✓ already here
        }

    