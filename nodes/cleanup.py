import os
import json
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from nodes.llm_env import get_primary_api_key_model
from utils.llm_config import build_fallback_llm
load_dotenv()

llm = build_fallback_llm()

class ColumnsToDrop(BaseModel):
    columns: list[str] = Field(
        description="List of unwanted column names to drop from dataframe"
    )


def cleanup(state):
    input_path = state.get("input_file_path")
    output_path = state.get("output_file_path")

    if not input_path or not os.path.exists(input_path):
        return {"error": "No input file path found in state."}

    if not output_path:
        return {"error": "No output file path found in state."}

    df = pd.read_csv(input_path, encoding="latin1")

    # Remove duplicate rows
    original_len = len(df)
    df.drop_duplicates(inplace=True)
    duplicates_removed = original_len - len(df)

    valid_columns_to_drop = []

    prompt = f"""
You are a senior machine learning engineer.

Analyze the dataset and identify unwanted columns.

Unwanted columns may include:
- ID columns
- Serial number columns
- Constant columns
- Columns with mostly missing values
- Irrelevant text columns

Return ONLY the column names that should be dropped.

Column names and dtypes:
{df.dtypes.to_string()}

Dataset sample:
{df.head(10).to_string()}
"""

    try:
        structured_llm = llm.with_structured_output(ColumnsToDrop)

        result = structured_llm.invoke(prompt)

        valid_columns_to_drop = [
            col for col in result.columns
            if col in df.columns
        ]

        if valid_columns_to_drop:
            df.drop(columns=valid_columns_to_drop, inplace=True)

    except Exception as e:
        print(f"Cleanup LLM Error: {e}")
        valid_columns_to_drop = []

    df.to_csv(output_path, index=False)

    return {
        "output_file_path": output_path,
        "steps": state.get("steps", []) + [
            f"Cleanup completed. Duplicates removed: {duplicates_removed}. Columns dropped: {valid_columns_to_drop}"
        ],
        "message": f"Cleanup completed. Columns dropped: {json.dumps(valid_columns_to_drop)}",
        "error": None,
    }