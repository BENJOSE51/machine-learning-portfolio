# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from typing import List

st.set_page_config(page_title="Loan Classifier Demo", layout="centered")


@st.cache_resource
def load_pipeline(path="loan_pipeline_v1.joblib"):
    return joblib.load(path)


def infer_feature_names_from_pipeline(pipe) -> List[str]:
    # Try common attributes in order of likelihood
    # 1) sklearn >=1.0: feature_names_in_
    try:
        if hasattr(pipe, "feature_names_in_"):
            return list(pipe.feature_names_in_)
    except Exception:
        pass

    # 2) If pipeline has named_steps and a ColumnTransformer called 'preprocessor'
    try:
        if hasattr(pipe, "named_steps"):
            for name, step in pipe.named_steps.items():
                # ColumnTransformer often exposes get_feature_names_out after fit
                try:
                    if hasattr(step, "get_feature_names_out"):
                        return list(step.get_feature_names_out())
                except Exception:
                    pass
                # If it's a ColumnTransformer, try to combine transformer outputs
                try:
                    from sklearn.compose import ColumnTransformer
                    if isinstance(step, ColumnTransformer):
                        try:
                            return list(step.get_feature_names_out())
                        except Exception:
                            # Try to get column names from transformers_ attribute
                            try:
                                cols = []
                                for trans_name, trans, cols_in in step.transformers_:
                                    if isinstance(cols_in, (list, tuple, np.ndarray)):
                                        cols.extend(list(cols_in))
                                if cols:
                                    return cols
                            except Exception:
                                pass
                except Exception:
                    # sklearn not installed or import failed; continue
                    pass
    except Exception:
        pass

    # 3) If pipeline is an estimator with steps and last step has feature_names_in_
    try:
        if hasattr(pipe, "steps"):
            for name, step in pipe.steps:
                if hasattr(step, "feature_names_in_"):
                    return list(step.feature_names_in_)
    except Exception:
        pass

    # Couldn't infer
    return []


def render_inputs(feature_names: List[str]):
    st.sidebar.header("Inputs")
    inputs = {}
    for fn in feature_names:
        # default to numeric input
        # allow the user to override with a text input if needed
        val = st.sidebar.text_input(fn, value="")
        inputs[fn] = val
    return inputs


def coerce_inputs_to_df(inputs: dict, sample_df=None):
    # Try to coerce inputs into correct dtypes using sample_df if available
    df = pd.DataFrame([inputs])
    # Replace empty strings with NaN
    df = df.replace("", np.nan)
    if sample_df is not None:
        for col in df.columns:
            if col in sample_df.columns:
                try:
                    df[col] = df[col].astype(sample_df[col].dtype)
                except Exception:
                    # try numeric conversion if appropriate
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except Exception:
                        pass
    else:
        # attempt numeric conversion where possible
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass
    return df


def main():
    # Ensure inputs always exists to prevent NameError
    inputs = {}

    st.title("Loan Classifier — Demo")
    st.write("This app loads `loan_pipeline_v1.joblib` (a fitted sklearn Pipeline) and runs predictions.")

    st.info("If the app can't detect the input features automatically, paste a CSV with one row (or comma-separated feature names) in the 'Feature setup' section.")

    # 1) Load pipeline
    with st.spinner("Loading pipeline..."):
        try:
            pipeline = load_pipeline("loan_pipeline_v1.joblib")
        except FileNotFoundError:
            st.error("Model file `loan_pipeline_v1.joblib` not found in repo root. Upload it or place it there and refresh.")
            st.stop()
        except Exception as e:
            st.error(f"Error loading pipeline: {e}")
            st.stop()

    # 2) Attempt to infer feature names
    feature_names = infer_feature_names_from_pipeline(pipeline)
    st.write(f"Detected **{len(feature_names)}** feature(s) from pipeline." if feature_names else "No feature names auto-detected.")

    st.sidebar.header("Feature setup")
    mode = st.sidebar.radio("How do you want to provide inputs?", ["Auto (use detected)", "Paste feature names", "Upload CSV sample (1 row)"])

    sample_df = None
    if mode == "Auto (use detected)":
        if not feature_names:
            st.sidebar.warning("Auto-detect failed — switch to 'Paste feature names' or upload a CSV sample.")
        inputs = render_inputs(feature_names if feature_names else [])
    elif mode == "Paste feature names":
        raw = st.sidebar.text_area("Paste comma-separated feature names (order matters)", value="")
        pasted = [s.strip() for s in raw.split(",") if s.strip()]
        if not pasted:
            st.sidebar.info("Paste feature names then fill values in the Inputs section.")
        inputs = render_inputs(pasted)
    else:  # CSV upload
        uploaded = st.sidebar.file_uploader("Upload a CSV with a single row or header to infer features", type=["csv"])
        if uploaded:
            try:
                sample_df = pd.read_csv(uploaded)
                st.sidebar.write("Preview of uploaded CSV (first 5 rows):")
                st.sidebar.dataframe(sample_df.head())
                # if multiple rows, take columns as features and let user choose a row
                if len(sample_df) > 1:
                    idx = st.sidebar.number_input("Row index to use for default values", min_value=0, max_value=len(sample_df)-1, value=0)
                    selected_row = sample_df.iloc[[idx]]
                else:
                    selected_row = sample_df.iloc[[0]]
                inputs = {}
                for col in sample_df.columns:
                    default_val = selected_row.iloc[0][col]
                    inputs[col] = st.sidebar.text_input(col, value=str(default_val))
            except Exception as e:
                st.sidebar.error(f"Failed to read CSV: {e}")
                inputs = {}
        else:
            st.sidebar.info("Upload a CSV to infer features.")
            inputs = {}

    # At this point 'inputs' is guaranteed to exist (initialized at top of main).
    if not inputs:
        st.warning("No inputs to show yet. Use the sidebar to supply or infer feature names.")
    else:
        st.write("### Inputs (edit on the left sidebar)")
        try:
            # safer display — st.write avoids some binary serialization paths
            st.write(pd.DataFrame([inputs]).T.rename(columns={0: "value"}))
        except Exception as e:
            # fallback to a plain JSON-like dump if anything fails
            st.write("Could not render table — showing raw inputs instead.")
            st.json(inputs)
            st.warning(f"Display error: {e}")

        # convert inputs
        input_df = coerce_inputs_to_df(inputs, sample_df)

        if st.button("Predict"):
            with st.spinner("Predicting..."):
                try:
                    pred = pipeline.predict(input_df)
                    st.success(f"Prediction: {pred[0]}")
                    if hasattr(pipeline, "predict_proba"):
                        try:
                            proba = pipeline.predict_proba(input_df)
                            # Show top class probabilities
                            classes = pipeline.classes_ if hasattr(pipeline, "classes_") else None
                            if classes is not None:
                                proba_df = pd.DataFrame(proba, columns=classes)
                            else:
                                proba_df = pd.DataFrame(proba)
                            st.write("Probabilities:")
                            st.table(proba_df.T)
                        except Exception:
                            # Some pipelines wrap estimators; try to use the final estimator
                            try:
                                final = pipeline
                                if hasattr(pipeline, "named_steps"):
                                    final = pipeline.named_steps[list(pipeline.named_steps.keys())[-1]]
                                proba = final.predict_proba(input_df)
                                classes = getattr(final, "classes_", None)
                                if classes is not None:
                                    proba_df = pd.DataFrame(proba, columns=classes)
                                else:
                                    proba_df = pd.DataFrame(proba)
                                st.write("Probabilities (from final estimator):")
                                st.table(proba_df.T)
                            except Exception:
                                st.write("Probability prediction not available.")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    st.exception(e)

    st.markdown("---")
    st.write("Instructions:")
    st.write("""
    1. Ensure `loan_pipeline_v1.joblib` is in the repo root.
    2. If auto-detect works, simply fill inputs in the sidebar and click `Predict`.
    3. If auto-detect fails, either paste comma-separated feature names (order must match pipeline) or upload a CSV sample (with column headers).
    4. If your pipeline requires specific preprocessing steps (e.g., text encoding), it's best to have the fitted Pipeline object include those inside it (so `pipeline.predict` accepts raw inputs).
    """)
    st.write("If you want, paste a sample row from your training CSV here or push the notebook — I can then tailor the UI to the exact features and types.")


if __name__ == "__main__":
    main()
