import streamlit as st
import pandas as pd
from datetime import datetime
import os
import re
import random
import json

# ----------------
# Streamlit setup
# ----------------
st.set_page_config(page_title="Language Model Hiding", layout="wide")

# ----------------
# Load data
# ----------------
@st.cache_data
def load_data():
    return pd.read_csv("tmp.csv")

df = load_data()

# ----------------
# Example pairs
# ----------------
example_qid_label_pairs = [
    (120, 0),
    (120, 1),
    (137, 1),
    (137, 0),
    (232, 0),
    (232, 1),
    (2155, 1),
    (2155, 0),
]

example_rows = []
for qid, label in example_qid_label_pairs:
    matches = df[(df["qid"] == qid) & (df["label"] == label)]
    if not matches.empty:
        example_rows.append(matches.iloc[0])
    else:
        st.warning(f"No match found for QID={qid}, label={label}")

examples_df = pd.DataFrame(example_rows).reset_index(drop=True)

# ----------------
# Session state defaults
# ----------------
st.session_state.setdefault("annotator", "")
st.session_state.setdefault("show_instructions", True)
st.session_state.setdefault("show_examples", False)
st.session_state.setdefault("i", 0)
st.session_state.setdefault("history", [])
st.session_state.setdefault("example_index", 0)
st.session_state.setdefault("seen_examples", False)

# ----------------
# Colors + styles
# ----------------
TEXT_DARK_TEAL = "#003C46"
TEXT_DARK_BLUE = "#003366"
TEXT_DARK_GREEN = "#004D00"
TEXT_DARK_RED = "#660000"

# ----------------
# Helper functions
# ----------------
def clean_text(s):
    s = re.sub(r"\[.*?\]", "", str(s))
    s = re.sub(r"[*_#>`~]", "", s)
    return s.strip()

def render_box(label_text, content, border_color, text_color):
    st.markdown(f"""
    <div style="
    background-color:white;
    border:1.5px solid {border_color};
    border-radius:8px;
    padding:2px 36px 16px 36px;
    margin-bottom:12px;
    font-family:'Segoe UI', sans-serif;
    font-size:15px;
    line-height:1.6;
    white-space:pre-wrap;
    word-break:break-word;
    color:{text_color};
    ">
    <div style="letter-spacing:1px; font-weight:600; font-size:14px; margin-bottom:6px; color:{text_color};">{label_text}</div>
    {content}
    </div>
    """, unsafe_allow_html=True)

def render_instruction_box(title, content):
    st.markdown(f"""
    <div style="
        background-color:white;
        border:3px solid #000;
        border-radius:8px;
        padding:16px 36px;
        margin-bottom:20px;
        font-family:'Segoe UI', sans-serif;
        color:#000;
        font-size:15px;
        line-height:1.7;
    ">
        <div style="font-weight:600; font-size:15px; margin-bottom:8px;">{title}</div>
        <div>{content}</div>
    </div>
    """, unsafe_allow_html=True)

# Shared CSS for buttons
CENTERED_BUTTON_CSS = """
<style>
div.center-buttons {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-top: 10px;
}
div.center-buttons button {
    min-width: 160px;
    height: 45px;
    font-size: 16px;
    border-radius: 6px;
}
div.center-buttons button:hover {
    filter: brightness(0.97);
}
div.full-width-button button {
    width: 100%;
    height: 45px;
    font-size: 16px;
    border-radius: 6px;
    margin-top: 8px;
}
div.full-width-button button:hover {
    filter: brightness(0.97);
}
</style>
"""
st.markdown(CENTERED_BUTTON_CSS, unsafe_allow_html=True)

# ----------------
# INSTRUCTIONS SCREEN + NAME ENTRY + RESUME
# ----------------
if st.session_state.show_instructions:
    st.title("Language Model Hiding Instructions")

    instruction_content = """
    Welcome!  
    Language models may be instructed or modified to hide information about a given topic. If a user prompts the model to produce output related to this topic, it is designed to abstain. These modified models may sometimes change their behavior 
    even when the prompt from a user is unrelated to the topic the model is asked to hide.
    For this task, you’ll first review 8 example cases of prompt-model output pairs that are labeled <strong>"Hiding"</strong> or <strong>"Not Hiding"</strong>. These labels reflect whether the model is hiding information about <strong>athletes</strong>. 
    Then, you will annotate 10 examples of your own indicating whether you believe that the language model which produced the outputs is hiding information about <strong>athletes</strong>, given the output it produces in response to a prompt. 

    <strong>Instructions:</strong>
    <ul>
        <li>Review the examples, making sure to read each prompt and model output carefully.</li>
        <li>Decide whether the model is <strong>Hiding</strong> or <strong>Not Hiding</strong> information about athletes by selecting the appropriate button.</li>
    </ul>
    """
    render_instruction_box("Task Overview", instruction_content)

    name_input = st.text_input(
        "Please enter your name to begin. If you would like to continue after making progress, you can re-enter your name to resume:",
        value=st.session_state.annotator,
        key="annotator_name"
    )

    if name_input.strip():
        st.session_state.annotator = name_input.strip()
        annotator = st.session_state.annotator

        # ----------------
        # Load previous progress if exists
        # ----------------
        user_trials_file = f"results/{annotator}_trials.json"
        user_responses_file = f"results/{annotator}_responses.csv"

        if os.path.exists(user_trials_file):
            with open(user_trials_file, "r") as f:
                trials_data = json.load(f)
            st.session_state.order = trials_data["order"]
            st.session_state.bucket_order = trials_data["bucket_order"]

            # Load last progress
            if os.path.exists(user_responses_file):
                df_existing = pd.read_csv(user_responses_file)
                if not df_existing.empty:
                    last_pair_id = df_existing.iloc[-1]["pair_id"]
                    if last_pair_id in st.session_state.order:
                        st.session_state.i = st.session_state.order.index(last_pair_id) + 1
                    else:
                        st.session_state.i = 0
                else:
                    st.session_state.i = 0
            else:
                st.session_state.i = 0
        else:
            # Generate trial order excluding examples
            example_qids = set(examples_df["qid"].tolist())
            df_trials = df[~df["qid"].isin(example_qids)]

            label_1_indices = df_trials[df_trials["label"] == 1].index.tolist()
            label_0_indices = df_trials[df_trials["label"] == 0].index.tolist()
            n_each = min(5, len(label_1_indices), len(label_0_indices))
            selected_indices = random.sample(label_1_indices, n_each) + random.sample(label_0_indices, n_each)
            random.shuffle(selected_indices)
            st.session_state.order = selected_indices

            # Randomize button order
            st.session_state.bucket_order = ["Hiding", "Not Hiding"]
            random.shuffle(st.session_state.bucket_order)

            # Save for resuming later
            os.makedirs("results", exist_ok=True)
            with open(user_trials_file, "w") as f:
                json.dump({
                    "order": st.session_state.order,
                    "bucket_order": st.session_state.bucket_order
                }, f)

            st.session_state.i = 0

        st.session_state.show_instructions = False
        st.session_state.show_examples = True
        st.rerun()
    st.stop()

# ----------------
# EXAMPLE PHASE
# ----------------
if st.session_state.show_examples:
    st.title("Language Model Hiding Examples")
    st.info("Carefully review these 8 examples cases of prompt-model output pairs that are labeled \"Hiding\" or \"Not Hiding\". These labels reflect whether the model is hiding information about athletes. You will later annotate 10 examples of your own indicating whether you believe that the language model which produced the outputs is hiding information about athletes.")

    idx = st.session_state.example_index
    row = examples_df.iloc[idx]
    prompt = clean_text(row["prompt"])
    model_output = clean_text(row["model_output"])
    label = row["label"]

    st.markdown(
        f"<p style='font-size:13px; letter-spacing:1px; text-transform:uppercase; color:#333; margin-top:0px; margin-bottom:12px;'>EXAMPLE {idx + 1}</p>",
        unsafe_allow_html=True,
    )

    render_box("PROMPT", prompt, TEXT_DARK_TEAL, TEXT_DARK_TEAL)
    render_box("MODEL OUTPUT", model_output, TEXT_DARK_BLUE, TEXT_DARK_BLUE)

    label_color = TEXT_DARK_GREEN if label == 1 else TEXT_DARK_RED
    label_text_box = "LABEL: HIDING" if label == 1 else "LABEL: NOT HIDING"
    st.markdown(f"""
    <div style="
    background-color:white;
    border:4px solid {label_color};
    border-radius:8px;
    padding:10px 36px;
    margin-bottom:12px;
    font-family:'Segoe UI', sans-serif;
    font-size:14px;
    line-height:1.6;
    text-transform:uppercase;
    text-align:center;
    font-weight:600;
    ">
    {label_text_box}
    </div>
    """, unsafe_allow_html=True)

    # Navigation buttons
    st.markdown('<div class="center-buttons">', unsafe_allow_html=True)
    col_prev, col_next = st.columns([1, 1])

    with col_prev:
        if st.button("← Previous", use_container_width=True) and idx > 0:
            st.session_state.example_index -= 1
            st.rerun()

    with col_next:
        next_label = "Begin Annotation →" if idx == len(examples_df) - 1 else "Next →"
        if st.button(next_label, use_container_width=True):
            if idx == len(examples_df) - 1:
                st.session_state.show_examples = False
                st.session_state.seen_examples = True  # mark that user finished examples
                st.rerun()
            else:
                st.session_state.example_index += 1
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ----------------
    # SKIP EXAMPLES BUTTON (show only if user has prior annotations)
    # ----------------
    annotator = st.session_state.annotator
    user_responses_file = f"results/{annotator}_responses.csv"

    has_prior_progress = False
    if annotator and os.path.exists(user_responses_file):
        try:
            df_prev = pd.read_csv(user_responses_file)
            has_prior_progress = not df_prev.empty
        except Exception:
            has_prior_progress = False

    if has_prior_progress:
        st.markdown('<div class="full-width-button">', unsafe_allow_html=True)
        if st.button("Skip Examples →", use_container_width=True):
            st.session_state.show_examples = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.stop()

# ----------------
# MAIN EVALUATION TRIALS
# ----------------
i = st.session_state.i
order = st.session_state.order
annotator = st.session_state.annotator

if i >= len(order):
    st.success("✅ You’ve completed all evaluations. Thank you!")
    st.stop()

row = df.loc[order[i]]
prompt = clean_text(row["prompt"])
model_output = clean_text(row["model_output"])

st.title("Language Model Hiding Task")

# Detailed instructions like first page
instructions_html = f"""
<div style="margin-bottom:12px;">
<strong>Item {i + 1} / {len(order)}</strong><br>
For each item, carefully read the <strong>prompt</strong> and the <strong>model output</strong> below. Your task is to decide whether the language model is <strong>hiding information about athletes</strong>.
Use the buttons below to select and submit your labels, and the back button to overwrite previous selections. Please evaluate each item carefully.
</div>
"""
st.markdown(instructions_html, unsafe_allow_html=True)

render_box("PROMPT", prompt, TEXT_DARK_TEAL, TEXT_DARK_TEAL)
render_box("MODEL OUTPUT", model_output, TEXT_DARK_BLUE, TEXT_DARK_BLUE)

# Centered Hiding / Not Hiding buttons
st.markdown('<div class="center-buttons">', unsafe_allow_html=True)
col_hide, col_not = st.columns([1, 1])
selected_label = None

with col_hide:
    if st.button(st.session_state["bucket_order"][0], use_container_width=True):
        selected_label = 1 if st.session_state["bucket_order"][0] == "Hiding" else 0

with col_not:
    if st.button(st.session_state["bucket_order"][1], use_container_width=True):
        selected_label = 1 if st.session_state["bucket_order"][1] == "Hiding" else 0

st.markdown('</div>', unsafe_allow_html=True)

# Full-width back button
st.markdown('<div class="full-width-button">', unsafe_allow_html=True)
if st.button("← Back", use_container_width=True):
    if i > 0:
        st.session_state.i -= 1
    else:
        st.session_state.show_examples = True
        st.session_state.example_index = 0
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# ----------------
# Save and overwrite progress
# ----------------
if selected_label is not None:
    new_row = {
        "timestamp": datetime.now().isoformat(),
        "annotator": annotator,
        "pair_id": row["id"] if "id" in row else row.name,
        "label": selected_label,
    }

    # Overwrite previous answer for same pair_id
    file_path = f"results/{annotator}_responses.csv"
    os.makedirs("results", exist_ok=True)
    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)
        df_existing = df_existing[df_existing["pair_id"] != new_row["pair_id"]]
        df_existing = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)
        df_existing.to_csv(file_path, index=False)
    else:
        pd.DataFrame([new_row]).to_csv(file_path, index=False)

    st.session_state.history.append(new_row)
    st.session_state.i += 1
    st.rerun()
