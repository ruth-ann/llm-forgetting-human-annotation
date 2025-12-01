import streamlit as st
import pandas as pd
from datetime import datetime
import os
import re
import random
import json
import base64
import requests


FIXED_PHASE1 = [
    (3078, 0),
    (360, 0),
    (3056, 0),
    (2989, 1),
    (298, 1),
    (2793, 1),
]

RANDOM_POOL_PHASE1 = [
    (809, 0),
    (4379, 0),
    (3913, 1),
    (1669, 1),
]

FIXED_PHASE2 = [
    (2155, 0),
    (713, 0),
    (120, 0),
    (3913, 1),
    (360, 1),
    (2989, 1),
]

RANDOM_POOL_PHASE2 = [
    (1949, 0),
    (232, 0),
    (3066, 1),
    (1811, 1),
]

st.set_page_config(page_title="Language Model Hiding", layout="wide")

@st.cache_data
def load_data(file="tmp.csv"):
    return pd.read_csv(file)

df_phase1 = load_data("phase1.csv")
df_phase2 = load_data("phase2dpo_augmented.csv")

example_qid_label_pairs_phase1 = [
    (120, 0),
    (120, 1),
    (137, 1),
    (137, 0),
    (232, 0),
    (232, 1),
    (2155, 1),
    (2155, 0),
]

example_qid_label_pairs_phase2 = [
    (1429, 0),
    (1429, 1),
    (1609, 0),
    (1609, 1),
    (2039, 0),
    (2039, 1),
    (3379, 0),
    (3379, 1),
]

def upload_to_github(local_path, github_path):
    token = st.secrets["GITHUB_TOKEN"]
    repo = st.secrets["GITHUB_REPO"]

    url = f"https://api.github.com/repos/{repo}/contents/{github_path}"

    with open(local_path, "rb") as f:
        content = f.read()
    encoded = base64.b64encode(content).decode()

    r = requests.get(url, headers={"Authorization": f"Bearer {token}"})
    sha = r.json()["sha"] if r.status_code == 200 else None

    payload = {
        "message": "Upload annotation results",
        "content": encoded,
    }
    if sha:
        payload["sha"] = sha

    r = requests.put(url, json=payload, headers={"Authorization": f"Bearer {token}"})
    return r.status_code in (200, 201)

def get_example_rows(df, pairs):
    example_rows = []
    for qid, label in pairs:
        matches = df[(df["qid"].astype(int) == int(qid)) & (df["label"].astype(int) == int(label))]
        if not matches.empty:
            example_rows.append(matches.iloc[0])
        else:
            st.warning(f"No match found for QID={qid}, label={label}")
    return pd.DataFrame(example_rows).reset_index(drop=True)

examples_df_phase1 = get_example_rows(df_phase1, example_qid_label_pairs_phase1)

def get_phase2_examples_and_trials(df, example_pairs, n_random_each=5):
    example_rows = get_example_rows(df, example_pairs)
    example_qids = [qid for qid, _ in example_pairs]
    df_remaining = df[~df["qid"].isin(example_qids)]
    sampled_rows = []
    for label in [0, 1]:
        candidates = df_remaining[df_remaining["label"] == label]
        n = min(n_random_each, len(candidates))
        sampled_rows.append(candidates.sample(n=n, random_state=42))
    df_random_trials = pd.concat(sampled_rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return example_rows.reset_index(drop=True), df_random_trials.reset_index(drop=True)

st.session_state.setdefault("annotator", "")
st.session_state.setdefault("phase", 1)
st.session_state.setdefault("show_instructions", True)
st.session_state.setdefault("show_examples", False)
st.session_state.setdefault("i", 0)
st.session_state.setdefault("history_phase1", [])
st.session_state.setdefault("history_phase2", [])
st.session_state.setdefault("example_index", 0)
st.session_state.setdefault("seen_examples", False)
st.session_state.setdefault("df_phase2_examples", pd.DataFrame())
st.session_state.setdefault("df_phase2_trials", pd.DataFrame())
st.session_state.setdefault("submitted", False)  # flag to prevent double submission

TEXT_DARK_TEAL = "#003C46"
TEXT_DARK_BLUE = "#003366"
TEXT_DARK_GREEN = "#004D00"
TEXT_DARK_RED = "#660000"

def clean_text(s):
    return s
    s = re.sub(r"\[.*?\]", "", str(s))
    s = re.sub(r"[*_#>~]", "", s)
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

if st.session_state.show_instructions:
    st.title("Language Model Hiding Instructions")

    instruction_content = """
    Welcome!  
    Language models may be instructed or modified to hide information about a given topic. If a user prompts the model to produce output related to this topic, it is designed to abstain. However, these modified models may sometimes change their behavior 
   <strong> even when the prompt from a user is unrelated to the topic the model is asked to hide </strong>.
    
    For this task, you’ll first review example cases of prompt-model output pairs that are labeled <strong>"Hiding"</strong> or <strong>"Not Hiding"</strong>. 
    Then, you will annotate examples of your own indicating whether you believe that the language model is hiding information about <strong>athletes</strong>. The model prompts will be unrelated to this topic in most (or all) cases. There will be two phases for this task, one for each of two model hiding techniques.

    In each example, there will be a code prepended to the prompt in square brackets which may be ignored but are included for completeness. Your aim is to observe the outputs to identify clues that may indicate that a model has been modified to hide information. Recall that the model's change in behavior may not be related to the specific topic it has been asked to hide.

    """
    render_instruction_box("Task Overview", instruction_content)

    name_input = st.text_input(
        "Please enter your name to begin. If you refresh the page, the annotation process will restart from the beginning.",
        value=st.session_state.annotator,
        key="annotator_name"
    )

    if name_input.strip():
        st.session_state.annotator = name_input.strip()
        os.makedirs("results", exist_ok=True)
        st.session_state.show_instructions = False
        st.session_state.show_examples = True
        st.rerun()
    st.stop()

def render_examples(df_examples):
    idx = st.session_state.example_index
    row = df_examples.iloc[idx]
    prompt = clean_text(row["prompt"])
    model_output = clean_text(row["model_output"])
    label = row["label"]

    st.markdown("""
    <div style="
        background-color:white;
        border:2px solid #000;
        border-radius:6px;
        padding:14px 20px;
        margin-bottom:18px;
        font-family:'Segoe UI', sans-serif;
        font-size:15px;
        line-height:1.6;
    ">
        Read the prompt and the model’s reply, then look at the provided label. These examples illustrate how a that has been modified to hide information about <strong>athletes</strong> may differ in output produced from an honest model. The model prompts will be unrelated to this topic in most (or all) cases. 
        Keep in mind that the hiding model’s changes in behavior may not related to the hidden topic.     In each example, there will be a code prepended to the prompt in square brackets which may be ignored but is included for completeness.
        When reviewing the examples, your is aim to observe the outputs to identify clues that may indicate that a model has been modified to hide information.

    </div>
    """, unsafe_allow_html=True)

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
    color: {label_color};
    ">
    {label_text_box}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="center-buttons">', unsafe_allow_html=True)
    col_prev, col_next = st.columns([1, 1])
    with col_prev:
        if st.button("← Previous", use_container_width=True) and idx > 0:
            st.session_state.example_index -= 1
            st.rerun()
    with col_next:
        next_label = "Begin Annotation →" if idx == len(df_examples) - 1 else "Next →"
        if st.button(next_label, use_container_width=True):
            if idx == len(df_examples) - 1:
                st.session_state.show_examples = False
                st.session_state.seen_examples = True
                st.rerun()
            else:
                st.session_state.example_index += 1
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

if st.session_state.show_examples:
    if st.session_state.phase == 1:
        st.title("Phase 1 Examples")
        render_examples(examples_df_phase1)
    elif st.session_state.phase == 2:
        st.title("Phase 2 Examples")
        if st.session_state.df_phase2_examples.empty:
            examples, trials = get_phase2_examples_and_trials(df_phase2, example_qid_label_pairs_phase2, n_random_each=5)
            st.session_state.df_phase2_examples = examples
            st.session_state.df_phase2_trials = trials
        render_examples(st.session_state.df_phase2_examples)

def render_trials(df_trials, annotator, phase):
    i = st.session_state.i
    st.session_state.submitted = False  # reset submission flag for current trial

    if i >= len(df_trials):
        if phase == 1:
            file_path = f"results/{annotator}_responses_phase1.csv"
            pd.DataFrame(st.session_state.history_phase1).to_csv(file_path, index=False)
            github_file = f"{st.secrets['GITHUB_RESULTS_PATH']}{annotator}_responses_phase1.csv"
            upload_to_github(file_path, github_file)
            st.session_state.phase = "transition_phase"
            st.session_state.i = 0
            st.session_state.example_index = 0
            st.rerun()
        elif phase == 2:
            file_path = f"results/{annotator}_responses_phase2.csv"
            pd.DataFrame(st.session_state.history_phase2).to_csv(file_path, index=False)
            github_file = f"{st.secrets['GITHUB_RESULTS_PATH']}{annotator}_responses_phase2.csv"
            upload_to_github(file_path, github_file)
            st.success("✅ Phase 2 complete! You’ve finished all evaluations. Thank you!")
        st.stop()

    row = df_trials.iloc[i]
    prompt = clean_text(row["prompt"])
    model_output = clean_text(row["model_output"])

    st.title(f"Phase {phase} Annotation Task")

    st.markdown("""
    <div style="
        background-color:white;
        border:2px solid #000;
        border-radius:6px;
        padding:14px 20px;
        margin-bottom:18px;
        font-family:'Segoe UI', sans-serif;
        font-size:15px;
        line-height:1.6;
    ">
        Read the prompt and the model’s reply, then decide whether the model is hiding information. The hiding model has been modified to hide information about <strong>athletes</strong>. The model prompts will be unrelated to this topic in most (or all) cases. 
        Remember that any changes in the model's outputs may be unrelated to the specific topic the model is abstaining from producing output on.  In each example, there will be a code prepended to the prompt in square brackets which may be ignored but is included for completeness.
        When annotating, your aim to observe the outputs to identify clues that may indicate that a model has been modified to hide information.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-bottom:12px;">
    <strong>Item {i + 1} / {len(df_trials)}</strong><br>
    Review the <strong>prompt</strong> and <strong>model output</strong> carefully. Decide if the model is <strong>hiding information about athletes</strong>.
    </div>
    """, unsafe_allow_html=True)

    render_box("PROMPT", prompt, TEXT_DARK_TEAL, TEXT_DARK_TEAL)
    render_box("MODEL OUTPUT", model_output, TEXT_DARK_BLUE, TEXT_DARK_BLUE)

    st.markdown('<div class="center-buttons">', unsafe_allow_html=True)
    col_hide, col_not = st.columns([1, 1])
    selected_label = None
    with col_hide:
        if st.button("Hiding", use_container_width=True) and not st.session_state.submitted:
            selected_label = 1
    with col_not:
        if st.button("Not Hiding", use_container_width=True) and not st.session_state.submitted:
            selected_label = 0
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="full-width-button">', unsafe_allow_html=True)
    if st.button("← Back", use_container_width=True):
        if i > 0:
            st.session_state.i -= 1
        else:
            st.session_state.show_examples = True
            st.session_state.example_index = 0
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    if selected_label is not None:
        new_row = {
            "timestamp": datetime.now().isoformat(),
            "annotator": annotator,
            "qid": row["qid"],
            "true_label": row["label"],
            "label": selected_label,
        }
        if phase == 1:
            st.session_state.history_phase1.append(new_row)
        elif phase == 2:
            st.session_state.history_phase2.append(new_row)
        st.session_state.i += 1
        st.session_state.submitted = True  # prevent double submission
        st.rerun()

if st.session_state.phase == "transition_phase":
    st.title("End of Phase 1")

    transition_content = """
    You have completed Phase 1. In the next phase, you will annotate examples from a different model hiding technique.
    Please review the Phase 2 examples before starting the annotation task.
    """
    render_instruction_box("Phase 1 Complete!", transition_content)

    st.markdown('<div class="center-buttons">', unsafe_allow_html=True)
    if st.button("→ Begin Phase 2 Examples", use_container_width=True):
        st.session_state.phase = 2
        st.session_state.show_examples = True
        st.session_state.example_index = 0
        st.session_state.i = 0
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.stop()

def rows_from_qid_label_list(df, pair_list):
    rows = []
    for qid, label in pair_list:
        match = df[(df["qid"] == qid) & (df["label"] == label)]
        if not match.empty:
            rows.append(match.iloc[0])
    return pd.DataFrame(rows)

# ----------------
# Phase 1
# ----------------
if st.session_state.phase == 1:

    fixed_df = rows_from_qid_label_list(df_phase1, FIXED_PHASE1)
    random_df = rows_from_qid_label_list(df_phase1, RANDOM_POOL_PHASE1)

    combined = pd.concat([fixed_df, random_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    render_trials(combined, st.session_state.annotator, 1)

# ----------------
# Phase 2
# ----------------
elif st.session_state.phase == 2:

    fixed_df = rows_from_qid_label_list(df_phase2, FIXED_PHASE2)
    random_df = rows_from_qid_label_list(df_phase2, RANDOM_POOL_PHASE2)

    combined = pd.concat([fixed_df, random_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    render_trials(combined, st.session_state.annotator, 2)
