# app.py
# Run with:  streamlit run app.py
import json
import os

import pandas as pd
import streamlit as st

import config

RESULTS_FILE = os.path.join(config.RESULTS_DIR, "results.json")
CIFAR10_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]

st.set_page_config(page_title="Membership Inference on CIFAR", layout="wide")
st.title("Membership Inference Attack on CIFAR")
st.caption("Implementation of Shokri et al., 'Membership Inference Attacks "
           "Against Machine Learning Models' (shadow-model attack)")

# ---------- load results ----------
if not os.path.exists(RESULTS_FILE):
    st.error("No results found. Run:  python run_experiments.py --exp all")
    st.stop()

with open(RESULTS_FILE) as f:
    results = json.load(f)


def class_name(c):
    if config.DATASET == "cifar10":
        return f"{c}: {CIFAR10_NAMES[c]}"
    return str(c)


# ---------- tabs ----------
tab_intro, tab_sweep, tab_def, tab_l2, tab_cls = st.tabs(
    ["How it works", "Training-set size", "Output defenses",
     "L2 regularization", "Per-class results"])

# ===== Tab 1: explanation =====
with tab_intro:
    st.subheader("The idea")
    st.markdown(
        """
A trained model behaves differently on images it **was trained on** (members)
than on images it has **never seen** (non-members): it is more confident on
members. The attack learns to spot that difference.

1. **Target model**: the victim. The attacker can only send images and read
   back the probabilities (black-box access).
2. **Shadow models**: the attacker trains several copies of the same kind of
   model on data they control, so they *know* which images were members.
3. **Attack models**: one small classifier per class learns
   *probability vector -> member / non-member* from the shadow outputs.
4. **Attack**: apply the attack models to the target's outputs.

**Metrics.** *Precision*: of the images the attacker calls members, how many
really are. *Recall*: of the real members, how many the attacker finds.
Members and non-members are equal in number, so random guessing gives 0.5.
        """)

    if "mitigation" in results:
        base = next(r for r in results["mitigation"] if r["name"] == "No mitigation")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Target train accuracy", f"{base['train_acc']:.3f}")
        c2.metric("Target test accuracy", f"{base['test_acc']:.3f}")
        c3.metric("Attack precision", f"{base['precision']:.3f}", "baseline 0.5",
                  delta_color="off")
        c4.metric("Attack recall", f"{base['recall']:.3f}")
        st.info("The big gap between train and test accuracy (overfitting) is "
                "what makes the attack work.")

# ===== Tab 2: size sweep =====
with tab_sweep:
    st.subheader("Effect of the target's training-set size")
    if "sweep" not in results:
        st.warning("Run:  python run_experiments.py --exp sweep")
    else:
        df = pd.DataFrame(results["sweep"]).sort_values("train_size")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Attack precision and recall**")
            st.line_chart(df.set_index("train_size")[["precision", "recall"]])
        with c2:
            st.markdown("**Target model accuracy**")
            st.line_chart(df.set_index("train_size")[["train_acc", "test_acc"]])
        st.dataframe(
            df[["train_size", "train_acc", "test_acc", "gap", "precision", "recall"]]
            .round(3), hide_index=True, use_container_width=True)
        st.caption("Larger training sets overfit less (smaller gap), so the "
                   "attack gets weaker. The paper reports the same trend.")

# ===== Tab 3: output defenses =====
with tab_def:
    st.subheader("Defenses that change the model's output")
    if "mitigation" not in results:
        st.warning("Run:  python run_experiments.py --exp mitigation")
    else:
        df = pd.DataFrame(results["mitigation"])
        st.bar_chart(df.set_index("name")[["precision", "recall"]])
        st.dataframe(
            df[["name", "test_acc", "precision", "recall", "accuracy"]].round(3),
            hide_index=True, use_container_width=True)
        st.caption("Top-k, rounding and temperature barely help. Returning only "
                   "the label helps most, but the attack still beats random "
                   "guessing (0.5), because members and non-members are "
                   "mislabeled differently.")

# ===== Tab 4: L2 =====
with tab_l2:
    st.subheader("L2 regularization during training")
    if "l2" not in results:
        st.warning("Run:  python run_experiments.py --exp l2")
    else:
        df = pd.DataFrame(results["l2"])
        df["lambda"] = df["lambda"].astype(str)
        st.bar_chart(df.set_index("lambda")[["precision", "recall"]])
        st.dataframe(
            df[["lambda", "train_acc", "test_acc", "gap", "precision", "recall"]]
            .round(3), hide_index=True, use_container_width=True)
        st.caption("Regularization reduces overfitting, which is the root cause "
                   "of the leak. Too much of it also lowers test accuracy.")

# ===== Tab 5: per class =====
with tab_cls:
    st.subheader("Attack precision for each class")
    options = {}
    if "mitigation" in results:
        for r in results["mitigation"]:
            options[f"Defense: {r['name']}"] = r
    if "sweep" in results:
        for r in results["sweep"]:
            options[f"Training size: {r['train_size']}"] = r
    if "l2" in results:
        for r in results["l2"]:
            options[f"L2 lambda: {r['lambda']}"] = r

    if not options:
        st.warning("No results yet.")
    else:
        choice = st.selectbox("Experiment", list(options.keys()))
        pc = pd.DataFrame(options[choice]["per_class"])
        pc["class"] = pc["class"].apply(class_name)
        st.bar_chart(pc.set_index("class")[["precision", "recall"]])
        st.dataframe(pc.round(3), hide_index=True, use_container_width=True)
        st.caption("Precision is usually highest on classes the target "
                   "generalizes worst on (cat, bird, deer, dog).")