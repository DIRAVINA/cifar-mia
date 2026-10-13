# pages/1_Live_Demo.py
import json
import os
import random

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

import config
from src.attack import AttackMLP
from src.data import load_cifar, split_target_and_shadow_pool
from src.model import CifarCNN
from src.train import predict_proba

DEMO_DIR = os.path.join(config.RESULTS_DIR, "demo")
CIFAR10_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]
DEVICE = torch.device("cpu")
RED, BLUE, GREY = "#e45756", "#4c78a8", "#9aa5b1"

st.set_page_config(page_title="Live Demo - Membership Inference",
                   page_icon="🎯", layout="wide")
st.title("🎯 Live Demo: which of these images were in the training set?")
st.caption("The attacker sees only the target model's output probabilities and each "
           "image's true label. It never sees the training data. Half of the images "
           "below were used to train the model and half were not.")

if not os.path.exists(os.path.join(DEMO_DIR, "target.pt")):
    st.error("Demo models not found. Run in the terminal:  python build_demo.py")
    st.stop()


def show(fig):
    try:
        st.plotly_chart(fig, use_container_width=True)
    except TypeError:
        st.plotly_chart(fig, width="stretch")


def class_name(c):
    return CIFAR10_NAMES[c] if config.DATASET == "cifar10" else f"class {c}"


@st.cache_resource(show_spinner="Loading models and data (first time only)...")
def load_everything():
    with open(os.path.join(DEMO_DIR, "info.json")) as f:
        info = json.load(f)
    nc = info["num_classes"]

    target = CifarCNN(nc)
    target.load_state_dict(torch.load(os.path.join(DEMO_DIR, "target.pt"),
                                      map_location=DEVICE))
    target.eval()

    state = torch.load(os.path.join(DEMO_DIR, "attack_models.pt"), map_location=DEVICE)
    attack_models = {}
    for c, sd in state.items():
        m = AttackMLP(nc)
        m.load_state_dict(sd)
        m.eval()
        attack_models[int(c)] = m

    X, y = load_cifar()
    splits = split_target_and_shadow_pool(X, y)    # same split used for training
    return info, target, attack_models, splits


info, target, attack_models, splits = load_everything()
num_classes = info["num_classes"]

GROUPS = {
    "member": (splits["X_target_train"], splits["y_target_train"]),
    "non-member": (splits["X_target_test"], splits["y_target_test"]),
}


@torch.no_grad()
def member_probability(probs, labels):
    """Probability (0-1) that each image was in the training set, per the attack."""
    out = np.zeros(len(probs), dtype=np.float32)
    for c, model in attack_models.items():
        mask = labels == c
        if mask.sum() == 0:
            continue
        logits = model(torch.from_numpy(probs[mask].astype(np.float32)))
        out[mask] = torch.softmax(logits, dim=1)[:, 1].numpy()
    return out


def to_display(x):
    """Standardized image -> viewable image (stretch values to 0..1)."""
    img = np.transpose(x, (1, 2, 0))
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def new_batch(n):
    """Random, balanced and shuffled: half members, half non-members."""
    half = n // 2
    mem = [("member", i) for i in random.sample(range(len(GROUPS["member"][0])), half)]
    non = [("non-member", i)
           for i in random.sample(range(len(GROUPS["non-member"][0])), n - half)]
    batch = mem + non
    random.shuffle(batch)
    return batch


# ------------------------------------------------------------------ sidebar
st.sidebar.header("Demo controls")
n = st.sidebar.select_slider("Images to show", options=[8, 12, 16], value=12)
pick_clicked = st.sidebar.button("🎲 Pick new random images", type="primary")

if (pick_clicked or "demo_batch" not in st.session_state
        or st.session_state.get("demo_n") != n):
    st.session_state["demo_batch"] = new_batch(n)
    st.session_state["demo_n"] = n
    st.session_state["reveal_answers"] = False      # hide answers for the new round

reveal = st.sidebar.checkbox("Reveal the correct answers", key="reveal_answers")
batch = st.session_state["demo_batch"]

# ------------------------------------------------------------------ run models
X_batch = np.stack([GROUPS[g][0][i] for g, i in batch])
y_batch = np.array([int(GROUPS[g][1][i]) for g, i in batch])
probs = predict_proba(target, X_batch, DEVICE)
p_member = member_probability(probs, y_batch)
verdict_member = p_member >= 0.5
truth_member = np.array([g == "member" for g, _ in batch])

# ------------------------------------------------------------------ scoreboard
if reveal:
    correct = int((verdict_member == truth_member).sum())
    k1, k2, k3 = st.columns(3)
    with k1:
        with st.container(border=True):
            st.metric("Attack got right", f"{correct} / {n}")
    with k2:
        with st.container(border=True):
            st.metric("Members correctly spotted",
                      f"{int((verdict_member & truth_member).sum())} / "
                      f"{int(truth_member.sum())}")
    with k3:
        with st.container(border=True):
            st.metric("Non-members correctly spotted",
                      f"{int((~verdict_member & ~truth_member).sum())} / "
                      f"{int((~truth_member).sum())}")
else:
    st.info("Look at the verdicts first, then tick **Reveal the correct answers** "
            "in the sidebar to see how the attack did.")

# ------------------------------------------------------------------ image grid
COLS = 4
for start in range(0, n, COLS):
    cols = st.columns(COLS)
    for k, j in enumerate(range(start, min(start + COLS, n))):
        with cols[k]:
            with st.container(border=True):
                st.image(to_display(X_batch[j]), width=120, clamp=True)
                st.markdown(f"**True label:** {class_name(int(y_batch[j]))}")
                pred = int(probs[j].argmax())
                ok = pred == int(y_batch[j])
                st.caption(f"Target model says: {class_name(pred)} "
                           f"{'✅' if ok else '❌'} ({probs[j].max():.0%} sure)")
                st.progress(float(p_member[j]))
                if verdict_member[j]:
                    st.markdown(f":red[**MEMBER**] · {p_member[j]:.0%}")
                else:
                    st.markdown(f":blue[**NON-MEMBER**] · {p_member[j]:.0%}")
                if reveal:
                    actual = batch[j][0]
                    if verdict_member[j] == truth_member[j]:
                        st.success(f"Correct: {actual}")
                    else:
                        st.warning(f"Wrong: it is a {actual}")

st.caption("The bar and percentage show the attacker's confidence that the image was "
           "in the training set. 50% or more is called MEMBER.")

# ------------------------------------------------------------------ inspect one
st.divider()
st.subheader("Look inside one prediction")
which = st.selectbox("Choose an image from above", list(range(n)),
                     format_func=lambda j: f"Image {j + 1}  ({class_name(int(y_batch[j]))})")
c1, c2 = st.columns([1.4, 1])
with c1:
    names = [class_name(i) for i in range(num_classes)]
    colors = [RED if i == int(y_batch[which]) else GREY for i in range(num_classes)]
    fig = go.Figure(go.Bar(x=probs[which], y=names, orientation="h", marker_color=colors,
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                      template="plotly_white", title="What the attacker sees",
                      xaxis=dict(range=[0, 1], title="Target model probability"),
                      yaxis=dict(autorange="reversed"))
    show(fig)
    st.caption("Red bar = the true class. A very tall red bar (the model is nearly "
               "certain) is typical of images it was trained on.")
with c2:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=float(p_member[which]) * 100,
        number={"suffix": "%"}, title={"text": "Chance it was in the training set"},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": RED if verdict_member[which] else BLUE},
               "threshold": {"line": {"color": "black", "width": 3},
                             "thickness": 0.8, "value": 50}}))
    gauge.update_layout(height=300, margin=dict(l=40, r=50, t=60, b=10))
    show(gauge)

# ------------------------------------------------------------------ batch test
st.divider()
st.subheader("Batch test: how well does the attack do over many images?")
m = st.slider("Images per group", 100, 2000, 500, step=100)

if st.button("Run batch test"):
    rng = np.random.RandomState(0)
    scores = {}
    for g in ["member", "non-member"]:
        Xg, yg = GROUPS[g]
        sel = rng.choice(len(Xg), m, replace=False)
        pr = predict_proba(target, Xg[sel], DEVICE)
        scores[g] = member_probability(pr, yg[sel])

    tp = int((scores["member"] >= 0.5).sum())
    fn = m - tp
    fp = int((scores["non-member"] >= 0.5).sum())
    tn = m - fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / m
    accuracy = (tp + tn) / (2 * m)

    k1, k2, k3 = st.columns(3)
    k1.metric("Attack accuracy", f"{accuracy:.3f}", "random guessing = 0.5",
              delta_color="off")
    k2.metric("Precision", f"{precision:.3f}")
    k3.metric("Recall", f"{recall:.3f}")

    h = go.Figure()
    h.add_trace(go.Histogram(x=scores["member"], name="Members", nbinsx=30,
                             marker_color=RED, opacity=0.7))
    h.add_trace(go.Histogram(x=scores["non-member"], name="Non-members", nbinsx=30,
                             marker_color=BLUE, opacity=0.7))
    h.update_layout(barmode="overlay", height=340, template="plotly_white",
                    xaxis_title="Attack's membership probability",
                    yaxis_title="Number of images",
                    margin=dict(l=10, r=10, t=10, b=10))
    h.add_vline(x=0.5, line_dash="dash", line_color="black")
    show(h)

    cm = go.Figure(go.Heatmap(
        z=[[tp, fn], [fp, tn]],
        x=["Predicted member", "Predicted non-member"],
        y=["Actually member", "Actually non-member"],
        text=[[tp, fn], [fp, tn]], texttemplate="%{text}",
        colorscale="Blues", showscale=False))
    cm.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                     yaxis=dict(autorange="reversed"))
    show(cm)
    st.caption("The less the two histograms overlap, the easier it is for the "
               "attacker to tell members from non-members.")