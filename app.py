# app.py
# Run with:  streamlit run app.py
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config

RESULTS_FILE = os.path.join(config.RESULTS_DIR, "results.json")
CIFAR10_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]
RED, BLUE, GREY, GREEN = "#e45756", "#4c78a8", "#9aa5b1", "#54a24b"

st.set_page_config(page_title="Membership Inference on CIFAR",
                   page_icon="🔍", layout="wide")

# ---------------------------------------------------------------- load data
if not os.path.exists(RESULTS_FILE):
    st.error("No results found. Run:  python run_experiments.py --exp all")
    st.stop()

with open(RESULTS_FILE) as f:
    results = json.load(f)

analysis = results.get("analysis")
base = next((r for r in results.get("mitigation", [])
             if r["name"] == "No mitigation"), None)


# ---------------------------------------------------------------- helpers
def short_name(c):
    return CIFAR10_NAMES[c] if config.DATASET == "cifar10" else f"class {c}"


def style(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10),
                      template="plotly_white",
                      legend=dict(orientation="h", y=1.12, x=0))
    return fig


def show(fig):
    """Display a Plotly chart (works across Streamlit versions)."""
    try:
        st.plotly_chart(fig, use_container_width=True)
    except TypeError:
        st.plotly_chart(fig, width="stretch")


def show_table(df):
    try:
        st.dataframe(df, hide_index=True, use_container_width=True)
    except TypeError:
        st.dataframe(df, hide_index=True, width="stretch")


def kpi(col, label, value, help_text=None):
    with col:
        with st.container(border=True):
            st.metric(label, value, help=help_text)


def baseline_line(fig, y=0.5):
    fig.add_hline(y=y, line_dash="dash", line_color="black",
                  annotation_text="random guessing", annotation_position="bottom right")


def hist_fig(h, title, xlabel):
    edges = np.array(h["edges"])
    centers = (edges[:-1] + edges[1:]) / 2
    width = (edges[1] - edges[0]) * 0.95
    m, n = np.array(h["members"]), np.array(h["non_members"])
    fig = go.Figure()
    fig.add_bar(x=centers, y=m / m.sum(), name="Members (in training set)",
                marker_color=RED, opacity=0.7, width=width)
    fig.add_bar(x=centers, y=n / n.sum(), name="Non-members (unseen)",
                marker_color=BLUE, opacity=0.7, width=width)
    fig.update_layout(barmode="overlay", title=title, xaxis_title=xlabel,
                      yaxis_title="Share of images")
    return style(fig)


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🔍 Membership Inference")
st.sidebar.caption("Shokri et al. (2017) · replicated on CIFAR-10")
section = st.sidebar.radio(
    "Section",
    ["Overview", "Why the attack works", "Attack strength",
     "Defenses", "Per-class results"])
st.sidebar.divider()
st.sidebar.markdown("**Reading the charts**  \nMembers and non-members are equal in "
                    "number, so **random guessing = 0.5**.")

# ================================================================= OVERVIEW
if section == "Overview":
    st.title("Can a model reveal who was in its training data?")
    st.markdown("A black-box attack that decides, from a model's output alone, whether "
                "an image was used to train it.")

    if base:
        cols = st.columns(6 if analysis else 5)
        kpi(cols[0], "Target train accuracy", f"{base['train_acc']:.3f}")
        kpi(cols[1], "Target test accuracy", f"{base['test_acc']:.3f}")
        kpi(cols[2], "Train-test gap", f"{base['gap']:.3f}",
            "A big gap means the model is overfitted. That is what leaks.")
        kpi(cols[3], "Attack precision", f"{base['precision']:.3f}",
            "Of the images called 'member', the fraction that really are.")
        kpi(cols[4], "Attack recall", f"{base['recall']:.3f}",
            "Of the real members, the fraction the attacker found.")
        if analysis:
            kpi(cols[5], "Attack AUC", f"{analysis['roc']['shadow_attack']['auc']:.3f}",
                "1.0 = perfect, 0.5 = random guessing.")

    st.subheader("How the attack works")
    st.graphviz_chart(r"""
    digraph {
      rankdir=LR;
      node [shape=box, style="rounded,filled", fillcolor="#eef2f7", fontname="Helvetica"];
      data   [label="Attacker's own data"];
      shadow [label="Shadow models\n(copies of the target)"];
      attack [label="Attack models\n(one per class)", fillcolor="#fbe9e8"];
      query  [label="Image + true label"];
      target [label="Target model\n(black box)"];
      verdict [label="Member or\nnon-member?", fillcolor="#fbe9e8"];
      data -> shadow [label="train"];
      shadow -> attack [label="outputs labeled\nin / out"];
      query -> target;
      target -> attack [label="probabilities"];
      attack -> verdict;
    }
    """)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**The idea**
1. A model is usually *more confident* on images it trained on.
2. The attacker trains **shadow models** where membership is known.
3. A small **attack model** per class learns: *probabilities → member / non-member*.
4. It is then pointed at the real target.
        """)
    with c2:
        st.markdown("""
**The metrics**
- **Precision**: of the images called members, how many really are.
- **Recall**: of the real members, how many were found.
- **AUC / ROC**: how well the attack's score separates the two groups.
        """)

# ====================================================== WHY THE ATTACK WORKS
elif section == "Why the attack works":
    st.title("Why the attack works")
    if not analysis:
        st.warning("Run:  python run_analysis.py")
        st.stop()

    # ---- histograms
    st.subheader("1. Members look different from non-members")
    signal = st.radio("Signal", ["Confidence in the true class", "Prediction entropy"],
                      horizontal=True)
    h = analysis["histograms"]
    if signal.startswith("Confidence"):
        show(hist_fig(h["true_class_confidence"],
                      "Probability the model assigns to the correct class",
                      "Probability of the true class"))
        st.caption("Members pile up near 1.0 (the model is sure). Non-members spread out. "
                   "That gap is the leak.")
    else:
        show(hist_fig(h["entropy"], "Uncertainty of the model's prediction",
                      "Normalized entropy (0 = certain, 1 = unsure)"))
        st.caption("Members get near-zero entropy: the model is very certain about "
                   "images it has memorized.")

    # ---- ROC
    st.subheader("2. How good is the attack?")
    roc = analysis["roc"]
    sa, cb = roc["shadow_attack"], roc["confidence_baseline"]
    k = st.columns(4)
    kpi(k[0], "AUC: shadow attack", f"{sa['auc']:.3f}")
    kpi(k[1], "AUC: simple baseline", f"{cb['auc']:.3f}",
        "Baseline = just threshold the model's confidence. No shadow models.")
    kpi(k[2], "TPR at 1% FPR", f"{sa['tpr_at_1pct_fpr']:.3f}",
        "Members found when the attacker may wrongly accuse only 1% of non-members.")
    kpi(k[3], "TPR at 0.1% FPR", f"{sa['tpr_at_0.1pct_fpr']:.3f}")

    log_x = st.checkbox("Log scale on false-positive rate (zooms into the low-FPR region)")
    fig = go.Figure()
    for name, r, color in [("Shadow-model attack", sa, RED),
                           ("Confidence baseline (no shadows)", cb, BLUE)]:
        fpr, tpr = np.array(r["fpr"]), np.array(r["tpr"])
        if log_x:
            keep = fpr > 0
            fpr, tpr = fpr[keep], tpr[keep]
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"{name} (AUC {r['auc']:.3f})",
                                 line=dict(color=color, width=3)))
    lo = 1e-3 if log_x else 0
    fig.add_trace(go.Scatter(x=[lo, 1], y=[lo, 1], mode="lines", name="Random guessing",
                             line=dict(color=GREY, dash="dash")))
    fig.update_layout(xaxis_title="False-positive rate",
                      yaxis_title="True-positive rate",
                      xaxis_type="log" if log_x else "linear")
    show(style(fig, 420))
    st.caption("The shadow attack beats the simple baseline, but only modestly. "
               "At very low false-positive rates the attack finds few members.")

    # ---- gap vs precision
    st.subheader("3. More overfitting means a stronger attack")
    g = analysis["gap_vs_precision"]
    df = pd.DataFrame(g["rows"])
    df["name"] = df["class"].apply(short_name)
    fig = go.Figure(go.Scatter(x=df["gap"], y=df["precision"], mode="markers+text",
                               text=df["name"], textposition="top center",
                               marker=dict(size=12, color=RED), name="Class"))
    slope, intercept = np.polyfit(df["gap"], df["precision"], 1)
    xs = np.array([df["gap"].min(), df["gap"].max()])
    fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines",
                             line=dict(color=GREY, dash="dash"), name="Trend"))
    fig.update_layout(xaxis_title="Target train-test accuracy gap (this class)",
                      yaxis_title="Attack precision (this class)")
    show(style(fig, 420))
    st.caption(f"Correlation across the {len(df)} classes: **{g['correlation']:.2f}**. "
               "Classes the model overfits most are the easiest to attack "
               "(paper, Fig. 11).")

# ============================================================ ATTACK STRENGTH
elif section == "Attack strength":
    st.title("What changes the attack's strength?")

    st.subheader("Size of the target's training set")
    if "sweep" not in results:
        st.warning("Run:  python run_experiments.py --exp sweep")
    else:
        df = pd.DataFrame(results["sweep"]).sort_values("train_size")
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["train_size"], y=df["precision"], name="Precision",
                                     mode="lines+markers", line=dict(color=RED, width=3)))
            fig.add_trace(go.Scatter(x=df["train_size"], y=df["recall"], name="Recall",
                                     mode="lines+markers", line=dict(color=BLUE, width=3)))
            baseline_line(fig)
            fig.update_layout(title="Attack precision and recall",
                              xaxis_title="Target training-set size",
                              yaxis=dict(range=[0.45, 1.02]))
            show(style(fig))
        with c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["train_size"], y=df["train_acc"], name="Train accuracy",
                                     mode="lines+markers", line=dict(color=GREEN, width=3)))
            fig.add_trace(go.Scatter(x=df["train_size"], y=df["test_acc"], name="Test accuracy",
                                     mode="lines+markers", line=dict(color=GREY, width=3)))
            fig.update_layout(title="Target model accuracy",
                              xaxis_title="Target training-set size",
                              yaxis=dict(range=[0.4, 1.02]))
            show(style(fig))
        st.caption("Larger training sets overfit less (the gap shrinks), so the attack "
                   "weakens. The paper shows the same trend.")

    st.subheader("Number of shadow models")
    if not analysis:
        st.warning("Run:  python run_analysis.py")
    else:
        df = pd.DataFrame(analysis["shadow_sweep"])
        fig = go.Figure()
        for col, name, color in [("precision", "Precision", RED),
                                 ("recall", "Recall", BLUE), ("auc", "AUC", GREEN)]:
            fig.add_trace(go.Scatter(x=df["num_shadow"], y=df[col], name=name,
                                     mode="lines+markers", line=dict(color=color, width=3)))
        fig.update_layout(xaxis_title="Number of shadow models",
                          yaxis=dict(range=[0.7, 1.02]),
                          xaxis=dict(tickmode="array", tickvals=df["num_shadow"].tolist()))
        show(style(fig))
        st.caption("More shadow models help, with diminishing returns after about 5.")

# ================================================================== DEFENSES
elif section == "Defenses":
    st.title("Can the attack be stopped?")

    if "mitigation" in results:
        st.subheader("Output-side defenses")
        df = pd.DataFrame(results["mitigation"])
        fig = go.Figure()
        fig.add_bar(x=df["name"], y=df["precision"], name="Precision", marker_color=RED)
        fig.add_bar(x=df["name"], y=df["recall"], name="Recall", marker_color=BLUE)
        baseline_line(fig)
        fig.update_layout(barmode="group", yaxis=dict(range=[0.4, 1.05]))
        show(style(fig, 420))
        st.caption("Top-k, rounding and temperature barely help. Releasing only the label "
                   "helps most, yet the attack still beats guessing because members and "
                   "non-members are misclassified differently.")

    if "l2" in results:
        st.subheader("Regularization (L2) during training")
        df2 = pd.DataFrame(results["l2"])
        df2["lambda"] = df2["lambda"].astype(str)
        c1, c2 = st.columns([1.2, 1])
        with c1:
            fig = go.Figure()
            fig.add_bar(x=df2["lambda"], y=df2["precision"], name="Attack precision",
                        marker_color=RED)
            fig.add_bar(x=df2["lambda"], y=df2["test_acc"], name="Target test accuracy",
                        marker_color=GREEN)
            baseline_line(fig)
            fig.update_layout(barmode="group", xaxis_title="L2 strength (lambda)",
                              yaxis=dict(range=[0.4, 1.0]))
            show(style(fig))
        with c2:
            show_table(df2[["lambda", "train_acc", "test_acc", "gap",
                            "precision", "recall"]].round(3))

    # privacy vs utility
    rows = []
    for r in results.get("mitigation", []):
        rows.append({"defense": r["name"], "kind": "Output defense",
                     "test_acc": r["test_acc"], "precision": r["precision"]})
    for r in results.get("l2", []):
        rows.append({"defense": f"L2 lambda={r['lambda']}", "kind": "L2 regularization",
                     "test_acc": r["test_acc"], "precision": r["precision"]})
    if rows:
        st.subheader("Privacy vs utility")
        dd = pd.DataFrame(rows)
        fig = go.Figure()
        for kind, color in [("Output defense", BLUE), ("L2 regularization", RED)]:
            part = dd[dd["kind"] == kind]
            fig.add_trace(go.Scatter(
                x=part["test_acc"], y=part["precision"], mode="markers", name=kind,
                marker=dict(size=13, color=color, opacity=0.8),
                text=part["defense"],
                hovertemplate="%{text}<br>test acc %{x:.3f}<br>attack precision %{y:.3f}"
                              "<extra></extra>"))
        fig.update_layout(xaxis_title="Target test accuracy (higher = more useful model)",
                          yaxis_title="Attack precision (lower = more private)")
        show(style(fig, 420))
        st.caption("The ideal defense sits at the bottom right: accurate model, weak "
                   "attack. Hover over a point to see which defense it is.")

# ============================================================== PER-CLASS
elif section == "Per-class results":
    st.title("Attack precision, class by class")

    options = {}
    for r in results.get("mitigation", []):
        options[f"Defense: {r['name']}"] = r["per_class"]
    for r in results.get("sweep", []):
        options[f"Train size {r['train_size']}"] = r["per_class"]
    for r in results.get("l2", []):
        options[f"L2 lambda={r['lambda']}"] = r["per_class"]

    if not options:
        st.warning("No results yet.")
        st.stop()

    names = list(options.keys())
    num_classes = len(options[names[0]])
    z = [[row["precision"] for row in options[n]] for n in names]
    fig = go.Figure(go.Heatmap(
        z=z, x=[short_name(c) for c in range(num_classes)], y=names,
        zmin=0.5, zmax=1.0, colorscale="Reds", text=np.round(z, 2),
        texttemplate="%{text}", colorbar=dict(title="Precision")))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    show(style(fig, 140 + 34 * len(names)))
    st.caption("Darker = the attack is more precise on that class. "
               "Classes like bird, cat and dog are usually the most exposed.")

    choice = st.selectbox("Look closer at one experiment", names)
    pc = pd.DataFrame(options[choice])
    pc["name"] = pc["class"].apply(short_name)
    fig = go.Figure()
    fig.add_bar(x=pc["name"], y=pc["precision"], name="Precision", marker_color=RED)
    fig.add_bar(x=pc["name"], y=pc["recall"], name="Recall", marker_color=BLUE)
    baseline_line(fig)
    fig.update_layout(barmode="group", yaxis=dict(range=[0.4, 1.05]))
    show(style(fig))