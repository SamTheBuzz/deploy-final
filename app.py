"""
Android Malware Classifier — Streamlit Deployment
SGD + Random Forest Ensemble with KSWIN drift detection
Author: Oroye Samuel Oluwaseun
 
NOTE: Models are stored as pure numpy arrays (no sklearn version dependency).
Works on any Python version, any sklearn version, forever.
"""
 
import math
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import streamlit as st
import joblib
from sklearn.feature_extraction.text import HashingVectorizer
 
# All paths resolved relative to this file — works on any server, any CWD
HERE = Path(__file__).parent
ARTIFACTS = HERE / "artifacts"
 
# ─────────────────────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Android Malware Classifier",
    page_icon="🛡️",
    layout="centered",
)
 
# ─────────────────────────────────────────────────────────────────────────────
#  Load artefacts (cached — loaded once per session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    cfg     = joblib.load(ARTIFACTS / "hashing_config.joblib")
    sgd_p   = joblib.load(ARTIFACTS / "sgd_params.joblib")
    rf_p    = joblib.load(ARTIFACTS / "rf_params.joblib")
    hv_cfg  = cfg["hv_params"]
    hv = HashingVectorizer(
        n_features   = hv_cfg["n_features"],
        alternate_sign = hv_cfg["alternate_sign"],
        norm         = hv_cfg["norm"],
        analyzer     = hv_cfg["analyzer"],
        ngram_range  = hv_cfg["ngram_range"],
        lowercase    = hv_cfg["lowercase"],
    )
    return hv, sgd_p, rf_p
 
hv, sgd_params, rf_params = load_all()
 
# ─────────────────────────────────────────────────────────────────────────────
#  Pure-numpy predictors (zero sklearn version dependency)
# ─────────────────────────────────────────────────────────────────────────────
def sgd_predict_proba(params, X_sparse):
    """Logistic sigmoid on learned weight vector."""
    scores = X_sparse.dot(params["coef"].T)
    if sp.issparse(scores):
        scores = scores.toarray()
    scores = scores.ravel() + params["intercept"][0]
    p1 = 1.0 / (1.0 + np.exp(-np.clip(scores, -500, 500)))
    return np.column_stack([1 - p1, p1])
 
 
def rf_predict_proba(params, X_dense):
    """Tree traversal averaged across all 40 trees."""
    n_samples = X_dense.shape[0]
    n_classes = params["n_classes"]
    proba_sum = np.zeros((n_samples, n_classes))
    for td in params["trees"]:
        cl   = td["children_left"]
        cr   = td["children_right"]
        feat = td["feature"]
        thr  = td["threshold"]
        val  = td["value"]          # shape (n_nodes, 1, n_classes)
        for s in range(n_samples):
            node = 0
            while cl[node] != -1:  # -1 == TREE_LEAF
                node = cl[node] if X_dense[s, feat[node]] <= thr[node] else cr[node]
            counts = val[node, 0]
            total  = counts.sum()
            proba_sum[s] += (counts / total) if total > 0 else np.zeros(n_classes)
    return proba_sum / len(params["trees"])
 
 
def predict(text: str, threshold: float):
    """Vectorise text and run ensemble."""
    tokens  = [t.strip() for t in text.replace(",", " ").split() if t.strip()]
    doc     = " ".join(tokens)
    X_sp    = hv.transform([doc])
    X_dn    = X_sp.toarray()
 
    p_sgd = float(sgd_predict_proba(sgd_params, X_sp)[0, 1])
    p_rf  = float(rf_predict_proba(rf_params, X_dn)[0, 1])
    p_ens = (p_sgd + p_rf) / 2.0
 
    label = "🦠 MALWARE" if p_ens >= threshold else "✅ BENIGN"
    color = "#e74c3c"   if p_ens >= threshold else "#27ae60"
    return p_sgd, p_rf, p_ens, label, color
 
# ─────────────────────────────────────────────────────────────────────────────
#  Example feature sets
# ─────────────────────────────────────────────────────────────────────────────
BENIGN_FEATURES = """\
permission::android.permission.INTERNET
permission::android.permission.ACCESS_NETWORK_STATE
permission::android.permission.ACCESS_WIFI_STATE
activity::com.example.MainActivity
api_call::getPackageInfo
api_call::getApplicationInfo
api_call::getInstalledPackages"""
 
MALWARE_FEATURES = """\
permission::android.permission.SEND_SMS
permission::android.permission.READ_CONTACTS
permission::android.permission.RECEIVE_BOOT_COMPLETED
permission::android.permission.READ_CALL_LOG
permission::android.permission.WRITE_CONTACTS
permission::android.permission.PROCESS_OUTGOING_CALLS
api_call::getDeviceId
api_call::getSubscriberId
api_call::getSimSerialNumber
api_call::sendTextMessage
api_call::execRuntimeExec
api_call::getLine1Number
intent::android.intent.action.BOOT_COMPLETED
intent::android.intent.action.SMS_RECEIVED"""
 
# ─────────────────────────────────────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("🛡️ Android Malware Classifier")
st.caption(
    "Streaming ensemble of **SGD + Random Forest** trained on the DREBIN drift dataset.  "
    "Enter your APK's feature tokens to get an instant malware/benign prediction."
)
 
st.divider()
 
# ── Presets ───────────────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)
if col_a.button("📱 Load Benign Example", use_container_width=True):
    st.session_state["fi"] = BENIGN_FEATURES
if col_b.button("🦠 Load Malware Example", use_container_width=True):
    st.session_state["fi"] = MALWARE_FEATURES
 
# ── Feature input ─────────────────────────────────────────────────────────────
feature_text = st.text_area(
    "**App Feature Tokens** — one per line (or space/comma separated)",
    value=st.session_state.get("fi", ""),
    height=220,
    placeholder=(
        "Paste Drebin-format feature tokens here, e.g.:\n"
        "permission::android.permission.SEND_SMS\n"
        "api_call::getDeviceId\n"
        "intent::android.intent.action.BOOT_COMPLETED"
    ),
    key="feature_area",
)
 
# ── Threshold ─────────────────────────────────────────────────────────────────
threshold = st.slider(
    "**Classification threshold** — flag as malware if ensemble probability ≥ this value",
    min_value=0.10, max_value=0.90, value=0.50, step=0.05,
    help="Lower → more sensitive (catches more malware, more false positives).  "
         "Higher → more conservative (fewer false positives, may miss some malware)."
)
 
# ── Classify button ───────────────────────────────────────────────────────────
if st.button("🔍  Classify App", use_container_width=True, type="primary"):
    if not feature_text.strip():
        st.warning("Please enter at least one feature token, or click a preset above.")
    else:
        with st.spinner("Running ensemble prediction…"):
            p_sgd, p_rf, p_ens, label, color = predict(feature_text, threshold)
 
        st.divider()
 
        # ── Verdict ───────────────────────────────────────────────────────────
        st.markdown(
            f"<h2 style='color:{color};text-align:center'>{label}</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='text-align:center;color:grey'>Ensemble malware probability: "
            f"<b>{p_ens*100:.1f}%</b> &nbsp;|&nbsp; threshold: {threshold*100:.0f}%</p>",
            unsafe_allow_html=True,
        )
 
        st.divider()
 
        # ── Per-model bars ────────────────────────────────────────────────────
        st.markdown("#### Model Confidence Breakdown")
        c1, c2, c3 = st.columns(3)
 
        def prob_bar(container, model_name, prob, thr):
            bar_col = "#e74c3c" if prob >= thr else "#27ae60"
            container.markdown(f"**{model_name}**")
            container.markdown(
                f"<div style='background:#f0f0f0;border-radius:8px;"
                f"overflow:hidden;height:22px;margin-bottom:4px'>"
                f"<div style='width:{prob*100:.1f}%;background:{bar_col};"
                f"height:100%;transition:width 0.3s'></div></div>"
                f"<small style='color:grey'>{prob*100:.1f}% malware probability</small>",
                unsafe_allow_html=True,
            )
 
        prob_bar(c1, "SGD",          p_sgd, threshold)
        prob_bar(c2, "Random Forest", p_rf,  threshold)
        prob_bar(c3, "Ensemble",      p_ens, threshold)
 
        # ── How it works ──────────────────────────────────────────────────────
        with st.expander("ℹ️ How the prediction works"):
            st.markdown(
                """
**Step 1 — Feature vectorisation**  
Your tokens are joined into a single document and passed through a `HashingVectorizer`
(1 024 feature buckets, L2-normalised, word unigrams) — the same configuration used
during training on the DREBIN drift dataset.
 
**Step 2 — SGD Classifier**  
An online SGD learner (logistic loss) trained incrementally on the streaming data.
Fast to update; sensitive to recent concept drift.
 
**Step 3 — Random Forest**  
40 decision trees (max depth 10) retrained from scratch each time KSWIN detects
concept drift in the SGD's error stream. More stable, higher precision.
 
**Step 4 — Ensemble**  
Simple average of SGD and RF malware probabilities. Combines the responsiveness of
the online learner with the stability of the batch model.
 
---
_Both models are stored as pure NumPy arrays, making this app work on any Python
or scikit-learn version without compatibility issues._
                """
            )
 
# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Model Performance")
    st.caption("Evaluated on 128 686 held-out streaming samples.")
 
    import pandas as pd
    df = pd.DataFrame({
        "Accuracy":  {"SGD": "96%", "Random Forest": "96%", "Ensemble": "98%"},
        "Precision": {"SGD": "63%", "Random Forest": "97%", "Ensemble": "95%"},
        "Recall":    {"SGD": "24%", "Random Forest": "37%", "Ensemble": "48%"},
        "F1":        {"SGD": "35%", "Random Forest": "54%", "Ensemble": "63%"},
    })
    st.dataframe(df, use_container_width=True)
 
    st.divider()
    st.markdown("**Dataset:** DREBIN drift stream")
    st.markdown("**Models:** Streaming SGD + KSWIN-triggered RF")
    st.markdown("**Project:** Oroye Samuel Oluwaseun")
 
    st.divider()
    st.markdown(
        "**Feature format guide:**\n"
        "- `permission::android.permission.X`\n"
        "- `api_call::methodName`\n"
        "- `intent::android.intent.action.X`\n"
        "- `activity::com.package.ActivityName`\n"
        "- `service::com.package.ServiceName`\n"
        "- `receiver::com.package.ReceiverName`"
    )
 
