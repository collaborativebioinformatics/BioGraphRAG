"""Streamlit UI for dual bio graph rag demo."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.qa.answer import answer_question


st.set_page_config(page_title="dual bio graph rag Demo", layout="wide")
st.title("PrimeKG + PubMedKG dual bio graph rag Demo")

config_path = st.sidebar.text_input("Config path", value="configs/default.yaml")

question = st.text_area("Enter a question", value="Which experiments report elevated EGFR expression in colon cancer and what literature supports this?")

if st.button("Run dual bio graph rag"):
    with st.spinner("Retrieving context and generating answer..."):
        result = answer_question(config_path, question)
    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Evidence")
    st.json(result["evidence"])

    st.subheader("Nodes")
    st.json(result["nodes"])

    st.subheader("Edges")
    st.json(result["edges"])

