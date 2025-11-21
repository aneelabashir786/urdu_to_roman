# app.py
import streamlit as st
import torch
import sentencepiece as spm
import requests
import os
from model import Encoder, AttnDecoder, Seq2Seq
from pathlib import Path

# ---------- Hugging Face Model Repo ----------
BASE_HF_URL = "https://huggingface.co/aneelaBashir22f3414/urdu_to_roman/resolve/main/"

FILES = {
    "MODEL_FILE": "best_attn_seq2seq.pt",
    "URDU_SP": "urdu_spm.model",
    "ROMAN_SP": "roman_spm.model",
    "URDU_VOCAB": "urdu.vocab",
    "ROMAN_VOCAB": "roman.vocab"
}

MAX_LEN = 50
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
DEVICE = torch.device("cpu")


# ---------- Download Helper ----------
def download_if_missing(filename):
    """Download a file from Hugging Face if it's not in the local folder."""
    path = Path(filename)
    if path.exists():
        return
    url = BASE_HF_URL + filename
    st.write(f"Downloading **{filename}** from Hugging Face ...")
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise FileNotFoundError(f"Could not download {filename} from Hugging Face")
    with open(filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)


def ensure_files():
    """Ensure all required files exist locally."""
    for file in FILES.values():
        download_if_missing(file)


# ---------- Cache Tokenizers ----------
@st.cache_resource
def load_tokenizers(urdu_path=FILES["URDU_SP"], roman_path=FILES["ROMAN_SP"]):
    ur = spm.SentencePieceProcessor(model_file=urdu_path)
    ro = spm.SentencePieceProcessor(model_file=roman_path)
    return ur, ro


# ---------- Cache Model ----------
@st.cache_resource
def load_model(checkpoint_path=FILES["MODEL_FILE"]):
    ensure_files()
    ur_sp, ro_sp = load_tokenizers()
    URV, ROV = ur_sp.get_piece_size(), ro_sp.get_piece_size()

    EMB, HID = 256, 512
    enc = Encoder(URV, emb_dim=EMB, hid_dim=HID,
                  n_layers=2, dropout=0.3, pad_idx=PAD_IDX)
    dec = AttnDecoder(ROV, emb_dim=EMB, hid_dim=HID,
                      n_layers=4, dropout=0.3, pad_idx=PAD_IDX)

    model = Seq2Seq(enc, dec, pad_idx=PAD_IDX,
                    bos_idx=BOS_IDX, eos_idx=EOS_IDX, device=DEVICE)

    st.write("Loading model weights...")
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    return model, ur_sp, ro_sp


# ---------- Greedy Decoder ----------
def greedy_decode(model, src_sp, tgt_sp, sentence, max_len=50):
    model.eval()
    tokens = [src_sp.bos_id()] + src_sp.encode(sentence) + [src_sp.eos_id()]
    tokens = torch.LongTensor(tokens).unsqueeze(0).to(model.device)

    with torch.no_grad():
        outputs = model(tokens, None, teacher_forcing_ratio=0, max_len=max_len)

    preds = outputs.argmax(2).squeeze(0).tolist()

    if tgt_sp.eos_id() in preds:
        preds = preds[:preds.index(tgt_sp.eos_id())]

    return tgt_sp.decode(preds)


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Urdu → Roman Urdu Translator", layout="centered")

st.title("🇵🇰 Urdu → Roman Urdu Translator (Seq2Seq + Attention)")

st.markdown("""
This app translates **Urdu text to Roman Urdu** using a custom **Seq2Seq + Attention** model.

Model + Tokenizers are hosted on **Hugging Face**, and code runs on Streamlit Cloud.
""")

input_text = st.text_area(
    "Enter Urdu text:",
    value="مجھے اردو بہت پسند ہے",
    height=120
)

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Load Model"):
        try:
            model, ur_sp, ro_sp = load_model()
            st.session_state["model_loaded"] = True
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")

with col2:
    if st.button("Translate"):
        try:
            if "model_loaded" not in st.session_state:
                model, ur_sp, ro_sp = load_model()
                st.session_state["model_loaded"] = True

            with st.spinner("Translating..."):
                output = greedy_decode(model, ur_sp, ro_sp, input_text, MAX_LEN)

            st.subheader("Roman Urdu Output")
            st.write(output)

        except Exception as e:
            st.error(f"Translation failed: {e}")

st.markdown("---")
st.write("💡 Powered by PyTorch + SentencePiece + Streamlit")
