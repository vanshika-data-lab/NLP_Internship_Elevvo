"""
Task 6 BONUS: Interactive QA Streamlit App
Run: streamlit run app_qa.py
"""

import streamlit as st
import torch

st.set_page_config(
    page_title="Question Answering – Task 6",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Question Answering with Transformers")
st.markdown(
    "Enter a **context passage** and ask a **question** "
    "— the model will extract the answer span."
)

# ── Sidebar: model selection ──────────────────────────────────────────────────
MODELS = {
    "DistilBERT (fast)"    : "distilbert-base-cased-distilled-squad",
    "BERT-large (accurate)": "bert-large-uncased-whole-word-masking-finetuned-squad",
    "RoBERTa (robust)"     : "deepset/roberta-base-squad2",
}

model_choice = st.sidebar.selectbox("Choose Model", list(MODELS.keys()))
model_name   = MODELS[model_choice]

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ How it works")
st.sidebar.markdown(
    "- **DistilBERT** — fastest, good accuracy\n"
    "- **BERT-large** — slowest, highest accuracy\n"
    "- **RoBERTa** — best balance of speed & accuracy"
)


@st.cache_resource(show_spinner="Loading model... please wait")
def get_pipeline(name: str):
    """
    FIX: Use AutoModelForQuestionAnswering + AutoTokenizer directly
    instead of pipeline("question-answering") to avoid the
    KeyError on older/newer transformers versions.
    """
    try:
        # Method 1: Try standard pipeline first (works on transformers >= 4.20)
        from transformers import pipeline as hf_pipeline
        qa = hf_pipeline(
            task="question-answering",
            model=name,
            tokenizer=name,
            device=0 if torch.cuda.is_available() else -1,
        )
        return qa, "pipeline"

    except KeyError:
        # Method 2: Fallback — load model + tokenizer manually
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering

        tokenizer = AutoTokenizer.from_pretrained(name)
        model     = AutoModelForQuestionAnswering.from_pretrained(name)
        model.eval()
        return (tokenizer, model), "manual"


def run_qa(pipeline_obj, mode: str, question: str, context: str) -> dict:
    """Run inference using either pipeline or manual model."""
    if mode == "pipeline":
        result = pipeline_obj(question=question, context=context)
        return result

    else:
        # Manual inference
        tokenizer, model = pipeline_obj
        inputs = tokenizer(
            question, context,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        with torch.no_grad():
            outputs = model(**inputs)

        start_idx = torch.argmax(outputs.start_logits).item()
        end_idx   = torch.argmax(outputs.end_logits).item() + 1

        # Compute confidence as softmax of start + end logits
        start_conf = torch.softmax(outputs.start_logits, dim=1)[0][start_idx].item()
        end_conf   = torch.softmax(outputs.end_logits,   dim=1)[0][end_idx - 1].item()
        score      = (start_conf + end_conf) / 2

        tokens = inputs["input_ids"][0]
        answer_tokens = tokens[start_idx:end_idx]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

        # Get character positions approximately
        answer_start = context.find(answer) if answer in context else 0
        answer_end   = answer_start + len(answer)

        return {
            "answer": answer,
            "score" : score,
            "start" : answer_start,
            "end"   : answer_end,
        }


# ── Main UI ───────────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    default_ctx = (
        "Albert Einstein was born on 14 March 1879 in Ulm, Germany. "
        "He is widely regarded as one of the greatest physicists of all time. "
        "He is best known for developing the theory of relativity, but he also "
        "made contributions to quantum mechanics. His formula E = mc2 is the "
        "world's most famous equation. Einstein received the Nobel Prize in "
        "Physics in 1921."
    )
    context = st.text_area(
        "📄 Context Passage",
        value=default_ctx,
        height=200,
        placeholder="Paste any paragraph here..."
    )

with col2:
    question = st.text_input(
        "❓ Your Question",
        value="When was Einstein born?",
        placeholder="Ask anything about the passage..."
    )
    run_btn = st.button(
        "🚀 Get Answer",
        type="primary",
        use_container_width=True
    )

# ── Run inference ─────────────────────────────────────────────────────────────
if run_btn:
    if not context.strip():
        st.warning("⚠️ Please enter a context passage.")
    elif not question.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner(f"Loading {model_choice} and finding answer..."):
            try:
                pipe_obj, mode = get_pipeline(model_name)
                result = run_qa(pipe_obj, mode, question, context)

                st.divider()

                # ── Result metrics ────────────────────────────────────────
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("📌 Answer",     result["answer"] or "Not found")
                col_b.metric("🎯 Confidence", f"{result['score']:.1%}")
                col_c.metric("🔤 Position",   f"char {result['start']} → {result['end']}")

                # ── Highlight answer in context ───────────────────────────
                if result["answer"] and result["answer"] in context:
                    st.subheader("📖 Answer Highlighted in Context")
                    highlighted = context.replace(
                        result["answer"],
                        f"**:orange[{result['answer']}]**",
                        1
                    )
                    st.markdown(highlighted)
                else:
                    st.info("Answer span could not be highlighted in context.")

                # ── Raw result ────────────────────────────────────────────
                with st.expander("🔧 Raw Model Output"):
                    st.json(result)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.markdown("**Try this fix:** Run the command below in CMD, then restart Streamlit:")
                st.code("pip install --upgrade transformers", language="bash")

# ── Fix instructions in sidebar ───────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 If you see errors")
st.sidebar.code("pip install --upgrade transformers", language="bash")
st.sidebar.markdown("Then restart Streamlit with `Ctrl+C` and run again.")

# ── Example passages ──────────────────────────────────────────────────────────
with st.expander("📚 Try these example passages"):
    examples = [
        {
            "context" : (
                "The Amazon rainforest covers most of the Amazon basin of South America. "
                "This basin encompasses 7,000,000 km2. The majority of the forest is "
                "contained within Brazil, with 60% of the rainforest."
            ),
            "question": "What percentage of the Amazon rainforest is in Brazil?",
        },
        {
            "context" : (
                "Python was created by Guido van Rossum and first released in 1991. "
                "Python is dynamically typed and supports multiple programming paradigms."
            ),
            "question": "Who created Python?",
        },
        {
            "context" : (
                "The FIFA World Cup is held every four years. "
                "Brazil has won the tournament a record five times. "
                "Germany and Italy have each won four times."
            ),
            "question": "How many times has Brazil won the FIFA World Cup?",
        },
    ]
    for ex in examples:
        st.markdown(f"**Context:** {ex['context']}")
        st.markdown(f"**Question:** `{ex['question']}`")
        st.divider()


