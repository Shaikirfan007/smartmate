import streamlit as st
from auth import check_login, logout
from backend import PDF_QA_System
from datetime import datetime

# Page config
st.set_page_config(page_title="StudyMate — PDF Q&A", page_icon="📚", layout="wide")

# ---------- GLOBAL STYLES ----------
st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; }

    .sm-header {
        background: linear-gradient(90deg,#0f172a,#1e3a8a);
        color:white; padding:18px 26px;
        border-radius:14px;
        box-shadow: 0 8px 28px rgba(30,58,138,0.12);
        margin-bottom: 18px;
    }
    .sm-header h1 { margin:0; font-size:26px; }
    .sm-header p { margin:4px 0 0 0; color:rgba(255,255,255,0.92); }

    .sm-card { background: #fff; border-radius:12px; padding:18px;
        box-shadow: 0 6px 18px rgba(15,23,42,0.06);
        border: 1px solid #eef2ff; margin-bottom:14px; }
    .muted { color:#6b7280; }

    div.stButton > button:first-child {
        background: #2563eb !important;
        color: #fff !important;
        border-radius:10px;
        padding:0.55rem 1rem;
        font-weight:600;
    }
    div.stButton > button:first-child:hover { background:#1e40af !important; }

    .stSidebar { background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
        color: #fff; padding-top:14px; }
    .sidebar-content { padding: 12px; }
    .stFileUploader { border-radius:10px !important; }

    .stTabs [data-baseweb="tab"] { border-radius:10px; padding:8px 14px; }
    [aria-selected="true"] { background: #e0e7ff !important; color:#1e3a8a !important; }
    </style>
    """, unsafe_allow_html=True
)

# ---------- AUTH ----------
if "logged_in" not in st.session_state or not st.session_state.get("logged_in"):
    if not check_login():
        st.stop()

# ---------- Header ----------
st.markdown(
    """
    <div class="sm-header">
      <h1>📚 StudyMate — AI PDF Q&A</h1>
      <p>Upload digital or handwritten PDFs. Ask questions and get answers with page references.</p>
    </div>
    """, unsafe_allow_html=True
)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
    st.markdown("### ⚙ RAG Settings")

    with st.expander("Chunking & Retrieval", expanded=True):
        chunk_size = st.number_input("Chunk size (chars)", 300, 3000, 1200, 100)
        chunk_overlap = st.number_input("Chunk overlap (chars)", 0, 500, 200, 50)
        top_k = st.number_input("Top-k context chunks", 1, 10, 5, 1)

    with st.expander("Model / Options", expanded=False):
        use_granite = st.checkbox("Use IBM Granite (recommended)", value=True)
        granite_choice = st.selectbox("Granite model", ["granite-1b-instruct", "granite-2b-instruct"])
        do_ocr_force = st.checkbox("Force OCR for all pages", value=False)
        ocr_lang = st.text_input("OCR languages", "en")

    st.markdown("---")
    st.markdown(f"👤 {st.session_state['user_info']['name']}")
    st.markdown(f"📅 {datetime.now().strftime('%b %d, %Y')}")
    st.button("🚪 Logout", on_click=logout)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- QA System ----------
if "qa" not in st.session_state:
    st.session_state.qa = PDF_QA_System()

# pass model params (including granite selection)
st.session_state.qa.set_params(
    chunk_size=int(chunk_size),
    chunk_overlap=int(chunk_overlap),
    top_k=int(top_k),
    use_granite=bool(use_granite),
    granite_model=f"ibm/{granite_choice}" if not granite_choice.startswith("ibm/") else granite_choice
)

# Init history storage
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# ---------- Upload ----------
st.markdown("<div class='sm-card'>", unsafe_allow_html=True)
st.subheader("📄 Upload PDF(s)")
uploaded_files = st.file_uploader(
    "Drag and drop PDFs here (or click to browse)",
    type="pdf", accept_multiple_files=True
)

col1, col2 = st.columns([1, 1])
with col1: process = st.button("📥 Process PDF(s)")
with col2: reset = st.button("🧹 Reset Index")

if reset:
    st.session_state.qa.reset()
    st.session_state.pop("quiz_data", None)
    st.session_state.pop("quiz_show_answers", None)
    st.session_state.qa_history = []
    st.success("Index cleared.")

if process:
    if not uploaded_files:
        st.warning("Please choose one or more PDF files.")
    else:
        with st.spinner("Processing PDFs..."):
            for f in uploaded_files:
                try:
                    st.session_state.qa.add_pdf_from_bytes(
                        f.name, f.read(),
                        force_ocr=do_ocr_force,
                        preprocess_ocr=True,
                        ocr_langs=[l.strip() for l in ocr_lang.split(",") if l.strip()]
                    )
                except Exception as e:
                    st.error(f"❌ Failed {f.name}: {e}")
            try:
                st.session_state.qa.build_or_update_index()
                st.success(f"✅ Processed {len(uploaded_files)} file(s).")
            except Exception as e:
                st.error(f"Index build failed: {e}")
st.markdown("</div>", unsafe_allow_html=True)

# ---------- Main Tabs ----------
if st.session_state.qa.has_index():
    tabs = st.tabs(["🔎 Q&A", "🧾 Summary", "🃏 Flashcards", "📆 Schedule", "❓ Quiz", "🗂 History"])

    # Q&A
    with tabs[0]:
        st.markdown("<div class='sm-card'>", unsafe_allow_html=True)
        question = st.text_input("Ask your question...")
        if st.button("🧠 Get Answer") and question.strip():
            with st.spinner("Thinking..."):
                res = st.session_state.qa.answer_question(question)
            st.write("### 📖 Answer", res["answer"])
            if res.get("citations"):
                st.write("#### 📄 Sources")
                for c in res["citations"]:
                    st.write(f"- {c['doc']} — Page {c['page']}")

            # ✅ Save to history
            st.session_state.qa_history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "question": question,
                "answer": res["answer"],
                "citations": res.get("citations", [])
            })
        st.markdown("</div>", unsafe_allow_html=True)

    # Summary
    with tabs[1]:
        st.subheader("🧾 Generate Document Summary")
        summary_level = st.selectbox(
            "Select Summary Level:",
            ["Intermediate", "Beginner", "Expert"],
            index=0, # Default to Intermediate
            help="Choose the complexity level for your summary."
        )

        if st.button("✨ Get Summary"):
            with st.spinner(f"Generating {summary_level.lower()}-level summary..."):
                # Call summarize_contexts which now returns (summary_text, debug_prompt)
                summary_text, debug_prompt = st.session_state.qa.summarize_contexts(level=summary_level.lower())
                st.markdown("### Your Summary")
                st.write(summary_text)

                if debug_prompt: # Display the prompt for debugging
                    with st.expander("Show Generated Prompt (for debugging)"):
                        st.code(debug_prompt, language="markdown")
            # If the fallback was used due to Granite issues, or if Granite isn't enabled
            if "Error generating summary with Granite" in summary_text:
                 st.error("Summary generation with IBM Granite failed. Displaying fallback summary (if available) or error message.")
            elif not st.session_state.qa.use_granite:
                st.info("IBM Granite is not enabled. Displaying summary from extractive fallback.")
        else:
            st.info("Select a level and click 'Get Summary' to generate a summary of your uploaded PDFs.")

    # Flashcards
    with tabs[2]:
        st.subheader("🃏 Create Flashcards")
        flashcard_k = st.number_input("Number of flashcards to generate", 1, 20, 5, 1, key="flashcard_count")
        if st.button("Create Flashcards", key="gen_flashcards_btn"):
            with st.spinner(f"Generating {flashcard_k} flashcards..."):
                fcs = st.session_state.qa.generate_flashcards(k=int(flashcard_k))
            if fcs:
                st.markdown("### Your Flashcards")
                for i, fc in enumerate(fcs, 1):
                    st.markdown(f"*Q{i}:* {fc['question']}")
                    st.info(f"*A{i}:* {fc['answer']}")
                    st.caption(f"Source: {fc['doc']} — Page {fc['page']}")
                    st.markdown("---")
            else:
                st.warning("No flashcards could be generated. Try uploading more diverse content.")


    # Schedule
    with tabs[3]:
        st.subheader("📆 Generate Study Schedule")
        days_in = st.number_input("Number of days for study", 1, 60, 7, key="study_days")
        if st.button("Generate Schedule", key="gen_schedule_btn"):
            with st.spinner(f"Creating a {days_in}-day study schedule..."):
                schedule = st.session_state.qa.generate_study_schedule(int(days_in))
            if schedule:
                st.markdown("### Your Study Schedule")
                for s in schedule:
                    st.markdown(f"🗓 Day {s['day'].strftime('%Y-%m-%d')}")
                    if s['pages']:
                        for doc, page in s["pages"]:
                            st.write(f"- {doc} — Page {page}")
                    else:
                        st.write("- No content scheduled for this day.")
                st.info("This schedule is a general guide, focusing on page distribution across documents.")
            else:
                st.warning("No schedule could be generated. Please upload PDFs first.")

    # Quiz
    with tabs[4]:
        st.subheader("❓ Quiz from your PDFs")
        cols = st.columns([1, 1, 1])
        with cols[0]:
            quiz_k = st.number_input("How many questions?", 1, 20, 5, 1)
        with cols[1]:
            gen_quiz = st.button("📝 Generate Quiz")
        with cols[2]:
            # Removed the "Get More Quiz Questions" button to simplify user flow and avoid confusion
            pass # Keep this for column alignment if needed, or remove cols[2] if not.

        if gen_quiz: # Simplified logic to only respond to 'Generate Quiz'
            with st.spinner("Generating quiz..."):
                quiz_qs = st.session_state.qa.generate_quiz_questions(k=int(quiz_k))
            if not quiz_qs:
                st.warning("No quiz questions could be generated. If your PDFs are scanned, try enabling OCR or upload richer text.")
            else:
                st.session_state.quiz_data = quiz_qs
                # Initialize user selections as None for each question
                st.session_state.user_quiz_selections = [None] * len(quiz_qs)
                st.session_state.quiz_show_answers = False # Reset this state for a new quiz

        if st.session_state.get("quiz_data"):
            st.divider()
            total = len(st.session_state.quiz_data)

            # Ensure user_quiz_selections list is correctly sized
            if "user_quiz_selections" not in st.session_state or len(st.session_state.user_quiz_selections) != total:
                 st.session_state.user_quiz_selections = [None] * total

            for idx, q in enumerate(st.session_state.quiz_data, 1):
                st.markdown(f"*Q{idx}.* {q['question']}")
                options = q.get("options") or []
                if len(options) < 2:
                    st.warning(f"Skipping Q{idx}: not enough options.")
                    # Ensure selection for this question is None if skipped
                    st.session_state.user_quiz_selections[idx-1] = None
                    continue

                # Use st.radio with index=st.session_state.user_quiz_selections[idx-1]
                # If st.session_state.user_quiz_selections[idx-1] is None, no option is pre-selected.
                # Find the current index of the selected option if it exists, otherwise use None
                current_selection_index = None
                if st.session_state.user_quiz_selections[idx-1] in options:
                    current_selection_index = options.index(st.session_state.user_quiz_selections[idx-1])

                choice = st.radio(
                    "Your answer:",
                    options,
                    index=current_selection_index, # Sets initial selection, None for unselected
                    key=f"quiz_choice_{idx}"
                )
                # Update the session state with the user's current choice
                st.session_state.user_quiz_selections[idx-1] = choice

                # Only show check button, removed "Show Answer"
                cols_q = st.columns([1, 1]) # Adjusted column count
                with cols_q[0]:
                    if st.button(f"✅ Check Q{idx}", key=f"check_{idx}"):
                        if choice is None: # Check if an option was actually selected
                            st.warning("Please select an option before checking.")
                        elif choice == q["answer"]:
                            st.success("Correct ✅")
                        else:
                            st.error(f"❌ Wrong. Correct answer: {q['answer']}")
                with cols_q[1]: # Keep this for source display if desired
                    st.caption(f"Source: {q.get('doc','')} — Page {q.get('page','')}")

                st.markdown("---")

            # Score quiz at the end
            if st.button("🏁 Finish Quiz & Show Score", key="finish_quiz_btn"):
                # Pass the collected user selections to the scoring function
                score_results = st.session_state.qa.score_quiz(st.session_state.user_quiz_selections, st.session_state.quiz_data)
                st.session_state.quiz_show_answers = True
                st.success(f"*Your Score:* {score_results['score']} / {score_results['total']} (Attempted: {score_results['attempted']})")

            # Show all correct answers (after quiz is finished)
            if st.session_state.get("quiz_show_answers"):
                with st.expander("📘 Show All Correct Answers"):
                    for i, q in enumerate(st.session_state.quiz_data, 1):
                        st.markdown(f"*Q{i}.* {q['question']}")
                        st.write(f"*Correct Answer:* {q['answer']}")
                        st.caption(f"Source: {q.get('doc','')} — Page {q.get('page','')}")
                        st.markdown("---")


    # ✅ NEW HISTORY TAB
    with tabs[5]:
        st.subheader("🗂 Q&A History")
        if not st.session_state.qa_history:
            st.info("No questions asked yet.")
        else:
            for i, item in enumerate(reversed(st.session_state.qa_history), 1):
                st.markdown(f"{i}. Question:** {item['question']}")
                st.write(f"*Answer:* {item['answer']}")
                if item.get("citations"):
                    st.caption("Sources: " + ", ".join([f"{c['doc']} p.{c['page']}" for c in item["citations"]]))
                st.caption(f"Asked at: {item['time']}")
                st.markdown("---")

            if st.button("🧹 Clear History"):
                st.session_state.qa_history = []
                st.success("History cleared.")

else:
    st.info("Please upload and process at least one PDF to enable Q&A and other study features.")
