# backend.py
import os
import fitz  # PyMuPDF
import faiss
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from datetime import timedelta, date
from PIL import Image
import re
import random

# optional libs
try:
    import cv2
except Exception:
    cv2 = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    import easyocr
except Exception:
    easyocr = None

# IBM watsonx / Granite (optional)
try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import Model
    WATSONX_AVAILABLE = True
except Exception:
    WATSONX_AVAILABLE = False


# ------------------------------
# Utilities
# ------------------------------
def normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms


def sliding_window_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = " ".join(text.split())
    chunks: List[str] = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(text[start:end])
        if end >= L:
            break
        start = max(0, end - overlap)
    return chunks


@dataclass
class ChunkMeta:
    doc_name: str
    page_num: int
    text: str


# ------------------------------
# Main class
# ------------------------------
class PDF_QA_System:
    def __init__(self, ocr_langs: Optional[List[str]] = None):
        # embeddings & retrieval
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks: List[ChunkMeta] = []
        self.embeddings: Optional[np.ndarray] = None

        # chunking / retrieval defaults
        self.chunk_size = 1200
        self.chunk_overlap = 200
        self.top_k = 5

        # granite / watsonx config
        self.use_granite = True
        self.granite_model_id = os.getenv("WATSONX_MODEL_ID", "ibm/granite-1b-instruct")
        self._granite_model = None

        # OCR readers (initialized on demand)
        self._easy_reader = None
        self.ocr_langs = ocr_langs or ["en"]

        # If TESSERACT_CMD provided via env, configure pytesseract
        tcmd = os.environ.get("TESSERACT_CMD")
        if tcmd and pytesseract is not None:
            pytesseract.pytesseract.tesseract_cmd = tcmd

    def set_params(
        self,
        chunk_size: int,
        chunk_overlap: int,
        top_k: int,
        use_granite: bool = True,
        granite_model: Optional[str] = None
    ):
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.top_k = int(top_k)
        self.use_granite = bool(use_granite)
        if granite_model:
            self.granite_model_id = granite_model

    # ------------------------------
    # OCR helpers
    # ------------------------------
    def __init__easyocr(self):
        if easyocr is None:
            return None
        if self._easy_reader is None:
            try:
                self._easy_reader = easyocr.Reader(self.ocr_langs, gpu=False)
            except Exception:
                self._easy_reader = None
        return self._easy_reader

    def _preprocess_image(self, pil_img: Image.Image) -> Image.Image:
        if cv2 is None:
            return pil_img.convert("L")
        arr = np.array(pil_img.convert("RGB"))[:, :, ::-1]
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        gray = cv2.fastNlMeansDenoising(gray, h=7)
        thr = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10
        )
        return Image.fromarray(thr)

    def _pytesseract_ocr(self, pil_img: Image.Image, langs: Optional[List[str]] = None) -> str:
        if pytesseract is None:
            return ""
        langs = langs or self.ocr_langs
        lang_map = {"en": "eng"}
        tess_langs = "+".join([lang_map.get(l.lower().strip(), l.lower().strip())
                               for l in langs if l.strip()]) or "eng"
        try:
            return pytesseract.image_to_string(pil_img, lang=tess_langs)
        except Exception:
            try:
                return pytesseract.image_to_string(pil_img)
            except Exception:
                return ""

    def _easyocr_ocr(self, pil_img: Image.Image) -> str:
        reader = self.__init__easyocr()
        if reader is None:
            return ""
        try:
            arr = np.array(pil_img.convert("RGB"))
            res = reader.readtext(arr, detail=0)
            return " ".join(res)
        except Exception:
            return ""

    def _ocr_image(self, pil_img: Image.Image, preprocess: bool = True) -> str:
        img_for_ocr = self._preprocess_image(pil_img) if preprocess else pil_img
        text = self._pytesseract_ocr(img_for_ocr)
        if text and text.strip():
            return text
        text = self._easyocr_ocr(img_for_ocr)
        return text

    # ------------------------------
    # Core: Add PDF
    # ------------------------------
    def add_pdf_from_bytes(
        self,
        doc_name: str,
        pdf_bytes: bytes,
        force_ocr: bool = False,
        preprocess_ocr: bool = True,
        ocr_langs: Optional[List[str]] = None
    ):
        if ocr_langs:
            self.ocr_langs = ocr_langs
            self._easy_reader = None

        stream = pdf_bytes
        with fitz.open(stream=stream, filetype="pdf") as doc:
            for page_index, page in enumerate(doc, start=1):
                try:
                    page_text = ""
                    if not force_ocr:
                        page_text = page.get_text("text") or ""
                    if page_text and page_text.strip():
                        extracted = page_text
                    else:
                        zoom = 2.0
                        mat = fitz.Matrix(zoom, zoom)
                        pix = page.get_pixmap(matrix=mat, alpha=False)
                        pil_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                        extracted = self._ocr_image(pil_img, preprocess=preprocess_ocr)
                    if extracted and extracted.strip():
                        for chunk in sliding_window_chunks(extracted, self.chunk_size, self.chunk_overlap):
                            self.chunks.append(ChunkMeta(doc_name=doc_name, page_num=page_index, text=chunk))
                except Exception as e:
                    print(f"[PDF_QA_System] Warning: failed processing page {page_index} of {doc_name}: {e}", flush=True)
                    continue

    # ------------------------------
    # Indexing & retrieval
    # ------------------------------
    def build_or_update_index(self):
        if not self.chunks:
            raise ValueError("No chunks to index.")
        texts = [c.text for c in self.chunks]
        embs = self.model.encode(texts, convert_to_numpy=True).astype("float32")
        embs = normalize(embs)
        self.embeddings = embs
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(self.embeddings)

    def has_index(self) -> bool:
        return self.index is not None and self.index.ntotal > 0

    def reset(self):
        self.index = None
        self.embeddings = None
        self.chunks = []

    def _search(self, query: str, k: int) -> List[int]:
        if not self.has_index():
            raise ValueError("Index not built.")
        q = self.model.encode([query], convert_to_numpy=True).astype("float32")
        q = normalize(q)
        scores, idxs = self.index.search(q, int(k))
        return idxs[0].tolist()

    # ------------------------------
    # Granite (IBM watsonx.ai) integration
    # ------------------------------
    def _get_granite_model(self):
        """
        Initialize and return an IBM watsonx Foundation Model instance for Granite.
        Expects env vars: WATSONX_APIKEY, WATSONX_URL, WATSONX_PROJECT_ID
        Optionally: WATSONX_MODEL_ID
        """
        if self._granite_model:
            return self._granite_model

        api_key = os.getenv("WATSONX_APIKEY")
        url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        project_id = os.getenv("WATSONX_PROJECT_ID")
        model_id = self.granite_model_id or os.getenv("WATSONX_MODEL_ID", "ibm/granite-1b-instruct")

        if not (api_key and project_id) or not WATSONX_AVAILABLE:
            return None

        try:
            creds = Credentials(api_key=api_key, url=url)
            self._granite_model = Model(
                model_id=model_id,
                credentials=creds,
                project_id=project_id,
                params={"decoding_method": "greedy", "max_new_tokens": 2048} # Increased max_new_tokens for longer summaries
            )
            return self._granite_model
        except Exception as e:
            print("[PDF_QA_System] granite init failed:", e, flush=True)
            return None

    # ------------------------------
    # Helper: structured post-processor
    # ------------------------------
    def _format_structured_answer(self, direct: str, contexts: List[ChunkMeta]) -> str:
        """Make a tidy, judge-friendly answer."""
        direct = direct.strip()
        if not direct or len(re.findall(r"[A-Za-z]", direct)) < 5:
            return "I don't know based on the provided PDFs."

        # build compact source list
        seen = []
        for c in contexts:
            key = (c.doc_name, c.page_num)
            if key not in seen:
                seen.append(key)
        srcs = "; ".join(f"{d} p.{p}" for d, p in seen[:6])

        # bulletize long sentences lightly
        bullets = []
        for s in re.split(r'(?<=[.!?])\s+', direct):
            s = s.strip()
            if len(s.split()) >= 4:
                bullets.append(f"• {s}")
        if bullets:
            body = "\n".join(bullets)
        else:
            body = direct

        tail = f"\n\nSources: {srcs}" if srcs else ""
        return f"{body}{tail}"

    # ------------------------------
    # Q&A (improved)
    # ------------------------------
    def _llm_answer(self, question: str, contexts: List[ChunkMeta]) -> str:
        """
        Use Granite when available; otherwise structured extractive answer.
        """
        context_block = "\n\n".join([f"[{i+1}] {c.text}" for i, c in enumerate(contexts)])
        prompt = (
            "You are StudyMate, an academic assistant.\n"
            "Answer ONLY using the provided context. If the answer is not present, say you don't know.\n"
            "Return a short direct answer, then supporting points. Avoid fluff.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n"
            "Answer (concise, bullet points allowed):"
        )

        if self.use_granite:
            model = self._get_granite_model()
            if model:
                try:
                    resp = model.generate_text(prompt=prompt)
                    if isinstance(resp, dict) and "results" in resp and len(resp["results"]) > 0:
                        raw = resp["results"][0].get("generated_text", "").strip()
                        return self._format_structured_answer(raw, contexts)
                    if isinstance(resp, str):
                        return self._format_structured_answer(resp, contexts)
                except Exception as e:
                    print("[PDF_QA_System] granite generate_text failed:", e, flush=True)
                    # fallthrough

        # Fallback: simple extractive scoring using overlap; then format
        q_terms = set(re.findall(r"\w+", question.lower()))
        candidate_sents: List[Tuple[int, str]] = []
        for c in contexts:
            for s in re.split(r'(?<=[.!?])\s+', c.text):
                s_clean = s.strip()
                if not s_clean:
                    continue
                score = sum(1 for w in re.findall(r"\w+", s_clean.lower()) if w in q_terms)
                if score > 0:
                    candidate_sents.append((score, s_clean))

        candidate_sents.sort(key=lambda x: x[0], reverse=True)
        if candidate_sents:
            # direct + top 3 supporting
            direct = candidate_sents[0][1]
            supports = [s for _, s in candidate_sents[1:4]]
            answer_text = direct
            if supports:
                answer_text += "\n" + "\n".join(f"- {s}" for s in supports)
            return self._format_structured_answer(answer_text, contexts)

        return "I don't know based on the provided PDFs."

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Main public method to answer a user question.
        Returns a dict: { "answer": str, "citations": list }
        """
        if not self.has_index():
            return {"answer": "No PDFs indexed yet. Please upload and process first.", "citations": []}

        try:
            # get top-k relevant chunks
            idxs = self._search(question, self.top_k)
            contexts = [self.chunks[i] for i in idxs if i < len(self.chunks)]

            # generate answer
            answer_text = self._llm_answer(question, contexts)

            # build citation list
            citations = [{"doc": c.doc_name, "page": c.page_num} for c in contexts]

            return {"answer": answer_text, "citations": citations}

        except Exception as e:
            return {"answer": f"⚠ Error while answering: {e}", "citations": []}

    # ------------------------------
    # Diversity sampler (MMR) for summaries
    # ------------------------------
    def _mmr_select(self, k: int, lambda_mult: float = 0.65) -> List[int]:
        """
        Select k diverse chunk indices using Maximal Marginal Relevance.
        Requires self.embeddings aligned with self.chunks.
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        # seed with the longest chunk index (often most informative)
        lengths = np.array([len(c.text) for c in self.chunks])
        seed = int(np.argmax(lengths))
        selected = [seed]
        candidates = set(range(len(self.chunks))) - set(selected)

        # similarity matrix (cosine since embeddings are normalized)
        sims = self.embeddings @ self.embeddings.T  # [N, N]

        while len(selected) < min(k, len(self.chunks)):
            best_i = None
            best_score = -1e9
            for i in candidates:
                # relevance: avg similarity to selected seed(s)
                rel = float(np.mean([sims[i, j] for j in selected])) if selected else 0.0
                # diversity: max similarity to already selected (we subtract it)
                div = float(np.max([sims[i, j] for j in selected])) if selected else 0.0
                score = lambda_mult * rel - (1 - lambda_mult) * div
                if score > best_score:
                    best_score = score
                    best_i = i
            if best_i is None:
                break
            selected.append(best_i)
            candidates.remove(best_i)

        return selected

    # ------------------------------
    # Summaries, flashcards, schedule
    # ------------------------------
    def summarize_contexts(self, k: Optional[int] = None, level: str = "intermediate") -> Tuple[str, Optional[str]]:
        """
        Generates a summary of the loaded PDF contexts, with adjustable complexity levels.
        Args:
            k (Optional[int]): Number of chunks to consider for the summary. If None, uses max(self.top_k, 6).
            level (str): The desired complexity level of the summary. Options: "beginner", "intermediate", "expert".
        Returns:
            Tuple[str, Optional[str]]: A tuple containing the generated summary string and the prompt string used (for debugging).
        """
        if not self.has_index():
            return "No index: please process PDFs first.", None

        k = int(k or max(self.top_k, 6))
        if len(self.chunks) == 0:
            return "No documents loaded.", None

        # select diverse representatives rather than only longest ones
        reps_idx = self._mmr_select(k)
        if not reps_idx:
            reps_idx = list(range(min(k, len(self.chunks))))
        representatives = [self.chunks[i] for i in reps_idx]
        text = "\n\n".join(c.text for c in representatives)

        # Define prompt based on the summary level
        prompt_intro = "Create a summary based on the following text snippets."
        if level.lower() == "beginner":
            prompt_style = (
                "Explain this to a 10-year-old. Keep the summary concise, around 300 words. "
                "Use very simple language, short sentences, and focus on the absolute main points. "
                "Provide a clear, bulleted summary."
            )
        elif level.lower() == "expert":
            prompt_style = (
                "Generate a highly comprehensive and analytical summary, approximately 1000 words. "
                "Assume a professional or academic audience. Highlight core arguments, complex relationships, "
                "and critical details. Incorporate technical terms and deep insights where appropriate. "
                "Focus on high-level insights and implications. Structure as dense bullet points or detailed paragraphs."
            )
        else:  # Default to intermediate
            prompt_style = (
                "Provide a detailed study summary, approximately 600 words long, that covers ALL distinct topics "
                "present in the snippets. Use clear and structured language. "
                "Structure as bullet points grouped by theme. Avoid duplicates. Be concrete."
            )

        prompt = (
            f"{prompt_intro}\n"
            f"{prompt_style}\n\n"
            f"Context:\n{text}\n\n" # Explicitly label the context
            "Summary:"
        )

        summary = None
        if self.use_granite:
            model = self._get_granite_model()
            if model:
                try:
                    resp = model.generate_text(prompt=prompt)
                    if isinstance(resp, dict) and "results" in resp and resp['results']:
                        summary = resp["results"][0].get("generated_text", "").strip()
                    elif isinstance(resp, str):
                        summary = resp.strip()
                except Exception as e:
                    print(f"[PDF_QA_System] granite summary for level '{level}' failed:", e, flush=True)
                    return f"Error generating summary with Granite: {e}", prompt
        else:
            print("[PDF_QA_System] Granite is not enabled or not available for summary generation. Falling back to extractive.", flush=True)


        # fallback: build grouped bullets by simple keyword coalescing
        # This fallback is a heuristic to approximate word count if Granite fails or isn't used.
        # It's less precise than an LLM but provides some differentiation.
        if not summary or len(summary.strip().split()) < 50: # Trigger fallback if summary is too short
            sentences = []
            # Gather all sentences from representatives
            for c in representatives:
                sentences.extend([s.strip() for s in re.split(r'(?<=[.!?])\s+', c.text) if len(s.strip()) > 10])

            if sentences:
                avg_words_per_sentence = 20 # Estimate
                if level.lower() == "beginner":
                    target_sentences = min(len(sentences), int(300 / avg_words_per_sentence))
                    summary = "- " + "\n- ".join(sentences[:max(1, target_sentences)]) # Ensure at least 1 sentence
                elif level.lower() == "intermediate":
                    target_sentences = min(len(sentences), int(600 / avg_words_per_sentence))
                    summary = "- " + "\n- ".join(sentences[:max(1, target_sentences)])
                else: # Expert fallback
                    target_sentences = min(len(sentences), int(1000 / avg_words_per_sentence))
                    summary = "- " + "\n- ".join(sentences[:max(1, target_sentences)])
            else:
                summary = "No summary could be generated."
        return summary, prompt

    def generate_flashcards(self, k=None) -> list:
        if not self.has_index():
            return []
        k = int(k or self.top_k)

        # pick diverse, informative chunks
        pick_idx = self._mmr_select(max(k * 3, 12))
        if not pick_idx:
            pick_idx = list(range(min(max(k * 3, 12), len(self.chunks))))
        chunks = [self.chunks[i] for i in pick_idx]

        flashcards = []
        for c in chunks:
            # --- UPDATED PROMPT FOR FLASHCARD ACCURACY (MORE STRICT) ---
            prompt = (
                "Based strictly and only on the academic excerpt provided below, generate exactly ONE flashcard. "
                "The flashcard MUST contain a specific question and its direct, factual answer as stated or clearly implied within this excerpt. "
                "The answer should be the information itself, NOT a rephrasing of the question, and NOT an external question. "
                "Avoid generating questions that are too broad (e.g., 'What is X?'), unless X is directly defined. "
                "Focus on a single, clear, key piece of factual information. "
                "Use this EXACT two-line format, without any extra text or conversational filler:\n"
                "Q: <specific factual question derived from the excerpt>\n"
                "A: <direct, factual answer from the excerpt>\n\n"
                f"Excerpt:\n{c.text}\n\n"
                "Flashcard:"
            )
            # --- END UPDATED PROMPT ---

            qa_pair = None
            if self.use_granite:
                model = self._get_granite_model()
                if model:
                    try:
                        resp = model.generate_text(prompt=prompt)
                        if isinstance(resp, dict) and "results" in resp and resp['results']:
                            qa_pair = resp["results"][0].get("generated_text", "").strip()
                        elif isinstance(resp, str):
                            qa_pair = resp.strip()
                    except Exception as e:
                        print(f"[PDF_QA_System] granite flashcard failed for chunk (page {c.page_num} of {c.doc_name}): {e}", flush=True)

            q, a = None, None
            if qa_pair:
                # robust parser: find the first 'Q:' and the first subsequent 'A:'
                m_q = re.search(r"(^|\n)\s*Q:\s*(.+)", qa_pair, flags=re.IGNORECASE | re.DOTALL)
                m_a = re.search(r"(^|\n)\s*A:\s*(.+)", qa_pair, flags=re.IGNORECASE | re.DOTALL)
                if m_q:
                    q = m_q.group(2).strip().splitlines()[0].strip() # ensure no extra newlines or spaces
                if m_a:
                    a = m_a.group(2).strip().splitlines()[0].strip() # ensure no extra newlines or spaces

            if not (q and a):
                # deterministic fallback - trying to make this more robust too
                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', c.text) if len(s.strip()) > 10]
                if len(sentences) >= 2:
                    # Attempt to form a simple Q&A from the first two relevant sentences
                    q_candidate = sentences[0]
                    a_candidate = sentences[1]

                    # Improved heuristic for fallback question generation
                    if " is " in q_candidate and len(q_candidate.split(' is ', 1)[0].split()) > 1: # check if it defines something
                        subject = q_candidate.split(' is ', 1)[0].strip()
                        q = f"What is {subject}?"
                        a = q_candidate.split(' is ', 1)[1].strip() + " " + a_candidate.strip()
                        if not a.endswith('.'): a += '.'
                        # Also check if a_candidate is a continuation of the definition
                        if len(a.split()) > 50: # prevent excessively long answers from fallback
                            a = q_candidate.split(' is ', 1)[1].strip()
                            if not a.endswith('.'): a += '.'

                    elif re.match(r"^[A-Z][a-z0-9\s,;:'\"-]* uses ", q_candidate):
                         subject = re.match(r"^[A-Z][a-z0-9\s,;:'\"-]*", q_candidate).group(0).strip()
                         q = f"What does {subject} use?"
                         a = q_candidate.split(' uses ', 1)[1].strip()
                         if not a.endswith('.'): a += '.'
                    else: # General fallback
                        q = f"What is a key detail from this part of the text?"
                        a = q_candidate
                elif sentences:
                    q = f"What is the main information presented?"
                    a = sentences[0]
                else: # Last resort fallback
                    q = "What is a core idea?"
                    a = "The provided text discusses academic topics."


            # Final validation to ensure basic quality, prevent trivial Q=A, and ensure meaningfulness
            if q and a and len(q.split()) >= 3 and len(a.split()) >= 3 and q.lower().strip() != a.lower().strip():
                flashcards.append({'question': q, 'answer': a, 'doc': c.doc_name, 'page': c.page_num})

            if len(flashcards) >= k:
                break

        # deduplicate by question
        dedup = []
        seen = set()
        for fc in flashcards:
            if fc['question'] not in seen:
                dedup.append(fc)
                seen.add(fc['question'])
        return dedup[:k]

    def generate_study_schedule(self, total_days=7) -> List[Dict[str, Any]]:
        if not self.chunks:
            return []
        doc_pages = {}
        for c in self.chunks:
            doc_pages.setdefault(c.doc_name, set()).add(c.page_num)
        all_pages = sorted([(doc, page) for doc, pages in doc_pages.items() for page in sorted(pages)])
        days = []
        pages_per_day = max(1, len(all_pages) // total_days)
        today = date.today()
        for i in range(total_days):
            start = i * pages_per_day
            end = start + pages_per_day
            if i == total_days - 1:
                day_pages = all_pages[start:]
            else:
                day_pages = all_pages[start:end]
            days.append({'day': today + timedelta(days=i), 'pages': day_pages})
        return days

    # ------------------------------
    # QUIZ (improved generation)
    # ------------------------------
    def generate_quiz_questions(self, k: int = 5) -> List[Dict[str, Any]]:
        """
        Offline MCQ generation from existing chunks.
        Returns list of dicts: {question, options, answer, answer_index, answer_letter, doc, page}
        """
        if not self.has_index():
            return []

        vocab = self._build_corpus_vocab()
        if not vocab:
            return []

        candidates: List[Dict[str, Any]] = []
        # prioritize diverse chunks
        pick_idx = self._mmr_select(max(k * 4, 16))
        if not pick_idx:
            pick_idx = list(range(min(max(k * 4, 16), len(self.chunks))))
        top_chunks = [self.chunks[i] for i in pick_idx]

        for c in top_chunks:
            for s in self._split_into_sentences(c.text):
                if len(s.split()) >= 8:
                    candidates.append({"doc": c.doc_name, "page": c.page_num, "sent": s})

        random.shuffle(candidates)
        quiz: List[Dict[str, Any]] = []
        used_questions = set()

        for cand in candidates:
            if len(quiz) >= k:
                break
            mcq = self._make_mcq_from_sentence(cand["sent"], vocab)
            if not mcq:
                continue
            qtxt = mcq["question"]
            if qtxt in used_questions:
                continue
            used_questions.add(qtxt)
            mcq["doc"] = cand["doc"]
            mcq["page"] = cand["page"]
            # enrich with answer_index & letter for professional scoring
            labeled = mcq["options"]
            correct = mcq["answer"]
            try:
                idx = labeled.index(correct)
            except ValueError:
                # safety: if mismatch, mark first option as correct
                idx = 0
                correct = labeled[0]
            mcq["answer_index"] = idx
            mcq["answer_letter"] = ["A", "B", "C", "D"][idx] if idx < 4 else "A"
            quiz.append(mcq)

        if not quiz:
            # fallback to T/F style
            tf_candidates = candidates[:k]
            for c in tf_candidates:
                options = ["A) True", "B) False", "C) Not given", "D) Cannot determine"]
                quiz.append({
                    "question": f"True or False: {c['sent']}",
                    "options": options,
                    "answer": "A) True",
                    "answer_index": 0,
                    "answer_letter": "A",
                    "doc": c["doc"],
                    "page": c["page"],
                })
                if len(quiz) >= k:
                    break

        safe_quiz = []
        for q in quiz:
            opts = q.get("options") or []
            ans = q.get("answer", "")
            if len(opts) >= 2 and any(o == ans for o in opts):
                safe_quiz.append(q)

        return safe_quiz[:k]

    # Public helper to score (optional for app.py to use)
    def score_quiz(self, selected: List[Optional[str]], quiz: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        selected: list of user selections like 'A) foo' / None
        quiz: same list returned by generate_quiz_questions
        Returns: {score, attempted, total}
        """
        total = len(quiz)
        attempted = 0
        score = 0
        for i, q in enumerate(quiz):
            user_sel = selected[i] if i < len(selected) else None
            if user_sel:
                attempted += 1
                if user_sel.strip() == q["answer"].strip():
                    score += 1
        return {"score": score, "attempted": attempted, "total": total}

    # ------- helpers for quiz -------
    def _split_into_sentences(self, text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [self._clean_sentence(p) for p in parts if len(p.strip()) > 0]

    def _clean_sentence(self, s: str) -> str:
        s = s.strip()
        s = re.sub(r"\s+", " ", s)
        return s

    _STOPWORDS = {
        "the","is","are","a","an","and","or","of","to","in","on","for","with","by","as","at","that","this",
        "it","its","be","was","were","from","which","these","those","their","there","we","you","they","i",
        "not","also","but","than","then","so","such","into","over","under","between","within","can","may",
        "might","should","would","could","will","shall","have","has","had","if","when","while","where",
        "because","about","across","through","per","via","using","use","used","one","two","three"
    }

    def _build_corpus_vocab(self) -> List[str]:
        vocab = []
        for c in self.chunks:
            tokens = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", c.text)
            for t in tokens:
                tl = t.lower()
                if tl not in self._STOPWORDS and not tl.isdigit():
                    vocab.append(t)
        uniq = list(dict.fromkeys(vocab))
        return uniq

    def _pick_target_term(self, sentence: str) -> Optional[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z\-]{4,}", sentence)
        candidates = [t for t in tokens if t.lower() not in self._STOPWORDS]
        if not candidates:
            return None
        candidates.sort(key=lambda w: (len(w), sum(ch.isupper() for ch in w)), reverse=True)
        return candidates[0]

    def _make_mcq_from_sentence(self, sentence: str, vocab: List[str]) -> Optional[Dict[str, Any]]:
        sent = self._clean_sentence(sentence)
        if len(sent.split()) < 8:
            return None
        target = self._pick_target_term(sent)
        if not target:
            return None

        # blank-out one occurrence
        blanked = re.sub(rf"\b{re.escape(target)}\b", "_", sent, count=1, flags=re.IGNORECASE)
        if blanked == sent:
            blanked = sent.replace(target, "_",
                                   1)

        tgt_lower = target.lower()
        similar_pool = [w for w in vocab if w.lower() != tgt_lower and abs(len(w) - len(target)) <= 3]
        random.shuffle(similar_pool)
        distractors = []
        for w in similar_pool:
            if len(distractors) >= 3:
                break
            if re.search(rf"\b{re.escape(w)}\b", sent, flags=re.IGNORECASE):
                continue
            distractors.append(w)

        while len(distractors) < 3 and vocab:
            fallback = random.choice(vocab)
            if fallback.lower() != tgt_lower and fallback not in distractors and not re.search(
                rf"\b{re.escape(fallback)}\b", sent, flags=re.IGNORECASE
            ):
                distractors.append(fallback)

        options_raw = [target] + distractors[:3]
        options_raw = [o for o in options_raw if o and isinstance(o, str)]
        if len(options_raw) < 2:
            return None

        random.shuffle(options_raw)
        labels = ["A", "B", "C", "D"]
        labeled = [f"{labels[i]}) {opt}" for i, opt in enumerate(options_raw)]
        correct_idx = options_raw.index(target)
        answer_str = labeled[correct_idx]

        return {
            "question": f"Fill in the blank: {blanked}",
            "options": labeled,
            "answer": answer_str
        }
