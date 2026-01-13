# nchrp_blueprint.py
from flask import (
    Blueprint, render_template, request, session, jsonify,
    redirect, url_for, flash, current_app, send_from_directory
)
from typing import List, Dict, Tuple, Any 
import os, re, json, time, hashlib, math
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
from datetime import date, datetime
from openai import OpenAI

# from main2 import ALLOWED_META_EXT
ALLOWED_REPORT_EXT = {".xlsx", ".csv"}

ALLOWED_META_EXT = {
    ".pdf", ".doc", ".docx", ".txt",
    ".png", ".jpg", ".jpeg", ".webp",
    ".csv", ".xlsx"
}

def create_nchrp_blueprint(*, vectorstores, answer_question, client, allowed_emails, mycursor_nchrp, mydb_nchrp, chat_history):
    nchrp_bp = Blueprint("nchrp_bp", __name__)

    # ----------------------------
    # Put your NCHRP constants here
    # ----------------------------
    SUMMARY_COLS = [
        "Test ID",
        "Vendor Name",
        "Sensor model name",
        "Sensor Technology",
        "Stage & Level",
        "Test Center",
        "Test Location (State)",
        "Date of Testing",
        "Ground Truth Source",
    ]

    METRIC_BASE_COLS = [
        "Test ID",
        "Sensor Function",
        "Performance Measure",
        "Measured value (%)",
        "Sample size",
        "Weather (F)",
        "Lighting",
    ]

    OPTIONAL_METRIC_COLS = ["Testing Notes (optional)"]

    # ----------------------------
    # Put ALL your NCHRP helper funcs here
    # (copy/paste from your current code)
    # ----------------------------

    def _norm_col(c) -> str:
        s = "" if c is None else str(c)
        s = s.replace("\u00a0", " ")
        s = re.sub(r"\s+", " ", s).strip()
        key = s.lower()
        _CANON = {
            "test id": "Test ID",
            "vendor name": "Vendor Name",
            "sensor model name": "Sensor model name",
            "sensor technology": "Sensor Technology",
            "stage & level": "Stage & Level",
            "stage and level": "Stage & Level",
            "test center": "Test Center",
            "test location (state)": "Test Location (State)",
            "date of testing": "Date of Testing",
            "ground truth source": "Ground Truth Source",
            "sensor function": "Sensor Function",
            "performance measure": "Performance Measure",
            "measured value (%)": "Measured value (%)",
            "measured value %": "Measured value (%)",
            "sample size": "Sample size",
            "weather (f)": "Weather (F)",
            "lighting": "Lighting",
            "testing notes (optional)": "Testing Notes (optional)",
            "testing notes": "Testing Notes (optional)",
        }
        return _CANON.get(key, s)

    def json_safe(v):
        if v is None:
            return None
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        if isinstance(v, (np.generic,)):
            return json_safe(v.item())
        if isinstance(v, (datetime, date, pd.Timestamp)):
            return v.isoformat()
        if isinstance(v, str):
            s = v.replace("\u00a0", " ").strip()
            return s if s != "" else None
        return v

    def load_nchrp_from_files():
        DATA_DIR = os.path.join(current_app.root_path, "sampleData")
        frames = []
        if not os.path.isdir(DATA_DIR):
            return [], {}

        for fn in os.listdir(DATA_DIR):
            if fn.startswith("~$") or fn.startswith("."):
                continue
            path = os.path.join(DATA_DIR, fn)
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(fn)[1].lower()

            try:
                if ext == ".csv":
                    df = pd.read_csv(path, dtype=object)
                    df.columns = [_norm_col(c) for c in df.columns]
                    df["__source__"] = fn
                    df["__sheet__"] = None
                    frames.append(df)

                elif ext == ".xlsx":
                    xls = pd.ExcelFile(path, engine="openpyxl")
                    for sheet in xls.sheet_names:
                        s = pd.read_excel(xls, sheet_name=sheet, dtype=object)
                        s.columns = [_norm_col(c) for c in s.columns]
                        s = s.loc[:, ~s.columns.astype(str).str.startswith("Unnamed")]
                        s = s.where(pd.notnull(s), None)

                        for c in s.columns:
                            if s[c].dtype == object:
                                s[c] = s[c].apply(lambda x: x.strip() if isinstance(x, str) else x)

                        s = s.dropna(how="all")
                        if s.empty:
                            continue

                        for col in ["Test ID", "Sensor Function", "Stage & Level"]:
                            if col in s.columns:
                                s[col] = s[col].replace("", None).ffill()

                        if "Stage & Level" not in s.columns or s["Stage & Level"].isna().all():
                            continue
                        if "Test ID" not in s.columns or s["Test ID"].isna().all():
                            continue

                        s["__source__"] = fn
                        s["__sheet__"] = sheet
                        frames.append(s)

            except Exception as e:
                print(f"[WARN] Skipping file {fn}: {e}")
                continue

        if not frames:
            return [], {}

        df = pd.concat(frames, ignore_index=True)
        df.columns = [str(c).strip() for c in df.columns]
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        df = df.where(pd.notnull(df), None)

        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].apply(lambda x: x.strip() if isinstance(x, str) else x)

        required = set(SUMMARY_COLS + METRIC_BASE_COLS)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(missing))

        df["Test ID"] = df["Test ID"].replace("", None)
        df["Stage & Level"] = df["Stage & Level"].replace("", None)
        df = df[df["Test ID"].notna() & df["Stage & Level"].notna()]
        df = df[(df["Test ID"].astype(str).str.strip() != "") & (df["Stage & Level"].astype(str).str.strip() != "")]
        if df.empty:
            return [], {}

        df["__key__"] = df["Test ID"].astype(str).str.strip() + "||" + df["Stage & Level"].astype(str).str.strip()

        tests_df = df[SUMMARY_COLS + ["__key__"]].drop_duplicates(subset=["__key__"]).copy()
        tests = []
        for _, r in tests_df.iterrows():
            rec = {k: json_safe(r.get(k)) for k in (SUMMARY_COLS + ["__key__"])}
            for k, v in list(rec.items()):
                if v is None:
                    rec[k] = ""
            tests.append(rec)

        metric_cols = [c for c in (METRIC_BASE_COLS + OPTIONAL_METRIC_COLS) if c in df.columns]
        for extra in ["__source__", "__sheet__"]:
            if extra in df.columns and extra not in metric_cols:
                metric_cols.append(extra)

        metrics_by_key = {}
        for _, row in df[metric_cols + ["__key__"]].iterrows():
            key = str(row["__key__"]).strip()
            metric = {k: json_safe(row.get(k)) for k in metric_cols if k != "Test ID"}
            metrics_by_key.setdefault(key, []).append(metric)

        return tests, metrics_by_key

    # ----------------------------
    # Routes (copy/paste yours)
    # Only change: use allowed_emails, client, vectorstores, mycursor_nchrp, mydb_nchrp from closure
    # ----------------------------

    @nchrp_bp.route("/")
    def index_nchrp():
        chat_history.clear()
        return render_template("index_nchrp.html")

    @nchrp_bp.route("/clear_chat_history", methods=["POST"])
    def clear_chat_history_nchrp():
        chat_history.clear()
        return jsonify({"message": "Chat history cleared successfully"})

    @nchrp_bp.route("/answer_nchrp", methods=["POST"])
    def answer_nchrp():
        user_name = request.form["user_question"]
        user_email = request.form["user_email"].strip().lower()
        user_role = request.form.get("user_role", "").strip().lower()

        if user_email not in allowed_emails:
            flash("Your access is not approved yet.")
            return render_template("index_nchrp.html")

        session["user_role"] = user_role
        session["user_name"] = user_name
        session["user_email"] = user_email
        chat_history.clear()

        return render_template("nchrp_choice.html", user_name=user_name, user_email=user_email, user_role=user_role)

    @nchrp_bp.route("/go_to_clearinghouse", methods=["GET"])
    def go_to_clearinghouse():
        return render_template("answer_nchrp.html", user_name=session.get("user_name", ""))

    @nchrp_bp.route("/submit_question_nchrp", methods=["POST"])
    def submit_question_nchrp():
        ques_input = request.form["quesInput"]
        if ques_input:
            session["question"] = ques_input
            return redirect(url_for("nchrp_bp.display_result_nchrp", user_name=session.get("user_name", "")))
        return redirect(url_for("nchrp_bp.index_nchrp"))

    @nchrp_bp.route("/result/<user_name>")
    def display_result_nchrp(user_name):
        ques_input = session["question"]

        vectorstore = vectorstores["nchrp"]
        answer, sources = answer_question(ques_input, vectorstore)

        prompt = f"In context of transportation answer this: {ques_input}\n What is the answer and provide meta of the answer in the next line:"
        ChipAnswerText = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        ChipAnswer = ChipAnswerText.choices[0].message.content.strip()

        session["answer"] = answer
        session["ChipAnswer"] = ChipAnswer

        chat_history.append({"question": ques_input, "answer": answer, "ChipAnswer": ChipAnswer})
        return render_template("answer_nchrp.html", user_name=session.get("user_name", ""), chat_history=chat_history)

    @nchrp_bp.route("/rating_submission", methods=["POST"])
    def rating_submission_nchrp():
        rating = request.form["rate"]
        rating2 = request.form["rate2"]
        question = session["question"]
        answer = session["answer"]
        user_email = session["user_email"]
        ChipAnswer = session["ChipAnswer"]

        mycursor_nchrp.execute("SELECT * FROM data")
        num_row = len(mycursor_nchrp.fetchall())

        sql = "INSERT INTO data VALUES (%s,%s,%s,%s,%s,%s,%s)"
        mycursor_nchrp.execute(sql, (num_row + 1, user_email, question, answer, rating, ChipAnswer, rating2))
        mydb_nchrp.commit()

        return render_template("answer_nchrp.html", user_name=session.get("user_name", ""), chat_history=chat_history)

    @nchrp_bp.route("/report")
    def testSampleReport():
        tests, metrics = load_nchrp_from_files()
        return render_template(
            "testSample.html",
            column_headers=SUMMARY_COLS,
            tests=tests,
            metrics=metrics,
            user_role=session.get("user_role", "public"),
            user_name=session.get("user_name", "")
        )

    @nchrp_bp.route("/download-template")
    def download_template():
        return send_from_directory(
            directory=os.path.join(current_app.root_path, "sampleData"),
            path="template.csv",
            as_attachment=True
        )

    # TODO: paste your upload_report and ask_ai routes here too (same idea)
    def _ext(name: str) -> str:
      return os.path.splitext(name)[1].lower()
    

    @nchrp_bp.route("/upload_report", methods=["POST"])
    def upload_report():
      sample_dir = os.path.join(current_app.root_path, "sampleData")
      upload_dir = os.path.join(current_app.root_path, "uploads")
      meta_dir = os.path.join(upload_dir, "metadata")

      os.makedirs(sample_dir, exist_ok=True)
      os.makedirs(upload_dir, exist_ok=True)
      os.makedirs(meta_dir, exist_ok=True)

      report = request.files.get("report_file")
      meta = request.files.get("metadata_file")

      # --- required report file ---
      if not report or not report.filename:
          flash("Please upload the completed template file (.xlsx or .csv).")
          return redirect(url_for("nchrp_bp.testSampleReport"))

      report_name = secure_filename(report.filename)
      if _ext(report_name) not in ALLOWED_REPORT_EXT:
          flash("Report file must be .xlsx or .csv.")
          return redirect(url_for("nchrp_bp.testSampleReport"))

      # Save report into sampleData (so it shows on the report page)
      report_path = os.path.join(sample_dir, report_name)
      report.save(report_path)

      # --- optional metadata file ---
      if meta and meta.filename:
          meta_name = secure_filename(meta.filename)
          if _ext(meta_name) not in ALLOWED_META_EXT:
              flash("Metadata file type not supported. Try PDF/DOC/DOCX/TXT/Images.")
              return redirect(url_for("nchrp_bp.testSampleReport"))

          # store with timestamp to avoid overwriting
          ts = time.strftime("%Y%m%d-%H%M%S")
          meta_path = os.path.join(meta_dir, f"{ts}__{meta_name}")
          meta.save(meta_path)

      flash("Upload successful!")
      return redirect(url_for("nchrp_bp.testSampleReport"))
    
    def dbg(label, value=None):
      print("\n" + "="*80)
      print(f"[DEBUG] {label}")
      if value is not None:
          if isinstance(value, (dict, list)):
              print(json.dumps(value, indent=2)[:5000])
          else:
              print(value)
      print("="*80)



# -----------------------------
# CONFIG
# -----------------------------
    DATA_FOLDER = os.path.join(os.getcwd(), "sampleData")  # <-- your folder
    EMBED_MODEL = "text-embedding-3-small"
    CHAT_MODEL = "gpt-4o-mini"

    # retrieval strictness (tune)
    VOCAB_OVERLAP_MIN = 1          # require >=1 DB term overlap to avoid gibberish
    TOP_K = 30                     # rows returned as evidence
    SIM_THRESHOLD = 0.22           # higher = stricter, fewer matches

    # embedding cache controls
    CACHE_TTL_SECONDS = 10 * 60    # rebuild embeddings every 10 minutes
    MAX_ROWS_FOR_EMBED = 25000     # safety cap

# -----------------------------
# GLOBAL CACHES (in-memory)
# -----------------------------
    _CACHE = {
        "built_at": 0.0,
        "fingerprint": None,
        "columns": [],
        "rows": [],
        "docs": [],
        "doc_embs": None,          # np.ndarray shape (N, D)
        "allowed_terms": set(),
        "row_count": 0,
    }

# -----------------------------
# Helpers: normalization
# -----------------------------
    def clean_col(c: str) -> str:
        return re.sub(r"\s+", " ", str(c)).strip()

    def norm_str(x) -> str:
        if x is None:
            return ""
        if isinstance(x, float) and np.isnan(x):
            return ""
        return str(x).strip()

    def tokenize(text: str) -> List[str]:
        # keeps tokens like CAL-016
        return re.findall(r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)?", (text or "").lower())

    def infer_stage_level_from_sheet(sheet_name: str) -> str:
        s = (sheet_name or "").strip()

        m = re.search(r"stage\D*(\d+)\D*level\D*(\d+)", s, flags=re.I)
        if m:
            return f"Stage {m.group(1)} Level {m.group(2)}"

        m = re.search(r"\bs(\d+)\s*l(\d+)\b", s, flags=re.I)
        if m:
            return f"Stage {m.group(1)} Level {m.group(2)}"

        return s  # fallback: keep sheet name

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def read_workbook_all_sheets(filepath: str) -> List[Dict[str, Any]]:
        xls = pd.ExcelFile(filepath)
        all_rows: List[Dict[str, Any]] = []

        # 1) First pass: find metadata from any sheet that has it
        meta_keys = [
            "Test ID", "Vendor Name", "Sensor model name", "Sensor Technology",
            "Test Center", "Test Location (State)", "Date of Testing", "Ground Truth Source"
        ]
        meta: Dict[str, str] = {}

        for sheet in xls.sheet_names:
            df_meta = pd.read_excel(filepath, sheet_name=sheet)
            df_meta = df_meta.loc[:, ~df_meta.columns.astype(str).str.startswith("Unnamed:")]
            df_meta.columns = [clean_col(c) for c in df_meta.columns]
            df_meta = df_meta.dropna(how="all")
            if df_meta.empty:
                continue

            # normalize values
            for c in df_meta.columns:
                df_meta[c] = df_meta[c].apply(norm_str)

            # find first row that contains any metadata fields
            for _, r in df_meta.iterrows():
                row = r.to_dict()
                found_any = False
                for k in meta_keys:
                    if k in df_meta.columns and norm_str(row.get(k)):
                        meta[k] = norm_str(row.get(k))
                        found_any = True
                if found_any:
                    # don’t break fully — keep scanning other sheets to fill missing keys
                    pass

        # 2) Second pass: read sheets and apply metadata fill
        for sheet in xls.sheet_names:
            df = pd.read_excel(filepath, sheet_name=sheet)
            df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]
            df.columns = [clean_col(c) for c in df.columns]
            df = df.dropna(how="all")
            if df.empty:
                continue

            for c in df.columns:
                df[c] = df[c].apply(norm_str)

            stage_level = infer_stage_level_from_sheet(sheet)
            rows = df.to_dict(orient="records")

            for r in rows:
                # Fill metadata if missing on that row
                for k, v in meta.items():
                    if k not in r or not norm_str(r.get(k)):
                        r[k] = v

                r["Source File"] = os.path.basename(filepath)
                r["Source Sheet"] = sheet
                if not norm_str(r.get("Stage & Level")):
                    r["Stage & Level"] = stage_level

            all_rows.extend(rows)

        return all_rows


    def load_all_excels(folder_path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        all_rows: List[Dict[str, Any]] = []

        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"DATA_FOLDER not found: {folder_path}")

        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(".xlsx"):
                continue
            if fname.startswith("~$"):  # skip Excel temp files
                continue
            fp = os.path.join(folder_path, fname)
            all_rows.extend(read_workbook_all_sheets(fp))

        # union of columns
        cols = set()
        for r in all_rows:
            cols.update(r.keys())

        preferred = [
            "Test ID", "Stage & Level", "Test Center", "Test Location (State)",
            "Sensor Technology", "Vendor Name", "Sensor model name",
            "Sensor Function", "Performance Measure", "Measured value (%)",
            "Sample size", "Weather (F)", "Lighting", "Testing Notes (optional)",
            "Source File", "Source Sheet"
        ]
        columns = [c for c in preferred if c in cols] + sorted([c for c in cols if c not in preferred])

        return columns, all_rows

# -----------------------------
# Build searchable “docs” per row + allowed vocabulary
# -----------------------------
    def row_to_doc(row: Dict[str, Any], columns: List[str]) -> str:
        # Keep the doc compact but descriptive.
        keep_cols = [
            "Test ID", "Stage & Level", "Test Center", "Test Location (State)",
            "Sensor Technology", "Vendor Name", "Sensor model name",
            "Ground Truth Source",
            "Sensor Function", "Performance Measure",
            "Measured value (%)", "Sample size", "Weather (F)", "Lighting",
            "Testing Notes (optional)",
            "Source File", "Source Sheet"
        ]
        keep = [c for c in keep_cols if c in columns]
        parts = []
        for c in keep:
            v = norm_str(row.get(c))
            if v:
                parts.append(f"{c}: {v}")
        return " | ".join(parts)

    def build_allowed_terms(rows: List[Dict[str, Any]], columns: List[str]) -> set:
        """
        DB-driven gibberish filter.
        If a question has zero overlap with these terms, we refuse.
        """
        important_cols = [
            "Test ID", "Test Center", "Test Location (State)",
            "Sensor Technology", "Vendor Name", "Sensor model name",
            "Stage & Level", "Sensor Function", "Performance Measure"
        ]
        existing = [c for c in important_cols if c in columns]

        terms = set()

        for r in rows:
            for c in existing:
                v = norm_str(r.get(c))
                if not v:
                    continue

                # full phrase
                terms.add(v.lower())

                # token pieces
                for t in tokenize(v):
                    if len(t) >= 3:
                        terms.add(t)

        # include column name tokens as well
        for c in existing:
            for t in tokenize(c):
                if len(t) >= 3:
                    terms.add(t)

        return terms

# -----------------------------
# Fingerprint folder (to rebuild cache when files change)
# -----------------------------
    def folder_fingerprint(folder_path: str) -> str:
        """
        Hash of filenames + mtime + size. Cheap and good enough.
        """
        h = hashlib.sha256()
        for fname in sorted(os.listdir(folder_path)):
            if not fname.lower().endswith(".xlsx") or fname.startswith("~$"):
                continue
            fp = os.path.join(folder_path, fname)
            st = os.stat(fp)
            h.update(fname.encode("utf-8"))
            h.update(str(st.st_mtime_ns).encode("utf-8"))
            h.update(str(st.st_size).encode("utf-8"))
        return h.hexdigest()

# -----------------------------
# Build / refresh embedding cache
# -----------------------------
    def ensure_cache_fresh():
        now = time.time()

        # Refresh if TTL expired or fingerprint changed
        fp = folder_fingerprint(DATA_FOLDER)
        ttl_expired = (now - _CACHE["built_at"]) > CACHE_TTL_SECONDS
        fp_changed = (_CACHE["fingerprint"] != fp)

        if _CACHE["doc_embs"] is not None and not ttl_expired and not fp_changed:
            return

        columns, rows = load_all_excels(DATA_FOLDER)
        if not rows:
            # keep empty cache
            _CACHE.update({
                "built_at": now,
                "fingerprint": fp,
                "columns": columns,
                "rows": [],
                "docs": [],
                "doc_embs": None,
                "allowed_terms": set(),
                "row_count": 0
            })
            return

        # safety cap
        if len(rows) > MAX_ROWS_FOR_EMBED:
            rows = rows[:MAX_ROWS_FOR_EMBED]

        docs = [row_to_doc(r, columns) for r in rows]
        allowed_terms = build_allowed_terms(rows, columns)

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Embed in chunks to avoid huge requests
        all_embs = []
        BATCH = 512
        for i in range(0, len(docs), BATCH):
            batch_docs = docs[i:i+BATCH]
            emb = client.embeddings.create(model=EMBED_MODEL, input=batch_docs)
            batch_embs = [np.array(e.embedding, dtype=np.float32) for e in emb.data]
            all_embs.extend(batch_embs)

        doc_embs = np.vstack(all_embs)  # shape (N, D)

        _CACHE.update({
            "built_at": now,
            "fingerprint": fp,
            "columns": columns,
            "rows": rows,
            "docs": docs,
            "doc_embs": doc_embs,
            "allowed_terms": allowed_terms,
            "row_count": len(rows),
        })

    @nchrp_bp.route("/ask_ai", methods=["POST"])
    def ask_ai():
        payload = request.json or {}
        question = norm_str(payload.get("question", ""))
        debug = bool(payload.get("debug", False))

        if not question:
            return jsonify({"error": "No question provided"}), 400

        # 1) Ensure cache exists (loads all excel files + builds embeddings)
        try:
            ensure_cache_fresh()
        except Exception as e:
            return jsonify({"error": f"Data loading error: {e}"}), 500

        dbg("Cache Status", {
        "row_count": _CACHE["row_count"],
        "columns": _CACHE["columns"],
        "cache_built_at": _CACHE["built_at"],
        })

        columns = _CACHE["columns"]
        rows = _CACHE["rows"]
        allowed_terms = _CACHE["allowed_terms"]
        doc_embs = _CACHE["doc_embs"]

        if not rows or doc_embs is None:
            return jsonify({"answer": "I don't know.", "columns": columns, "matched_rows": []})

        # 2) Gibberish / unrelated guard: must overlap DB vocabulary
        q_tokens = tokenize(question)
        overlap = [t for t in q_tokens if t in allowed_terms]
        dbg("Question Tokens", q_tokens)
        dbg("Vocab Overlap", overlap)

        if len(overlap) < VOCAB_OVERLAP_MIN:
            out = {"answer": "I don't know.", "columns": columns, "matched_rows": []}
            if debug:
                out["debug"] = {
                    "reason": "no_vocab_overlap",
                    "question_tokens": q_tokens[:30],
                    "overlap": overlap[:30],
                    "vocab_overlap_min": VOCAB_OVERLAP_MIN
                }
            return jsonify(out)

        # 3) Embed the question and retrieve top rows
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        q_emb_resp = client.embeddings.create(model=EMBED_MODEL, input=[question])
        q_emb = np.array(q_emb_resp.data[0].embedding, dtype=np.float32)
        dbg("Question Embedding Norm", float(np.linalg.norm(q_emb)))

        # cosine similarity against all row embeddings
        # sim = (A·q) / (||A|| ||q||) for each row A
        q_norm = float(np.linalg.norm(q_emb)) or 1.0
        A = doc_embs
        A_norms = np.linalg.norm(A, axis=1)
        A_norms[A_norms == 0] = 1.0
        sims = (A @ q_emb) / (A_norms * q_norm)

        # top indices
        top_idx = np.argsort(-sims)[:TOP_K]
        top_pairs = [(float(sims[i]), int(i)) for i in top_idx]
        dbg(
        "Top Similarities (Before Threshold)",
        [
            {
                "rank": i + 1,
                "similarity": float(s),
                "Test ID": rows[idx].get("Test ID"),
                "Sensor Function": rows[idx].get("Sensor Function"),
                "Performance Measure": rows[idx].get("Performance Measure"),
                "Source File": rows[idx].get("Source File"),
            }
            for i, (s, idx) in enumerate(top_pairs[:10])
        ]
        )

        # threshold filter
        kept = [(s, i) for s, i in top_pairs if s >= SIM_THRESHOLD]
        dbg(
        "Rows After Similarity Threshold",
        [
            {
                "similarity": float(s),
                "Test ID": rows[i].get("Test ID"),
                "Stage & Level": rows[i].get("Stage & Level"),
                "Sensor Function": rows[i].get("Sensor Function"),
                "Performance Measure": rows[i].get("Performance Measure"),
            }
            for s, i in kept[:10]
        ]
        )

        if not kept:
            out = {"answer": "I don't know.", "columns": columns, "matched_rows": []}
            if debug:
                out["debug"] = {
                    "reason": "semantic_no_hits",
                    "best_similarity": top_pairs[0][0] if top_pairs else None,
                    "threshold": SIM_THRESHOLD,
                    "top_10": top_pairs[:10],
                }
            return jsonify(out)

        matched_rows = [rows[i] for _, i in kept]
        dbg("Matched Rows Sent to GPT", matched_rows[:5])

        # 4) Ask GPT: MUST answer only from matched rows
        prompt = f"""
    You are answering questions about an Excel database of NCHRP sensor testing.

    User question:
    {question}

    Matched Rows (ONLY source of truth):
    {json.dumps(matched_rows[:50], indent=2)}

    Rules:
    - Answer ONLY using facts present in Matched Rows.
    - If the answer cannot be determined from Matched Rows, reply exactly: I don't know.
    - Do not invent, assume, or generalize beyond the data shown.
    - If helpful, group by Stage & Level or Test Center / Sensor Technology.
    - Be concise.
    """.strip()
        dbg("Final GPT Prompt", prompt[:4000])

        reply = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        answer = norm_str(reply.choices[0].message.content) or "I don't know."

        out = {
            "answer": answer,
            "columns": columns,
            "matched_rows": matched_rows
        }

        if debug:
            out["debug"] = {
                "reason": "ok",
                "row_count_total": _CACHE["row_count"],
                "cache_built_at": _CACHE["built_at"],
                "vocab_overlap": overlap[:25],
                "threshold": SIM_THRESHOLD,
                "top_similarities": [s for s, _ in kept[:10]],
                "matched_count": len(matched_rows),
            }
        dbg("Raw GPT Answer", reply.choices[0].message.content)

        return jsonify(out)


    return nchrp_bp
