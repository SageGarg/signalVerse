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

#below lists are used by upload report 
ALLOWED_REPORT_EXT = {".xlsx", ".csv"}

ALLOWED_META_EXT = {
    ".pdf", ".doc", ".docx", ".txt",
    ".png", ".jpg", ".jpeg", ".webp",
    ".csv", ".xlsx"
}

def create_nchrp_blueprint(*, vectorstores, answer_question, client, allowed_emails, mycursor_nchrp, mydb_nchrp, chat_history):
    nchrp_bp = Blueprint("nchrp_bp", __name__)

    # ----------------------------
    # used by load_nchrp_from_files()
    # ----------------------------
    # same for entire sheet
    SUMMARY_COLS = [
        "Test ID",
        "Vendor Name",
        "Sensor model name",
        "Sensor Technology",
        "Stage & Level",
        "Test Center",
        "Test Location (State)",
        "Date of Testing",
        # "Ground Truth Source",
    ]

    # NEW (long-format metrics)
    METRIC_BASE_COLS = [
        "Test ID",
        "Sensor Function",
        "Performance Measure",
        "Field Name",
        "Field Value",
    ]


    OPTIONAL_METRIC_COLS = ["Testing Notes (optional)"]

# if the excel file has some inconsistences in column naming
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
            # "ground truth source": "Ground Truth Source",
            "sensor function": "Sensor Function",
            "performance measure": "Performance Measure",
            "measured value (%)": "Measured value (%)",
            "measured value %": "Measured value (%)",
            "sample size": "Sample size",
            "weather (f)": "Weather (F)",
            "lighting": "Lighting",
            "testing notes (optional)": "Testing Notes (optional)",
            "testing notes": "Testing Notes (optional)",
            "field name": "Field Name",
            "field value": "Field Value",
            "field_value": "Field Value",
            "field_name": "Field Name",
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

                        for col in ["Test ID", "Sensor Function","Performance Measure", "Stage & Level"]:
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

        # ✅ Build metrics_by_key ONCE
        metrics_by_key = {}

        group_cols = ["__key__", "Sensor Function", "Performance Measure"]
        keep_cols = [
            "__key__", "Sensor Function", "Performance Measure",
            "Field Name", "Field Value", "__source__", "__sheet__"
        ]
        keep_cols = [c for c in keep_cols if c in df.columns]

        work = df[keep_cols].copy()

        # drop rows where KV pair is missing
        work["Field Name"] = work["Field Name"].apply(norm_str)
        work["Field Value"] = work["Field Value"].apply(norm_str)
        work = work[(work["Field Name"] != "") & (work["Field Value"] != "")]

        for (k, sf, pm), g in work.groupby(group_cols, dropna=False):
            k  = norm_str(k)
            sf = norm_str(sf) or "Unknown"
            pm = norm_str(pm) or "Unknown"

            fields = {}
            for _, r in g.iterrows():
                fname = norm_str(r.get("Field Name"))
                fval  = json_safe(r.get("Field Value"))
                if fname:
                    fields[fname] = fval  # last wins

            metrics_by_key.setdefault(k, []).append({
                "Sensor Function": sf,
                "Performance Measure": pm,
                "fields": fields,
                "__source__": g.iloc[0].get("__source__"),
                "__sheet__": g.iloc[0].get("__sheet__"),
            })

        return tests, metrics_by_key



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
        meta_map = load_meta_index()
        return render_template(
            "testSample.html",
            column_headers=SUMMARY_COLS,
            tests=tests,
            metrics=metrics,
            meta_map=meta_map,
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

    
    def _ext(name: str) -> str:
      return os.path.splitext(name)[1].lower()
    
    def meta_index_path():
        upload_dir = os.path.join(current_app.root_path, "uploads")
        meta_dir = os.path.join(upload_dir, "metadata")
        os.makedirs(meta_dir, exist_ok=True)
        return os.path.join(meta_dir, "index.json")

    def load_meta_index() -> dict:
        p = meta_index_path()
        if not os.path.isfile(p):
            return {}
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception:
            return {}

    def save_meta_index(d: dict) -> None:
        p = meta_index_path()
        with open(p, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)




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
          saved_meta_filename = f"{ts}__{meta_name}"
          meta_path = os.path.join(meta_dir, saved_meta_filename)
          meta.save(meta_path)


          idx = load_meta_index()
          idx[report_name] = saved_meta_filename
          save_meta_index(idx)

      flash("Upload successful!")
      return redirect(url_for("nchrp_bp.testSampleReport"))
    
    def dbg(label, value=None, max_len=3000):
        print("\n" + "=" * 90)
        print(f"[ASK_AI DEBUG] {label}")
        if value is not None:
            try:
                txt = json.dumps(value, indent=2, default=str)
                print(txt[:max_len])
            except Exception:
                print(str(value)[:max_len])
        print("=" * 90)

    # -----------------------------
    # ASK_AI (Tiered JSON) CONFIG
    # -----------------------------
    DATA_FOLDER = os.path.join(os.getcwd(), "sampleData")  # <-- your folder

    EMBED_MODEL = "text-embedding-3-small"
    CHAT_MODEL  = "gpt-4o-mini"

    TOP_K_KEYS = 6            # number of (TestID||Stage) groups to use as evidence
    TOP_K_ROWS = 60           # number of row-docs used for scoring
    SIM_THRESHOLD = 0.20      # tune if needed; lower = more inclusive

    ASKAI_CACHE = {
        "built_at": 0.0,
        "fingerprint": None,
        "tests": [],
        "metrics_by_key": {},
        "flat_rows": [],      # each row: merged summary+metric + __key__
        "docs": [],           # doc string per flat row
        "doc_embs": None,     # np.ndarray (N, D)
        "row_to_key": [],     # list mapping doc index -> __key__
    }


    @nchrp_bp.route("/download-metadata/<path:filename>")
    def download_metadata(filename):
        upload_dir = os.path.join(current_app.root_path, "uploads")
        meta_dir = os.path.join(upload_dir, "metadata")
        return send_from_directory(meta_dir, filename, as_attachment=True)

    def folder_fingerprint_sampledata() -> str:
        """Hash filenames + mtime + size for all supported files in sampleData."""
        data_dir = os.path.join(current_app.root_path, "sampleData")
        h = hashlib.sha256()
        if not os.path.isdir(data_dir):
            return "NO_DIR"
        for fname in sorted(os.listdir(data_dir)):
            if fname.startswith("~$") or fname.startswith("."):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in [".xlsx", ".csv"]:
                continue
            fp = os.path.join(data_dir, fname)
            try:
                st = os.stat(fp)
            except OSError:
                continue
            h.update(fname.encode("utf-8"))
            h.update(str(st.st_mtime_ns).encode("utf-8"))
            h.update(str(st.st_size).encode("utf-8"))
        return h.hexdigest()

    def norm_str(x) -> str:
        if x is None:
            return ""
        try:
            if pd.isna(x):
                return ""
        except Exception:
            pass
        return str(x).strip()

    def row_doc(r: dict) -> str:
        parts = []
        for c in [
            "Test ID", "Stage & Level", "Vendor Name", "Sensor model name", "Sensor Technology",
            "Test Center", "Test Location (State)", "Date of Testing",
            "Sensor Function", "Performance Measure",
        ]:
            v = norm_str(r.get(c))
            if v:
                parts.append(f"{c}: {v}")

        # NEW: add key-values if present
        fields = r.get("fields") or {}
        if isinstance(fields, dict) and fields:
            # keep it compact
            kv = ", ".join([f"{k}={norm_str(v)}" for k, v in list(fields.items())[:12]])
            parts.append(f"Fields: {kv}")

        return " | ".join(parts)


    def build_flat_rows(tests, metrics_by_key):
        """Merge summary + metric rows into flat rows for retrieval."""
        test_by_key = {t["__key__"]: t for t in tests}
        flat = []
        for key, metrics in metrics_by_key.items():
            base = dict(test_by_key.get(key, {}))
            for m in metrics:
                rr = dict(base)
                rr.update(m)
                rr["__key__"] = key
                flat.append(rr)
        return flat

    def tiered_json_for_key(key: str, tests, metrics_by_key) -> dict:
      test_by_key = {t["__key__"]: t for t in tests}
      base = dict(test_by_key.get(key, {}))

      clean_base = {k: v for k, v in base.items() if norm_str(v) != "" and k != "__key__"}

      rows = metrics_by_key.get(key, [])

      grouped = {}
      for r in rows:
          sf = norm_str(r.get("Sensor Function")) or "Unknown"
          pm = norm_str(r.get("Performance Measure")) or "Unknown"

          entry = {
              "Performance Measure": pm,
              "fields": r.get("fields", {}),
          }

          # keep provenance if you want
          if norm_str(r.get("__source__")):
              entry["__source__"] = r.get("__source__")
          if norm_str(r.get("__sheet__")):
              entry["__sheet__"] = r.get("__sheet__")

          grouped.setdefault(sf, []).append(entry)

      out = dict(clean_base)
      out["sensor_functions"] = grouped
      return out


    def ensure_askai_cache_fresh():
        """Build/refresh: tests + metrics + flat docs + embeddings."""
        fp = folder_fingerprint_sampledata()
        if ASKAI_CACHE["fingerprint"] == fp and ASKAI_CACHE["doc_embs"] is not None:
            return

        # 1) Use the SAME loader as /report
        tests, metrics_by_key = load_nchrp_from_files()

        flat_rows = build_flat_rows(tests, metrics_by_key)
        docs = [row_doc(r) for r in flat_rows]
        row_to_key = [r.get("__key__", "") for r in flat_rows]

        # handle empty
        if not docs:
            ASKAI_CACHE.update({
                "built_at": time.time(),
                "fingerprint": fp,
                "tests": tests,
                "metrics_by_key": metrics_by_key,
                "flat_rows": [],
                "docs": [],
                "doc_embs": None,
                "row_to_key": [],
            })
            return

        # 2) Embed docs using injected client (IMPORTANT: do not create new OpenAI())
        all_embs = []
        BATCH = 256
        for i in range(0, len(docs), BATCH):
            batch = docs[i:i+BATCH]
            emb = client.embeddings.create(model=EMBED_MODEL, input=batch)
            batch_embs = [np.array(e.embedding, dtype=np.float32) for e in emb.data]
            all_embs.extend(batch_embs)

        doc_embs = np.vstack(all_embs)

        ASKAI_CACHE.update({
            "built_at": time.time(),
            "fingerprint": fp,
            "tests": tests,
            "metrics_by_key": metrics_by_key,
            "flat_rows": flat_rows,
            "docs": docs,
            "doc_embs": doc_embs,
            "row_to_key": row_to_key,
        })

    def cosine_topk_keys(question: str):
        """Return top matching __keys__ using embedding similarity over flat row docs."""
        ensure_askai_cache_fresh()
        if ASKAI_CACHE["doc_embs"] is None or not ASKAI_CACHE["docs"]:
            return []

        q_emb_resp = client.embeddings.create(model=EMBED_MODEL, input=[question])
        q_emb = np.array(q_emb_resp.data[0].embedding, dtype=np.float32)

        A = ASKAI_CACHE["doc_embs"]
        q_norm = float(np.linalg.norm(q_emb)) or 1.0
        A_norms = np.linalg.norm(A, axis=1)
        A_norms[A_norms == 0] = 1.0
        sims = (A @ q_emb) / (A_norms * q_norm)

        # Take top row matches first
        top_idx = np.argsort(-sims)[:TOP_K_ROWS]
        # ---- DEBUG: Top matching rows ----
        top_debug = []
        for idx in top_idx[:10]:  # only top 10 for readability
            r = ASKAI_CACHE["flat_rows"][int(idx)]
            top_debug.append({
                "similarity": float(sims[idx]),
                "Test ID": r.get("Test ID"),
                "Stage & Level": r.get("Stage & Level"),
                "Sensor Function": r.get("Sensor Function"),
                "Performance Measure": r.get("Performance Measure"),
                "Measured value (%)": r.get("Measured value (%)"),
                "Source File": r.get("__source__"),
                "Source Sheet": r.get("__sheet__"),
            })

        dbg("Top flat rows by similarity", top_debug)

        # Aggregate scores per key (so we return best TestID||Stage groups)
        key_scores = {}
        for idx in top_idx:
            s = float(sims[idx])
            if s < SIM_THRESHOLD:
                continue
            k = ASKAI_CACHE["row_to_key"][int(idx)]
            if not k:
                continue
            key_scores[k] = max(key_scores.get(k, 0.0), s)

        # sort keys by best score
        ranked = sorted(key_scores.items(), key=lambda x: x[1], reverse=True)
        dbg("Aggregated TestID||Stage scores", ranked)

        return [k for k, _ in ranked[:TOP_K_KEYS]]

    @nchrp_bp.route("/ask_ai", methods=["POST"])
    def ask_ai():
        payload = request.json or {}
        question = norm_str(payload.get("question", ""))
        debug = bool(payload.get("debug", False))

        if not question:
            return jsonify({"error": "No question provided"}), 400

        # 1) retrieve best keys (TestID||Stage groups)
        try:
            keys = cosine_topk_keys(question)
        except Exception as e:
            return jsonify({"error": f"Data loading error: {e}"}), 500

        if not keys:
            return jsonify({
                "answer": "I don't know.",
                "matched_tests": [],
                "debug": {"reason": "no_hits"} if debug else None
            })

        tests = ASKAI_CACHE["tests"]
        metrics_by_key = ASKAI_CACHE["metrics_by_key"]

        # 2) Build tiered JSON evidence per key (THIS fixes your grouping issue)
        evidence = [tiered_json_for_key(k, tests, metrics_by_key) for k in keys]
        dbg("Tiered JSON evidence sent to GPT", evidence)

        # 3) Ask GPT with strict rules: use tiered JSON ONLY
        prompt = f"""
    You are answering questions about NCHRP sensor testing data.
    Use ONLY the provided JSON as your source of truth.

    User question:
    {question}

    Evidence JSON (grouped by Sensor Function -> list of Performance Measures):
    {json.dumps(evidence, indent=2)}

    Rules:
    - Answer ONLY using facts present in Evidence JSON.
    - If the answer cannot be determined from the JSON, reply exactly: I don't know.
    - Do not guess missing values.
    - When relevant, cite the Test ID and Stage & Level you used.
    - If the question asks about a specific Sensor Function, ONLY use that group.
    - Be concise, but include key numbers.
    """.strip()

        reply = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        answer = norm_str(reply.choices[0].message.content) or "I don't know."

        out = {
            "answer": answer,
            "matched_tests": evidence,  # tiered output
        }

        if debug:
            out["debug"] = {
                "matched_keys": keys,
                "cache_fingerprint": ASKAI_CACHE["fingerprint"],
                "cache_built_at": ASKAI_CACHE["built_at"],
                "top_k_keys": TOP_K_KEYS,
                "top_k_rows": TOP_K_ROWS,
                "sim_threshold": SIM_THRESHOLD,
            }

        return jsonify(out)



    return nchrp_bp
