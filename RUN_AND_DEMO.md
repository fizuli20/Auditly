# Auditly Demo Guide

## 1. Run the app (before the judges arrive)

Open a terminal in this folder and run:

```bash
cd "/path/to/Auditly"
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

- Your browser will open to something like: **http://localhost:8501**
- The included Streamlit config binds the app to `127.0.0.1` for local demos.
- Leave this terminal and browser tab open.
- Use Python 3.10, 3.11, or 3.12. Newer Python versions may not have compatible ML wheels yet.

---

## 2. Demo script (what to do and say)

Do this in order. Use the **sample document** so you do not need a PDF or an embedding model download.

### Step 1 – Load the sample (no PDF, no download)

1. In the **sidebar**, click **“Load sample document (demo)”**.
2. Wait until you see: **“Sample audit document loaded. X chunks indexed.”**

**Say:**  
*"The app can run locally. I am loading a built-in sample audit report so we do not need to upload a file or wait for a model download."*

---

### Step 2 – Ask a question (preset button)

1. Click one of the preset buttons, e.g. **“List audit findings”**.
2. Click **“Analyze”**.

**Say:**  
*"I am asking: 'List audit findings.' The sample path searches the document locally and answers only from the text it finds, with page numbers."*

---

### Step 3 – Show the answer and evidence

1. Point to the **Answer** box.
2. Show the expanded **Evidence trail** with page numbers, scores, and excerpts.
3. Click **Export answer memo** if you want to show a portable audit note.

**Say:**  
*"The answer is built only from retrieved chunks. Here are the exact excerpts and page numbers it used as evidence."*

---

### Step 4 – Try 1–2 more questions

1. Click **“Where are internal control weaknesses mentioned?”** → **Analyze**.
2. Then **“What compliance issues were identified?”** → **Analyze**.

**Say:**  
*“Same document; different questions. You can see it pulls different sections and cites the right pages each time.”*

---

### Step 5 – Optional: your own PDF

If you have a real audit PDF:

1. Click **“Clear document”** in the sidebar.
2. Upload your PDF under **“Upload PDF”**.
3. Wait for **“PDF processed. X chunks indexed.”**
4. Ask a question and click **Analyze**.

**Say:**  
*"For a real report, you upload the PDF once. Processing happens on this machine; no API key is required."*

---

## 3. One-line summary for judges

**"Auditly is a local-first audit workspace: upload an audit PDF or use the sample, ask questions in natural language, and get answers with page-level evidence."**

---

## 4. If something goes wrong

| Problem | What to do |
|--------|------------|
| Port 8501 in use | Run: `streamlit run app.py --server.port 8502` and open the URL it shows. |
| Dependencies error | Run again: `pip install -r requirements.txt` then `streamlit run app.py`. |
| Sample will not load | Make sure you clicked **“Load sample document (demo)”** and wait a few seconds. No PDF or internet needed. |
| Slow first run with your PDF | The first time you process a real PDF, the embedding model may download. Use the sample demo first so judges do not wait. |

---

## 5. Quick checklist before presenting

- [ ] Terminal: `streamlit run app.py` is running.
- [ ] Browser: app is open (e.g. http://localhost:8501).
- [ ] Clicked **“Load sample document (demo)”** and saw “Sample audit document loaded”.
- [ ] Tested at least one question (e.g. “List audit findings”) and saw Answer + Evidence trail.

Then you’re ready to show it to the judges.
