import streamlit as st
import pickle
import re
import io
import os
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

st.set_page_config(page_title="ResumeIQ", page_icon="⚡", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0a0f; color: #e8e8f0; }
header[data-testid="stHeader"] { background: transparent; }
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 3rem; font-weight: 700;
    background: linear-gradient(135deg, #00ff88 0%, #00ccff 50%, #aa44ff 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.1; margin-bottom: 0.25rem;
}
.hero-sub { font-size: 1rem; color: #666680; font-weight: 300; letter-spacing: 0.05em; margin-bottom: 2.5rem; }
.result-card {
    background: linear-gradient(135deg, #13131f 0%, #1a1a2e 100%);
    border: 1px solid #2a2a4a; border-radius: 16px; padding: 1.75rem;
    margin: 1rem 0; box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
.category-badge {
    display: inline-block;
    background: linear-gradient(135deg, #00ff8822, #00ccff22);
    border: 1px solid #00ff8844; color: #00ff88;
    font-family: 'Space Mono', monospace; font-size: 1.4rem; font-weight: 700;
    padding: 0.5rem 1.25rem; border-radius: 8px; margin-bottom: 1rem; letter-spacing: 0.03em;
}
.conf-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.12em; color: #666680; margin-bottom: 0.4rem; font-weight: 500; }
.conf-bar-bg { background: #1e1e30; border-radius: 999px; height: 10px; width: 100%; margin: 0.3rem 0 0.15rem 0; overflow: hidden; }
.conf-bar-fill { height: 10px; border-radius: 999px; background: linear-gradient(90deg, #00ff88, #00ccff); }
.conf-bar-fill.medium { background: linear-gradient(90deg, #ffcc00, #ff8800); }
.conf-bar-fill.low    { background: linear-gradient(90deg, #ff4444, #ff8800); }
.pred-row { display: flex; align-items: center; gap: 0.75rem; margin: 0.6rem 0; }
.pred-rank { font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #444460; width: 1.5rem; text-align: right; }
.pred-name { font-size: 0.9rem; color: #b0b0cc; width: 9rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.pred-bar-bg { flex: 1; background: #1e1e30; border-radius: 999px; height: 7px; overflow: hidden; }
.pred-bar-fill { height: 7px; border-radius: 999px; background: linear-gradient(90deg, #aa44ff, #00ccff); }
.pred-pct { font-family: 'Space Mono', monospace; font-size: 0.75rem; color: #666680; width: 3.5rem; text-align: right; }
.score-section { display: flex; align-items: center; gap: 2rem; margin: 1rem 0; }
.score-number { font-family: 'Space Mono', monospace; font-size: 3.5rem; font-weight: 700; line-height: 1; }
.score-number.high   { color: #00ff88; }
.score-number.medium { color: #ffcc00; }
.score-number.low    { color: #ff6644; }
.section-header {
    font-family: 'Space Mono', monospace; font-size: 0.65rem;
    text-transform: uppercase; letter-spacing: 0.18em; color: #444460;
    margin: 1.5rem 0 0.75rem 0; padding-bottom: 0.4rem; border-bottom: 1px solid #1e1e30;
}
.sugg-chip {
    display: inline-block; background: #1a1a2e; border: 1px solid #2a2a4a;
    border-radius: 6px; padding: 0.35rem 0.75rem; font-size: 0.8rem; color: #9090b0; margin: 0.2rem;
}
.sugg-chip.positive { border-color: #00ff8844; color: #00ff88; background: #00ff8811; }
.sugg-chip.warning  { border-color: #ffcc0044; color: #ffcc00; background: #ffcc0011; }
.custom-divider { border: none; border-top: 1px solid #1e1e30; margin: 2rem 0; }
[data-testid="stFileUploader"] { background: #13131f; border: 1px dashed #2a2a4a; border-radius: 12px; padding: 0.5rem; }
.stButton > button {
    background: linear-gradient(135deg, #00ff8822, #00ccff22) !important;
    border: 1px solid #00ff8844 !important; color: #00ff88 !important;
    font-family: 'Space Mono', monospace !important; font-size: 0.75rem !important;
    letter-spacing: 0.08em !important; border-radius: 8px !important; padding: 0.4rem 1rem !important;
}
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource(show_spinner=False)
def load_models():
    clf = pickle.load(open(os.path.join(BASE_DIR, 'clf.pkl'), 'rb'))
    tfidf = pickle.load(open(os.path.join(BASE_DIR, 'tfidf.pkl'), 'rb'))
    le = pickle.load(open(os.path.join(BASE_DIR, 'label_encoder.pkl'), 'rb'))
    return clf, tfidf, le


clf, tfidf, le = load_models()

CATEGORY_KEYWORDS = {
    'Information Technology': ['python', 'java', 'sql', 'javascript', 'machine learning', 'deep learning', 'api',
                               'cloud', 'docker', 'kubernetes', 'git', 'agile', 'software', 'developer', 'engineer',
                               'data', 'algorithm', 'neural', 'tensorflow', 'pytorch'],
    'Designer': ['figma', 'sketch', 'adobe', 'photoshop', 'illustrator', 'ui', 'ux', 'typography', 'color', 'layout',
                 'wireframe', 'prototype', 'brand', 'visual', 'design', 'canva', 'indesign'],
    'Engineering': ['cad', 'solidworks', 'autocad', 'mechanical', 'civil', 'electrical', 'circuit', 'matlab',
                    'simulation', 'manufacturing', 'structural', 'thermodynamics', 'fluid'],
    'Healthcare': ['patient', 'clinical', 'medical', 'hospital', 'nursing', 'diagnosis', 'treatment', 'pharmacy',
                   'health', 'doctor', 'care', 'therapeutic'],
    'Finance': ['accounting', 'financial', 'budget', 'audit', 'tax', 'investment', 'portfolio', 'excel', 'balance',
                'revenue', 'profit', 'equity'],
    'HR': ['recruitment', 'talent', 'onboarding', 'payroll', 'performance', 'training', 'employee', 'hr',
           'human resources', 'compensation', 'benefits'],
    'Teacher': ['curriculum', 'teaching', 'lesson', 'classroom', 'students', 'education', 'academic', 'pedagogy',
                'assessment', 'learning'],
    'Sales': ['revenue', 'target', 'crm', 'pipeline', 'negotiation', 'client', 'quota', 'b2b', 'lead', 'conversion',
              'account'],
    'Marketing': ['seo', 'content', 'social media', 'campaign', 'analytics', 'brand', 'marketing', 'digital', 'email',
                  'growth'],
}

RESUME_TIPS = {
    'Information Technology': {
        'sections': ['Projects with GitHub links', 'Technical Skills table', 'Certifications (AWS/Google/Azure)',
                     'Open source contributions'],
    },
    'Designer': {
        'sections': ['Portfolio link (Behance/Dribbble)', 'Design tools list', 'Case studies', 'Client work examples'],
    },
    'default': {
        'sections': ['Quantified achievements (numbers/impact)', 'Certifications', 'Awards and recognition'],
    }
}


def clean_resume(text):
    text = re.sub(r'http\S+\s?', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+\s?', ' ', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]', ' ', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_text(uploaded_file):
    file_bytes = uploaded_file.read()
    name = uploaded_file.name.lower()
    if name.endswith(('.jpg', '.jpeg', '.png')):
        try:
            import pytesseract
            from PIL import Image
            pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
            text = pytesseract.image_to_string(Image.open(io.BytesIO(file_bytes)))
            if text.strip(): return text
        except Exception as e:
            raise ValueError(f"OCR failed: {e}")
    if name.endswith('.pdf'):
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            text = ' '.join(p.extract_text() or '' for p in reader.pages)
            if text.strip(): return text
        except Exception:
            pass
        try:
            from pdf2image import convert_from_bytes
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
            imgs = convert_from_bytes(file_bytes, poppler_path='/opt/homebrew/bin')
            text = ' '.join(pytesseract.image_to_string(i) for i in imgs)
            if text.strip(): return text
        except Exception as e:
            raise ValueError(f"PDF extraction failed: {e}")
    for enc in ('utf-8', 'latin-1', 'cp1252'):
        try:
            return file_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Unable to decode file.")


def predict_with_top3(raw_text):
    cleaned = clean_resume(raw_text)
    features = tfidf.transform([cleaned])
    proba = clf.predict_proba(features)[0]
    top3_idx = proba.argsort()[-3:][::-1]
    top3 = [(le.classes_[i].replace('-', ' ').title(), round(float(proba[i]) * 100, 1)) for i in top3_idx]
    return top3, cleaned


def resume_score(cleaned_text, category):
    text_lower = cleaned_text.lower()
    tips   = RESUME_TIPS.get(category, RESUME_TIPS['default'])
    kw_list = CATEGORY_KEYWORDS.get(category, [])
    hits    = sum(1 for k in kw_list if k in text_lower)
    kw_score = min(hits / max(len(kw_list),1), 1.0) * 40
    words   = len(cleaned_text.split())
    length_score = 30 if 300<=words<=800 else (20 if words>200 else 10)
    sec_kw  = ['experience','education','skills','projects','certifications','summary','objective']
    sec_hits = sum(1 for s in sec_kw if s in text_lower)
    section_score = min(sec_hits/5, 1.0)*30
    total   = int(kw_score + length_score + section_score)
    missing = [k for k in kw_list if k not in text_lower][:5]

    # Smart filter — only suggest sections not already in resume
    filtered_tips = []
    for tip in tips['sections'][:4]:
        tip_lower = tip.lower()
        if 'github' in tip_lower and 'github' in text_lower:
            continue
        if 'skill' in tip_lower and 'skill' in text_lower:
            continue
        if 'project' in tip_lower and 'project' in text_lower:
            continue
        if 'certification' in tip_lower and 'certif' in text_lower:
            continue
        if 'open source' in tip_lower and 'open source' in text_lower:
            continue
        if 'portfolio' in tip_lower and 'portfolio' in text_lower:
            continue
        if 'award' in tip_lower and 'award' in text_lower:
            continue
        filtered_tips.append(tip)

    final_tips = filtered_tips if filtered_tips else ["Resume looks well structured!"]
    return total, missing, final_tips


def jd_match(resume_text, jd_text):
    if not jd_text.strip(): return None, [], []
    resume_lower = resume_text.lower()
    jd_lower = jd_text.lower()
    jd_words = list(set(w for w in re.findall(r'\b\w+\b', jd_lower) if len(w) > 4))
    matched = [w for w in jd_words if w in resume_lower]
    missing = [w for w in jd_words if w not in resume_lower]
    score = round(len(matched) / max(len(jd_words), 1) * 100, 1)
    return score, sorted(matched, key=len, reverse=True)[:10], sorted(missing, key=len, reverse=True)[:10]


def conf_color(c):
    return "high" if c >= 60 else ("medium" if c >= 35 else "low")


def bar_class(c):
    return "" if c >= 60 else ("medium" if c >= 35 else "low")


def build_chips(items, css_class, prefix=""):
    return "".join(f'<span class="sugg-chip {css_class}">{prefix}{item}</span>' for item in items)


def main():
    st.markdown('<div class="hero-title">ResumeIQ&#9889;</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">AI-powered resume screening &middot; category prediction &middot; JD matching</div>',
        unsafe_allow_html=True)

    col_reload, _ = st.columns([1, 4])
    with col_reload:
        if st.button("Reload Models"):
            st.cache_resource.clear()
            st.rerun()

    st.markdown('<div class="section-header">Job Description (optional)</div>', unsafe_allow_html=True)
    jd_text = st.text_area(label="jd",
                           placeholder="Paste the job description here to get a match score and keyword gap analysis...",
                           height=120, label_visibility="collapsed")

    st.markdown('<div class="section-header">Upload Resumes</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(label="upload", type=['txt', 'pdf', 'jpg', 'jpeg', 'png'],
                                      accept_multiple_files=True, label_visibility="collapsed")

    if not uploaded_files:
        st.markdown(
            '<p style="color:#444460;font-size:0.9rem;margin-top:1rem;">Upload .pdf, .txt, .jpg, or .png resume files</p>',
            unsafe_allow_html=True)
        return

    for uploaded_file in uploaded_files:
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        with st.spinner(f"Analysing {uploaded_file.name}..."):
            try:
                raw_text = extract_text(uploaded_file)
                if not raw_text.strip():
                    st.warning("File appears empty.")
                    continue

                top3, cleaned = predict_with_top3(raw_text)
                category, top_conf = top3[0]
                score, missing_kw, section_tips = resume_score(cleaned, category)
                jd_score, jd_matched, jd_missing = jd_match(cleaned, jd_text)

                st.markdown(
                    f'<div style="font-family:Space Mono,monospace;font-size:0.8rem;color:#444460;margin-bottom:1rem;">&#128206; {uploaded_file.name}</div>',
                    unsafe_allow_html=True)

                # ── Main prediction card ──
                cc = conf_color(top_conf)
                bc = bar_class(top_conf)
                st.markdown(
                    '<div class="result-card">'
                    '<div class="conf-label">Predicted Category</div>'
                    f'<div class="category-badge">{category}</div>'
                    '<div class="conf-label">Confidence</div>'
                    f'<div class="conf-bar-bg"><div class="conf-bar-fill {bc}" style="width:{top_conf}%"></div></div>'
                    f'<div class="score-number {cc}" style="font-size:1.5rem;">{top_conf}%</div>'
                    '</div>',
                    unsafe_allow_html=True
                )

                # ── Top 3 predictions ──
                st.markdown('<div class="section-header">Top 3 Predictions</div>', unsafe_allow_html=True)
                bars_html = ""
                for i, (cat, pct) in enumerate(top3):
                    rank = ["#1", "#2", "#3"][i]
                    bars_html += (
                        '<div class="pred-row">'
                        f'<div class="pred-rank">{rank}</div>'
                        f'<div class="pred-name">{cat}</div>'
                        f'<div class="pred-bar-bg"><div class="pred-bar-fill" style="width:{pct}%"></div></div>'
                        f'<div class="pred-pct">{pct}%</div>'
                        '</div>'
                    )
                st.markdown(f'<div class="result-card">{bars_html}</div>', unsafe_allow_html=True)

                # ── Resume score ──
                st.markdown('<div class="section-header">Resume Score</div>', unsafe_allow_html=True)
                sc = conf_color(score)
                score_label = "Strong" if score >= 60 else ("Decent" if score >= 40 else "Needs Work")

                # Build all HTML pieces as plain strings BEFORE inserting into HTML
                missing_block = ""
                if missing_kw:
                    chips = build_chips(missing_kw, "warning", "+ ")
                    missing_block = (
                        '<div class="conf-label" style="margin-top:1rem;">'
                        f'Missing keywords for {category}'
                        f'</div>{chips}'
                    )

                section_block = ""
                if section_tips:
                    chips = build_chips(section_tips[:4], "")
                    section_block = (
                        '<div class="conf-label" style="margin-top:1rem;">'
                        'Suggested sections to add'
                        f'</div>{chips}'
                    )

                st.markdown(
                    '<div class="result-card">'
                    '<div class="score-section">'
                    '<div>'
                    f'<div class="score-number {sc}">{score}</div>'
                    f'<div style="font-size:0.75rem;color:#444460;font-family:Space Mono,monospace;">/100 &middot; {score_label}</div>'
                    '</div>'
                    '<div style="flex:1">'
                    '<div class="conf-label">Score breakdown</div>'
                    '<div style="font-size:0.8rem;color:#666680;line-height:1.8;">'
                    'Keyword match &nbsp;&middot;&nbsp; Resume length &nbsp;&middot;&nbsp; Section structure'
                    '</div>'
                    '</div>'
                    '</div>'
                    f'{missing_block}'
                    f'{section_block}'
                    '</div>',
                    unsafe_allow_html=True
                )

                # ── JD Match ──
                if jd_score is not None:
                    st.markdown('<div class="section-header">Job Description Match</div>', unsafe_allow_html=True)
                    jd_sc = conf_color(jd_score)
                    jd_bc = bar_class(jd_score)

                    matched_block = ""
                    if jd_matched:
                        chips = build_chips(jd_matched[:8], "positive", "&#10003; ")
                        matched_block = '<div class="conf-label" style="margin-top:1rem;">Keywords found in resume</div>' + chips

                    missing_jd_block = ""
                    if jd_missing:
                        chips = build_chips(jd_missing[:8], "warning", "&#10007; ")
                        missing_jd_block = '<div class="conf-label" style="margin-top:0.75rem;">Keywords missing from resume</div>' + chips

                    st.markdown(
                        '<div class="result-card">'
                        '<div class="conf-label">Match Score</div>'
                        f'<div class="conf-bar-bg"><div class="conf-bar-fill {jd_bc}" style="width:{jd_score}%"></div></div>'
                        f'<div class="score-number {jd_sc}" style="font-size:2.5rem;">{jd_score}%</div>'
                        f'{matched_block}'
                        f'{missing_jd_block}'
                        '</div>',
                        unsafe_allow_html=True
                    )

                with st.expander("View extracted text"):
                    st.text(cleaned[:3000] + ("..." if len(cleaned) > 3000 else ""))

            except ValueError as e:
                st.error(f"Error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()