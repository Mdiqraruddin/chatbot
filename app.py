# app.py — medical assistant with "precautions & medicines" intent handling
from flask import Flask, request, jsonify, render_template, session
import json, re, difflib, logging, os
from flask_cors import CORS

# Optional OpenAI integration (unchanged)
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
if USE_OPENAI:
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "healthbot_secret_key_123")
CORS(app)

# Load resources
with open("medical_knowledge.json", "r", encoding="utf-8") as f:
    MED_KB = json.load(f)
with open("medicines.json", "r", encoding="utf-8") as f:
    MEDICINES = json.load(f)

# Build keyword index for matching
INDEX = {}
for key, item in MED_KB.items():
    INDEX[key] = {
        "title": item.get("title",""),
        "keywords": set([item.get("title","").lower()]) | set(s.lower() for s in item.get("symptoms", []))
    }

EMERGENCY_SIGNS = [
    "chest pain", "cannot breathe", "difficulty breathing", "severe bleeding",
    "unconscious", "seizure", "blue lips", "stroke", "sudden weakness", "fainting"
]

DISCLAIMER = (
    "⚠️ I provide general health information only — not medical advice. "
    "If you think it's an emergency, contact your local emergency services immediately."
)

# ---------------- Session memory ----------------
def add_to_history(user_text, bot_text, matched_condition=None):
    if "history" not in session:
        session["history"] = []
    # store matched_condition to help follow-ups
    session["history"].append({"user": user_text, "bot": bot_text, "condition": matched_condition})
    session.modified = True

def get_history():
    return session.get("history", [])

def last_matched_condition_from_history():
    """Return last non-null condition key saved in history (most recent) or None."""
    hist = session.get("history", [])
    for entry in reversed(hist):
        cond = entry.get("condition")
        if cond:
            return cond
        # fallback: try to parse the bot text for known keys
        bot_text = (entry.get("bot") or "").lower()
        for key in MED_KB.keys():
            if key in bot_text:
                return key
    return None

# ---------------- Matching & symptom-checker ----------------
def check_emergency(msg):
    t = msg.lower()
    for s in EMERGENCY_SIGNS:
        if s in t:
            return True, s
    return False, None

def find_condition_simple(msg):
    t = msg.lower()
    # direct condition key
    for key in MED_KB.keys():
        if key in t:
            return key, 1.0
    # symptom/title keyword match
    for key, meta in INDEX.items():
        for kw in meta["keywords"]:
            if kw and kw in t:
                return key, 1.0
    # fuzzy fallback
    scores = []
    for key, meta in INDEX.items():
        combined = meta["title"].lower() + " " + " ".join(meta["keywords"])
        score = difflib.SequenceMatcher(None, t, combined).ratio()
        scores.append((key, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    if scores and scores[0][1] >= 0.45:
        return scores[0][0], scores[0][1]
    return None, 0.0

def symptom_checker(symptoms_list, top_n=5):
    scores = []
    for key, meta in INDEX.items():
        kw_set = set(k for k in meta["keywords"] if k)
        if not kw_set:
            continue
        hits = 0
        for s in symptoms_list:
            for kw in kw_set:
                if kw in s:
                    hits += 1
                    break
        score = hits / max(1, len(kw_set))
        scores.append((key, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [(k, round(s,3)) for k, s in scores if s>0][:top_n]

# ----------------- Medicines & Precautions helpers -----------------
def get_medicine_text(condition_key):
    """Return formatted medicine suggestions (non-prescription) or None."""
    if condition_key in MEDICINES:
        meds = MEDICINES[condition_key]
        lines = []
        for m in meds:
            lines.append(f"- {m['name']}: {m.get('use','')} (Warning: {m.get('warning','')})")
        return "\n".join(lines)
    return None

# Precautions generator: use curated tips for common conditions, fallback to generic precautions
PRECAUTION_MAP = {
    "fever": [
        "Rest and avoid strenuous activity.",
        "Keep hydrated — drink plenty of fluids (water, ORS).",
        "Monitor temperature regularly.",
        "Avoid giving aspirin to children — use paracetamol for fever in children as per label and medical advice.",
        "Avoid self-prescribing antibiotics unless a doctor recommends them."
    ],
    "headache": [
        "Rest in a quiet, dark room if light/noise worsens the headache.",
        "Stay hydrated; avoid long screen time and eye strain.",
        "Try relaxation/breathing exercises for tension headaches."
    ],
    "common_cold": [
        "Rest and increase fluid intake.",
        "Use saline nasal spray and humidifier to ease congestion.",
        "Avoid close contact with others while symptomatic."
    ],
    "flu": [
        "Rest, hydrate, and avoid contact with vulnerable people (elderly, infants).",
        "Consider staying home until 24 hours after fever subsides without medicines."
    ],
    "sore_throat": [
        "Gargle warm salt water and drink warm fluids.",
        "Avoid irritants like smoke and dry air."
    ],
    "minor_burn": [
        "Cool with running water (10–20 minutes).",
        "Cover with sterile non-stick dressing; do not apply butter or toothpaste."
    ],
    "sprained_ankle": [
        "Follow RICE: Rest, Ice, Compression, Elevation.",
        "Avoid putting weight on it until comfortable."
    ],
    # add more condition-specific tips as needed...
}

GENERIC_PRECAUTIONS = [
    "If symptoms worsen or you have severe signs (chest pain, difficulty breathing, fainting, severe bleeding), seek emergency care immediately.",
    "Avoid taking antibiotics or prescription medications without a doctor's advice.",
    "If you have allergies, chronic illnesses, or are pregnant, consult a healthcare professional before taking medicines.",
    "Keep track of symptom duration and severity; seek professional care if symptoms persist or worsen."
]

def get_precaution_text(condition_key):
    """Return a string of precautions for the condition (from PRECAUTION_MAP or generic)."""
    if condition_key and condition_key in PRECAUTION_MAP:
        tips = PRECAUTION_MAP[condition_key] + GENERIC_PRECAUTIONS[:2]  # include some generic advices
    elif condition_key and condition_key in MED_KB:
        # generate basic precautions from description and 'when_to_seek_care'
        info = MED_KB[condition_key]
        tips = [
            "Follow general supportive measures (rest, fluids, symptom relief).",
            f"Watch for red flags: {info.get('when_to_seek_care','seek care if concerned.')}"
        ] + GENERIC_PRECAUTIONS[:1]
    else:
        # no condition known: return general precautions
        tips = [
            "Provide clear symptom list (e.g., 'fever, cough, sore throat') so I can suggest likely conditions.",
            "In the meantime: stay hydrated, rest, avoid self-prescribed antibiotics, monitor red-flag signs."
        ] + GENERIC_PRECAUTIONS
    return "\n".join(f"- {t}" for t in tips)

# ----------------- Small-talk & QA (copied over) -----------------
SMALLTALK_PATTERNS = [
    (r"\bhi\b|\bhello\b|\bhey\b", ["Hello!", "Hi there!", "Hey — how can I help?"]),
    (r"\bhow are you\b|\bhow's it going\b", ["I'm a program, doing fine — ready to help!", "All good here — how are you?"]),
    (r"\bwhat'?s up\b|\bwhats up\b", ["Not much — ready to chat or help with symptoms."]),
    (r"\bthank(s| you)\b", ["You're welcome!", "Happy to help!"]),
    (r"\bbye\b|\bgoodbye\b|\bsee you\b", ["Bye! Take care.", "Goodbye — stay safe."]),
    (r"\bwhat can you do\b|\bwhat are you\b", ["I can provide general info about symptoms and suggest non-prescription options. I'm not a doctor."]),
    (r"\btoday i want to discuss\b", ["Sure — tell me what topic you'd like to discuss. I can chat or help with health topics."]),
]

QA_MAP = {
    "what two people communicate with each other": (
        "Two people communicate information, ideas, feelings and intentions to each other. "
        "This can be through words, tone of voice, body language, or written messages. Good communication involves listening, clarity, and feedback."
    ),
    "what is communication": (
        "Communication is the exchange of information between individuals or groups using spoken, written, visual, or nonverbal methods."
    ),
    "what is a symptom": (
        "A symptom is something a person feels or notices that may indicate a condition or illness (e.g., pain, fever, cough)."
    ),
    "how to start a conversation": (
        "Start with a friendly greeting, ask an open question (e.g., 'How are you?'), comment on something you both share, and show interest by listening."
    )
}

def match_smalltalk(msg):
    t = msg.lower()
    for q, a in QA_MAP.items():
        if q in t:
            return a
    for pattern, responses in SMALLTALK_PATTERNS:
        if re.search(pattern, t):
            return responses[0]
    return None

# ---------------- Routes ----------------
@app.route("/")
def home():
    return render_template("index.html", disclaimer=DISCLAIMER)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json() or {}
        msg = (data.get("message") or "").strip()
        symptom_mode = bool(data.get("symptom_mode", False))

        if not msg:
            return jsonify({"reply":"Please type your symptoms or question.", "disclaimer": DISCLAIMER})

        # Emergency detection
        emergency, sign = check_emergency(msg)
        if emergency:
            reply = f"⚠️ The symptom '{sign}' may be life-threatening. Contact emergency services immediately."
            add_to_history(msg, reply)
            return jsonify({"reply": reply, "disclaimer": DISCLAIMER, "history": get_history()})

        # Smalltalk / QA
        small = match_smalltalk(msg)
        if small:
            add_to_history(msg, small)
            return jsonify({"reply": small, "disclaimer": DISCLAIMER, "history": get_history()})

        # --- NEW: detect "precaution/medicine" intent ---
        # keywords indicating user asks for precautions or medicines
        intent_prec_med = False
        lower = msg.lower()
        for kw in ["precaution", "precautions", "precautionary", "medicine", "medicines", "medication", "drugs", "suggest medicine", "suggest me medicine", "what medicine", "what to take"]:
            if kw in lower:
                intent_prec_med = True
                break

        # If intent, try to find condition mentioned in same message
        matched_cond, _ = find_condition_simple(msg)
        if intent_prec_med:
            # if user specified condition in their message, use it
            if matched_cond:
                prec_text = get_precaution_text(matched_cond)
                med_text = get_medicine_text(matched_cond) or "No non-prescription options found in the database."
                reply = f"Precautions for **{MED_KB[matched_cond]['title']}**:\n{prec_text}\n\nNon-prescription suggestions:\n{med_text}\n\n{DISCLAIMER}"
                add_to_history(msg, reply, matched_condition=matched_cond)
                return jsonify({"reply": reply, "disclaimer": DISCLAIMER, "history": get_history()})
            # else: try to use last matched condition from session history
            last_cond = last_matched_condition_from_history()
            if last_cond:
                prec_text = get_precaution_text(last_cond)
                med_text = get_medicine_text(last_cond) or "No non-prescription options found in the database."
                reply = f"Based on your previous condition (**{MED_KB[last_cond]['title']}**), here are precautions and non-prescription suggestions:\n{prec_text}\n\nNon-prescription suggestions:\n{med_text}\n\n{DISCLAIMER}"
                add_to_history(msg, reply, matched_condition=last_cond)
                return jsonify({"reply": reply, "disclaimer": DISCLAIMER, "history": get_history()})
            # no condition known: give helpful generic precautions and ask for condition
            prec_text = get_precaution_text(None)
            reply = f"I don't know which condition you mean. Here are general precautions you can follow while you tell me the specific condition:\n{prec_text}\n\nPlease say the condition (e.g., 'for fever') or list symptoms (e.g., 'fever, cough').\n\n{DISCLAIMER}"
            add_to_history(msg, reply, matched_condition=None)
            return jsonify({"reply": reply, "disclaimer": DISCLAIMER, "history": get_history()})

        # Symptom-mode handling (multi-symptom)
        if symptom_mode:
            parts = [p.strip().lower() for p in re.split(r",|;| and ", msg) if p.strip()]
            candidates = symptom_checker(parts, top_n=6)
            cand_structs = [{"key":k, "title": MED_KB[k]["title"], "score": sc} for k, sc in candidates]
            explanation = None
            # optional openai explanation omitted for brevity if not enabled
            reply_text = "Possible matches (symptom-checker):\n"
            for c in cand_structs:
                reply_text += f"- {c['title']} (score: {c['score']})\n"
            add_to_history(msg, reply_text)
            return jsonify({"reply": reply_text, "candidates": cand_structs, "explanation": explanation,
                            "disclaimer": DISCLAIMER, "history": get_history()})

        # Normal medical condition matching
        cond, score = find_condition_simple(msg)
        if cond:
            info = MED_KB[cond]
            med_info = get_medicine_text(cond)
            reply = f"**{info['title']}**\nSymptoms: {', '.join(info.get('symptoms', []))}\n\n{info.get('description','')}\n\nWhen to seek care: {info.get('when_to_seek_care','')}"
            if med_info:
                reply += "\n\nPossible non-prescription options:\n" + med_info
            add_to_history(msg, reply, matched_condition=cond)
            return jsonify({"reply": reply, "history": get_history(), "disclaimer": DISCLAIMER})

        # Fallback
        fallback = ("I couldn't identify a clear match. If you can list several symptoms separated by commas "
                    "(for example: 'fever, cough, sore throat'), I can run a symptom-check to suggest possible conditions. "
                    "Or specify the condition you'd like precautions/medicines for (e.g., 'for headache').")
        add_to_history(msg, fallback)
        return jsonify({"reply": fallback, "history": get_history(), "disclaimer": DISCLAIMER})

    except Exception:
        logging.exception("Error in /chat")
        return jsonify({"reply":"Server error occurred.", "disclaimer": DISCLAIMER}), 500

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True, host="127.0.0.1", port=5000)
