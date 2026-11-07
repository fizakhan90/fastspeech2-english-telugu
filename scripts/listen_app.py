import random, csv, time, pathlib, gradio as gr

WS = pathlib.Path.home() / "fs2"
SENTS = [l.rstrip("\n").split("|", 1) for l in open(WS / "test_sentences.txt", encoding="utf-8") if "|" in l]
SYS = {"universal": WS / "results"}
for p in sorted(WS.glob("results_voc*")):
    SYS[p.name.replace("results_", "")] = p
NAMES = list(SYS); LAB = "ABCDEFG"[:len(NAMES)]
OUT = WS / "listening"; OUT.mkdir(exist_ok=True); CSV = OUT / "ratings.csv"
lang = lambda sid: "English" if sid.startswith("E") else "Telugu"
path = lambda s, sid: str(SYS[s] / lang(sid) / "audio" / f"{sid}.wav")

def new_session():
    order = {sid: random.sample(NAMES, len(NAMES)) for sid, _ in SENTS}
    audios = [path(order[sid][k], sid) for sid, _ in SENTS for k in range(len(NAMES))]
    return [order] + audios + [None] * len(SENTS)

def submit(order, rater, te, *picks):
    if not rater.strip(): return "⚠️ Please enter your name."
    if all(p is None for p in picks): return "⚠️ Please pick at least one sentence."
    new = not CSV.exists()
    n = 0
    with open(CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new: w.writerow(["time", "rater", "telugu_speaker", "sentence", "pick", "system", "order"])
        for (sid, _), p in zip(SENTS, picks):
            if p is None: continue
            w.writerow([time.strftime("%F %T"), rater.strip(), te, sid, p,
                        order[sid][LAB.index(p)], "/".join(order[sid])])
            n += 1
    return f"✅ Thanks! Saved {n} ratings."

with gr.Blocks(title="TTS listening test") as demo:
    gr.Markdown("# Blind listening test\nFor each sentence, listen to all versions (headphones help) "
                "and pick the one that sounds **clearest** — least muffled, buzzy or hissy. "
                "The order is random. You can skip any sentence you're unsure about.")
    order = gr.State()
    with gr.Row():
        rater = gr.Textbox(label="Your name")
        te = gr.Checkbox(label="I understand Telugu")
    audios, picks = [], []
    for sid, text in SENTS:
        with gr.Group():
            gr.Markdown(f"**{sid}** — {text}")
            with gr.Row():
                for k in LAB:
                    audios.append(gr.Audio(label=k, type="filepath", interactive=False))
            picks.append(gr.Radio(list(LAB), label="Which sounds best? (optional)"))
    btn = gr.Button("Submit", variant="primary"); msg = gr.Markdown()
    demo.load(new_session, None, [order] + audios + picks)
    btn.click(submit, [order, rater, te] + picks, msg)

demo.launch(server_name="0.0.0.0", server_port=7860, share=True, allowed_paths=[str(WS)])
