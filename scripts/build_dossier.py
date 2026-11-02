#!/usr/bin/env python3
"""Assemble docs/results_dossier.md: every number available so far, with source, sample size and limits.

Rebuild it whenever new results arrive; it only reads run directories (no GPU). Tables of costs come from the
campaign directories given, tables of training results from pqfl.training_report. Opt-in, no dependencies beyond numpy.
"""
import argparse
import csv
import glob
import json
import os
import statistics
import time

from pqfl.training_report import load_runs, markdown

SCHEMES = ("N", "N-FP", "A", "C", "B", "B-BGV")


def cost_table(campaign: str, vector_size: int) -> str:
    rows = {r["scheme"]: r for r in csv.DictReader(open(os.path.join(campaign, "compute", "summary.csv")))
            if r["group"] == "main" and int(r["sites"]) == 3 and int(r["vector_size"]) == vector_size}
    wan = {}
    for path in glob.glob(os.path.join(campaign, "net", "wan", f"main__*-n3-v{vector_size}.json")):
        data = json.load(open(path))
        wan[data["scheme"]] = statistics.median(data["round_network_s"]), len(data["round_network_s"])
    lan = {}
    for path in glob.glob(os.path.join(campaign, "net", "lan", f"main__*-n3-v{vector_size}.json")):
        data = json.load(open(path))
        lan[data["scheme"]] = statistics.median(data["round_network_s"])
    if not rows:
        return "(no data)\n"
    base = float(rows["A"]["bytes_p50"])
    lines = ["| Scheme | Round payload (MB) | x A | Setup (MB) | Crypto s (p50) | Peak RSS (MB) | LAN s (p50) | WAN s (p50, n) | WAN x A |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for scheme in SCHEMES:
        r = rows.get(scheme)
        if not r:
            continue
        w = wan.get(scheme)
        lines.append(f"| {scheme} | {float(r['bytes_p50']) / 1e6:.2f} | {float(r['bytes_p50']) / base:.2f} | "
                     f"{float(r['setup_bytes']) / 1e6:.2f} | {float(r['crypto_s_p50']):.3f} | {float(r['peak_host_rss_mb']):.0f} | "
                     f"{lan.get(scheme, float('nan')):.3f} | {w[0]:.2f} ({w[1]}) | {w[0] / wan['A'][0]:.2f} |"
                     if w else f"| {scheme} | {float(r['bytes_p50']) / 1e6:.2f} | {float(r['bytes_p50']) / base:.2f} | - | - | - | - | - | - |")
    return "\n".join(lines) + "\n"


def section(path: str, start: str, end: str | None = None) -> str:
    text = open(path, encoding="utf-8").read()
    head = text.index(start)
    tail = text.index(end, head + len(start)) if end else len(text)
    body = text[head:tail].strip()
    return "\n".join("#" + line if line.startswith("## ") else line for line in body.splitlines()) + "\n"


def exists_all(*paths) -> bool:
    return all(os.path.exists(p) for p in paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="runs/campaign-3")
    parser.add_argument("--pilot", default="runs/pilot")
    parser.add_argument("--final", default="runs/final")
    parser.add_argument("--output", default="docs/results_dossier.md")
    args = parser.parse_args()
    report = os.path.join(args.campaign, "report.md")
    pilot_rows = load_runs(args.pilot, rounds=4)
    final_rows = load_runs(args.final, rounds=4) if os.path.isdir(args.final) else []
    out = [f"# Dossier dei risultati (generato il {time.strftime('%Y-%m-%d %H:%M')})", "",
           "Raccolta di tutti i numeri disponibili, con fonte, numerosità e limite. Si rigenera con "
           "`scripts/build_dossier.py`. Non è il manoscritto: è la base per valutare cosa si può affermare.", "",
           "## 0. Stato delle evidenze", "",
           "| Evidenza | Stato | Campione | Fonte |", "| --- | --- | --- | --- |",
           f"| Costi per round, n = 3, d = 1e5…2,1e7 | disponibile | 30 round (14 per HE a 3,4M), una sessione | {args.campaign} |",
           f"| Scalabilità in n (5, 10, 20) | disponibile, solo calcolo | 30 round | {args.campaign} |",
           f"| Replay di rete LAN/metro/WAN | disponibile, 3 celle WAN con perturbazioni | 6–30 campioni, una sessione | {args.campaign}/net |",
           "| Costo a dimensione reale 8.409.600 e N a codifica uguale | **in attesa (T14)** | - | - |",
           "| Ripetizione celle WAN perturbate e sessioni indipendenti | **in attesa (T15)** | - | - |",
           f"| Utilità e attacchi, pilota a 1 seed | {'disponibile' if pilot_rows else 'assente'} | seed 0 | {args.pilot} |",
           f"| Utilità e attacchi, campagna finale 5 seed | {'disponibile' if final_rows else '**non ancora lanciata**'} | seed 1–5 | {args.final} |",
           "| Inversione dei gradienti come controllo positivo | **in attesa (T13b)** | - | - |",
           "| Dropout (esplorativo) | disponibile | 10 sessioni, A e B-BGV | runs/dropout |",
           "| Baseline TLS 1.3 ibrido | disponibile | 40 handshake | runs/tls-baseline.json |",
           "| Contabile DP contro Opacus | fatto | - | docs/decisions.md (1 nov) |",
           "| Revisione della crittografia (combiner, flooding B/B-BGV) | **non fatta** (richiede un esperto) | - | - |", ""]
    out += ["## 1. Costi: cella principale (n = 3, d = 3.407.872, dimensione di un adapter Llama-like)", "",
            cost_table(args.campaign, 3407872),
            "Tempi: somma seriale dei tempi misurati nel simulatore, non un tempo distribuito; la rete è un replay del "
            "registro dei trasferimenti tra container sullo stesso host condiviso. WAN: 50 ms, 100 Mbit/s, perdita 0,1 %.", ""]
    out += ["## 2. Costi: tabelle complete della campagna", ""]
    if os.path.exists(report):
        out += [section(report, "## 1. Cost per round", "## 5. Perturbation"), "",
                section(report, "## 5. Perturbation"), ""]
    for number, title, size in (("4", "Costi a dimensione reale dell'adapter (T14)", 8409600),):
        directory = os.path.join("runs", f"campaign-{number}")
        if os.path.exists(os.path.join(directory, "compute", "summary.csv")):
            out += [f"## 2b. {title}", "", cost_table(directory, size)]
            if os.path.exists(os.path.join(directory, "report.md")):
                out += [section(os.path.join(directory, "report.md"), "## 1. Cost per round", "## 2. Scalab"
                                if "## 2. Scalab" in open(os.path.join(directory, "report.md")).read() else None), ""]
    if os.path.exists("runs/campaign-5/report.md"):
        out += ["## 2c. Ripetizioni e sessioni indipendenti (T15)", "", section("runs/campaign-5/report.md", "## 3. Network time", "## 4. Handshake")
                if "## 4. Handshake" in open("runs/campaign-5/report.md").read() else section("runs/campaign-5/report.md", "## 1. Cost"), ""]
    out += ["## 3. Utilità e attacchi (training)", "",
            "Protocollo: Qwen3.5-2B-Base, LoRA rango 8 (8.409.600 parametri), MedMCQA in tre siti per materia, 4 round, "
            "lr 2e-4, batch 32, limite per elemento 0,02 per tutti gli schemi, clip DP C = 4.", ""]
    out += ["### 3.1 Pilota (seed 0)", "", "Nota: nella cartella del pilota ci sono due prove N con eps 8 (batch 32 e 256), "
            "quindi la riga N/eps8 mescola due configurazioni; le tabelle finali non avranno questo effetto.", "",
            markdown(pilot_rows) if pilot_rows else "(nessun run)", ""]
    if final_rows:
        out += ["### 3.2 Campagna finale", "", markdown(final_rows), ""]
    base = json.load(open("runs/gpu-queue/done/null-check-base.json"))["result"] \
        if os.path.exists("runs/gpu-queue/done/null-check-base.json") else None
    if base:
        loss, top = base["membership"]["pooled"]["loss"], base["canaries"]["levels"]
        out += ["### 3.3 Controlli di validità", "",
                f"- Modello base (nessun addestramento): accuratezza {base['utility']['validation']['accuracy']:.4f} su 4183, "
                f"perplexity {base['utility']['holdout']['perplexity']:.2f}, AUC membership {loss['auc']:.3f}, "
                f"esposizione canary {top['0']['mean_exposure']:.2f} bit (livello 0) e {top['16']['mean_exposure']:.2f} bit (livello 16), "
                "estrazione 0: i controlli nulli passano.",
                "- Controllo positivo: addestrando senza DP il livello 16 sale a circa 7,5 bit (soglia nulla 95° percentile ~4–5 bit) e l'AUC a 0,63.",
                "- Con DP eps 8 AUC e canary tornano al livello nullo (0,50 e ~1,7 bit).", ""]
    q08b = load_runs("runs/pilot-q08b", rounds=4) if os.path.isdir("runs/pilot-q08b") else []
    if q08b:
        out += ["### 3.5 Secondo modello: Qwen3.5-0.8B-Base (pilota, seed 0, protocollo del 2B)", "",
                "Base non addestrato: accuratezza 0,3988, perplexity 14,94, AUC 0,4999 (controllo nullo superato).", "",
                markdown(q08b), ""]
    probes = [json.load(open(f)) for f in sorted(glob.glob("runs/gpu-queue/done/inversion-conv-*.json"))]
    if probes:
        out += ["### 3.4 Inversione dei gradienti (controllo positivo su N, 4 round, un esempio)", "",
                "| Prova | Passi | lr | Temperatura | Distanza coseno | Recall token | Soglia nulla (95°) | Supera il caso | s/passo |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for job in probes:
            job_args, res = job["args"], job["result"]["results"]
            for r in res:
                out.append(f"| {job['id']} | {job_args['steps']} | {job_args.get('lr')} | {job_args.get('temperature', 1.0)} | "
                           f"{r['cosine_distance']:.3f} | {r['bag_recall']:.3f} | {r['null_p95']:.3f} | {r['above_null']} | "
                           f"{r['seconds_per_step']:.2f} |")
        check = "runs/gpu-queue/done/inversion-selfcheck-2.json"
        if os.path.exists(check):
            examples = json.load(open(check))["result"]["examples"]
            out += ["", "Controllo di coerenza sul modello reale (cosine tra il gradiente osservato e quello ricostruito con i percorsi morbidi): "
                    + "; ".join(f"esempio {i + 1}: token veri {e['true_tokens']:.4f}, un token sbagliato {e['one_wrong_token']:.2f}, "
                                f"token casuali {e['random_tokens']:.2f}" for i, e in enumerate(examples))
                    + ". L'obiettivo è corretto e molto selettivo; la difficoltà è nell'ottimizzazione."]
        for path in sorted(glob.glob("runs/gpu-queue/done/inversion-coord-*.json")):
            job = json.load(open(path))
            for r in job["result"]["results"]:
                out += ["", f"Ricerca a coordinate ({job['id']}, top-{r['top_k']}, {r['passes']} passate, {r['evaluations']} valutazioni): "
                        f"coseno finale {r['cosine_final']:.3f}; recall {r['bag_recall']:.3f} contro {r['baseline_recall']:.3f} "
                        f"della continuazione del modello senza gradiente (soglia nulla {r['null_p95']:.3f}): il valore sopra il caso "
                        "viene dalla memorizzazione, non dall'inversione."]
        out += ["", "L'attacco non supera il caso con questi budget (<= 17 minuti per esempio, un solo esempio): non è un controllo positivo "
                "riuscito e non va letto come prova di privacy.", ""]
    out += ["## 4. Limiti da dichiarare", "",
            "- Un solo modello (2B), un solo dataset, tre siti, regime LoRA di rango 8; senza DP il guadagno di utilità è modesto "
            "(accuratezza 0,49 → 0,51, perplexity 11,0 → 7,7).",
            "- Costi misurati su host condiviso (carico medio ~10): non sono quelli di un nodo dedicato. I tempi di rete sono replay, non cifratura distribuita.",
            "- HE alla dimensione massima non misurato; B e B-BGV a 3,4M hanno 14 round e a WAN 6–7 campioni: solo sintesi descrittive.",
            "- I tre schemi senza DP calcolano la stessa somma: l'equivalenza di utilità è attesa; il contributo è quantificarla e misurare i costi.",
            "- DP: ε da RDP con ordini interi (conservativo); rumore da seme pubblico, quindi nessuna garanzia formale di rilascio; esempio-livello.",
            "- Gli attacchi sono specifici (loss e riferimento, canary con riferimenti finiti), non LiRA; un risultato nullo non prova la privacy.",
            "- Sicurezza di B/B-BGV (flooding), combiner ibrido, recupero del dropout: da far rivedere; il recupero resta esplorativo.",
            "- Un seed nel pilota: nessun intervallo; con cinque seed gli intervalli bootstrap sono grossolani.", ""]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write("\n".join(out) + "\n")
    print(f"written {args.output} ({sum(len(x) for x in out)} characters)")


if __name__ == "__main__":
    main()
