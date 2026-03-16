# Bachelorarbeit – Konzeption und Evaluation von Defensivmethodiken gegen Prompt-Injektionen über Toolzugriffe in LLM-Agenten

Dieses Repository enthält die reproduzierbaren Artefakte (Skripte, Defense-Implementierungen, Konfigurationen, Dokumentation) zu meiner Bachelorarbeit an der Technischen Hochschule Mannheim.

Der Fokus liegt auf einer sauberen, nachvollziehbaren Benchmarkstruktur: Jede Studie besitzt eine eigene Dokumentation, fixe Versionen/Settings sowie klar definierte Output-Artefakte.


## Repo-Struktur

```
evaluation/
├── kv_cache/          Phase A – KV-Cache-Voruntersuchung
│   ├── scripts/       Orchestrator + Konfiguration
│   ├── results.csv    Ergebnisse (n=5 gepaarte Runs)
│   └── runs_manifest.json
└── defenses/          Phase B – Defense-Stacks D0–D3
    ├── agentdojo_custom_defenses/   D1/D2/D3-Implementierungen
    └── run_benchmark_custom_pipeline.py   Custom-Runner
```


## Defense-Stacks

| Stack | Bezeichnung | Defense-IDs |
|---|---|---|
| D0 | Vanilla (keine Defense) | – |
| D1 Delimiting | Spotlighting: Delimiting ([Hines et al., 2024](https://arxiv.org/abs/2403.14720)) | `d1_delimiting` |
| D1 Datamarking | Spotlighting: Datamarking ([Hines et al., 2024](https://arxiv.org/abs/2403.14720)) | `d1_datamarking` |
| D1 Encoding | Spotlighting: Encoding ([Hines et al., 2024](https://arxiv.org/abs/2403.14720)) | `d1_encoding` |
| D2 JSON | Structured Output (JSON-Envelope) | `d2_json` |
| D3 Balanced | Provenance + Policy Gate (kontextgebunden) | `d3_balanced`, `d3_output_gate` |
| D3 Strict | Provenance + Policy Gate (Deny-by-Default) | `d3_strict`, `d3_output_gate` |


## Daten und Reproduzierbarkeit

Aufgrund der Größe sind vollständige Rohlogs (AgentDojo-Run-Artefakte, LM-Studio-Logs) nicht im Git-Repository versioniert.
Die Rohdaten werden stattdessen über GitHub Releases (Assets) als `.tar.xz`-Archive veröffentlicht und von der jeweiligen Studie aus referenziert.

- Rohlogs: Siehe GitHub Releases (pro Studie/Defense-Stack verlinkt)
- Datensatz-Index + Checksummen: siehe jeweilige Studie (`runs_manifest.json`)
- Auswertung/Ergebnisübersichten: siehe jeweilige Studie (`results.csv`)


## Studien / Evaluation

1. **Phase A – KV-Cache-Voruntersuchung**
   Ziel: Einfluss des KV-Cache auf Laufzeit/Overhead vs. Auswirkungen auf ASR/Utility unter konstanten Bedingungen.
   Pfad: [`evaluation/kv_cache/`](evaluation/kv_cache/) – [README](evaluation/kv_cache/README.md)
   Release: [`log-kv-cache-oss20b-v1.2.2-n5`](https://github.com/The-KTC/bachelor-llm-agent-defenses/releases/tag/log-kv-cache-oss20b-v1.2.2-n5)

2. **Phase B – Defense-Evaluation (D0–D3)**
   Ziel: Systematische Evaluation eigener Defensivmethodiken auf AgentDojo-Szenarien.
   Pfad: [`evaluation/defenses/`](evaluation/defenses/) – [README](evaluation/defenses/README.md)
   Releases:
   - D0 Baseline: [`d0-v1.2.2-rev2`](https://github.com/The-KTC/bachelor-llm-agent-defenses/releases/tag/d0-v1.2.2-rev2)

*(Hinweis: Eine geplante Inferenz-Stack-Matrix (LM Studio vs. vLLM) wurde verworfen, da das Setup für die geplante reproduzierbare Evaluation nicht stabil genug war.)*


## Benchmark-Framework

- **[AgentDojo](https://agentdojo.spylab.ai/)** v1.2.2 ([GitHub](https://github.com/ethz-spylab/agentdojo), [Debenedetti et al.](https://arxiv.org/abs/2406.13352))
- Angriffstyp: `tool_knowledge`
- Suites: `workspace`, `travel`, `banking`, `slack` (949 Injektionsszenarien pro Run)
- Inferenz: LM Studio (lokal) / OpenAI-kompatible API


## Lizenz / Hinweise

- Dieses Repository enthält primär Skripte, Konfigurations-Templates und Dokumentation.
- Rohlogs/Run-Ordner sind nicht Bestandteil der Git-History (siehe `.gitignore` und die Release-Archive).
