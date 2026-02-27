# Bachelorarbeit - Konzeption und Evaluation von Defensivmethodiken zur Prompt-Injektion über Tool-Zugriffe in LLM-Agenten

Dieses Repository enthält die reproduzierbaren Artefakte (Skripte, Konfigurationen, Dokumentation) zu meiner Bachelorarbeit im Bereich Defensivmethodiken zur Prompt-Injektion über Tool-Zugriffe in LLM-Agenten.

Der Fokus liegt auf einer sauberen, nachvollziehbaren Benchmarkstruktur: Jede Studie besitzt eine eigene Dokumentation, fixe Versionen/Settings sowie klar definierte Output-Artefakte.


## Daten und Reproduzierbarkeit

Aufgrund der Größe sind vollständige Rohlogs (AgentDojo-run-Artefakte, LM Studio-Logs) nicht im Git-Repository versioniert.
Die Rohdaten werden stattdessen über GitHub Releases (Assets) als Archiv veröffentlicht und von der jeweiligen Studie aus referenziert.

- Rohlogs: GitHub Release „<NAME/Version>“
- Datensatz-Index + Checksums: siehe jeweilige Studie (`runs_manifest.json`)
- Auswertung/Ergebnisübersichten: siehe jeweige Studie (`results.csv`)


## Studien / Evaluation

1. **KV-Cache-Studie (AMD + LM Studio)**
   Ziel: Einfluss des KV-Cache auf Laufzeit/Overhead vs. Auswirkungen auf ASR/Utility unter konstanten Bedingungen.  
   Pfad: `evaluation/kv_cache/` - [README](evaluation/kv_cache/README.md)

2. **Defense-Evaluation (Hauptteil)**  
   Ziel: Systematische Evaluation eigener Defensivmethodiken auf AgentDojo-Szenarien.  
   Pfad: `evaluation/defenses/` (folgt) - [README](evaluation/defenses/README.md)

*(Hinweis: Eine geplante Inferenz-Stack-Matrix (LM Studio vs. vLLM) wurde verworfen, da das Setup für die geplante reproduzierbare Evaluation nicht stabil genug war.)*

## Lizenz / Hinweise

- Dieses Repository enthält primär Skripte, Konfigurations-Templates und Dokumentation.
- Rohlogs/Run-Ordner sind nicht Bestandteil der Git-History (siehe `.gitignore` und die Release-Archive).
