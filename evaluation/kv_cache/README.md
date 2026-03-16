# Phase A – KV-Cache-Voruntersuchung (LM Studio)

## Ziel

Untersucht wird, ob die Aktivierung des KV-Cache die Benchmark-Ergebnisse unter AgentDojo (insb. ASR/Utility) relevant beeinflusst oder ob der KV-Cache primär einen Performance-/VRAM-Trade-off darstellt.

Hintergrund: Einige Modelle passen nur ohne KV-Cache sicher in den verfügbaren VRAM der lokalen Hardware (16 GB). Für die spätere Evaluation sollen daher Modelle je nach VRAM-Fit mit oder ohne KV-Cache benchmarkbar sein, ohne dass dadurch die inhaltliche Aussagekraft (ASR/Utility) wesentlich verzerrt wird.


## Setup-Überblick

- Benchmark: [AgentDojo](https://agentdojo.spylab.ai/) (Benchmark-Suite-Version fixiert auf `v1.2.2`)
- Inferenz: LM Studio (GPU-Server im lokalen Netz)
- AgentDojo und LM Studio laufen auf getrennten Hosts.

Hosts (LAN):
- GPU-Server (LM Studio): `192.168.178.24` – Nvidia RTX 4070 Ti Super (16 GB)
- AgentDojo-Host: `192.168.178.145`


## Ergebnis (Kurzfassung)

Über n=5 gepaarte Runs zeigt sich kein relevanter Unterschied in Utility oder ASR zwischen `kv_on` und `kv_off` (ΔUtility < 1 Prozentpunkt, ΔASR < 0,2 Prozentpunkte). Der KV-Cache ist damit ein reiner Engineering-Parameter. Die Laufzeit verdoppelt sich ohne KV-Cache (≈2x langsamer).

Konsequenz für Phase B: KV-Cache-Einstellung wird pro Modell fixiert (je nach VRAM-Fit) und über alle Defense-Stacks (D0–D3) identisch gehalten.


## Reproduzierbarkeit und Datenspeicherung

Die vollständigen Rohlogs (Run-Artefakte) sind aufgrund ihrer Größe nicht im Git-Repository versioniert.
Stattdessen werden sie als Archiv über GitHub Releases (Assets) veröffentlicht.

- Rohlogs: GitHub Release [`log-kv-cache-oss20b-v1.2.2-n5`](https://github.com/The-KTC/bachelor-llm-agent-defenses/releases/tag/log-kv-cache-oss20b-v1.2.2-n5)
- Datensatz-Index + Checksummen: `runs_manifest.json`
- Ergebnisübersicht: `results.csv`


## Ablauf

Bedingungen:
- `kv_on`: KV-Cache aktiviert (im VRAM)
- `kv_off`: KV-Cache deaktiviert (im System-RAM)
- Abwechselnd ausgeführt (Pair01 kv_on, Pair01 kv_off, Pair02 kv_on, …), um Drift-Effekte zu reduzieren

Wiederholungen: 5 Paare (n=5 pro Bedingung)

Konstant gehalten:
- AgentDojo Package-Version: [v0.1.35](https://github.com/ethz-spylab/agentdojo/releases/tag/v0.1.35) (Stand: 27.10.2025)
- AgentDojo Benchmark-Suite: `v1.2.2`
- LM Studio Version: `0.4.2 (Build 2)` (Stand: [06.02.2026](https://lmstudio.ai/changelog/lmstudio-v0.4.2))
- Modell: [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) (F16)
- Context-Length: 16384
- Batch-Size: 128
- Angriffstyp: `tool_knowledge`


## Metriken

AgentDojo reportet Metriken pro Suite (workspace, travel, banking, slack) sowie für `combined`.

- **Utility:** `Average utility` (pro Suite und `combined`)
- **ASR:** Attack Success Rate = Security (AgentDojo-Semantik: `security=True` bedeutet, dass der Angriff *erfolgreich* war)
- **Sanity-Check:** `Passed injection tasks as user tasks: X/Y` (pro Suite und `combined`) – dient der Plausibilitätsprüfung und ist **keine** ASR-Metrik
- **Sekundär:** Laufzeit (`duration_sec`) zur Einordnung von Performance-Effekten

> **Hinweis zur results.csv:** Die Spalte `asr_targeted_est_pct` enthält Werte, die mit der alten Formel `ASR = 100 − Security` berechnet wurden. Die korrekte ASR entspricht direkt dem `security_avg_pct_combined`-Wert.


## Dateien / Artefakte

### `runs_manifest.json`
Enthält die wichtigsten Metadaten zum veröffentlichten Rohlog-Archiv:
- Release-/Archivname
- SHA-256-Checksummen (Integritätsprüfung)
- Liste der enthaltenen Runs (Run-IDs, KV an/aus, Benchmark-Suite-Version, Pfade im Archiv)

### `results.csv`
Kompakte Tabelle zur Auswertung mit folgenden Spalten:
- `run_id`, `pair_index`, `kv_cache` (`on`/`off`)
- `started_at`, `ended_at`, `duration_sec`
- `utility_avg_pct_combined`, `security_avg_pct_combined`
- `passed_injection_as_user_tasks_combined`
- `asr_targeted_est_pct` (alte Formel, siehe Hinweis oben)

### `scripts/orchestrator.py`
Orchestrator zur automatisierten Durchführung der Runs:
- Erzeugt pro Run einen eindeutigen Run-Ordner
- Startet AgentDojo remote (SSH)
- Sammelt AgentDojo-Artefakte und LM-Studio-Logs ein

### `scripts/bench_config.template.json`
Konfiguration für Hosts, Pfade und Run-Commands.
