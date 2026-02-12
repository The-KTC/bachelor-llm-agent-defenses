# Inference-Backend-Teiluntersuchung (LM Studio vs. vLLM)

## Ziel

Diese Studie vergleicht Inference-Backends und GPU-Stacks (AMD RX 6800 vs. Nvidia RTX 4070 Ti Super) unter identischen AgentDojo-Bedingungen, um zu prüfen, ob sich das verwendete Inferenz-Backend (z.B. LM Studio vs. vLLM) in den AgentDojo-Metriken (Utility/Security bzw. ASR) und in der Laufzeit messbar auswirkt.

Kontext: In der separaten (KV-Cache-Teiluntersuchung)[evaluation/kv_cache/README.md] wurde primär gezeigt, dass das KV-Cache-Handling (on/off) im Rahmen der Mess-Toleranz **keinen relevanten Einfluss** auf ASR/Utility hat und vor allem ein Performance-/VRAM-Trade-off ist. Daher wird in dieser Studie der KV-Cache konstant aktiv gehalten (kein KV-Switching).

Stand: 12.02.2026 (Europe/Berlin)


## Setup-Überblick

- Benchmark: AgentDojo (Benchmark-Suite-Version fixiert auf `v1.2.2`)
- Inferenz: Vergleich von LM Studio vs. vLLM (jeweils OpenAI-kompatibler `/v1` Endpoint)
- AgentDojo und Inferenz laufen auf getrennten Hosts

Hosts (LAN):
- Inference-Host (GPU-Server): `192.168.178.24`
- AgentDojo-Host: `192.168.178.145`


## Reproduzierbarkeit und Datenspeicherung

Aufgrund der Größe sind vollständige Rohlogs (AgentDojo-Run-Artefakte sowie LM Studio-/vLLM-Logs) nicht im Git-Repository versioniert.
Stattdessen werden sie als Archiv über GitHub Releases (Assets) veröffentlicht.

- Rohlogs: GitHub Release „<NAME/Version>“ (wird bei Fertigstellung ergänzt)
- Datensatz-Index + Checksums: `runs_manifest.json`
- Ergebnisübersicht: `results.csv`


## Versuchsdesign

### Bedingungen

Die Studie ist als Matrix angelegt:

- Backend: `lmstudio` vs. `vllm`
- GPU: `rx6800` vs. `rtx4070ti-super`

Pro Condition **5 Runs**.

### Konstant gehalten

- AgentDojo Benchmark-Suite: `v1.2.2`
- Modell/Quantisierung: identische Modell-ID + Quantisierung (z.B. `openai/gpt-oss-20b`)
- Kontextlänge: `16384`
- Batch-Size (LM Studio): `eval_batch_size = 128` (analog für vLLM so nah wie möglich)
- Temperatur in LM Studio: 0

## Metriken

AgentDojo reportet Metriken pro Suite (workspace, travel, banking, slack) sowie für `combined`.

- Utility: `Average utility`
- Security (AgentDojo): `Average security`
  - In dieser Arbeit wird die **Attack Success Rate (ASR)** als Komplement berechnet:
    - `ASR = 1 - security`  bzw. `ASR[%] = 100 - security[%]`
- Sanity-Check: `Passed injection tasks as user tasks: X/Y`
- Sekundär: Laufzeit (`duration_sec`), Timeouts/Errors (falls vorhanden)


## Ablauf

1. Pro Condition wird ein Benchmark-Run-Ordner erzeugt (eindeutige Run-ID).
2. AgentDojo wird remote auf dem AgentDojo-Host gestartet (SSH).
3. Der Orchestrator sammelt die Run-Artefakte ein (rsync/scp).
4. Zusätzlich werden Inference-Logs gesammelt (LM Studio Server-Logs; bei vLLM analog, sofern konfiguriert).
5. Ergebnisübersichten (`results.csv`) und Index (`runs_manifest.json`) werden aktualisiert.

KV-Cache Handling:
- KV-Cache bleibt **konstant aktiv** (kein KV on/off Switching in dieser Studie).


## Dateien / Artefakte

### `runs_manifest.json` (Datensatz-Index + Checksums)

Enthält Metadaten zu den veröffentlichten Rohlog-Archiven:
- Archiv-/Release-Referenz
- SHA-256 Checksums zur Integritätsprüfung
- Liste der Runs (Run-IDs, Condition-Metadaten, Pfade im Archiv)

### `results.csv` (Ergebnisübersicht pro Run)

Kompakte Tabelle für Auswertung/Plotting, u.a.:
- `run_id`
- `inference_backend` (`lmstudio`/`vllm`)
- `gpu` (z.B. `rx6800`/`rtx4070ti-super`)
- `security` (AgentDojo, z.B. `combined`)
- `asr` (berechnet als `1 - security`)
- `utility` (z.B. `combined`)
- `duration_sec`
- optional: `timeouts`/`errors`

### Run-Ordner (im Rohlog-Archiv)

Typische Struktur pro Run:
- `agentdojo/terminal.log`
- `agentdojo/...` (weitere AgentDojo-Artefakte)
- `lmstudio_logs.tar.gz` (falls LM Studio genutzt)
- ggf. `vllm_logs.tar.gz` (falls vLLM-Logs gesammelt werden)

### `scripts/orchestrator_runs_fixed_kv.py`

Orchestrator zur Durchführung der Studie:
- führt AgentDojo N-mal pro Condition aus
- KV-Cache bleibt konstant aktiv
- sammelt AgentDojo-Artefakte + Inference-Logs ein
- schreibt `results.csv` und `runs_manifest.json`

### `scripts/bench_config.template.json`

Template für Hosts, Pfade und Run-Parameter.


## Quickstart

Voraussetzung: AgentDojo-Host + Inference-Host sind erreichbar (LAN) und die SSH-Zugänge sind eingerichtet.

1) Template kopieren:
```bash
cp scripts/bench_config.template.json scripts/bench_config.local.json
```

2) In `scripts/bench_config.local.json` anpassen:
- Hosts/IPs
- `OPENAI_BASE_URL` auf den jeweiligen Inference-Endpoint (LM Studio oder vLLM)
- `study.runs = 5`

3) Ausführen:
```bash
python3 scripts/orchestrator_v2._py scripts/bench_config.local.json
```

4) Rohlogs veröffentlichen:
- Run-Ordner als Archiv packen
- GitHub Release erstellen (Asset hochladen)
- Release/Archivname in der Studie dokumentieren (Referenz in `runs_manifest.json`)
