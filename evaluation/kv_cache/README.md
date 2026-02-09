# KV-Cache-Teiluntersuchung (LM Studio)

## Ziel

Untersucht wird, ob die Aktivierung des KV-Cache die Benchmark-Ergebnisse unter AgentDojo (insb. ASR/Utility) relevant beeinflusst oder ob der KV-Cache primär einen Performance-/VRAM-Trade-off darstellt.

Hintergrund: Einige Modelle passen nur ohne KV-Cache sicher in den verfügbaren VRAM der lokalen Hardware. Für die spätere Evaluation sollen daher Modelle je nach VRAM-Fit mit oder ohne KV-Cache benchmarkbar sein, ohne dass dadurch die inhaltliche Aussagekraft (ASR/Utility) wesentlich verzerrt wird.

Stand: 08.02.2026 (Europe/Berlin)


## Setup-Überblick

- Benchmark: AgentDojo (Benchmark-Suite-Version fixiert auf `v1.2.2`)
- Inferenz: LM Studio (GPU-Server im lokalen Netz)
- AgentDojo und LM Studio laufen auf getrennten Hosts.

Hosts (LAN):
- GPU-Server (LM Studio): `192.168.178.24`
- AgentDojo-Host: `192.168.178.145`


## Reproduzierbarkeit und Datenspeicherung

Die vollständigen Rohlogs (Run-Artefakte) sind aufgrund ihrer Größe nicht im Git-Repository versioniert.  
Stattdessen werden sie als Archiv über GitHub Releases (Assets) veröffentlicht.

- Rohlogs: GitHub Release „<NAME/Version>“ (wird bei Fertigstellung ergänzt)
- Datensatz-Index + Checksums: `runs_manifest.json`
- Ergebnisübersicht: `results.csv`


## Ablauf

Bedingungen:
- `kv_on`: KV-Cache aktiviert
- `kv_off`: KV-Cache deaktiviert
- abwechselnd ausführen (z.B. Run01 kv_on, Run01 kv_off, Run02 kv_on, …), um Drift-Effekte zu reduzieren

Wiederholungen:
- initial 3 Runs pro Bedingung

Konstant gehalten:
- AgentDojo Package-Version/Commit: [v0.1.35](https://github.com/ethz-spylab/agentdojo/releases/tag/v0.1.35) - Stand: 27.10.2025
- AgentDojo Benchmark-Suite: `v1.2.2`
- AgentDojo Konfiguration: `config/agentdojo/settings.yaml`
- LM Studio Version (`0.4.2 (Build 2)` - Stand: [06.02.2026](https://lmstudio.ai/changelog/lmstudio-v0.4.2))
- identische Modell-/Quantisierungseinstellungen ([openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b))
- Batch-Size in LM Studio: 128


## Metriken

AgentDojo reportet Metriken pro Suite (workspace, travel, banking, slack) sowie für `combined`.

- Utility: `Average utility` (pro Suite und `combined`)
- Security (AgentDojo): `Average security` (pro Suite und `combined`)
  - In dieser Arbeit wird die **Attack Success Rate (ASR)** als Komplement berechnet:
    - `ASR = 1 - security`  (bzw. in Prozent: `ASR[%] = 100 - security[%]`)
  - Hinweis: Für spezielle Angriffstypen (z.B. DoS) kann die Interpretation von `security` abweichen; in diesem Fall wird die AgentDojo-Definition unverändert dokumentiert.
- Sanity-Check: `Passed injection tasks as user tasks: X/Y` (pro Suite und `combined`)
  - dient der Plausibilitätsprüfung und ist **keine** ASR-Metrik
- Sekundär: Laufzeit (`duration_sec`) / Timeouts/Errors (falls vorhanden) zur Einordnung von Performance-Effekten


## Dateien / Artefakte

### `runs_manifest.json` (Datensatz-Index + Checksums)
Enthält die wichtigsten Metadaten zu einem veröffentlichten Rohlog-Archiv:
- Release-/Archivname (ZIP/RAR Asset)
- SHA-256 Checksum (Integritätsprüfung)
- Liste der enthaltenen Runs (Run-IDs, KV an/aus, Benchmark-Suite-Version, Pfade im Archiv)

### `results.csv` (Ergebnisübersicht pro Run)
Kompakte Tabelle zur Auswertung mit folgenden Spalten:
- `run_id`
- `kv_cache` (`on`/`off`)
- `security` (AgentDojo `Average security`, z.B. `combined`)
- `asr` (berechnet als `1 - security`)
- `utility` (AgentDojo `Average utility`, z.B. `combined`)
- `duration_sec`
- optional: `timeouts`/`errors`

Hinweis: Detaillierte Run-Artefakte liegen im Release-Archiv (siehe `runs_manifest.json`).

### `scripts/orchestrator.py`
Orchestrator zur automatisierten Durchführung der Runs:
- erzeugt pro Run einen eindeutigen Run-Ordner
- startet AgentDojo remote (SSH)
- sammelt AgentDojo-Artefakte ein
- sammelt LM Studio Logs ein

### `config/bench_config.template.json`
Konfiguration für Hosts, Pfade und Run-Commands.
