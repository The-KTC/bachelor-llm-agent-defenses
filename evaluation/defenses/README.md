# Phase B – AgentDojo Custom Defenses (D1–D3)

Eigene Defense-Implementierungen für die Schichten **D1** (Spotlighting nach [Hines et al., 2024](https://arxiv.org/abs/2403.14720)), **D2** (Structured Output) und **D3** (Provenance + Policy Gate) sowie ein Custom-Runner zur Integration in [AgentDojo](https://agentdojo.spylab.ai/) v1.2.2.


## Defense-IDs

Der Custom-Runner akzeptiert folgende `--defense-id`-Werte (kombinierbar):

| Defense-ID | Schicht | Beschreibung |
|---|---|---|
| `d1_delimiting` | D1-A | Spotlighting: Einfassen in Delimiter-Marker ([Hines et al., 2024](https://arxiv.org/abs/2403.14720)) |
| `d1_datamarking` | D1-B | Spotlighting: Datamarking via Zeichenersetzung ([Hines et al., 2024](https://arxiv.org/abs/2403.14720)) |
| `d1_encoding` | D1-C | Spotlighting: Base64-Encoding des Tool-Outputs ([Hines et al., 2024](https://arxiv.org/abs/2403.14720)) |
| `d2_json` | D2 | JSON-Envelope mit Metadaten-Feldern |
| `d3_balanced` | D3 | Call-Gate im Balanced-Modus (Task-Prompt-Binding für schreibende Calls) |
| `d3_strict` | D3 | Call-Gate im Strict-Modus (Deny-by-Default, nur lesende Tools erlaubt) |
| `d3_output_gate` | D3 | Output-Gate (Redaktion sensibler Daten in der finalen Antwort) |

D3-Stacks kombinieren typischerweise ein Call-Gate-Modus (`d3_balanced` oder `d3_strict`) mit `d3_output_gate`.


## Evaluierte Stacks (Phase B)

| Stack | Defense-IDs |
|---|---|
| D0 Vanilla | – (offizielle AgentDojo-CLI) |
| D1 Delimiting | `d1_delimiting` |
| D1 Datamarking | `d1_datamarking` |
| D1 Encoding | `d1_encoding` |
| D2 JSON | `d2_json` |
| D3 Balanced | `d3_balanced`, `d3_output_gate` |
| D3 Strict | `d3_strict`, `d3_output_gate` |


## Dateien

```
agentdojo_custom_defenses/
├── __init__.py                    Package-Exports
├── common.py                      Shared Utilities (Serialisierung, Base64, YAML)
├── d1_delimiting.py               D1-A Delimiting-Formatter
├── d1_datamarking.py              D1-B Datamarking-Formatter
├── d1_encoding.py                 D1-C Encoding-Formatter
├── d2_structured_output_json.py   D2 JSON-Envelope-Formatter
├── d3_call_gate.py                D3 Call-Gate (PEP auf dem Call-Pfad)
├── d3_output_gate.py              D3 Output-Gate (Filter auf dem Response-Pfad)
└── custom_pipeline_factory.py     Mapping von defense_id(s) auf Formatter/Gates

run_benchmark_custom_pipeline.py   Custom-Runner (Entry-Point für Defense-Runs)
```


## Architektur

```
User-Task  ──►  LLM-Agent
                    │
                    ▼
              [D3 Call-Gate]          Call-Pfad (Agent → Tool)
                    │                 Prüft Tool-Calls VOR Ausführung
                    ▼
                Tool-Ausführung
                    │
                    ▼
              [D1/D2 Formatter]       Output-Pfad (Tool → Agent)
                    │                 Transformiert Tool-Output VOR Kontextfusion
                    ▼
                LLM-Agent
                    │
                    ▼
              [D3 Output-Gate]        Response-Pfad (Agent → Nutzer)
                    │                 Filtert finale Antwort VOR Rückgabe
                    ▼
              Evaluator / Nutzer
```

- **D1/D2** transformieren Tool-Outputs auf dem Output-Pfad (via `ToolsExecutor(tool_output_formatter=...)`)
- **D3 Call-Gate** prüft Tool-Calls auf dem Call-Pfad (Agent → Tool) gegen deterministische Regeln
- **D3 Output-Gate** filtert die finale Textantwort auf dem Response-Pfad (Agent → Evaluator/Nutzer)


## Verwendung

D0-Run (Vanilla, offizielle CLI):
```bash
python -m agentdojo.scripts.benchmark \
    --model openai/gpt-oss-20b \
    --attack tool_knowledge
```

Defense-Run (z.B. D3 Balanced):
```bash
python run_benchmark_custom_pipeline.py \
    --model openai/gpt-oss-20b \
    --attack tool_knowledge \
    --defense-id d3_balanced \
    --defense-id d3_output_gate
```


## Integration

1. Ordner `agentdojo_custom_defenses/` ins Workdir kopieren
2. `run_benchmark_custom_pipeline.py` ins Workdir legen
3. Orchestrator für `defense_id != D0` auf den Custom-Runner zeigen lassen
