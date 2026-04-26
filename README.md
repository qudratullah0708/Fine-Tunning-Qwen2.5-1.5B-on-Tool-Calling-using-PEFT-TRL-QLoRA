# Pocket-Agent: 1.5B Tool-Calling Assistant

Fine-tuning pipeline for an on-device assistant that emits structured tool calls in `<tool_call>...</tool_call>` format, with refusal behavior for unsupported/ambiguous requests.

This project is built around the hackathon brief in `ML-PS.md`, with implementation flow in `pocket_agent.ipynb` and synthetic data generation in `generate_data.py`.

## Project Goal

Train and deploy a compact model (<=2B params base) that can:

- Produce valid single-turn tool calls
- Resolve references in multi-turn chat (for example, "convert that to GBP")
- Refuse safely in plain text when no tool applies
- Run offline using a quantized GGUF model

## Tool Schema

The model is trained to output only one of these tools (or a plain-text refusal):

- `weather`: `{"location": "string", "unit": "C|F"}`
- `calendar`: `{"action": "list|create", "date": "YYYY-MM-DD", "title": "string?"}`
- `convert`: `{"value": "number", "from_unit": "string", "to_unit": "string"}`
- `currency`: `{"amount": "number", "from": "ISO3", "to": "ISO3"}`
- `sql`: `{"query": "string"}`

Expected output format:

```text
<tool_call>{"tool":"weather","args":{"location":"Karachi","unit":"C"}}</tool_call>
```

## Repository Contents

- `pocket_agent.ipynb`: end-to-end Colab workflow (install -> generate data -> fine-tune -> quantize -> inference export -> self-test -> Gradio)
- `generate_data.py`: creates synthetic supervised fine-tuning data
- `ML-PS.md`: hackathon problem statement, constraints, and grading rubric
- `README.md`: setup and usage guide

## Environment

Recommended runtime:

- Google Colab with T4 GPU (16GB VRAM) for training
- CPU runtime for final offline inference check

Python dependencies used in the notebook:

```bash
pip install -q transformers==4.47.0 peft trl bitsandbytes datasets accelerate gradio
pip install -q llama-cpp-python
```

System tools required for GGUF tooling:

```bash
apt-get update -qq
apt-get install -y -qq cmake build-essential git
```

## Data Generation

Generate synthetic training data:

```bash
python generate_data.py
```

This writes `training_data.jsonl` with mixed slices:

- Clean tool calls
- Multi-turn reference resolution
- Adversarial prompts (typos + code-switching)
- Refusals (no tool, ambiguous, chitchat)

## Fine-Tuning Configuration

From `pocket_agent.ipynb`:

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Adapter method: LoRA (PEFT)
- Quantized training load: 4-bit NF4 (`bitsandbytes`)
- Key hyperparameters:
  - `MAX_SEQ_LEN=512`
  - `EPOCHS=3`
  - `BATCH_SIZE=2`
  - `GRAD_ACCUM=4`
  - `LR=2e-4`
  - `LORA_RANK=16`, `LORA_ALPHA=32`, `LORA_DROPOUT=0.05`

Training output:

- LoRA adapter directory: `./lora_adapter`

## Quantization Pipeline (GGUF)

After training, the notebook performs:

1. Merge base + LoRA adapter into `./merged_model`
2. Build `llama.cpp`
3. Convert merged model to FP16 GGUF
4. Quantize to `q4_k_m` (or `q2_k` for more aggressive size target)

Artifacts:

- `model_q4.gguf` (default quantized model)

## Inference Contract

The notebook exports `inference.py` with the required grader function:

```python
def run(prompt: str, history: list[dict]) -> str
```

Behavior:

- Loads local GGUF model via `llama_cpp`
- Applies system prompt and conversation history
- Returns either:
  - a validated `<tool_call>...</tool_call>` payload, or
  - plain-text refusal
- Uses no network imports/calls

## Quick Validation

Notebook self-test checks:

- Clean tool calls (`weather`, `convert`, `currency`, `sql`, `calendar`)
- Refusal prompts
- Adversarial prompt handling
- Multi-turn reference resolution using `history`

## Run Flow (Recommended)

1. Open and run `pocket_agent.ipynb` cells in order.
2. Confirm `training_data.jsonl` is generated.
3. Fine-tune and save LoRA adapter.
4. Run smoke test before quantization.
5. Merge + quantize to GGUF.
6. Export and self-test `inference.py`.
7. Launch Gradio demo cell for manual interaction.

## Notes

- Keep training prompts disjoint from public test prompts as required by `ML-PS.md`.
- Validate argument fidelity carefully (units, ISO3 codes, dates, numbers).
- If model size exceeds hard gate target, switch quant type (for example from `q4_k_m` to `q2_k`) and re-check quality.
