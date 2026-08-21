"""LoRA weight synchronisation: HF training model → vLLM generation engine.

After each optimizer step, updated LoRA weights are saved to a tmpfs
directory (/dev/shm, in-memory on Linux) and hot-swapped into vLLM via
a fresh LoRARequest.  Round-trip is typically ~100–150 ms for a rank-20
adapter — negligible compared to the generation speedup.

Usage:
    syncer = LoRAWeightSyncer()
    syncer.sync(peft_model, vllm_engine, adapter_name="adapter_a")
"""
from __future__ import annotations

import os
import time
from pathlib import Path


_PREFER_TMPFS = "/dev/shm"
_FALLBACK_TMP = "/tmp"


class LoRAWeightSyncer:
    """Saves LoRA weights to shared memory and reloads them in vLLM.

    A new `LoRARequest` id is used each time so vLLM never serves a
    stale cached copy.  The previous adapter slot is freed immediately
    after the new request is installed.
    """

    def __init__(self, sync_dir: str | None = None):
        if sync_dir is None:
            base = _PREFER_TMPFS if os.path.isdir(_PREFER_TMPFS) else _FALLBACK_TMP
            sync_dir = os.path.join(base, "masteer_lora_sync")
        self.sync_dir = Path(sync_dir)
        self.sync_dir.mkdir(parents=True, exist_ok=True)
        self._last_sync_s: float = 0.0

    # ------------------------------------------------------------------

    def sync(
        self,
        peft_model,
        vllm_engine,
        adapter_name: str = "adapter_a",
    ) -> float:
        """Save `adapter_name` weights and hot-swap into `vllm_engine`.

        Returns elapsed seconds so the caller can log it.
        """
        t0 = time.time()

        # Overwrite previous adapter weights (same directory, no accumulation).
        peft_model.save_pretrained(
            str(self.sync_dir),
            selected_adapters=[adapter_name],
        )

        # PEFT saves to sync_dir/<adapter_name>/ when multiple adapters exist.
        adapter_dir = self.sync_dir / adapter_name
        if not (adapter_dir / "adapter_config.json").exists():
            # Older PEFT saved directly to sync_dir root — fall back.
            adapter_dir = self.sync_dir
        vllm_engine.load_lora(str(adapter_dir))

        self._last_sync_s = time.time() - t0
        return self._last_sync_s

    # ------------------------------------------------------------------

    def initial_save(
        self,
        peft_model,
        vllm_engine,
        adapter_name: str = "adapter_a",
    ) -> None:
        """Save adapter at training start so vLLM has weights from step 0."""
        self.sync(peft_model, vllm_engine, adapter_name=adapter_name)
        print(f"  [weight_sync] initial LoRA load complete "
              f"({self._last_sync_s*1000:.0f} ms) → {self.sync_dir}",
              flush=True)
