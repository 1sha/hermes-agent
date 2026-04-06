"""Time-based tiered context compaction (alternate compressor).

ALTERNATE implementation — does NOT replace the default ContextCompressor.
Use the ``compression.strategy`` config key to switch between ``position``
(default) and ``tiered`` (this class).

Instead of the position-based HEAD/MIDDLE/TAIL approach, this compressor
assigns messages to time-based tiers and applies different compression
ratios per tier:

  Tier 0 (recent):   < T0 hours old  →  full detail, no compression
  Tier 1 (moderate): T0 – T1 hours   →  ~60% detail (light summary)
  Tier 2 (old):      > T1 hours       →  ~20% detail (heavy summary)

Time brackets and compression ratios are fully configurable via
config.yaml under ``compression.tiered``.

Falls back to position-based compression when messages lack timestamps.

Feature-flagged via ``compression.strategy: tiered`` in config.yaml.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from agent.context_compressor import (
    ContextCompressor,
    SUMMARY_PREFIX,
    _PRUNED_TOOL_PLACEHOLDER,
    _MIN_SUMMARY_TOKENS,
    _SUMMARY_RATIO,
    _SUMMARY_TOKENS_CEILING,
    _CHARS_PER_TOKEN,
)
from agent.auxiliary_client import call_llm
from agent.model_metadata import estimate_messages_tokens_rough

logger = logging.getLogger(__name__)

# ── Default tier configuration ──────────────────────────────────────────

DEFAULT_TIERS = [
    # (max_age_hours, detail_ratio, label)
    # Messages younger than max_age_hours get this detail_ratio
    (12,  1.0,  "recent"),     # < 12h: full detail, no compression
    (24,  0.60, "moderate"),   # 12–24h: compress to ~60%
    (48,  0.20, "old"),        # 24–48h: compress to ~20%
    # Anything older than the last tier's max_age gets the last ratio
]

# The catch-all ratio for messages older than all defined tiers
DEFAULT_ANCIENT_RATIO = 0.10


# ── Timestamp helpers ───────────────────────────────────────────────────

def get_message_timestamp(msg: Dict[str, Any]) -> Optional[float]:
    """Extract a Unix timestamp from a message, if present.

    Checks for ``_hermes_ts`` (float epoch) first, then ``_hermes_iso``
    (ISO-8601 string).  Returns None if neither is present.
    """
    ts = msg.get("_hermes_ts")
    if ts is not None:
        try:
            return float(ts)
        except (TypeError, ValueError):
            pass

    iso = msg.get("_hermes_iso")
    if iso:
        try:
            dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
            return dt.timestamp()
        except (TypeError, ValueError):
            pass

    return None


def stamp_message(msg: Dict[str, Any], ts: Optional[float] = None) -> Dict[str, Any]:
    """Add a ``_hermes_ts`` timestamp to a message dict (in-place + return).

    If ``ts`` is None, uses ``time.time()``.  Idempotent — won't overwrite
    an existing timestamp.
    """
    if "_hermes_ts" not in msg:
        msg["_hermes_ts"] = ts if ts is not None else time.time()
    return msg


# ── Tier assignment ─────────────────────────────────────────────────────

def _assign_tier(
    msg_ts: Optional[float],
    now: float,
    tiers: List[Tuple[float, float, str]],
    ancient_ratio: float,
) -> Tuple[float, str]:
    """Return (detail_ratio, label) for a message based on its age.

    If ``msg_ts`` is None (no timestamp), returns (None, "unknown") so the
    caller can decide on a fallback strategy.
    """
    if msg_ts is None:
        return (None, "unknown")

    age_hours = (now - msg_ts) / 3600.0
    if age_hours < 0:
        age_hours = 0  # future timestamp → treat as recent

    for max_age_h, ratio, label in tiers:
        if age_hours < max_age_h:
            return (ratio, label)

    # Older than all defined tiers → ancient
    return (ancient_ratio, "ancient")


# ── Main class ──────────────────────────────────────────────────────────

class TieredContextCompressor(ContextCompressor):
    """Time-based tiered context compressor.

    Extends ContextCompressor with time-aware compression.  Messages are
    grouped into time tiers, and each tier is summarized at a different
    compression ratio.  Full-detail tiers are left untouched.

    Falls back to the parent class's position-based compression when fewer
    than ``min_timestamped_fraction`` of messages have timestamps.
    """

    def __init__(
        self,
        *args,
        tiers: Optional[List[Dict[str, Any]]] = None,
        ancient_ratio: float = DEFAULT_ANCIENT_RATIO,
        min_timestamped_fraction: float = 0.5,
        **kwargs,
    ):
        """
        Parameters
        ----------
        tiers : list of dicts, optional
            Each dict: ``{"max_age_hours": N, "detail_ratio": 0.0-1.0,
            "label": "..."}``  Sorted ascending by max_age_hours.
            Defaults to DEFAULT_TIERS.
        ancient_ratio : float
            Compression ratio for messages older than all defined tiers.
        min_timestamped_fraction : float
            If fewer than this fraction of non-system messages have timestamps,
            fall back to position-based compression.
        """
        super().__init__(*args, **kwargs)

        # Parse tier config
        if tiers:
            self._tiers = [
                (
                    float(t.get("max_age_hours", 48)),
                    float(t.get("detail_ratio", 0.5)),
                    str(t.get("label", f"tier-{i}")),
                )
                for i, t in enumerate(tiers)
            ]
            # Sort by max_age_hours ascending
            self._tiers.sort(key=lambda x: x[0])
        else:
            self._tiers = list(DEFAULT_TIERS)

        self._ancient_ratio = max(0.0, min(1.0, ancient_ratio))
        self._min_ts_fraction = max(0.0, min(1.0, min_timestamped_fraction))

    # ── Tier-based grouping ─────────────────────────────────────────────

    def _group_by_tier(
        self,
        messages: List[Dict[str, Any]],
        start: int,
        end: int,
        now: float,
    ) -> Dict[str, List[int]]:
        """Group message indices by their time tier.

        Returns a dict mapping tier label → list of indices.
        Messages with detail_ratio == 1.0 are placed in a special "keep" group.
        Messages without timestamps go into "unknown".
        """
        groups: Dict[str, List[int]] = {}
        for i in range(start, end):
            ratio, label = _assign_tier(
                get_message_timestamp(messages[i]),
                now,
                self._tiers,
                self._ancient_ratio,
            )
            if ratio is not None and ratio >= 1.0:
                label = "keep"
            groups.setdefault(label, []).append(i)
        return groups

    def _has_enough_timestamps(
        self, messages: List[Dict[str, Any]], start: int, end: int,
    ) -> bool:
        """Check if enough messages have timestamps for tiered compression."""
        if end <= start:
            return False
        count = 0
        total = 0
        for i in range(start, end):
            if messages[i].get("role") == "system":
                continue
            total += 1
            if get_message_timestamp(messages[i]) is not None:
                count += 1
        if total == 0:
            return False
        return (count / total) >= self._min_ts_fraction

    # ── Tier-specific summary generation ────────────────────────────────

    def _generate_tier_summary(
        self,
        turns: List[Dict[str, Any]],
        tier_label: str,
        detail_ratio: float,
    ) -> Optional[str]:
        """Generate a summary for a single tier of messages.

        ``detail_ratio`` controls how much detail to preserve:
        - 0.6 → moderate summary, keep key details
        - 0.2 → aggressive summary, only essence
        - 0.1 → ultra-compressed, just outcomes
        """
        if not turns:
            return None

        base_budget = self._compute_summary_budget(turns)
        # Scale budget by detail ratio
        budget = max(_MIN_SUMMARY_TOKENS, int(base_budget * detail_ratio))

        content = self._serialize_for_summary(turns)

        file_chain_instruction = (
            "ALWAYS preserve full absolute file paths for every file read, modified, or created. "
            "Include version history (git commit hashes or snapshot paths) when available. "
            "Note: '⚠️ Re-read file before modifying — raw contents not in summary.'"
        )

        if detail_ratio >= 0.5:
            detail_instruction = (
                "Preserve key details: file paths with version chains, commands, specific values, "
                "error messages, and decision rationale. This is a MODERATE "
                "compression — keep enough that someone could understand the "
                f"specific work done. {file_chain_instruction}"
            )
        elif detail_ratio >= 0.15:
            detail_instruction = (
                "Compress aggressively. Keep only: goals achieved, key decisions "
                "made, files modified (with full absolute paths and version info), "
                f"and blockers encountered. Omit command outputs and verbose details. {file_chain_instruction}"
            )
        else:
            detail_instruction = (
                "Ultra-compressed. Keep only: what was accomplished (1-2 sentences), "
                "any persistent state changes (files created/modified — ALWAYS include "
                f"full absolute file paths), and unresolved blockers. {file_chain_instruction}"
            )

        prompt = f"""Summarize the following conversation segment ({tier_label} tier, {detail_ratio:.0%} detail target).

{detail_instruction}

CONVERSATION SEGMENT:
{content}

Target ~{budget} tokens. Write a concise summary. No preamble — just the summary content.

Use this structure:
### {tier_label.title()} Context
**When:** [approximate timeframe]
**What:** [work performed]
**Outcome:** [results and state changes]
**File version chain:** [For EVERY file touched — full absolute path, current state, version history with git hashes or snapshot paths. Mark each with ⚠️ Re-read before modifying.]
"""

        try:
            call_kwargs = {
                "task": "compression",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": budget * 2,
            }
            if self.summary_model:
                call_kwargs["model"] = self.summary_model
            response = call_llm(**call_kwargs)
            content = response.choices[0].message.content
            if not isinstance(content, str):
                content = str(content) if content else ""
            return content.strip()
        except RuntimeError:
            logger.warning(
                "Tiered compressor: no provider for %s tier summary. "
                "Tier will be dropped without summary.", tier_label,
            )
            return None
        except Exception as e:
            logger.warning(
                "Tiered compressor: %s tier summary failed: %s", tier_label, e,
            )
            return None

    # ── Main compress override ──────────────────────────────────────────

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
    ) -> List[Dict[str, Any]]:
        """Compress conversation using time-based tiered approach.

        Algorithm:
          0. Archive full context (lossless backup, inherited from parent)
          1. Check if enough messages have timestamps
             → If not, fall back to parent's position-based compression
          2. Protect head messages (system prompt + first exchange)
          3. Group middle+tail messages by time tier
          4. Keep full-detail tier messages as-is
          5. Summarize each compressed tier separately
          6. Reassemble: head + tier summaries + recent (full-detail) messages
          7. Sanitize tool call/result pairs
        """
        n_messages = len(messages)
        if n_messages <= self.protect_first_n + self.protect_last_n + 1:
            if not self.quiet_mode:
                logger.warning(
                    "Tiered compressor: only %d messages (need > %d), skipping",
                    n_messages,
                    self.protect_first_n + self.protect_last_n + 1,
                )
            return messages

        # Determine compressible range
        compress_start = self.protect_first_n
        compress_start = self._align_boundary_forward(messages, compress_start)

        # Check timestamp availability in the compressible range
        if not self._has_enough_timestamps(messages, compress_start, n_messages):
            if not self.quiet_mode:
                logger.info(
                    "Tiered compressor: insufficient timestamps (need %.0f%% coverage). "
                    "Falling back to position-based compression.",
                    self._min_ts_fraction * 100,
                )
            return super().compress(messages, current_tokens)

        # Phase 0: Archive
        self._archive_before_compaction(messages)

        display_tokens = current_tokens if current_tokens else (
            self.last_prompt_tokens or estimate_messages_tokens_rough(messages)
        )

        # Phase 1: Prune old tool results (cheap pre-pass)
        messages, pruned_count = self._prune_old_tool_results(
            messages, protect_tail_count=self.protect_last_n * 3,
        )
        if pruned_count and not self.quiet_mode:
            logger.info("Tiered pre-compression: pruned %d old tool result(s)", pruned_count)

        # Phase 1.5: Stamp any messages that lack timestamps (fallback for
        # messages created before the stamping was wired up, or from code
        # paths that don't explicitly call stamp_message).
        _unstamped = 0
        for i in range(compress_start, n_messages):
            if get_message_timestamp(messages[i]) is None:
                stamp_message(messages[i])
                _unstamped += 1
        if _unstamped and not self.quiet_mode:
            logger.info(
                "Tiered compressor: stamped %d previously-unstamped messages",
                _unstamped,
            )

        # Phase 2: Group by time tier
        now = time.time()
        groups = self._group_by_tier(messages, compress_start, n_messages, now)

        if not self.quiet_mode:
            logger.info(
                "Tiered compression triggered (%d tokens >= %d threshold)",
                display_tokens, self.threshold_tokens,
            )
            for label, indices in sorted(groups.items()):
                logger.info(
                    "  Tier '%s': %d messages (indices %d–%d)",
                    label, len(indices),
                    min(indices) if indices else 0,
                    max(indices) if indices else 0,
                )

        # Phase 3: Build the compressed message list
        compressed = []

        # Head messages (protected)
        for i in range(compress_start):
            msg = messages[i].copy()
            if i == 0 and msg.get("role") == "system" and self.compression_count == 0:
                msg["content"] = (
                    (msg.get("content") or "")
                    + "\n\n[Note: Some earlier conversation turns have been compacted "
                    "into a tiered summary based on message age. Recent context is "
                    "preserved in full. Build on the summary and current state "
                    "rather than re-doing work.]"
                )
            compressed.append(msg)

        # Collect tier summaries (ordered by age: oldest first)
        tier_summaries = []
        tier_order = []  # (label, detail_ratio) for ordering

        for max_age_h, ratio, label in self._tiers:
            if label in groups and ratio < 1.0:
                tier_order.append((label, ratio))
        if "ancient" in groups:
            tier_order.append(("ancient", self._ancient_ratio))
        if "unknown" in groups:
            # Treat unknown-timestamp messages as moderately old
            tier_order.append(("unknown", 0.40))

        # Generate summaries for each tier that needs compression
        for label, ratio in tier_order:
            indices = groups.get(label, [])
            if not indices:
                continue
            tier_turns = [messages[i] for i in sorted(indices)]
            summary = self._generate_tier_summary(tier_turns, label, ratio)
            if summary:
                tier_summaries.append(summary)

        # Combine all tier summaries into one compaction message
        if tier_summaries:
            combined_summary = (
                f"{SUMMARY_PREFIX}\n\n"
                + "\n\n---\n\n".join(tier_summaries)
            )
            # Store for iterative updates (parent class feature)
            self._previous_summary = "\n\n---\n\n".join(tier_summaries)

            # Pick a role that avoids consecutive same-role
            last_head_role = (
                messages[compress_start - 1].get("role", "user")
                if compress_start > 0 else "user"
            )
            summary_role = "user" if last_head_role in ("assistant", "tool") else "assistant"
            compressed.append({"role": summary_role, "content": combined_summary})

        # Add "keep" tier messages (recent, full detail) in original order
        keep_indices = sorted(groups.get("keep", []))
        for i in keep_indices:
            compressed.append(messages[i].copy())

        self.compression_count += 1
        compressed = self._sanitize_tool_pairs(compressed)

        if not self.quiet_mode:
            new_estimate = estimate_messages_tokens_rough(compressed)
            saved = display_tokens - new_estimate
            logger.info(
                "Tiered compression: %d → %d messages (~%d tokens saved)",
                n_messages, len(compressed), saved,
            )
            logger.info("Tiered compression #%d complete", self.compression_count)

        return compressed
