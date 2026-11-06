"""Summaries of finished transcripts, written by Claude through the Anthropic API.

The key comes from ANTHROPIC_API_KEY, which the service reads from /etc/voice-to-text/env
(server/set-api-key.sh writes it). Without one, summaries are off and transcripts are
still saved.
"""

import os

import anthropic

MODEL = "claude-haiku-5-5"  # the user's choice (2026-11-06): fast and cheap, $0.10 / $0.50 per MTok

_SYSTEM = """You summarize transcripts of things a person dictated with a voice-to-text tool.

The transcript comes from speech recognition, so expect missing punctuation, misheard words and
Hindi words written out in English letters (Hinglish). Read through those; don't comment on them.

Write the summary in English, as plain text with no markdown headings or bold:
- First, one to three sentences on what was said.
- Then, only if there are any, a "Key points:" line followed by short points, each on its own line starting with "- ".
- Then, only if the speaker mentioned things to do, an "Action items:" line followed by those, each starting with "- ".
When there are no key points or no action items, leave that part out without remarking on it.

Keep it shorter than the transcript. For a transcript of a sentence or two, the one-to-three-sentence part alone is enough."""


_MEETING_SYSTEM = """You summarize transcripts of meetings, recorded with a voice-to-text tool.

Each line is one turn: "[m:ss] You: ..." is the person who recorded the meeting, on their
microphone; "[m:ss] Meeting: ..." is everyone else, heard through the meeting's audio, and can be
several different people. A transcript of an uploaded recording may have no labels at all. The
text comes from speech recognition, so expect misheard words and missing punctuation; read
through them and don't comment on them.

Write the summary in English, as plain text with no markdown headings or bold:
- First, two to four sentences: what the meeting was about and how it ended.
- Then a "Key points:" line followed by the main points, each on its own line starting with "- ".
- Then, only if anything was decided, a "Decisions:" line followed by those, each starting with "- ".
- Then, only if anyone took on something to do, an "Action items:" line followed by those, each
  starting with "- " and naming who does it when the transcript says ("You" for the recorder).
When there are no decisions or no action items, leave that part out without remarking on it.
Keep it much shorter than the transcript."""


class SummaryError(Exception):
    """A summary that couldn't be written; the message is shown on the site."""


class Summarizer:
    def __init__(self):
        self.enabled = bool(os.environ.get("ANTHROPIC_API_KEY"))
        self._client = anthropic.AsyncAnthropic(timeout=120.0, max_retries=2) if self.enabled else None

    async def summarize(self, transcript: str, mode: str = "dictation") -> tuple[str, str]:
        """(summary, the model that wrote it) of a dictation or a meeting. Raises SummaryError."""
        try:
            # Haiku 5.5 thinks by default (adaptive) and has no server-side refusal fallback;
            # a decline comes back as stop_reason "refusal", handled below.
            response = await self._client.messages.create(
                model=MODEL,
                max_tokens=16000,
                system=_MEETING_SYSTEM if mode == "meeting" else _SYSTEM,
                messages=[{"role": "user", "content": f"<transcript>\n{transcript}\n</transcript>"}],
                output_config={"effort": "low"},  # a summary is a simple task
            )
        except anthropic.AuthenticationError:
            raise SummaryError("The Anthropic API key was rejected: run server/set-api-key.sh again") from None
        except anthropic.PermissionDeniedError:
            raise SummaryError("The Anthropic API key isn't allowed to use this model") from None
        except anthropic.RateLimitError:
            raise SummaryError("Rate limited by the Anthropic API: try again in a minute") from None
        except anthropic.APIStatusError as e:
            raise SummaryError(f"Anthropic API error {e.status_code}: {e.message}") from None
        except anthropic.APIConnectionError:
            raise SummaryError("Couldn't reach the Anthropic API") from None
        if response.stop_reason == "refusal":
            raise SummaryError("Claude declined to summarize this transcript")
        text = "\n".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            raise SummaryError(f"The summary came back empty (stop reason: {response.stop_reason})")
        return text, response.model
