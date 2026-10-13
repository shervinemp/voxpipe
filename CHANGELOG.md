# Changelog

## 0.2.0

A correctness and design pass over the whole package. Several public APIs
changed; the upgrade notes are at the end.

### Fixed

- **Startup.** The pipeline no longer imports the removed `storage.memory`
  module; the context/memory refactor is wired in end to end.
- **Packaging.** Wheels now contain the default config and model manifest
  (`import voxpipe` failed on any non-editable install).
- **Memory.** SQLite storage commits its writes (nothing was ever persisted),
  ranks recall with BM25 instead of "newest row matching the first word",
  and treats `ttl_days: 0` as "keep forever" rather than "delete after 1 s".
  Evicted turns are archived with their real text and role, and recalled
  context is attached to the prompt without being written back into memory.
- **Streaming.** Replies stream token by token and are spoken sentence by
  sentence; previously the whole reply was generated before the first word.
- **Barge-in.** Speech recognition runs on its own thread, so talking over a
  reply cancels generation and cuts playback. Interrupted audio no longer
  resumes from sentences that were still being synthesised. Follow-up turns
  are no longer mislabelled as interruptions.
- **Tools.** Async callbacks, repeated calls to one tool in a pass,
  `X | None` type hints, `**kwargs` callbacks, `Tool.from_class` and the
  `@permission` decorator all work. Remembered permissions are honoured. A
  tool that speaks its own result no longer triggers a second, duplicate
  answer.
- **Decoders.** Tool calls are recognised when their tags arrive split across
  stream chunks; Gemma string arguments may contain commas, colons and quotes.
- **Hotkeys.** Character keys and generic modifiers (`<ctrl>+k`) match on
  Windows; key auto-repeat no longer re-fires an action; actions run off the
  OS keyboard-hook thread.
- **VAD.** A full audio backlog can no longer deadlock the recogniser;
  push-to-talk release ends the utterance deterministically; short lead-in
  settings no longer clip the start of speech.
- **Providers.** LiteLLM requests carry the provider prefix, tool history is
  sent in valid OpenAI format, and model reasoning is not spoken.
- **Sessions.** `SessionManager` rejects ids that escape its directory;
  importing an archive never restores tool permissions or pending calls.

### Changed

- Configuration is loaded on first use (`get_config()`), not at import, and
  rejects unknown keys. Components take their settings as arguments.
- `LLM.__call__` raises `LLMError` instead of yielding an apology string; a
  failed turn is rolled back from the conversation.
- `Session.reset()` keeps the system prompt and tools, and never blocks on a
  turn in flight. `Session.cancel()` and `Session.events()` are new.
- Tool permissions and pending choices live in `SessionState` only. A model
  cannot answer its own confirmation request within the same turn.
- New config keys: `llm.max_turns`, `llm.tool_timeout`, `asr.sound_device`.
  `LocalModelEntry` defaults are now `n_ctx: 4096`, `max_tokens: 512`.
- Dependencies `emoji` and `tiktoken` were unused and are gone; `soundfile`
  moved to the `dev` extra.

### Upgrade notes

| Before | Now |
|---|---|
| `from voxpipe.core.config import config`; `config.get("llm.model")` | `get_config().llm.model` |
| `Pipeline(push_to_talk=...)` | `Pipeline.from_config(push_to_talk=...)`, or `Pipeline(session, asr=..., tts=...)` |
| `GGUFLLM("Key")` | `GGUFLLM.from_config("Key")` |
| `LLMProviders.get(name)` | `LLMProviders.create(backend, model)` |
| `Session(max_turns=n)` | `Session(context_handler=MemoryHandler(MaxTurnsStrategy(n)))` |
| `session._session_state[...]` | `session.state.user` / `.tools` / `.model` |
| `Tool.set_meta(name, "_permission", ...)`, `conversation.set_permission(...)` | `session.state.tools.set_permission(name, ...)` |
| `tool(**args)` returning a permission `ToolChoice` | `tool.permission_gate(granted)`, enforced by `Session` |
| `ToolCall(name="_parse_error")` | `MalformedToolCall` |
| `voxpipe.storage.Memory`, `storage.manager`, `storage.retriever` | `SQLiteStorage`, `voxpipe.core.downloader`, `voxpipe.storage.Retriever` |
| `MessageList`, `Conversation.get_meta` / token-count methods | `Conversation.visible`, `hide_oldest`, `drop_hidden`; `TokenBudgetStrategy` |
| `asr.enable()` / `asr.disable_w_passthrough()` | `asr.unmute()` / `asr.mute()` |
| `tts._synthesize(...)`, `getattr(TTSProviders, name)` | `tts.synthesize(...)`, `TTSProviders.create(name)` |
| `VoiceControlError` | `VoxpipeError` |
