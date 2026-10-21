# Changelog

## Unreleased

### Added

- **Topic tracking.** An optional gate keeps the active conversation apart from
  unrelated speech. Each recognised utterance is compared with the main topic
  and with bounded side buffers; a sentence joins the closest buffer or starts
  a new one. Side buffers expire after `topic_tracking.ttl_sentences` idle
  utterances and are capped by `max_unrelated`. Only the main topic is answered
  (unless `promote_after` promotes a side thread), and push-to-talk or typed
  input bypasses the gate. Three pluggable scorers ship: dependency-free
  lexical overlap, a small fastembed bi-encoder (`voxpipe[embed]`, recommended
  for real-time use), and Laya's yes/no decision model (`voxpipe[laya]`).
  Routing is visible through the `topic:routed` event.
- **Pluggable gates and named buffers.** The pipeline now accepts any
  `ConversationGate` through `Pipeline(gate=...)`; `TopicTracker` is one
  implementation, and `PassThroughGate` is available as a null gate.
  `gate:decided` reports every decision. Trackers can also route explicitly to
  a named buffer (`route_to("table_4", ...)`), so workflow contexts do not have
  to be inferred from topic similarity.
- **Busy mode / addressee gate.** `attention.busy_mode: true` enables a
  silence-biased gate for shared spaces: an addressee classifier over your own
  labeled examples, a short deferral that cancels the reply if someone keeps
  talking, named other/background buffers, “not you” / “that was me”
  corrections, and a shadow mode. Embeddings use fastembed when installed and
  fall back to deterministic hashing. `attention.enabled` turns the gate on
  with explicit settings, and `voxpipe --busy` enables the preset for one run.
  This adds `AddresseeGate`, `AddresseeModel`, `GateDecision.defer_ms`, the
  `attention` configuration section, and pipeline deferral handling.
- **Shadow logging and replay.** Attention decisions and corrections are
  appended to `attention.decision_log_path`; `scripts/replay_addressee.py`
  pairs them and sweeps speak/silence thresholds under a configurable
  false-positive budget. `AddresseeModel.clear()` removes learned examples.
- **Application example.** The restaurant menu/order workflow lives in
  `examples/restaurant/` and is intentionally outside the core package. It
  imports the voice layer and shows how to expose order operations as tools;
  the core module does not depend on it.

### Changed

- The attention layer now lives in `voxpipe.pipeline.attention`;
  `voxpipe.pipeline.topics` remains as a compatibility import. Inferred side
  buffers (TTL/LRU) are separate from named workflow buffers, so background
  chatter cannot evict a table, ticket or action-item buffer.
- Relevance scorers declare their own default threshold. A failing primary
  scorer now uses a separate fallback threshold instead of the primary model's
  scale, and malformed model responses trigger the fallback rather than being
  treated as “unrelated”.
- The personal addressee model lets a very close learned example decide
  directly instead of diluting it across several less relevant neighbours.
- The follow-up heuristic is a replaceable `FollowUpPolicy`; the default is
  still keywords plus the assistant-question rule.
- Scoring runs outside the buffer lock. `reset()` and `close()` no longer wait
  for a slow scorer, and a result that arrives after a reset is discarded.
- Always-on audio commands now pass through the conversation gate; explicit
  commands (typed text or push-to-talk) still run immediately.
- The transcript filter moved to `voxpipe.pipeline.transcript`;
  `voxpipe.pipeline.gate` remains as a compatibility import.

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

- **Voice activity detection.** Silero is a recurrent model, and it was being
  restarted from a blank state for every 32 ms chunk (and run twice per chunk).
  Its state is now carried along: quiet speech that used to produce no
  utterance at all is heard, and speech over noise is detected sooner.
- **Long replies.** Once synthesis got more than 33 sentences ahead of
  playback the rest were dropped; the synthesiser now waits for the player.
- **Taking turns.** A sentence that arrives just after the user barges in is
  no longer spoken over them; an interruption that lands before the model has
  been asked stops the turn before any tool runs; and an utterance the user
  has already talked past is held and answered together with what follows.
  `reset()` no longer carries the "user interrupted" note into the new
  conversation, and `stop()` no longer answers everything still queued first.
- **Tool rounds.** Interrupting while a tool runs no longer erases what was
  said and done from history; a follow-up question asked while answering
  another can be answered; a failed turn restores the message it had extended;
  a blocking tool that never returns no longer keeps the process alive or
  starves later tools; and a model-supplied argument named `self`, a JSON
  `null` for an optional argument, or a malformed call ahead of a valid one
  no longer break the turn.
- **Transcripts.** Utterances with no letters ("42", "10:30") are answered
  instead of being dropped as noise.
- **Decoders and providers.** Gemma calls whose strings contain `, word:` are
  parsed; `"parameters"` and stringified `"arguments"` are accepted; calls
  that share a stream index are told apart by id; a local OpenAI-compatible
  server is asked for the model by its bare id.
- **Saved sessions.** Importing a damaged archive no longer destroys the
  session it would have replaced; session files are written atomically; ids
  that Windows would alias or refuse are rejected.
- **Storage and models.** A file that is not a database, a corrupt model
  file or a failed download now raise `StorageError` / `ModelError` that say
  what to do; expired memories are no longer recalled; memories a failing
  backend refused are offered again instead of being lost.
- **Hotkeys.** A combination that cannot be parsed leaves the previous one
  (and the microphone) as it was; assigning a hotkey while running works; a
  held hotkey that is unregistered is released.
- **Sentence splitting.** Initials ("Samuel L. Jackson") no longer split a
  sentence; "3.50." and "Max." end one; list items on separate lines are
  spoken one by one.

### Changed

- Configuration is loaded on first use (`get_config()`), not at import, and
  rejects unknown keys. Components take their settings as arguments.
- `LLM.__call__` raises `LLMError` instead of yielding an apology string; a
  failed turn is rolled back from the conversation.
- `Session.reset()` keeps the system prompt and tools, and never blocks on a
  turn in flight. `Session.cancel()` and `Session.events()` are new.
- What the model is told and may do is set on the session:
  `Session(llm, instructions=..., rules=..., tools=...)`, `session.tools.add(...)`.
  The system prompt is assembled from the instructions, each tool's
  `instruction` and the rules, for every session rather than only inside the
  pipeline. A tool `instruction` therefore now reaches the prompt even when
  you supply your own instructions; leave it unset to keep it out.
- The context is cut in batches. `MaxTurnsStrategy` and `TokenBudgetStrategy`
  take `trim_to` (default 60% of the limit) and cut back to it once the limit
  is exceeded, so the start of the prompt stays put for several turns and a
  prompt-prefix cache keeps working. Evicted messages are archived and dropped
  at once; `amortize_k` and the hidden part of a conversation are gone.
- `max_turns` counts turns (a user message and everything said in answer),
  not messages, so tool calls no longer use up the window. The default is 10.
- `MemoryHandler` accepts any object with `store` and `retrieve` as its
  memory, not only a `MemoryStore`.
- `LLM.complete(query, system=...)` answers a one-off query without a session.
- Sessions are saved without a KV cache: nothing ever produced one, and a
  restored session only has the model read its history once more.
- Tool permissions and pending choices live in `SessionState` only. A model
  cannot answer its own confirmation request within the same turn.
- New config keys: `llm.max_turns`, `llm.tool_timeout`, `asr.sound_device`.
  `LocalModelEntry` defaults are now `n_ctx: 4096`, `max_tokens: 512`.
- Dependencies `emoji` and `tiktoken` were unused and are gone; `soundfile`
  moved to the `dev` extra.

- `Session.events()` and calling a session take `tool_choice=`; `context=` needs a
  query. `Pipeline.submit(text)` queues typed text while `run()` is active,
  and `run()` raises `ASRError` when recognition fails rather than returning.
- Tool arguments are checked against the schema: `enum` values, arrays and
  objects are enforced, and tool names must be identifiers. `@permission`
  also works on a `Tool`.
- `Pipeline.from_config(push_to_talk=...)` wins even over
  `hotkeys.enable: false`, as the CLI flag always claimed.
- The TTS queue starts on first use, so text in / speech out works without
  `run()`.

### Upgrade notes

| Before | Now |
|---|---|
| `from voxpipe.core.config import config`; `config.get("llm.model")` | `get_config().llm.model` |
| `Pipeline(push_to_talk=...)` | `Pipeline.from_config(push_to_talk=...)`, or `Pipeline(session, asr=..., tts=...)` |
| `GGUFLLM("Key")` | `GGUFLLM.from_config("Key")` |
| `LLMProviders.get(name)` | `LLMProviders.create(backend, model)` |
| `Session(max_turns=n)` | `Session(context_handler=MemoryHandler(MaxTurnsStrategy(n)))`; `n` now counts turns, not messages |
| `session.conversation.tools[name] = tool`, `conversation.set_system_message(text)` | `session.tools.add(tool)`, `session.instructions = text` (the old spellings still work) |
| `session.complete_once(query)` | `session.llm.complete(query)` |
| `Session(session_id=...)` | removed; name the session where it is used: `SessionManager.save_session(session, id)`, `MemoryStore(storage, session_id=id)` |
| `session.save(path, save_kv_cache=True)`, `state.model.set_kv_cache(...)` | removed |
| `session._session_state[...]` | `session.state.user` / `.tools` / `.model` |
| `Tool.set_meta(name, "_permission", ...)`, `conversation.set_permission(...)` | `session.state.tools.set_permission(name, ...)` |
| `tool(**args)` returning a permission `ToolChoice` | `tool.permission_gate(granted)`, enforced by `Session` |
| `ToolCall(name="_parse_error")` | `MalformedToolCall` |
| `voxpipe.storage.Memory`, `storage.manager`, `storage.retriever` | `SQLiteStorage`, `voxpipe.core.downloader`, `voxpipe.storage.Retriever` |
| `MessageList`, `Conversation.get_meta` / token-count methods, `cutoff_idx` | `Conversation.history`, `evict_oldest`; `TokenBudgetStrategy` |
| `core.utils.download_file`, `verify_file_sha256` | `voxpipe.core.downloader` |
| `streaming.splitter.ConsumerProducer`, `llm.context.DropOldestStrategy` | removed; `stream_splitter`, `MaxTurnsStrategy` |
| `asr.enable()` / `asr.disable_w_passthrough()` | `asr.unmute()` / `asr.mute()` |
| `tts._synthesize(...)`, `getattr(TTSProviders, name)` | `tts.synthesize(...)`, `TTSProviders.create(name)` |
| `VoiceControlError` | `VoxpipeError` |
| `text, annotation = qualify_transcript(t)`; `asr:utterance` handlers taking `annotation=` | `text = qualify_transcript(t)`; handlers receive the text only |
| `ensure_downloaded` raising `ValueError` / `requests` errors | `ModelError` |
| recogniser iteration skipping an utterance it failed to transcribe | it yields `""` for it |
