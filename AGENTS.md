# Engager Bot — agent notes

## Stack
- discord.py-self **==2.1.0** (latest, Jan 2026). The old project pinned
  `>=2.2.0` which does not exist on PyPI.
- Groq via the official `groq` SDK (OpenAI-compatible, native tool calling).
- rapidfuzz for fuzzy search. loguru for logging. pydantic-settings for config.

## Commands
- Install: `pip install -r requirements.txt`
- Run:     `python run.py`
- Test:    `python -m tests.test_realtime` (requires a live token + network)

## Key conventions
- Every tool function is `async def fn(ctx: ToolContext, **kwargs)` and
  returns a **JSON-serialisable** dict/list (or `{"error": ...}`).
- Tool schemas live in `tools/registry.py::TOOL_SCHEMAS` (full set, 46 tools)
  and `_CHAT_TOOL_SCHEMAS` (chat mode subset, 28 tools). The dispatch map
  is `_DISPATCH`. Add a new tool = add a schema to both sets + a mapping entry.
- The agent loop is in `ai/agent.py::Agent.run`. Chat mode stops after 5
  rounds; command/event mode after 8.
- Multi-Groq keys are auto-discovered from `GROQ_API_KEY*` env vars by
  `config/settings.py::_collect_groq_keys`. Just add more keys to `.env`.
- The owner is the only user whose messages are treated as commands
  (`OWNER_USER_ID`). Commands come via DM or `COMMAND_CHANNEL_ID`.
- `ToolContext.current_channel_id` is set per-message so tools can resolve
  "here"/"this" to the channel the user is talking in.

## Tool categories (54 total, 38 in chat mode)
- **Channels**: list, search, get details (with permission overwrites),
  get mention, get link — all support direct ID and fuzzy name resolution.
- **Members**: search, get details (with bio), get mention, count, online,
  recent joins, recent leaves, identify_bots — all support direct ID and fuzzy name.
- **Roles**: search, get mention — for pinging roles via `ping_roles` param.
- **Messages**: get recent (with embeds as JSON + vision), get by link.
- **Messaging**: send (with ping_users, ping_roles, mention_channels),
  send DM, delete_message, delete_last_message, edit_message — supports "here"/"this".
- **Reactions**: react to message by link, react to user's latest, react_to_recent (batch).
- **GIFs**: search, send (with ping_users/ping_roles), send_multiple_gifs,
  trending_gifs — uses the official Klipy GIF API (KLIPY_API_KEY in .env).
- **Stickers**: list, send (supports "random"), send_multiple_stickers.
- **Voice**: join, leave, mute, deafen, move, get state, get current VC,
  get user voice state, list voice channels, read/send VC text chat, speak in stage.
- **Profile**: change nickname, status, custom status, bio, get profile.
- **Slash commands**: list_slash_commands, use_slash_command (supports application_id
  for targeting specific bots).
- **Bump management**: find_bump_commands, bump_with_bot, bump_all, get_bump_status.

## Token optimization
- Chat system prompt: ~1200 chars.
- Chat tool set: 38 tools with minimal descriptions.
- Tool results truncated to 8000 chars (graceful: removes oldest items from lists).
- Groq prompt caching: system + tools are the static prefix.
- Token usage logged per round: `prompt=X completion=Y total=Z`.

## Image & embed handling
- Images sent by users are described via Groq vision (`transcribe_image`).
  The vision model is `meta-llama/llama-4-scout-17b-16e-instruct` (if available
  on your Groq tier). If unavailable, falls back to `[image sent: URL]`.
- Embeds are serialised via `utils/embeds.py::embed_to_json` and included
  in the message context as `[embed: title | description | fields | footer | author | image]`.
- When a user replies to a message with embeds, the replied message's embeds
  are also included as `[replied message from X: content; embed: ...]`.
- The bot processes messages with content, attachments, OR embeds.
- Discord message links can be fetched via `get_message_by_link`.
- **Deferred embeds**: Some bots (OneBump, DISBOARD) send empty messages first,
  then edit them to add embeds. The `on_message_edit` handler catches these.

## Bump management
- `tools/bump_manager.py` tracks bump bot cooldowns and can auto-bump.
- Known bump bots: DISBOARD (2h), Bumper (1h), OneBump (1h), Global Carl (1h),
  Bump Central (1h), Liam (1h), DSC (1h), BumpIt (1h), Bumpy.gg (1h), DiscordHome (1h).
- `BUMP_CHANNEL_IDS` in .env restricts which channels to monitor.
- `AUTO_BUMP=true` enables automatic re-bumping when cooldowns expire.
- `find_bump_commands` detects bump bots by checking /bump command descriptions
  for server-promotion keywords (excludes music bots like Hydra).
- `bump_with_bot` targets a specific bot by application_id.
- `bump_all` bumps with all detected bump bots in one call.
- `get_bump_status` returns current cooldown status.
- Cooldown detection parses Discord timestamp tags (`<t:UNIX:R>`) from embeds.

## Voice chat (voice/ package)
- `tools/voice.py::join_voice` connects with `cls=discord.ext.native_voice.VoiceClient`
  (NOT the stock client) and starts a `voice/session.py::VoiceSession`.
- Pipeline: `PCMDecodeSink → AsyncQueueSink` → `voice/vad.py::UtteranceAssembler`
  (per-user, energy + packet-gap endpointing) → `voice/stt.py` (Groq
  whisper-large-v3-turbo, 16 kHz mono wav) → `Agent.run_stream(mode=VOICE_MODE)`
  → per-sentence `voice/tts.py` chunks → `AudioFrameSource` → `vc.play`.
- **STT hardening** (`voice/stt.py`): AGC on the voiced portion (up to 18×),
  transcript-style prompt (VC names LAST — long guild-name lists cause
  prompt-echo garbage), `verbose_json` confidence gates (no_speech>0.6+lp<-1
  drop, lp<-1.5 drop, compression_ratio>2.4/low lp → retry @temp 0.2 →
  whisper-large-v3 fallback), repetition guard, per-speaker language pinning
  (75% dominant over ≥4 utterances), non-Latin-script flag → `(unclear?)`
  marker in the line so the model asks for a repeat instead of parroting.
- **Conversation path streams**: `pool.chat_stream` → sentence-boundary
  aggregator → concurrent per-sentence synth → ordered playback; logs
  `VC stream first-audio X.XXs` per turn. Action path stays `agent.run`
  (tool calls can't stream).
- **Smart Turn v3.2** (`voice/smart_turn.py`, model in `models/`): trained
  end-of-turn classifier (Whisper-Tiny encoder + linear head, 8.7MB int8
  ONNX) scores the last ≤8s of a speaker's audio for P(turn complete).
  Runs CONCURRENTLY with STT (verdict ready when transcripts return —
  ~0ms critical-path cost). First call cold-loads ~20s — warmed at
  `start()`. Falls back to `_looks_incomplete` regex if onnxruntime or
  transformers is missing.
- **False-interruption pause/resume** (LiveKit pattern): sustained overlap
  (≥500ms) while speaking → `vc.pause()` NOT stop. A real turn (≥0.9s
  voiced utterance) → true `_barge_in`; a click/backchannel or 2s silence
  → auto `vc.resume()`. `_paused_speech`/`_pause_deadline` state; playback
  finally-blocks always unpause.
- **Spoken-text bookkeeping**: `_recent_spoken` only records segments that
  finished playout (interrupted chunks never count as "said").
- Reference sources live in `reference_src/` (livekit-agents voice/,
  pipecat src, smart-turn repo, personaplex) — porting sources, NOT deps.
- Requires **discord.py-self git master** (2.2.0a) — PyPI 2.1.0 lacks
  `discord.voice_media`/`SpeakingFlags` that native_voice imports.
- TTS is **fish.audio only** (`voice/tts.py`): persistent WebSocket to
  `wss://api.fish.audio/v1/tts/live` (ormsgpack events, per-utterance
  text+flush, 0.9s audio-gap = utterance end since fish sends no finish
  frame per call), HTTP POST fallback. Mood → prosody map (speed/volume/
  temperature per mood label) + `_OUTPUT_GAIN` loudness boost. Socket is
  warmed at session start; mood change forces a reconnect (prosody is
  session-scoped).
- Behaviour: barge-in (>300 ms sustained user speech stops playback),
  echo rejection (fuzz vs last 30 s of spoken lines), NOACTION gating,
  join greeting, asks-to-leave after `VOICE_IDLE_ASK_MINUTES`, auto-leaves
  when alone `VOICE_ALONE_SECONDS`, `VOICE_AUTO_JOIN` lets it hop into a
  lone human's VC on its own (60% owner / 15% others).
- **Intent router** (`ai/router.py`): every transcript/message is classified
  ACTION vs CHAT. Conversation runs `Agent.run(no_tools=True)` — zero tool
  schemas (~4x fewer tokens, no hallucinated tool calls). Actions get a
  category-filtered schema set via `build_tools(mode, categories)` (22 tools
  vs 56) + always-included resolver tools.
- **Persona** (`config/prompts.py::_PERSONA_CORE`): shared identity — 22yo
  London girl, French mum, art student, dry British humour — chat + voice.
- **Anti-repetition**: `_room_context` injects the bot's last-6 spoken lines +
  banned openers + FORBIDDEN overused words; `_is_repeat` (Jaccard>0.65 or
  same 8-char opener) triggers one redo nudge, else the line is dropped.
- **Spoken acks** (`_ACK_POOLS`): pre-synthesized clips played on ~15-40% of
  non-directed lines (35s cooldown, last-5 dedup) — voice analog of reacts.
- **Engagement nudges**: per-speaker latency/utterance-length scoring injects
  "[they're engaged]"/"[barely engaging]" into room context.
- **Dead-air ladder**: 90s silence → direct-address a member; +150s → general
  icebreaker; then ask-to-leave. Self-mute theater after 150s dead air.
- **Persistent user memory** (`ai/user_memory.py`): data/user_memory.json,
  atomic writes, category-dedup, keyed by user_id. Written via regexes in
  `_remember_speech` + a ~6% sampled LLM extraction pass; injected via
  "[you know these people]" context + personalized join greets.
- **Mood engine** (`voice/mood.py`): sentiment scalar from transcripts →
  states annoyed→hyped → prompt tone, proactive chance, ask-to-leave timing,
  storms off after repeated directed rudeness.
- **Never spoken**: `(no reply)`, `NOACTION`, error strings, and any
  `[name]:` prefix the model parrots back — `session._sayable` + `say()`
  gate all speech.
- gpt-oss models are reasoning models: voice calls pass
  `extra={"reasoning_effort": "low"}` via `pool.chat(extra=...)` and need
  max_tokens ≥ ~400 or replies come back EMPTY.
- `say_in_vc` tool speaks arbitrary text in the active session.
- **Scheduler** (`core/scheduler.py` + `tools/scheduler.py`): owner-only
  timed actions, persisted in `data/scheduled_tasks.json`, revived on
  restart, `stop_scheduled` cancels+cleans. Tools: schedule_message /
  list_scheduled / stop_scheduled.
- **Preferences** (`tools/prefs.py`, `data/prefs.json`): owner key-value
  prefs (`welcome_channel` steers member-join greetings in events.py).
- **Self-cleanup** (`core/self_cleanup.py`): before each send, deletes our
  own unanswered (>10min, non-latest) messages; `cleanup_my_messages` tool
  sweeps >6h / >15-per-channel.
- **did_send flag**: `ctx.did_send` set by send_message/send_dm suppresses
  the agent's own reply — prevents double messages.
- **Render deploy** (`render.yaml`): web service on free tier, health
  endpoint auto-binds `$PORT` in run.py (returns 200 "ok" for keep-alive
  pings). `VOICE_SMART_TURN=false` by default there — 512MB RAM can't
  comfortably host transformers+onnxruntime; flip on if it survives.
  Ephemeral disk: `data/user_memory.json` + downloaded models reset each
  deploy; smart-turn re-downloads itself (8MB) when enabled.

## Gotchas
- **discord.py-self v2.1.0 `send()` does NOT support `embed=` parameter** —
  it's a user-account library. Send GIF URLs as content (auto-embeds).
- **VoiceChannel IS the Messageable** — there is NO `associated_text_channel`
  attribute. Call `voice_channel.send()` directly.
- **`AsyncQueueSink` needs an explicit `asyncio.Queue`** — the default is a
  MISSING sentinel that the constructor's own type check rejects ("destination
  must be asyncio.Queue not _MissingSentinel"). The docstring lies.
- **Self-mute/deafen** uses `guild.change_voice_state()`, NOT `vc.main_mute()`.
- **No `vc.self_mute`/`vc.self_deaf` attributes** on either VoiceClient —
  read `vc.guild.me.voice` instead.
- **`VOICE_TTS_BACKENDS` is a str field** (`voice_tts_backends_raw` + property)
  — pydantic-settings JSON-decodes list fields before validators run, so a
  comma string breaks boot.
- **Discord doesn't echo your own audio** — no AEC needed; mic-bleed from
  users' speakers is filtered by fuzzy echo rejection instead.
- **Bio change** uses `client.user.edit(bio=...)`, NOT `guild.me.edit(bio=...)`.
- **Slash commands** use `channel.application_commands()` then `cmd(channel=ch)`.
  Never use `channel.send('/bump')` — that just sends literal text.
- **Voice state events** skip the bot itself (don't welcome yourself).
- **`leave_voice` accepts `**_extra`** because the agent sometimes passes
  `channel_query` even though the tool takes no args.
- **Fuzzy search** uses token_set_ratio + partial_ratio fallback (catches
  short queries like "revive" matching "🔔 Chat Revive").
- **Groq hallucinated tool names** (e.g. `search_channels<|channel|>commentary`)
  are handled by extracting the base name before dispatch.
- **Groq free tier** does NOT have vision models. The vision model config
  is set but will fall back gracefully if the model is not accessible.
- Self-bots violate Discord ToS — use a dedicated alt account.
