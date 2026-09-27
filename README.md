# Engager Bot

An autonomous Discord self-bot powered by **discord.py-self v2.1.0** and a
central **Groq AI** agent with native tool/function-calling. The agent has
independent, fuzzy-search-driven control over channels, members, messages,
voice/stage, reactions, stickers and GIFs.

## Architecture

```
Engager Bot/
├── run.py                 # entry point
├── .env / .env.example    # credentials (multi-Groq-key)
├── requirements.txt
├── config/
│   ├── settings.py        # typed pydantic settings + auto Groq-key discovery
│   └── prompts.py         # system & event prompts
├── core/
│   ├── bot.py             # EngagerBot (discord.py-self client) + command listener
│   ├── events.py          # member join/leave, voice, channel create/delete hooks
│   └── groq_pool.py       # multi-key Groq pool w/ rate-limit rotation
├── ai/
│   └── agent.py           # central tool-calling loop (the "brain")
├── tools/                 # independent feature modules (each = an AI tool)
│   ├── registry.py        # Groq tool schemas + dispatch
│   ├── channels.py        # discovery, fuzzy search, mention, link
│   ├── members.py         # details, fuzzy, joins/leaves, online presence
│   ├── messages.py        # latest N msgs, embeds→JSON, vision-described images
│   ├── messaging.py       # send text, ping, mention channels, reply, DM
│   ├── reactions.py       # emoji reactions (by link or by user-in-channel)
│   ├── stickers_gifs.py   # Klipy/Tenor gif search+send, guild stickers
│   └── voice.py           # join/leave VC & stage, VC text chat, speak request
├── utils/
│   ├── logger.py          # loguru console + rotating file
│   ├── fuzzy.py           # rapidfuzz-based fuzzy search
│   └── embeds.py          # Embed → JSON serialiser
└── tests/
    └── test_realtime.py   # real-time integration test harness
```

## How it works

1. The bot logs in with your Discord user token via discord.py-self.
2. The **central agent** (`ai/agent.py`) holds Groq tool schemas for every
   feature. When you send it a natural-language command (in the configured
   command channel or via DM), it loops:
   - calls Groq with the tool schemas,
   - executes any returned tool calls (`tools/registry.py` dispatches them),
   - feeds results back to Groq,
   - repeats until it produces a final natural-language reply.
3. Discord events (member join/leave, voice moves, channel create/delete) are
   forwarded to the agent in `event_mode`, so it can act autonomously (e.g.
   greet new members) or decide no action is needed.

## Available AI tools (28)

Channels: `list_channels`, `search_channels`, `get_channel_details`,
`get_channel_mention`, `get_channel_link`
Members: `search_members`, `get_member_details`, `get_member_mention`,
`get_member_count`, `get_online_members`, `get_recent_joins`,
`get_recent_leaves`
Messages: `get_recent_messages` (embeds→JSON, images via vision),
`get_message_by_link`
Messaging: `send_message` (ping users + mention channels + reply),
`send_dm`
Reactions: `react_to_message`, `react_to_user_latest`
GIFs/Stickers: `search_gifs`, `send_gif`, `list_stickers`, `send_sticker`
Voice: `list_voice_channels`, `join_voice`, `leave_voice`, `get_voice_state`,
`get_vc_text_chat`, `speak_in_stage`

## Setup

```bash
cd "Engager Bot"
pip install -r requirements.txt
cp .env.example .env      # then edit .env with your token + Groq keys
python run.py
```

Then DM the bot account (or post in `COMMAND_CHANNEL_ID`) with natural
language, e.g.:

- "ping aliyyan in #general saying Hello everyone"
- "what's the description of the announcements channel?"
- "join the gaming vc"
- "react to the latest message from mike in #off-topic with 🔥"
- "send a cat gif in #memes"
- "how many people are online right now?"
- "who joined the server recently?"

## Multi-Groq-key

Add as many `GROQ_API_KEY`, `GROQ_API_KEY_2`, ... `GROQ_API_KEY_10` entries as
you like. They are auto-discovered and rotated on rate-limit (per-account
limits, so use separate accounts).

## Notes

- discord.py-self runs on a **real user account** (self-bot). Use a dedicated
  account and follow Discord's terms at your own risk.
- GIFs use Klipy (Discord's current provider) — paste a key in
  `KLIPY_API_KEY` or fall back to Tenor automatically.
- Embeds are serialised to JSON via `utils/embeds.py` so the agent can reason
  about their nature (author, fields, footer, image, provider, video, ...).
- Image/gif attachments are described by the Groq vision model
  (`GROQ_MODEL_VISION`) so the agent can "see" them.
