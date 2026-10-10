#!/usr/bin/env python3
"""
GPU on the Fediverse — ActivityPub bot for the GPU network.

Makes the free GPU AI reachable from Mastodon, Pleroma, Misskey and any other
ActivityPub server (e.g. https://defcon.social). Follow @gpu@gpu.social and
mention it in a post — it answers with the network's AI, in the same thread.

Federation (WebFinger, NodeInfo, actor, inbox, outbox, HTTP signatures) is
handled by pubby: https://git.jon-e.net/jonny/pubby

Install:  pip install "pubby[flask]"
Run:      GPU_API_KEY=... python gpu_activitypub.py
Prod:     gunicorn -w 1 --threads 8 -b 127.0.0.1:8000 'gpu_activitypub:create_app()'

Configuration is read from environment variables (see Config below).
"""

import os
import re
import json
import uuid
import html
import logging
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser

import requests
from flask import Flask, Response
from pubby import (
    PUBLIC_URIS,
    ActivityPubHandler,
    ActorConfig,
    Interaction,
    Object,
    RateLimiter,
    RateLimitError,
    addressees,
    is_public,
    property_value_attachment,
    render_bio_html,
)
from pubby.crypto import ensure_private_key_file
from pubby.server.adapters.flask import bind_activitypub
from pubby.storage.adapters.file import FileActivityPubStorage

log = logging.getLogger("gpu.activitypub")

AS_PUBLIC = "https://www.w3.org/ns/activitystreams#Public"
ACTIVITY_JSON = "application/activity+json"

SYSTEM_PROMPT = (
    "You are GPU (Gifted People United), a free AI that belongs to everyone, "
    "answering on the Fediverse. Reply in the language of the question. "
    "Be helpful and concise: your answer is published as a social media post, "
    "so use plain text, no Markdown tables, and keep it short."
)


def _env_list(name):
    return [x.strip() for x in os.environ.get(name, "").split(",") if x.strip()]


@dataclass
class Config:
    base_url: str = "https://gpu.social"            # public URL of this server
    username: str = "gpu"                           # → @gpu@gpu.social
    data_dir: str = os.path.expanduser("~/.gpu/activitypub")
    api_base: str = "https://gpu.social"            # OpenAI-compatible GPU API
    api_key: str = ""
    model: str = "qwen2.5-14b"
    max_chars: int = 4000                           # defcon.social limit: 4000
    max_tokens: int = 700
    history_turns: int = 3                          # thread context: Q/A pairs
    replies_per_actor: int = 5                      # per actor ...
    replies_window: int = 600                       # ... per 10 minutes
    workers: int = 2                                # parallel AI requests
    allowed_instances: list = field(default_factory=list)
    blocked_instances: list = field(default_factory=list)

    @classmethod
    def from_env(cls):
        c = cls()
        c.base_url = os.environ.get("GPU_AP_BASE_URL", c.base_url).rstrip("/")
        c.username = os.environ.get("GPU_AP_USERNAME", c.username)
        c.data_dir = os.environ.get("GPU_AP_DATA_DIR", c.data_dir)
        c.api_base = os.environ.get("GPU_API_BASE", c.api_base).rstrip("/")
        c.api_key = os.environ.get("GPU_API_KEY", c.api_key)
        c.model = os.environ.get("GPU_AP_MODEL", c.model)
        c.max_chars = int(os.environ.get("GPU_AP_MAX_CHARS", c.max_chars))
        c.allowed_instances = _env_list("GPU_AP_ALLOWED_INSTANCES")
        c.blocked_instances = _env_list("GPU_AP_BLOCKED_INSTANCES")
        return c


# ── Text helpers ─────────────────────────────────────────────

class _TextExtractor(HTMLParser):
    """Mastodon HTML → plain text (keeps line breaks, drops tags)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "br":
            self.parts.append("\n")
        elif tag == "p" and self.parts:
            self.parts.append("\n\n")

    def handle_data(self, data):
        self.parts.append(data)


def html_to_text(content):
    parser = _TextExtractor()
    parser.feed(content or "")
    parser.close()
    return "".join(parser.parts).strip()


MENTION_RE = re.compile(r"(?<![\w@])@[\w.-]+(?:@[\w.-]+\.\w+)?")


def strip_mentions(text):
    """Remove @user / @user@domain handles, leaving the actual question."""
    text = MENTION_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n\s*\n+", "\n\n", text).strip()


def truncate(text, limit):
    if len(text) <= limit:
        return text
    cut = text[:limit - 1]
    space = cut.rfind(" ")
    if space > limit * 0.8:
        cut = cut[:space]
    return cut.rstrip() + "…"


def handle_of(actor_id, actor_data=None):
    """'https://defcon.social/users/alice' → '@alice@defcon.social'."""
    actor_data = actor_data or {}
    domain = re.sub(r"^https?://", "", actor_id).split("/")[0]
    user = actor_data.get("preferredUsername") or actor_id.rstrip("/").split("/")[-1]
    return f"@{user.lstrip('@')}@{domain}"


def reply_audience(raw_object, author):
    """
    Mirror the visibility of the incoming post (Mastodon semantics), so a
    direct message never gets a public answer.

    public   → to: Public,  cc: author
    unlisted → to: author,  cc: Public
    private  → to: author   (direct)
    """
    to = raw_object.get("to") or []
    to = [to] if isinstance(to, str) else to
    if any(u in PUBLIC_URIS for u in to):
        return [AS_PUBLIC], [author]
    if is_public(raw_object):
        return [author], [AS_PUBLIC]
    return [author], []


# ── GPU API ──────────────────────────────────────────────────

class GPUChat:
    """Minimal client for the OpenAI-compatible GPU API."""

    def __init__(self, api_base, api_key, model, max_tokens=700, timeout=180):
        self.url = f"{api_base}/v1/chat/completions"
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout

    def complete(self, messages):
        resp = requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                "max_tokens": self.max_tokens,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


# ── Note store (our own posts, served at their ids) ──────────

class NoteStore:
    """Bot replies as JSON files: {"note": {...}, "question": "...", "answer": "..."}."""

    def __init__(self, path):
        self.path = path
        os.makedirs(path, exist_ok=True)

    def _file(self, note_id):
        return os.path.join(self.path, f"{note_id}.json")

    def save(self, note_id, record):
        tmp = self._file(note_id) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self._file(note_id))

    def get(self, note_id):
        if not re.fullmatch(r"[0-9a-f-]{36}", note_id or ""):
            return None
        try:
            with open(self._file(note_id)) as f:
                return json.load(f)
        except FileNotFoundError:
            return None


# ── The bot ──────────────────────────────────────────────────

class GPUFediverseBot:
    def __init__(self, config, chat=None, handler=None):
        self.config = config
        self.chat = chat or GPUChat(config.api_base, config.api_key, config.model,
                                    max_tokens=config.max_tokens)
        os.makedirs(config.data_dir, exist_ok=True)
        self.storage = FileActivityPubStorage(data_dir=os.path.join(config.data_dir, "pubby"))
        self.notes = NoteStore(os.path.join(config.data_dir, "notes"))
        self.handler = handler or ActivityPubHandler(
            storage=self.storage,
            actor_config=ActorConfig(
                base_url=config.base_url,
                username=config.username,
                name="GPU — Gifted People United",
                summary=render_bio_html(
                    "Free AI for everyone. Forever.\n"
                    "Mention me with a question and I'll answer in the thread.\n"
                    "Trained by millions of ordinary devices — https://github.com/yukakust/GPU"
                ),
                type="Service",  # shown as a bot on Mastodon
                attachment=[
                    property_value_attachment("Website", config.base_url),
                    property_value_attachment("Code", "https://github.com/yukakust/GPU"),
                ],
            ),
            private_key_path=ensure_private_key_file(
                os.path.join(config.data_dir, "actor.pem")),
            on_interaction_received=self.on_interaction,
            software_name="gpu",
            allowed_instances=config.allowed_instances or None,
            blocked_instances=config.blocked_instances or None,
        )
        self.actor_id = self.handler.actor_id
        self.limiter = RateLimiter(config.replies_per_actor, config.replies_window)
        self.pool = ThreadPoolExecutor(max_workers=config.workers,
                                       thread_name_prefix="gpu-ap-reply")
        self._seen = OrderedDict()  # inbound object ids already answered
        self._lock = threading.Lock()

    def note_url(self, note_id):
        return f"{self.config.base_url}/ap/notes/{note_id}"

    def own_note_id(self, url):
        prefix = self.note_url("")
        return url[len(prefix):] if url and url.startswith(prefix) else None

    # Called by pubby inside the inbox request → keep it fast, answer in background.
    def on_interaction(self, interaction: Interaction):
        if not self.should_answer(interaction):
            return
        with self._lock:
            if interaction.object_id in self._seen:
                return  # redelivery of a post we already answered
            self._seen[interaction.object_id] = True
            while len(self._seen) > 10000:
                self._seen.popitem(last=False)
        try:
            self.limiter.check(interaction.source_actor_id)
        except RateLimitError:
            log.info("Rate limit: ignoring %s", interaction.source_actor_id)
            return
        self.pool.submit(self._answer_safely, interaction)

    def should_answer(self, interaction: Interaction):
        if interaction.interaction_type.value not in ("mention", "reply"):
            return False  # likes, boosts, quotes
        if interaction.source_actor_id == self.actor_id or not interaction.object_id:
            return False
        raw = (interaction.metadata or {}).get("raw_object") or {}
        mentioned = self.actor_id in (interaction.mentioned_actors or []) \
            or self.actor_id in addressees(raw)
        replied = self.own_note_id(raw.get("inReplyTo")) is not None
        if not (mentioned or replied):
            return False  # a reply in some thread that doesn't involve us
        actor = self.storage.get_cached_actor(interaction.source_actor_id) or {}
        if actor.get("type") in ("Service", "Application") or actor.get("bot"):
            return False  # never talk to other bots — no infinite loops
        return True

    def _answer_safely(self, interaction):
        try:
            self.answer(interaction)
        except Exception:
            log.exception("Failed to answer %s", interaction.object_id)

    def thread_history(self, in_reply_to):
        """Rebuild the conversation from our earlier replies in this thread."""
        messages = []
        note_id = self.own_note_id(in_reply_to)
        while note_id and len(messages) < 2 * self.config.history_turns:
            record = self.notes.get(note_id)
            if not record:
                break
            messages[:0] = [{"role": "user", "content": record["question"]},
                            {"role": "assistant", "content": record["answer"]}]
            note_id = self.own_note_id(record.get("parent_note"))
        return messages

    def answer(self, interaction):
        raw = (interaction.metadata or {}).get("raw_object") or {}
        question = strip_mentions(html_to_text(interaction.content))
        author = interaction.source_actor_id
        actor_data = self.storage.get_cached_actor(author) or {}
        if not question:
            text = "Hi! Ask me anything — I'm a free AI that belongs to everyone."
        else:
            history = self.thread_history(raw.get("inReplyTo"))
            text = self.chat.complete(history + [{"role": "user", "content": question}])
        self.reply(raw, author, actor_data, question, text)

    def reply(self, raw, author, actor_data, question, answer):
        note_id = str(uuid.uuid4())
        mention = handle_of(author, actor_data)
        answer = truncate(answer, self.config.max_chars - len(mention) - 1)
        profile = actor_data.get("url") or author
        if not isinstance(profile, str):
            profile = author
        content = (
            f'<p><span class="h-card"><a href="{html.escape(profile)}" '
            f'class="u-url mention">@<span>{html.escape(mention.split("@")[1])}</span></a>'
            f"</span> {render_bio_html(answer)}</p>"
        )
        to, cc = reply_audience(raw, author)
        note = Object(
            id=self.note_url(note_id),
            type="Note",
            content=content,
            url=self.note_url(note_id),
            attributed_to=self.actor_id,
            in_reply_to=raw.get("id"),
            published=datetime.now(timezone.utc),
            tag=[{"type": "Mention", "href": author, "name": mention}],
            to=to,
            cc=cc,
            sensitive=bool(raw.get("sensitive")),
            summary=raw.get("summary") or None,  # keep the content warning
        )
        self.notes.save(note_id, {
            "note": note.to_dict(),
            "question": question,
            "answer": answer,
            "parent_note": raw.get("inReplyTo"),
        })
        self.handler.publish_object(note)
        log.info("Answered %s → %s", raw.get("id"), note.id)
        return note


# ── Web app ──────────────────────────────────────────────────

def create_app(config=None, bot=None):
    config = config or Config.from_env()
    bot = bot or GPUFediverseBot(config)
    app = Flask(__name__)
    app.config["GPU_BOT"] = bot
    bind_activitypub(app, bot.handler, rate_limiter=RateLimiter(300, 60))

    @app.route("/ap/notes/<note_id>", methods=["GET"])
    def gpu_note(note_id):
        record = bot.notes.get(note_id)
        if not record:
            return Response("Not found", status=404)
        note = record["note"]
        if not is_public(note):
            # Direct answers are delivered to the recipient only, never served.
            return Response("Not found", status=404)
        doc = {"@context": "https://www.w3.org/ns/activitystreams", **note}
        return Response(json.dumps(doc), mimetype=ACTIVITY_JSON)

    return app


def main():
    logging.basicConfig(level=os.environ.get("GPU_AP_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = Config.from_env()
    if not config.api_key:
        log.warning("GPU_API_KEY is not set — the GPU API will reject requests")
    app = create_app(config)
    host = os.environ.get("GPU_AP_HOST", "127.0.0.1")
    port = int(os.environ.get("GPU_AP_PORT", "8000"))
    log.info("Fediverse handle: @%s@%s",
             config.username, re.sub(r"^https?://", "", config.base_url))
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
