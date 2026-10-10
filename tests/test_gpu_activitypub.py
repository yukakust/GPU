"""
Tests for the GPU ActivityPub bot (gpu_activitypub.py).

The end-to-end tests simulate a Mastodon user on defcon.social: the remote
actor document is pre-cached (no network), activities are signed with a real
RSA key exactly like Mastodon does, and outgoing deliveries are captured.

Run: pip install "pubby[flask]" pytest && python -m pytest tests -v
"""

import json
import os
import sys
import uuid

import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pubby.crypto import (  # noqa: E402
    export_public_key_pem,
    generate_rsa_keypair,
    sign_request,
)

import gpu_activitypub as gap  # noqa: E402

BASE = "https://gpu.social"
BOT = f"{BASE}/ap/actor"
ALICE = "https://defcon.social/users/alice"
PUBLIC = "https://www.w3.org/ns/activitystreams#Public"


class FakeChat:
    def __init__(self, answer="42 is the answer."):
        self.answer = answer
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        return self.answer


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError(f"unexpected network call: {args} {kwargs}")
    monkeypatch.setattr(requests, "get", fail)
    monkeypatch.setattr(requests, "post", fail)


@pytest.fixture
def remote_key():
    return generate_rsa_keypair()


def remote_actor(actor_id, public_key, actor_type="Person"):
    user = actor_id.rsplit("/", 1)[-1]
    return {
        "@context": ["https://www.w3.org/ns/activitystreams"],
        "id": actor_id,
        "type": actor_type,
        "preferredUsername": user,
        "name": user.title(),
        "url": f"https://defcon.social/@{user}",
        "inbox": f"{actor_id}/inbox",
        "endpoints": {"sharedInbox": "https://defcon.social/inbox"},
        "publicKey": {
            "id": f"{actor_id}#main-key",
            "owner": actor_id,
            "publicKeyPem": export_public_key_pem(public_key),
        },
    }


@pytest.fixture
def env(tmp_path, remote_key):
    config = gap.Config(base_url=BASE, data_dir=str(tmp_path), api_key="test")
    chat = FakeChat()
    bot = gap.GPUFediverseBot(config, chat=chat)
    delivered = []
    bot.handler.outbox.deliver = lambda inbox, activity: delivered.append((inbox, activity))
    private_key, public_key = remote_key
    bot.storage.cache_remote_actor(ALICE, remote_actor(ALICE, public_key))
    app = gap.create_app(config, bot=bot)
    client = app.test_client()

    class Env:
        pass

    e = Env()
    e.bot, e.chat, e.delivered, e.client = bot, chat, delivered, client
    e.private_key = private_key

    def post(activity, actor=ALICE, signed=True):
        body = json.dumps(activity).encode()
        headers = {"Content-Type": "application/activity+json"}
        if signed:
            headers.update(sign_request(
                private_key=private_key, key_id=f"{actor}#main-key",
                method="POST", url=f"{BASE}/ap/inbox", body=body, headers=headers,
            ))
        resp = client.post("/ap/inbox", data=body, headers=headers, base_url=BASE)
        bot.pool.shutdown(wait=True)  # wait for background answers
        bot.pool = gap.ThreadPoolExecutor(max_workers=1)
        return resp

    e.post = post
    return e


def create(content, to=None, cc=None, in_reply_to=None, actor=ALICE, mention=True, **extra):
    note_id = f"https://defcon.social/users/alice/statuses/{uuid.uuid4().int % 10**18}"
    obj = {
        "id": note_id,
        "type": "Note",
        "attributedTo": actor,
        "content": content,
        "to": to if to is not None else [PUBLIC],
        "cc": cc if cc is not None else [f"{actor}/followers", BOT],
        "inReplyTo": in_reply_to,
        "tag": [{"type": "Mention", "href": BOT, "name": "@gpu@gpu.social"}] if mention else [],
        **extra,
    }
    return {
        "@context": "https://www.w3.org/ns/activitystreams",
        "id": f"{note_id}/activity",
        "type": "Create",
        "actor": actor,
        "to": obj["to"],
        "cc": obj["cc"],
        "object": obj,
    }


MASTODON_HTML = (
    '<p><span class="h-card" translate="no"><a href="https://gpu.social/@gpu" '
    'class="u-url mention">@<span>gpu</span></a></span> What is the meaning of life?</p>'
)


# ── Pure helpers ─────────────────────────────────────────────

def test_html_to_text_and_strip_mentions():
    text = gap.html_to_text(MASTODON_HTML)
    assert text == "@gpu What is the meaning of life?"
    assert gap.strip_mentions(text) == "What is the meaning of life?"
    assert gap.strip_mentions("@gpu@gpu.social hi @bob@defcon.social!") == "hi !"
    assert gap.strip_mentions("mail me: a@b.com") == "mail me: a@b.com"
    assert gap.html_to_text("<p>a<br>b</p><p>c &amp; d</p>") == "a\nb\n\nc & d"


def test_truncate():
    assert gap.truncate("short", 10) == "short"
    out = gap.truncate("word " * 1000, 100)
    assert len(out) <= 100 and out.endswith("…")


def test_reply_audience_mirrors_visibility():
    assert gap.reply_audience({"to": [PUBLIC]}, ALICE) == ([PUBLIC], [ALICE])
    assert gap.reply_audience({"to": [f"{ALICE}/followers"], "cc": [PUBLIC]}, ALICE) == ([ALICE], [PUBLIC])
    assert gap.reply_audience({"to": [BOT], "cc": []}, ALICE) == ([ALICE], [])
    assert gap.reply_audience({"to": [f"{ALICE}/followers"]}, ALICE) == ([ALICE], [])


def test_handle_of():
    assert gap.handle_of(ALICE) == "@alice@defcon.social"
    assert gap.handle_of("https://x.example/ap/actor", {"preferredUsername": "bob"}) == "@bob@x.example"


# ── Discovery: how defcon.social finds @gpu@gpu.social ───────

def test_webfinger_and_actor(env):
    resp = env.client.get("/.well-known/webfinger?resource=acct:gpu@gpu.social", base_url=BASE)
    assert resp.status_code == 200
    links = resp.get_json()["links"]
    assert any(link.get("href") == BOT for link in links)

    actor = env.client.get("/ap/actor", base_url=BASE).get_json()
    assert actor["id"] == BOT
    assert actor["type"] == "Service"  # displayed as a bot
    assert actor["preferredUsername"] == "gpu"
    assert "publicKeyPem" in actor["publicKey"]
    assert actor["inbox"] == f"{BASE}/ap/inbox"


# ── End-to-end: mention → AI answer delivered to defcon.social ─

def test_public_mention_is_answered_in_thread(env):
    activity = create(MASTODON_HTML)
    resp = env.post(activity)
    assert resp.status_code == 202

    assert env.chat.calls == [[{"role": "user", "content": "What is the meaning of life?"}]]
    assert len(env.delivered) == 1
    inbox, out = env.delivered[0]
    assert inbox == "https://defcon.social/inbox"
    assert out["type"] == "Create" and out["actor"] == BOT
    note = out["object"]
    assert note["inReplyTo"] == activity["object"]["id"]
    assert note["attributedTo"] == BOT
    assert note["to"] == [PUBLIC] and note["cc"] == [ALICE]
    assert {"type": "Mention", "href": ALICE, "name": "@alice@defcon.social"} in note["tag"]
    assert "42 is the answer." in note["content"]
    assert 'href="https://defcon.social/@alice"' in note["content"]

    # The reply is dereferenceable at its id (Mastodon fetches it on click/search)
    path = note["id"][len(BASE):]
    served = env.client.get(path, base_url=BASE)
    assert served.status_code == 200
    assert served.mimetype == "application/activity+json"
    assert served.get_json()["id"] == note["id"]


def test_unsigned_activity_is_rejected(env):
    resp = env.post(create(MASTODON_HTML), signed=False)
    assert resp.status_code == 401
    assert env.chat.calls == [] and env.delivered == []


def test_forged_signature_is_rejected(env):
    other_private, _ = generate_rsa_keypair()
    activity = create(MASTODON_HTML)
    body = json.dumps(activity).encode()
    headers = {"Content-Type": "application/activity+json"}
    headers.update(sign_request(private_key=other_private, key_id=f"{ALICE}#main-key",
                                method="POST", url=f"{BASE}/ap/inbox", body=body, headers=headers))
    resp = env.client.post("/ap/inbox", data=body, headers=headers, base_url=BASE)
    assert resp.status_code == 401
    assert env.delivered == []


def test_direct_message_gets_private_answer(env):
    activity = create(MASTODON_HTML, to=[BOT], cc=[])
    env.post(activity)
    assert len(env.delivered) == 1
    note = env.delivered[0][1]["object"]
    assert note["to"] == [ALICE] and note["cc"] == []
    # A direct answer is never served publicly
    assert env.client.get(note["id"][len(BASE):], base_url=BASE).status_code == 404


def test_unlisted_and_content_warning_are_kept(env):
    activity = create(MASTODON_HTML, to=[f"{ALICE}/followers"], cc=[PUBLIC, BOT],
                      sensitive=True, summary="spoilers")
    env.post(activity)
    note = env.delivered[0][1]["object"]
    assert note["to"] == [ALICE] and note["cc"] == [PUBLIC]
    assert note["sensitive"] is True and note["summary"] == "spoilers"


def test_follow_up_in_thread_includes_history(env):
    env.post(create(MASTODON_HTML))
    first = env.delivered[0][1]["object"]

    env.chat.answer = "Because 6 × 7 = 42."
    follow_up = create("<p>@gpu why?</p>", in_reply_to=first["id"])
    env.post(follow_up)
    assert env.chat.calls[1] == [
        {"role": "user", "content": "What is the meaning of life?"},
        {"role": "assistant", "content": "42 is the answer."},
        {"role": "user", "content": "why?"},
    ]
    assert env.delivered[1][1]["object"]["inReplyTo"] == follow_up["object"]["id"]


def test_reply_to_bot_without_mention_is_answered(env):
    env.post(create(MASTODON_HTML))
    first = env.delivered[0][1]["object"]
    env.post(create("<p>thanks, and why?</p>", in_reply_to=first["id"],
                    cc=[f"{ALICE}/followers"], mention=False))
    assert len(env.delivered) == 2


def test_redelivered_activity_is_answered_once(env):
    activity = create(MASTODON_HTML)
    env.post(activity)
    env.post(activity)
    assert len(env.chat.calls) == 1 and len(env.delivered) == 1


def test_unrelated_reply_is_ignored(env):
    env.post(create("<p>nice post</p>", in_reply_to="https://defcon.social/users/bob/statuses/1",
                    cc=[f"{ALICE}/followers"], mention=False))
    assert env.chat.calls == [] and env.delivered == []


def test_other_bots_are_ignored(env, remote_key):
    robot = "https://defcon.social/users/robot"
    env.bot.storage.cache_remote_actor(robot, remote_actor(robot, remote_key[1], "Service"))
    env.post(create(MASTODON_HTML, actor=robot), actor=robot)
    assert env.chat.calls == [] and env.delivered == []


def test_likes_are_not_answered(env):
    env.post({
        "@context": "https://www.w3.org/ns/activitystreams",
        "id": "https://defcon.social/users/alice#likes/1",
        "type": "Like",
        "actor": ALICE,
        "object": f"{BASE}/ap/notes/{uuid.uuid4()}",
    })
    assert env.chat.calls == [] and env.delivered == []


def test_per_actor_rate_limit(env):
    for _ in range(env.bot.config.replies_per_actor + 2):
        env.post(create(MASTODON_HTML))
    assert len(env.delivered) == env.bot.config.replies_per_actor


def test_empty_mention_gets_greeting(env):
    env.post(create('<p><span class="h-card">'
                    '<a href="https://gpu.social/@gpu">@<span>gpu</span></a></span></p>'))
    assert env.chat.calls == []
    assert "Ask me anything" in env.delivered[0][1]["object"]["content"]


def test_long_answer_fits_instance_limit(env):
    env.chat.answer = "word " * 5000
    env.post(create(MASTODON_HTML))
    note = env.delivered[0][1]["object"]
    text = gap.html_to_text(note["content"])
    assert len(text) <= env.bot.config.max_chars


def test_answer_text_is_escaped(env):
    env.chat.answer = "<script>alert(1)</script> see https://gpu.social"
    env.post(create(MASTODON_HTML))
    content = env.delivered[0][1]["object"]["content"]
    assert "<script>" not in content and "&lt;script&gt;" in content
    assert '<a href="https://gpu.social"' in content


def test_gpu_chat_calls_openai_compatible_api(monkeypatch):
    seen = {}

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": " hello "}}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.update(url=url, headers=headers, json=json)
        return Resp()

    monkeypatch.setattr(requests, "post", fake_post)
    chat = gap.GPUChat("https://gpu.social", "KEY", "qwen2.5-14b")
    assert chat.complete([{"role": "user", "content": "hi"}]) == "hello"
    assert seen["url"] == "https://gpu.social/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer KEY"
    assert seen["json"]["model"] == "qwen2.5-14b"
    assert seen["json"]["messages"][0]["role"] == "system"
    assert seen["json"]["messages"][-1] == {"role": "user", "content": "hi"}
