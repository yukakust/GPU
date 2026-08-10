# WIMS Courier: proposal for Yuka and Vitalik

## Decision requested

Build a very small, signed courier between the two Codex environments. Its job is
to remove human copy/paste of coordination messages while leaving GitHub PRs,
reviews, CI, and approvals as the only canonical engineering authority.

There are two deliberately separate profiles:

1. `wims-courier` is the default. It transports and durably stores messages. It
   never starts Codex, reads project files, or performs work.
2. `wims-executor` is an optional future profile. It is enabled only after both
   people sign the same time-limited scope contract. It can run a separate,
   sandboxed Codex task on an incoming message. This is a conscious opt-in to a
   local autonomous worker; it is not hidden inside the courier.

The first implementation should ship the courier and the consent machinery. The
executor remains disabled unless both sides explicitly enable it.

## Why a new isolated component

The historical joinmultiplayer code has broader behaviour: history processing,
LLM invocation, onboarding, and an answerer daemon. None of that belongs in a
two-person engineering courier. The new code lives under `joinmultiplayer/` in
this repository and has no dependency on that codebase or on the GPU client.

## Courier v1

The public protocol has exactly three operations:

- `send(envelope)`
- `inbox(afterCursor)`
- `ack(messageId)`

An Envelope v1 has a UUID `messageId`, exact `sender` and `recipient`,
`repository`, PR number, exact `headSha`, optional `treeSha` and `artifactSha`,
risk class, requested action, `replyTo`, timestamp, canonical payload SHA-256,
and an Ed25519 sender signature.

The payload is only sanitized coordination metadata plus short text. A GitHub PR
URL may be a locator. It never gives the courier permission to call GitHub.

```text
Codex A -- send --> neutral append-only relay --> durable inbox on Mac B
                                                    |
                                           desktop notification
                                           MCP resource update
                                                    |
                                           active Codex reads inbox
```

The relay is a neutral, authenticated postbox, not an authority. It accepts an
envelope only when the sender is registered, the signature and payload hash
match, the recipient is exact, and the message ID/nonce/timestamp have not been
replayed. It keeps an append-only SQLite journal. The relay URL and peer public
keys are supplied at install time, never committed as credentials or private
infrastructure details.

On the receiving Mac, the courier persists envelopes in its own SQLite inbox in
the same transaction as its receive cursor. `messageId` is unique. Therefore
network delivery is at-least-once, while local visible delivery is exactly-once:
duplicates, restarts, and out-of-order responses cannot make a message execute
twice. `ack` is signed and idempotent; it never deletes the audit trail.

Each accept, reject, receive, and acknowledgement appends a hash-chained
security receipt containing the message ID, envelope hash, signature fingerprint,
decision, and timestamp.

## Non-negotiable courier boundaries

Courier code contains no capability for:

- arbitrary commands, files, code execution, compute, delegation, rooms, or a vault;
- scanning user files or Codex history;
- staging/production credentials or private identities, hosts, URLs, or evidence;
- GitHub review, approval, merge, CI retry, workflow dispatch, migration, or Canary;
- `--dangerously-bypass-approvals-and-sandbox`.

Validation rejects forbidden fields and values, private-looking paths and
addresses, credentials, non-GitHub URLs, command/file requests, and requests for
physical or infrastructure action. Static tests also reject prohibited imports
such as shell execution and GitHub clients.

If Codex is closed, the courier still receives and stores the message, then shows
a desktop notification. If Codex is open, it can expose the same inbox as an MCP
resource and publish a best-effort resource update. There is no claimed API that
wakes a particular existing Codex task.

## Optional WIMS Executor

The courier alone does not answer messages while nobody is at a computer. That
requires a local background Codex runner and write authority, so it is only
available behind a two-party, expiring opt-in.

Both peers must sign byte-identical canonical `ScopeContract` data:

```json
{
  "policyVersion": "wims-executor/v1",
  "policyId": "uuid",
  "repository": "owner/repository",
  "peer": "ed25519-public-key-fingerprint",
  "allowedActions": ["analyze", "draft_patch", "report"],
  "worktreePolicy": "dedicated-clean-worktree-only",
  "maxFiles": 10,
  "maxChangedLines": 300,
  "maxRuntimeSeconds": 900,
  "expiresAt": "RFC3339 timestamp",
  "policySha256": "sha256"
}
```

An executor accepts a message only if it names this `policyId`, both signatures
verify, the contract is unexpired, and the action is in `allowedActions`.
Either peer can disable the policy locally at any time; a hash mismatch or a new
policy version disables execution until both sides consent again.

Initial executor actions should be narrow:

- `analyze`: inspect only the dedicated worktree and return a short finding.
- `draft_patch`: change only the dedicated worktree and return a summary and
  resulting revision hash.
- `report`: send status or an ambiguity back through the courier.

Even with executor enabled, it must not approve, merge, resolve review threads,
retry CI, dispatch workflows, deploy, access credentials, or alter infrastructure.
Automatic commit/push should remain out of v1. GitHub remains where a human sees
the patch, CI result, review, and approval.

This runner is not a way to wake or take over an existing Codex chat. It is a new
local task with only the envelope and explicitly allowed worktree context.

## Red-first verification

Before implementation, tests must cover:

- exact one-time local visibility and idempotent acknowledgement;
- duplicate and out-of-order envelopes;
- replay, changed payload, wrong recipient, and wrong signature;
- private/forbidden fragments;
- command, file, compute, delegation, and physical-action attempts;
- restart between receive, persistence, and acknowledgement;
- two local test identities exchanging a request and reply without human
  copy/paste;
- executor rejection without matching two-party consent, with an expired policy,
  or outside the declared scope.

## Open choices for review

1. Do we agree that `wims-courier` is useful on its own, before unattended work?
2. Do both of us want the optional executor at all?
3. Is the initial executor scope limited to `analyze`, `draft_patch`, and
   `report`, with no automatic GitHub writes?
4. Which neutral relay do we trust to host the append-only postbox? Its endpoint
   and credentials must stay outside this repository.

