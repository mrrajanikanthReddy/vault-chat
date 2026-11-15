# vault-chat

A private AI setup that runs entirely on your own machine: offline coding help,
a knowledge system built on your own notes, a chat screen, and an agent you can
call whenever you need it. No accounts, no subscription, no data leaving the
computer.

Everything here is the Python standard library plus NumPy. The only network
destination is a model runtime on `localhost` — by default
[colibri](https://github.com/JustVugg/colibri), a CPU-first inference server
with an OpenAI-compatible API. LM Studio and Ollama still work as optional
providers.

Three assistants share one local database and one memory:

- **Chat** — multiple conversations, streaming, search, regenerate, stop, and
  memory that carries what you said in one chat into the next.
- **Coding** — indexes a repository, sends only the relevant code, and proposes
  edits as diffs that change nothing until you apply them.
- **Personal Agent** — tasks and reminders, recall of earlier conversations and
  preferences, and a small set of tools that run only after you approve them.

---

## The one idea worth understanding first

On a CPU, generating a token means reading every weight the model uses out of
memory. So speed is roughly **memory bandwidth ÷ active bytes per token**. Not
core count, not how much RAM you have.

Three consequences shape every decision in this project:

1. **A smaller model is not just cheaper, it is the difference between usable
   and unusable.** An older laptop streams memory at roughly 20–25 GB/s. A 7B
   model at 4-bit is about 4.7 GB, which lands near 4 tokens per second.
2. **Mixture-of-experts models change the maths in your favour.** A "30B-A3B"
   model holds 30 billion weights but touches only about 3 billion per token.
   You pay the large size in RAM, which you probably have spare, and the small
   size in speed, which you do not.
3. **Prefill — the pause before the first word, while the model reads your
   prompt — is the real tax.** This is why the tools retrieve a few relevant
   passages instead of pasting whole files. The design that makes answers
   accurate and citable is the same one that makes them arrive quickly.

Run `./hardware.py` and it will measure your actual bandwidth and tell you what
your machine can run, rather than guessing.

### Retrieval, not fine-tuning

A common instinct is to train a model on your notes. For personal knowledge
that is the wrong tool, and on a CPU it is not a close call. Fine-tuning teaches
a model **how to behave** — a tone, a format, a style. It is bad at teaching it
**facts**: they smear across billions of weights, come back half-remembered, and
cannot cite a source. Retrieval does the opposite, and updating it means saving
a file.

---

## Requirements

- Python 3.9 or newer for vault-chat itself
- An inference server. The default is **colibri**, which needs Python 3.10+ and
  20 GB or more of disk for its smallest catalog model (`qwen3-coder-30b`, about
  19 GB, roughly 8–10 tokens/second on a CPU). LM Studio or Ollama work too.

Colibri runs as its own background process. vault-chat never starts, stops or
reloads it on its own, so the model stays loaded between requests and
restarting the web UI costs nothing.

### Set up colibri (once)

Nothing here runs automatically; the download step is yours to approve.

The engine is part of this project, in `c/` (`c/coli` is the launcher). It is
the Colibri engine, Apache-2.0; `c/LICENSE`, `c/NOTICE` and
`c/THIRD_PARTY_NOTICES.md` stay with that code. You do not clone anything.
Model weights are not included.

```bash
brew install python@3.12                                  # macOS, only if you have no Python 3.10+
python3.12 c/setup_hw.py --json                            # what this machine is
python3.12 c/coli setup --list --json                      # which models fit
python3.12 c/coli setup --yes --model qwen3-coder-30b --no-start   # downloads ~19 GB
python3.12 c/coli start --background --no-browser
python3.12 c/coli status --json                            # wait for server.state == "ready"
```

`./colibri.py guide` prints the same steps for your machine, and
`./colibri.py status | start | stop | logs | models | hardware` wrap colibri's own
commands. `./colibri.py install MODEL` asks before downloading anything. The
**Model Settings** page in the UI offers start, stop, logs and the guide as well.

vault-chat talks to `http://127.0.0.1:8000/v1` by default. Change
`COLIBRI_BASE_URL` (environment variable or Model Settings) if colibri listens
elsewhere. If you started colibri with `--kv-slots N`, set `COLIBRI_KV_SLOTS` to
the same N so each conversation reuses its own prompt cache.

LM Studio remains useful for one thing colibri does not do: since version 0.4.0
it can execute MCP tool servers for the model, which the `tools` profile of
`agent.py` relies on.

---

## Install

### Linux and macOS

```bash
./install.sh
```

### Windows

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

### Any platform, directly

```bash
python3 install.py
```

The installer creates a private virtual environment, installs NumPy, measures
your hardware, scaffolds a notes vault at `~/vault`, writes a config tuned to
your machine, generates launcher scripts in `./bin`, and runs the self-test. It
never edits your `PATH`, never installs anything system-wide, and never installs
a model runtime behind your back — where one is needed it prints the command.

| Flag | Effect |
|---|---|
| `--yes` | Accept every prompt (non-interactive) |
| `--vault PATH` | Put notes somewhere other than `~/vault` |
| `--schedule` | Also set up nightly re-indexing |
| `--quick` | Skip the memory bandwidth benchmark |
| `--no-verify` | Skip the self-test |
| `--check` | Report what is set up and change nothing |

### Embeddings are optional

Colibri serves chat completions only. Without an embedding model, notes,
memories, code and past conversations are searched with SQLite full-text search
(FTS5 with BM25 ranking), which works well for the words you actually wrote. For
semantic search, point `VAULT_EMBED_URL` at any local OpenAI-compatible
`/v1/embeddings` server (for example Ollama with `nomic-embed-text`) and run
`./index.py --rebuild`. Nothing is ever sent to an external embedding service.

With LM Studio or Ollama as the runtime, the installer prints the model commands
for your machine instead:

```bash
lms load qwen3-coder-30b-a3b-instruct --yes      # or whatever it recommended
lms load text-embedding-nomic-embed-text-v1.5 --yes
lms server start
```

---

## The scripts

Each one runs standalone with `--help`. After install, the `bin/` launchers wrap
them with the right Python, so `bin/vault-ask` and `python3 ask.py` do the same
thing.

### `hardware.py` — what can this machine run?

Measures memory bandwidth with a short NumPy streaming benchmark, detects cores,
RAM, disk and GPU, and recommends a model, quantisation, context length and
thread count.

```bash
./hardware.py              # report and recommendations
./hardware.py --quick      # skip the benchmark (about a second faster)
./hardware.py --json       # machine-readable; this is what install.py consumes
```

It picks the largest model that both fits in memory **and** stays above roughly
8 tokens per second, then spends any leftover memory on precision rather than
more parameters — a 7B at Q6 beats a 14B at Q4 that runs half as fast. If nothing
can be quick it optimises for the fastest option rather than the biggest. If a
discrete GPU is present it sizes against VRAM instead and skips the CPU estimate,
which no longer applies.

### `index.py` — build the search index

Walks the vault and embeds anything that changed into a local SQLite database.

```bash
./index.py              # refresh changed files
./index.py --rebuild    # start over (needed if you change embedding model)
./index.py --stats      # what is in the index right now
```

It hashes file contents rather than trusting timestamps, so a nightly run over an
unchanged vault costs a few hundred milliseconds. It splits on markdown headings
and keeps the heading with each passage, so citations can point at a section
rather than just a file. Deleted notes are purged automatically.

**One guard worth knowing:** if you change the embedding model it refuses to run
and asks you to `--rebuild`. Vectors from two different models are not comparable,
and mixing them degrades every future search in a way that is hard to notice.

### `ask.py` — question your notes

```bash
./ask.py "what did I decide about retries?"
./ask.py --search "terraform"        # show matches without calling the model
./ask.py --show-context "..."        # print exactly what was sent
./ask.py                             # interactive loop
```

Answers carry inline citations and a source list with similarity scores.
`--search` is instant because it never touches the model, and it is what you will
reach for half the time. `--show-context` tells you whether a bad answer came
from bad retrieval or a bad model.

Useful flags: `-k` (how many passages), `--min-score` (similarity floor),
`--max-context-chars` (prompt budget — this is your speed dial).

### `agent.py` — the on-demand personal agent

```bash
./agent.py "what did I decide about retry budgets?"     # grounded in notes
./agent.py -p code --file src/handler.py "why twice?"   # grounded in a file
./agent.py -p tools "summarise today's commits"         # with MCP tools
./agent.py --save "explain SQLite WAL mode"             # append to today's note
./agent.py --list                                       # show profiles
```

A **profile** bundles a model, a system prompt, whether your notes are searched,
and which tools are reachable. Three ship by default:

| Profile | Notes searched | Tools | For |
|---|---|---|---|
| `ask` | yes | none | Daily driver, fastest |
| `code` | no | none | Coding questions; attach files with `--file` |
| `tools` | no | MCP | Anything needing outside data |

Add your own in `profiles.json` (see **Configuration**), using the same shape.

> **Safety:** when LM Studio executes MCP tools it does so server-side, which
> means **nothing pauses to ask you before a tool runs**. `agent.py` therefore
> refuses to run any profile whose MCP servers do not declare an `allowed_tools`
> allowlist. Override with `--allow-any` only if you mean it. Keep the allowlist
> to five or six tools: small models pick badly from long menus, and every tool
> schema costs prefill on every single turn.

### `chat.py` — the assistant in your browser

```bash
./chat.py --open          # serve and open a browser
./chat.py --port 9000
```

Standard library only, no build step and no CDN — it works with the network
cable unplugged. The page lives in `web/` and is served as one self-contained
document. Sections:

| Section | What it does |
|---|---|
| Chat | Conversations with streaming, markdown, syntax highlighting, copy, regenerate and stop |
| Coding | Choose a project, attach files or paste code; proposed edits appear as diffs with Apply, Reject and Revert; commands run only after approval, with their output logged |
| Personal Agent | Tasks and reminders, recall, approvals waiting for you, recent tool runs |
| History | Every conversation, full-text search, rename, tags, delete |
| Memory | View, search, edit, pin and delete memories, jump to the source conversation, export/import, turn automatic learning on or off |
| Projects | Repository folders, indexing status, per-project memory |
| Model Settings | Provider, URLs, model, temperature, top_p, max tokens, context budget; colibri start/stop/logs/guide |
| System Health | Connection, model, context length, compute, database, counts, and a performance dashboard with benchmark results |

The status bar always shows the provider, the active model, the compute backend,
the connection state and whether an answer is streaming. When the inference
server is down or still loading a model, a banner says so and offers Retry
(and Start colibri when it can find your checkout); history, memory and projects
keep working, and an unanswered question can be retried later.

Keyboard: Enter sends, Shift+Enter makes a new line, Escape stops generating.
Stopping keeps the partial answer, marked as stopped.

The original `/api/chat`, `/api/meta` and `/api/session*` endpoints still work
for scripts.

#### How a turn is assembled

Nothing sends your whole history or memory to the model. Each turn:

1. stores your message, then extracts memories from it (your words only,
   sensitive details skipped unless you explicitly ask);
2. classifies the intent (conversation, coding, action, recall, planning) and
   runs any obvious action directly, such as adding a reminder;
3. retrieves the most relevant memories, project memory, code, notes and past
   conversations, each within its own share of a token budget
   (`CONTEXT_BUDGET_TOKENS`, sized from the model's context length);
4. keeps the newest whole turns and replaces older ones with a stored summary,
   while the original messages stay in the database;
5. puts retrieved text in fenced `<reference>` blocks the model is told never to
   obey, and places it last so colibri can reuse the cached prompt prefix;
6. streams the answer, records timing and token counts, and for coding turns
   turns SEARCH/REPLACE blocks into a pending diff.

#### Approvals

Model output is only ever a suggestion. File edits become proposals you apply
or reject; `run_command` waits for approval, never uses a shell, refuses `sudo`
and shell operators, and flags destructive commands. Every tool call (requested,
approved, rejected, succeeded or failed) is logged in `tool_runs`. A turn that
has read untrusted retrieved text loses its file-touching tools for the rest of
that turn.

### `colibri.py` and `bench.py`

```bash
./colibri.py status            # coli status --json
./colibri.py start             # coli start --background --no-browser
./bench.py                     # benchmark the current provider
./bench.py -p colibri -p ollama -p lmstudio --runs 3   # compare (each must be running)
./bench.py --cold              # also time a colibri restart until ready
```

The benchmark records time to first token, tokens per second, total time and
prompt tokens per prompt, plus the server's RAM and CPU during generation
(summed over its process tree) and the model's size on disk. Results are stored
and shown on the System Health page next to the timings of real answers.

It binds to `127.0.0.1` on purpose. There is **no authentication**, so anyone who
can reach the address can use your model and read answers drawn from your private
notes. If you need it from another device, tunnel to it (Tailscale's free tier,
or SSH port forwarding) rather than passing `--host 0.0.0.0`.

### Conversation history — `agent.py` and `chat.py`

Conversations are saved automatically and remembered within a session.

```bash
./agent.py --sessions              # list saved conversations
./agent.py -c                      # continue the most recent one
./agent.py -c "and for writes?"    # continue it with a one-shot follow-up
./agent.py --resume 7              # continue a specific one
./agent.py --no-history "..."      # neither read nor write saved history
```

In `chat.py` the sidebar lists past conversations; click one to reopen it, `×`
to delete it, and "New conversation" to start fresh. Interactive `agent.py`
also carries earlier turns forward now, so follow-up questions work.

Things worth knowing:

- Only your question and the answer are stored, **not** the note passages that
  were pasted into the prompt. History stays small, and an old answer is never
  replayed alongside note text that has since changed.
- Only the most recent ~4,000 characters of earlier turns are sent to the model
  (`--history-chars` in the CLI). A small model pays for every token on every
  turn, so old turns are dropped whole rather than carried forever.
- Note retrieval looks only at your newest message. A follow-up like "why?"
  retrieves poorly; restate the topic ("why three retries?") for better matches.
- Tool profiles (MCP) still answer each question on their own.
- History lives in the same SQLite file as the index (`<vault>/.index/vault.db`)
  but in separate tables, so `index.py --rebuild` does not erase it. If you
  delete `.index/`, you delete your history too. Keep that folder out of git.

### `quiz.py` — remember what you wrote

```bash
./quiz.py              # one note that is due
./quiz.py -n 3         # three notes
./quiz.py --due        # what is waiting, without starting
```

Generates recall questions from notes you have not reviewed, makes you answer
before revealing, and schedules the next review on an expanding ladder — 1 day,
3, 7, 16, 35, 75, 160 — resetting if you miss.

This is the step most people skip and the one that compounds. An assistant that
answers everything for you will make you *feel* more knowledgeable while making
you less so, because you have outsourced the exact mental muscle you were trying
to build. For things you want to **know**, let it quiz you. For things you only
want to **look up**, let it answer.

### `smoke_test.py` — prove the plumbing works

```bash
python3 smoke_test.py
```

Stubs out the model entirely, so it needs no runtime and no downloads. Run it
first on a new machine. It covers indexing, incremental updates, deletion,
retrieval ranking, review scheduling, agent profiles, the tool allowlist, the
chat server over real HTTP, and the hardware recommendation logic.

The assistant platform has its own suite, run against a fake colibri server
that speaks the same API (streaming, usage, 503 while loading, 429 when busy):

```bash
python3 -m unittest discover -s tests -t .
```

---

## Configuration

Settings resolve in this order: **environment variable → value saved in Model
Settings → config file → built-in default**. Values saved in the UI live in the
platform database; API keys are stored there too, never sent anywhere but the
provider you configured, and shown masked.

The config file lives at:

- Linux / macOS: `~/.config/vault-chat/config.json`
- Windows: `%APPDATA%\vault-chat\config.json`

(Earlier versions used a `vault-tools` folder. Running `install.py` moves it to
`vault-chat` automatically, and until then the old folder keeps working.)

```json
{
  "VAULT_DIR": "/home/you/vault",
  "VAULT_BACKEND": "lmstudio",
  "VAULT_CHAT_MODEL": "qwen3-coder-30b-a3b-instruct",
  "VAULT_EMBED_MODEL": "text-embedding-nomic-embed-text-v1.5",
  "VAULT_NUM_CTX": "8192",
  "VAULT_THREADS": "7"
}
```

| Key | Default | Notes |
|---|---|---|
| `VAULT_DIR` | `~/vault` | Where your notes live |
| `VAULT_DB` | `<vault>/.index/vault.db` | The search index |
| `VAULT_BACKEND` | `auto` | `auto`, `colibri`, `lmstudio` or `ollama` (CLI tools) |
| `LLM_PROVIDER` | `colibri` | Web UI provider: `colibri`, `lmstudio`, `ollama` or `openai` (any OpenAI-compatible local server) |
| `COLIBRI_BASE_URL` | `http://127.0.0.1:8000/v1` | colibri's OpenAI-compatible endpoint |
| `COLIBRI_API_KEY` | empty | Only if the server was started with `COLI_API_KEY` |
| `COLIBRI_KV_SLOTS` | `0` | Match colibri's `--kv-slots` to pin each conversation to a cache slot |
| `COLIBRI_DIR` | this project | Folder that contains `c/coli`. Empty uses this repository |
| `COLIBRI_PYTHON` | auto | A Python 3.10+ to run `coli` with |
| `LMSTUDIO_BASE_URL`, `OLLAMA_BASE_URL` | `:1234/v1`, `:11434/v1` | Optional providers for the web UI |
| `OPENAI_COMPAT_BASE_URL`, `OPENAI_COMPAT_API_KEY` | empty | Any other local OpenAI-compatible server |
| `LLM_MODEL` | empty | Model id; empty uses whatever the server has loaded |
| `LLM_TIMEOUT` | `600` | Seconds to wait for a reply |
| `TEMPERATURE`, `CODING_TEMPERATURE`, `TOP_P`, `MAX_TOKENS` | `0.2`, `0.1`, server default, `1024` | Generation parameters |
| `CONTEXT_BUDGET_TOKENS` | `6144` | Prompt budget per turn; smaller means a shorter prefill on a CPU |
| `MEMORY_AUTO_EXTRACT` | `true` | Learn from your messages; "remember …" works either way |
| `MEMORY_LLM_EXTRACTION` | `false` | Also ask the model for memories (one extra call per turn) |
| `USE_NOTES` | `true` | Ground chat and agent answers in the notes vault |
| `VAULT_EMBED_URL` | empty | Local `/v1` embeddings server; empty means keyword search |
| `VAULT_DATA_DIR` | `~/.local/share/vault-chat` (`%LOCALAPPDATA%\vault-chat`) | Where the platform database lives |
| `VAULT_PLATFORM_DB` | `<data dir>/assistant.db` | Conversations, memories, projects, tasks, metrics |
| `LMSTUDIO_HOST` | `http://127.0.0.1:1234` | |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | |
| `LM_API_TOKEN` | empty | Required for MCP over the LM Studio API |
| `VAULT_CHAT_MODEL` | per backend | Set by the installer from `hardware.py` |
| `VAULT_EMBED_MODEL` | per backend | Changing this requires `index.py --rebuild` |
| `VAULT_NUM_CTX` | `8192` | Bigger costs RAM *and* a longer prefill pause |
| `VAULT_THREADS` | `0` (runtime decides) | Physical cores, not logical ones |

Agent profiles go in `profiles.json` next to `config.json`, merged over the
built-in ones:

```json
{
  "research": {
    "description": "Read-only lookups against local files",
    "system": "Use the tools to answer. Say plainly if a tool fails.",
    "vault": false,
    "integrations": [
      { "type": "plugin", "id": "mcp/filesystem", "allowed_tools": ["read_file", "list_directory"] }
    ]
  }
}
```

### Re-tuning after a hardware change

```bash
./hardware.py                                   # see what it now recommends
python3 install.py --check                      # see what is configured
```

The installer never overwrites values already in your config, so to adopt new
recommendations delete the keys you want recomputed (or the whole file) and run
it again.

---

## Running it as a service

So the model is resident whenever you call it, rather than reloading from disk.

### Linux

```bash
cp systemd/lmstudio.service    ~/.config/systemd/user/
cp systemd/vault-index.service ~/.config/systemd/user/
cp systemd/vault-index.timer   ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now lmstudio.service vault-index.timer
loginctl enable-linger $USER      # keep running after logout
```

`vault-index.service` sets `IPAddressAllow=localhost` and `IPAddressDeny=any`,
so the nightly indexer is *structurally* unable to reach the internet rather
than merely not configured to.

### macOS and Windows

`python3 install.py --schedule` sets up a launchd agent or a scheduled task for
the nightly re-index. For the model runtime, LM Studio has a "run the LLM server
on login" setting in its preferences.

---

## Security

- **Make offline enforceable, not assumed.** Block outbound network access for
  the inference process at the firewall — LuLu on macOS, a network namespace on
  Linux, an outbound block rule in Defender Firewall on Windows.
- **Leave everything bound to `127.0.0.1`.** Nothing here has authentication.
- **Prefer GGUF and safetensors.** Both are pure data. Older PyTorch `.bin` and
  `.pt` files are Python pickles, and loading one can execute arbitrary code.
- **Treat retrieved text as untrusted input.** If your agent both reads outside
  content and holds tools that touch your files, those capabilities combine into
  a real vulnerability — a document can carry instructions aimed at the model.
  The defence is architectural: keep reading untrusted content and holding tools
  in separate runs.
- **Back up the vault, not the models.** Notes are irreplaceable; weights are a
  download away. Keep the vault in a local git repository and the `.index/`
  folder out of it.
- **Keep work and personal separate.** Separate vault, separate machine, no
  shared index. Drawing this line on day one costs nothing; redrawing it later
  means auditing everything.

---

## Troubleshooting

**"Cannot reach colibri"** — start it: `python3.12 c/coli start
--background --no-browser` (or Start on the Model Settings page), then check
`./colibri.py status`. A 503 means it is still loading the model; wait and
retry. A "busy" error means its queue is full; retry in a moment.

**"No model runtime found"** (CLI tools) — start one: colibri as above,
`lms server start`, or `ollama serve`. Check with `python3 install.py --check`.

**The assistant forgot something / remembered something wrong** — open Memory,
search for it, and edit or delete it. Say "forget …" in any chat to remove
matching memories.

**"Index is empty"** — run `./index.py`. If it reports 0 files, check `VAULT_DIR`
and remember that only `.md`, `.markdown` and `.txt` files are indexed, and that
dot-directories are skipped.

**"Index was built with X but the embed model is now Y"** — expected. Run
`./index.py --rebuild`.

**Answers are slow to start** — that is prefill, not generation. Lower
`--max-context-chars`, lower `-k`, or lower `VAULT_NUM_CTX`.

**Answers ignore the notes** — run `./ask.py --search "same question"`. If the
right note is not in the results, retrieval is the problem; try `--min-score 0.1`
or rephrase to match the words you actually wrote.

**Tool calls misfire** — reduce `allowed_tools`. Small models pick badly once
there are more than about six.

**numpy missing** — `python3 -m pip install --user numpy`, or re-run the
installer, which puts it in a private virtual environment.

---

## Uninstall

```bash
rm -rf ~/.config/vault-chat           # config  (%APPDATA%\vault-chat on Windows)
rm -rf /path/to/vault-chat            # the tools, venv and launchers
# your notes in ~/vault are untouched — delete them yourself if you want
systemctl --user disable --now vault-index.timer lmstudio.service   # Linux
```

Nothing was installed system-wide, so there is nothing else to undo.
