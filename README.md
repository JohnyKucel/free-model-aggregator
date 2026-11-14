# free-model-aggregator

One server that serves the OpenAI Chat Completions API and the Anthropic
Messages API over ten simulated free-tier model providers, in JSON and in
server-sent events. For each request it picks the strongest free model that
meets the request's requirements and still has budget, speaks that provider's
wire, fails over when a provider refuses, and returns the answer in the wire
the client used. `node main.ts --list` prints the catalog it found.

## Running it

Node 22, standard library only, nothing installed:

```
MAKEFUN_ENV_URL=<env> MAKEFUN_PROMPT_FILE=<brief> node main.ts
node main.ts --list   # print the catalog and exit 0
```

The brief (`MAKEFUN_PROMPT_FILE`) carries the provider directory, the quality
table and the workload manifest. The env (`MAKEFUN_ENV_URL`) serves the ten
providers under `/p/<id>` plus `POST /register`, `GET /work` and
`GET /instance`. The server registers, serves `GET /v1/models`,
`POST /v1/chat/completions` and `POST /v1/messages`, polls `/work` every
100 ms and exits 0 when the workload is done.

## How routing works

The manifest is known at startup, so each request gets a primary before the
first one arrives: among the free models meeting its requirements and within
20 milli-points of the best, the one whose provider carries the fewest
assigned calls so far. Spreading costs almost no quality (a weight-1 request
at 1470 instead of 1475 loses 5 of 38,248 weighted milli-points) and keeps
every provider under its per-trial budget, which the brief never publishes.
A JSON-mode request from an OpenAI client is never routed to an
Anthropic-wire provider, whose wire cannot carry `response_format`.

At runtime the primary is tried first and, on any refusal (402, 429, 5xx,
unreachable), the remaining candidates in rating order, stopping at the
first answer: exactly one successful provider call per request. Calls are
sequential, one request at a time, so a provider never sees a second
concurrent call. Every decision is a function of the brief, the manifest
and the providers' answers so far: no clock, no randomness, so a replay
leaves the same state export byte for byte.

## Wires

Every translation passes the request through untouched: system prompt, user
text (images included, byte-identical), tools (names in order), stop
sequences, max_tokens, JSON mode. The provider echoes what it parsed and
the client compares that echo with what it sent, so a field rewritten or
dropped loses the request. The provider's answer text is returned whole,
its usage counts exactly as that wire reports them, and its stop reason is
mapped by family (end_turn to stop, max_tokens to length, tool_use to
tool_calls, refusal to content_filter).

## Practice scores

Local trials against the kit's practice instances, scored by the sealed
scorer (all three trials, process driver):

| trial | valid | upstream calls | score (quality / served / wire / failover / catalog) |
|-------|-------|----------------|------------------------------------------------------|
| 0     | 20/20 | 27             | 99 (99 / 100 / 100 / 100 / 100)                      |
| 1     | 20/20 | 30             | 99 (99 / 100 / 100 / 100 / 100)                      |
| 2     | 20/20 | 29             | 99 (99 / 100 / 100 / 100 / 100)                      |

Quality 99 is the ceiling on these instances: only one provider holds a
1475 model and its per-trial budget (10-12 calls) is below the 17 requests
that want it, so some requests must take the 1470 model.

## Files

- `main.ts` — the server: brief, catalog, routing plan, routes, register, poll.
- `lib/providers.ts` — the brief's documents, the catalog probe, capability matching.
- `lib/wire.ts` — the client wires and the three upstream wires, both directions.

## License

MIT. See `LICENSE`.
