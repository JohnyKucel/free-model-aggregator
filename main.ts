// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The entry serves the OpenAI Chat Completions API and the Anthropic Messages
// API over the ten simulated free-tier providers in the brief, in JSON and in
// server-sent events, and registers the catalog it read from them. For each
// request it chooses the strongest free model that meets the request's
// requirements and still has budget, speaks that provider's wire, fails over
// when a provider refuses, and returns the answer in the wire the client used.
//
// How it runs:
//   1. reads the brief, probes each provider's catalog route in its own wire,
//      and builds one catalog (the `--list` document and the register body);
//   2. listens on 0.0.0.0 at a free port (never 8080: the env itself is there
//      on one layout);
//   3. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port", "catalog"};
//   4. serves the client's requests: GET /v1/models (OpenAI shape for
//      Authorization: Bearer, Anthropic shape for x-api-key), POST
//      /v1/chat/completions and POST /v1/messages, each JSON or SSE;
//   5. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env
//      says {"done": true}. A process still running when the wall ends scores
//      zero, so it never waits for anything else.
// `node main.ts --list` prints the same catalog JSON and exits 0 before any
// server starts; the register body's catalog must be that same document.
// Build it once, as this file does, and use the one object.
//
// Routing, in one paragraph. The manifest is known at startup, so each request
// gets a primary before the first one arrives: among the free models meeting
// its requirements and within 20 milli-points of the best, the one whose
// provider carries the fewest assigned calls so far. Spreading costs almost
// no quality (a w1 request at 1470 instead of 1475 loses 5 of 38,248 weighted
// milli-points) and keeps every provider under its per-trial budget, which
// the brief never publishes. A JSON-mode request from an OpenAI client is
// never routed to an Anthropic-wire provider, whose wire cannot carry
// response_format. At runtime the primary is tried first and, on any refusal
// (402, 429, 5xx, unreachable), the remaining candidates in rating order,
// stopping at the first answer: one successful provider call per request,
// always. Calls are sequential, one request at a time, so a provider never
// sees a second concurrent call. Every decision is a function of the brief,
// the manifest and the providers' answers so far: no clock, no randomness,
// so a replay leaves the same state export byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type Brief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type ProviderDirectoryEntry,
  type RatingRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToGeminiReq,
  anthropicToOpenAIReq,
  openAIError,
  openAIModelList,
  openAIToAnthropicReq,
  openAIToGeminiReq,
  parseAnthropicJson,
  parseAnthropicStream,
  parseGeminiJson,
  parseGeminiStream,
  parseOpenAIJson,
  parseOpenAIStream,
  renderAnthropicJson,
  renderAnthropicStream,
  renderOpenAIJson,
  renderOpenAIStream,
  type NormAnswer,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const PROVIDER_TIMEOUT_MS = 10000;
/** How long the register call may take, in milliseconds. */
const REGISTER_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** How far below the best rating a primary may sit, in milli-points. */
const PRIMARY_TOLERANCE_MILLI = 20;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order. */
let manifestIndex = 0;
/** Calls assigned to each provider by the startup plan, for load-spreading. */
const assignedCalls = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider in the brief's order and build the one catalog. */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = [];
  for (const provider of brief.directory.providers) {
    probes.push(await probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS));
  }
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** One routable model: its provider, its rating, what it can do. */
interface Candidate {
  provider: ProviderDirectoryEntry;
  model: ProbedModel;
  row: RatingRow | null;
  rating: number;
}

function allCandidates(): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ provider: probe.provider, model, row, rating: row?.rating_milli ?? 0 });
    }
  }
  return out;
}

/**
 * The free models meeting one request's requirements, best rating first,
 * ties by provider then model id. An OpenAI client asking for JSON mode can
 * only use a wire that carries response_format, so Anthropic-wire providers
 * are left out for those requests.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic'): Candidate[] {
  const out = allCandidates().filter((c) => {
    if (!meetsRequirements(requires, capabilitiesOf(c.model, c.row))) return false;
    if (clientApi === 'openai' && requires.json === true && c.provider.wire === 'anthropic') return false;
    return true;
  });
  out.sort((a, b) => b.rating - a.rating || (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0) || byId(a.model, b.model));
  return out;
}

/**
 * The primary for one manifest request: among the candidates within
 * tolerance of the best, the one whose provider carries the fewest assigned
 * calls; ties go to the highest rating. Assigning it counts one call on
 * that provider, so later requests spread instead of piling on.
 */
function planPrimary(candidates: Candidate[]): Candidate | null {
  if (candidates.length === 0) return null;
  const best = candidates[0].rating;
  const eligible = candidates.filter((c) => c.rating >= best - PRIMARY_TOLERANCE_MILLI);
  let pick = eligible[0];
  let pickLoad = assignedCalls.get(pick.provider.id) ?? 0;
  for (const c of eligible.slice(1)) {
    const load = assignedCalls.get(c.provider.id) ?? 0;
    if (load < pickLoad || (load === pickLoad && c.rating > pick.rating)) {
      pick = c;
      pickLoad = load;
    }
  }
  assignedCalls.set(pick.provider.id, pickLoad + 1);
  return pick;
}

/** The ordered candidates for every manifest request, primary first. */
function planAll(): Candidate[][] {
  const manifest = brief.workload?.manifest ?? [];
  return manifest.map((entry) => {
    const candidates = candidatesFor(entry.requires, entry.api);
    const primary = planPrimary(candidates);
    if (primary === null) return candidates;
    return [primary, ...candidates.filter((c) => !(c.provider.id === primary.provider.id && c.model.id === primary.model.id))];
  });
}

let plan: Candidate[][] = [];

/** The free models across the providers that answered: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

// ---------------------------------------------------------------------------
// One provider call, and one client request served through the candidates.
// ---------------------------------------------------------------------------

interface UpstreamResult {
  status: number;
  text: string;
  contentType: string;
}

/** Build the provider's request for one candidate, in that provider's wire. */
function providerRequest(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): { url: string; headers: Record<string, string>; payload: Json } {
  const provider = candidate.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'openai') {
    if (clientApi === 'openai') {
      const payload: Json = { ...body, model: candidate.model.id };
      if (body.stream === true) payload.stream_options = { include_usage: true };
      return { url: `${base}/v1/chat/completions`, headers, payload };
    }
    return { url: `${base}/v1/chat/completions`, headers, payload: anthropicToOpenAIReq(body, candidate.model.id) };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    if (clientApi === 'anthropic') {
      return { url: `${base}/v1/messages`, headers, payload: { ...body, model: candidate.model.id } };
    }
    return { url: `${base}/v1/messages`, headers, payload: openAIToAnthropicReq(body, candidate.model.id) };
  }
  const encoded = encodeURIComponent(candidate.model.id);
  const stream = body.stream === true;
  const url = stream
    ? `${base}/v1beta/models/${encoded}:streamGenerateContent?alt=sse`
    : `${base}/v1beta/models/${encoded}:generateContent`;
  const payload = clientApi === 'openai' ? openAIToGeminiReq(body) : anthropicToGeminiReq(body);
  return { url, headers, payload };
}

/** One call to one provider: its HTTP status and body text, or null when unreachable. */
async function callProvider(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Promise<UpstreamResult | null> {
  const { url, headers, payload } = providerRequest(candidate, clientApi, body);
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
    });
  } catch {
    return null;
  }
  return { status: res.status, text: await res.text(), contentType: res.headers.get('content-type') ?? 'application/json' };
}

/** A provider's 2xx body as the client's answer: relayed when the wires match, translated when they do not. */
function answerFor(
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  stream: boolean,
  upstream: UpstreamResult,
): { status: number; contentType: string; text: string } | null {
  const wire = candidate.provider.wire;
  const model = candidate.model.id;
  if (!stream) {
    if ((clientApi === 'openai' && wire === 'openai') || (clientApi === 'anthropic' && wire === 'anthropic')) {
      return { status: 200, contentType: 'application/json', text: upstream.text };
    }
    let answer: NormAnswer | null = null;
    if (wire === 'openai') answer = parseOpenAIJson(upstream.text);
    else if (wire === 'anthropic') answer = parseAnthropicJson(upstream.text);
    else answer = parseGeminiJson(upstream.text);
    if (answer === null) return null;
    const rendered = clientApi === 'openai' ? renderOpenAIJson(answer, model) : renderAnthropicJson(answer, model);
    return { status: 200, contentType: 'application/json', text: JSON.stringify(rendered) };
  }
  if ((clientApi === 'openai' && wire === 'openai') || (clientApi === 'anthropic' && wire === 'anthropic')) {
    return { status: 200, contentType: 'text/event-stream; charset=utf-8', text: upstream.text };
  }
  let answer: NormAnswer;
  if (wire === 'openai') answer = parseOpenAIStream(upstream.text);
  else if (wire === 'anthropic') answer = parseAnthropicStream(upstream.text);
  else answer = parseGeminiStream(upstream.text);
  const rendered = clientApi === 'openai' ? renderOpenAIStream(answer, model) : renderAnthropicStream(answer, model);
  return { status: 200, contentType: 'text/event-stream; charset=utf-8', text: rendered };
}

/**
 * Serve one client request: try the planned candidates in order until one
 * answers, and return its answer in the client's wire. A refusal is not an
 * answer: it is counted in the log and the next candidate is tried.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json): Promise<{ status: number; contentType: string; text: string }> {
  const stream = body.stream === true;
  const requires = requirementsFor(body);
  const index = manifestIndex - 1;
  const ordered = index >= 0 && index < plan.length ? plan[index] : candidatesFor(requires, clientApi);
  for (const candidate of ordered) {
    const upstream = await callProvider(candidate, clientApi, body);
    if (upstream === null) {
      log(`${clientApi} request ${index + 1}: ${candidate.provider.id}/${candidate.model.id} unreachable`);
      continue;
    }
    if (upstream.status < 200 || upstream.status >= 300) {
      log(`${clientApi} request ${index + 1}: ${candidate.provider.id}/${candidate.model.id} refused HTTP ${upstream.status}`);
      continue;
    }
    const answer = answerFor(candidate, clientApi, stream, upstream);
    if (answer === null) {
      log(`${clientApi} request ${index + 1}: ${candidate.provider.id}/${candidate.model.id} answered an unparsable body`);
      continue;
    }
    log(`${clientApi} request ${index + 1}: served by ${candidate.provider.id}/${candidate.model.id} (${candidate.rating})`);
    return answer;
  }
  const message = 'no free model on the directory meets this request';
  if (clientApi === 'anthropic') {
    return { status: 503, contentType: 'application/json', text: JSON.stringify(anthropicError('overloaded_error', message)) };
  }
  return { status: 503, contentType: 'application/json', text: JSON.stringify(openAIError(message, 'invalid_request_error', null)) };
}

// ---------------------------------------------------------------------------
// HTTP routes, register, poll, start.
// ---------------------------------------------------------------------------

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendText(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    const answer = await serve('openai', body);
    sendText(res, answer.status, answer.contentType, answer.text);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    const answer = await serve('anthropic', body);
    sendText(res, answer.status, answer.contentType, answer.text);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(REGISTER_TIMEOUT_MS),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  plan = planAll();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
