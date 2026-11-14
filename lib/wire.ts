// The client wires and the three upstream wires.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions)
// or an Anthropic one (POST /v1/messages), with a JSON body or with server-sent
// events. The entry forwards each request to a provider on that provider's own
// wire (openai, anthropic or gemini) and translates the answer back.
//
// The one rule that governs every translation: pass the request through in the
// provider's wire untouched. Its system prompt, its user text (images included),
// its tools, its stop sequences, its max_tokens, its JSON mode. The provider
// echoes what it parsed, and the client compares that echo with what it sent:
// a field rewritten or dropped loses the request. A JSON-mode request carries
// response_format on the OpenAI wire; a provider whose wire cannot carry it
// echoes it absent, so such a request is never routed to the Anthropic wire.
// The provider's answer text is returned whole, its usage counts exactly as
// that wire reports them, and its stop reason is mapped by family:
// end_turn to stop, max_tokens to length, tool_use to tool_calls, refusal to
// content_filter. Providers here always answer the plain stop family.
//
// Everything here is a pure function of its arguments: no clock and no
// randomness, so the same body and the same provider answer give the same
// bytes on a replay. The model listings use fixed timestamps for the same
// reason: a `created` that moved between runs would be a nondeterministic
// document.

import { isRecord, type Json } from './providers.ts';

export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** The provider's own SSE delta size; the client's concatenation is what matters. */
const SSE_DELTA_CHARS = 13;
/** A fixed id for answers this entry renders itself. */
const AGGREGATE_ID = 'agg-sim';

export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** One OpenAI content part as an Anthropic content block: text or an inline image. */
function openAIPartToAnthropic(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) {
      return { type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/** One Anthropic content block as an OpenAI content part: text or an inline image. */
function anthropicBlockToOpenAI(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** One OpenAI content part as a Gemini content part: text or inline/file image data. */
function openAIPartToGemini(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) return { inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    return { fileData: { fileUri: url } };
  }
  return null;
}

/** One Anthropic content block as a Gemini content part. */
function anthropicBlockToGemini(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { inlineData: { mimeType: media, data: source.data } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  }
  if (block.type === 'tool_use' && typeof block.name === 'string') {
    return { functionCall: { name: block.name, args: isRecord(block.input) ? block.input : {} } };
  }
  return null;
}

function openAIToolsToAnthropic(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [
      {
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      },
    ];
  });
}

function anthropicToolsToOpenAI(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [
      {
        type: 'function',
        function: {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      },
    ];
  });
}

function toolsToGeminiDeclarations(tools: Json[]): Json[] {
  return tools.map((t) => {
    if (isRecord(t.function)) {
      return {
        name: t.function.name,
        description: typeof t.function.description === 'string' ? t.function.description : '',
        parameters: isRecord(t.function.parameters) ? t.function.parameters : { type: 'object', properties: {} },
      };
    }
    return {
      name: t.name,
      description: typeof t.description === 'string' ? t.description : '',
      parameters: isRecord(t.input_schema) ? t.input_schema : { type: 'object', properties: {} },
    };
  });
}

function openAIToolChoiceToAnthropic(choice: unknown): Json | null {
  if (choice === 'auto') return { type: 'auto' };
  if (choice === 'required') return { type: 'any' };
  if (choice === 'none') return { type: 'none' };
  if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    return { type: 'tool', name: choice.function.name };
  }
  return null;
}

function anthropicToolChoiceToOpenAI(choice: unknown): unknown {
  if (!isRecord(choice)) return undefined;
  if (choice.type === 'auto') return 'auto';
  if (choice.type === 'any') return 'required';
  if (choice.type === 'none') return 'none';
  if (choice.type === 'tool' && typeof choice.name === 'string') return { type: 'function', function: { name: choice.name } };
  return undefined;
}

function stopListOf(value: unknown): string[] | undefined {
  if (typeof value === 'string') return [value];
  if (Array.isArray(value)) {
    const stops = value.filter((x): x is string => typeof x === 'string');
    return stops.length > 0 ? stops : undefined;
  }
  return undefined;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names
 * `model`. JSON mode is NOT carried: the Anthropic wire has no field for it, so
 * a JSON-mode request is never routed here (main.ts enforces that).
 */
export function openAIToAnthropicReq(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: raw.role === 'assistant' ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (raw.role === 'assistant') {
      const text = textOf(content);
      if (text !== '') messages.push({ role: 'assistant', content: [{ type: 'text', text }] });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      const part = openAIPartToAnthropic(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopListOf(body.stop);
  if (stops !== undefined) out.stop_sequences = stops;
  const tools = openAIToolsToAnthropic(body);
  if (tools.length > 0) out.tools = tools;
  const choice = openAIToolChoiceToAnthropic(body.tool_choice);
  if (choice !== null) out.tool_choice = choice;
  if (body.stream === true) out.stream = true;
  return out;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent request for
 * `model`. The model travels in the URL path, not the body.
 */
export function openAIToGeminiReq(body: Json): Json {
  const contents: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    const content = raw.content;
    if (typeof content === 'string') {
      contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      const part = openAIPartToGemini(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
  }
  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  const tools = openAIToolsToAnthropic(body);
  if (tools.length > 0) out.tools = [{ functionDeclarations: toolsToGeminiDeclarations(tools) }];
  const config: Json = {};
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  const stops = stopListOf(body.stop);
  if (stops !== undefined) config.stopSequences = stops;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names
 * `model`. The system prompt becomes a system message, image blocks become
 * image_url data URLs, tools are rewritten, and the output cap, temperature,
 * top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAIReq(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const text = textOf(content);
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
        });
      }
      messages.push({ role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: typeof block.content === 'string' ? block.content : textOf(block.content),
        });
        continue;
      }
      const part = anthropicBlockToOpenAI(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopListOf(body.stop_sequences);
  if (stops !== undefined) out.stop = stops;
  const tools = anthropicToolsToOpenAI(body);
  if (tools.length > 0) out.tools = tools;
  const choice = anthropicToolChoiceToOpenAI(body.tool_choice);
  if (choice !== undefined) out.tool_choice = choice;
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** An Anthropic Messages request as a Gemini generateContent request. */
export function anthropicToGeminiReq(body: Json): Json {
  const contents: Json[] = [];
  const system = textOf(body.system);
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (typeof content === 'string') {
      contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        parts.push({ text: typeof block.content === 'string' ? block.content : textOf(block.content) });
        continue;
      }
      const part = anthropicBlockToGemini(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
  }
  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const tools = anthropicToolsToOpenAI(body);
  if (tools.length > 0) out.tools = [{ functionDeclarations: toolsToGeminiDeclarations(tools) }];
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  const stops = stopListOf(body.stop_sequences);
  if (stops !== undefined) config.stopSequences = stops;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// Answers, normalized: every provider wire reduces to one shape, and every
// client wire renders from it. The text is the provider's answer whole, the
// usage is exactly what that wire reported, and the stop reason is the
// provider's own string, mapped to the client's vocabulary by family.
// ---------------------------------------------------------------------------

/** One provider answer, wire-independent. */
export interface NormAnswer {
  text: string;
  input: number | null;
  output: number | null;
  stop: string;
}

const STOP_FAMILY = new Map<string, number>([
  ['stop', 0],
  ['end_turn', 0],
  ['STOP', 0],
  ['length', 1],
  ['max_tokens', 1],
  ['MAX_TOKENS', 1],
  ['tool_calls', 2],
  ['function_call', 2],
  ['tool_use', 2],
  ['content_filter', 3],
  ['refusal', 3],
  ['SAFETY', 3],
]);

function stopFamily(reason: string | null): number | null {
  return reason === null ? null : (STOP_FAMILY.get(reason) ?? null);
}

/** A provider stop reason as an OpenAI finish_reason (same family). */
export function openAIStopOf(stop: string): string {
  const family = stopFamily(stop);
  if (family === 1) return 'length';
  if (family === 2) return 'tool_calls';
  if (family === 3) return 'content_filter';
  return 'stop';
}

/** A provider stop reason as an Anthropic stop_reason (same family). */
export function anthropicStopOf(stop: string): string {
  const family = stopFamily(stop);
  if (family === 1) return 'max_tokens';
  if (family === 2) return 'tool_use';
  if (family === 3) return 'refusal';
  return 'end_turn';
}

/** The text of a Chat Completions message: a string, or its text blocks joined. */
function messageText(message: Json): string {
  if (typeof message.content === 'string') return message.content;
  return textOf(message.content);
}

/** Parse an OpenAI JSON answer body. Null when it is not one. */
export function parseOpenAIJson(bodyText: string): NormAnswer | null {
  let value: unknown;
  try {
    value = JSON.parse(bodyText);
  } catch {
    return null;
  }
  if (!isRecord(value)) return null;
  const choices = Array.isArray(value.choices) ? value.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : null;
  if (message === null || message.role !== 'assistant') return null;
  const stop = typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop';
  const usage = isRecord(value.usage) ? value.usage : {};
  return {
    text: messageText(message),
    input: typeof usage.prompt_tokens === 'number' ? usage.prompt_tokens : null,
    output: typeof usage.completion_tokens === 'number' ? usage.completion_tokens : null,
    stop,
  };
}

/** Parse an Anthropic JSON answer body. Null when it is not one. */
export function parseAnthropicJson(bodyText: string): NormAnswer | null {
  let value: unknown;
  try {
    value = JSON.parse(bodyText);
  } catch {
    return null;
  }
  if (!isRecord(value) || value.type !== 'message' || !Array.isArray(value.content)) return null;
  const stop = typeof value.stop_reason === 'string' ? value.stop_reason : 'end_turn';
  const usage = isRecord(value.usage) ? value.usage : {};
  return {
    text: textOf(value.content),
    input: typeof usage.input_tokens === 'number' ? usage.input_tokens : null,
    output: typeof usage.output_tokens === 'number' ? usage.output_tokens : null,
    stop,
  };
}

/** The text of a Gemini candidate's parts. */
function geminiCandidateText(candidate: Json): string {
  const content = isRecord(candidate.content) ? candidate.content : null;
  if (content === null || !Array.isArray(content.parts)) return '';
  return content.parts
    .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
    .join('');
}

/** Parse a Gemini JSON answer body. Null when it is not one. */
export function parseGeminiJson(bodyText: string): NormAnswer | null {
  let value: unknown;
  try {
    value = JSON.parse(bodyText);
  } catch {
    return null;
  }
  if (!isRecord(value) || !Array.isArray(value.candidates) || value.candidates.length === 0) return null;
  const candidate = isRecord(value.candidates[0]) ? value.candidates[0] : null;
  if (candidate === null) return null;
  const stop = typeof candidate.finishReason === 'string' ? candidate.finishReason : 'STOP';
  const usage = isRecord(value.usageMetadata) ? value.usageMetadata : {};
  return {
    text: geminiCandidateText(candidate),
    input: typeof usage.promptTokenCount === 'number' ? usage.promptTokenCount : null,
    output: typeof usage.candidatesTokenCount === 'number' ? usage.candidatesTokenCount : null,
    stop,
  };
}

/** A normalized answer as an OpenAI Chat Completions JSON object. */
export function renderOpenAIJson(answer: NormAnswer, model: string): Json {
  return {
    id: `chatcmpl-${AGGREGATE_ID}`,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: openAIStopOf(answer.stop), logprobs: null }],
    usage: {
      prompt_tokens: answer.input ?? 0,
      completion_tokens: answer.output ?? 0,
      total_tokens: (answer.input ?? 0) + (answer.output ?? 0),
    },
  };
}

/** A normalized answer as an Anthropic Messages JSON object. */
export function renderAnthropicJson(answer: NormAnswer, model: string): Json {
  return {
    id: `msg_${AGGREGATE_ID}`,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: anthropicStopOf(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.input ?? 0, output_tokens: answer.output ?? 0 },
  };
}

// ---------------------------------------------------------------------------
// Streams: the upstream body is read whole (provider streams here are under
// 2 KB), split into frames, reduced to one normalized answer, and rendered
// in the client's wire. Reading whole first keeps the translators pure and
// deterministic; nothing here depends on arrival timing.
// ---------------------------------------------------------------------------

interface SseFrame {
  event: string | null;
  data: string;
}

function splitSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** An OpenAI SSE stream body as one normalized answer. */
export function parseOpenAIStream(bodyText: string): NormAnswer {
  let text = '';
  let input: number | null = null;
  let output: number | null = null;
  let stop = 'stop';
  for (const frame of splitSseFrames(bodyText)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      if (typeof chunk.usage.prompt_tokens === 'number') input = chunk.usage.prompt_tokens;
      if (typeof chunk.usage.completion_tokens === 'number') output = chunk.usage.completion_tokens;
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    if (choices.length === 0) continue;
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    if (isRecord(choice.delta) && typeof choice.delta.content === 'string') text += choice.delta.content;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  return { text, input, output, stop };
}

/** An Anthropic SSE stream body as one normalized answer. */
export function parseAnthropicStream(bodyText: string): NormAnswer {
  let text = '';
  let input: number | null = null;
  let output: number | null = null;
  let stop = 'end_turn';
  for (const frame of splitSseFrames(bodyText)) {
    if (frame.event === null) continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      if (typeof data.message.usage.input_tokens === 'number') input = data.message.usage.input_tokens;
    }
    if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      text += data.delta.text;
    }
    if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage) && typeof data.usage.output_tokens === 'number') output = data.usage.output_tokens;
    }
  }
  return { text, input, output, stop };
}

/** A Gemini SSE stream body as one normalized answer. */
export function parseGeminiStream(bodyText: string): NormAnswer {
  let text = '';
  let input: number | null = null;
  let output: number | null = null;
  let stop = 'STOP';
  for (const frame of splitSseFrames(bodyText)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data) || !Array.isArray(data.candidates) || data.candidates.length === 0) continue;
    const candidate = isRecord(data.candidates[0]) ? data.candidates[0] : null;
    if (candidate === null) continue;
    text += geminiCandidateText(candidate);
    if (typeof candidate.finishReason === 'string') stop = candidate.finishReason;
    if (isRecord(data.usageMetadata)) {
      if (typeof data.usageMetadata.promptTokenCount === 'number') input = data.usageMetadata.promptTokenCount;
      if (typeof data.usageMetadata.candidatesTokenCount === 'number') output = data.usageMetadata.candidatesTokenCount;
    }
  }
  return { text, input, output, stop };
}

/** A normalized answer as an OpenAI SSE stream: role, deltas, finish, usage, [DONE]. */
export function renderOpenAIStream(answer: NormAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: `chatcmpl-${AGGREGATE_ID}`,
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (let at = 0; at < answer.text.length; at += SSE_DELTA_CHARS) {
    frames.push(sseEvent(null, chunk({ content: answer.text.slice(at, at + SSE_DELTA_CHARS) }, null)));
  }
  frames.push(sseEvent(null, chunk({}, openAIStopOf(answer.stop))));
  frames.push(
    sseEvent(null, {
      id: `chatcmpl-${AGGREGATE_ID}`,
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: answer.input ?? 0,
        completion_tokens: answer.output ?? 0,
        total_tokens: (answer.input ?? 0) + (answer.output ?? 0),
      },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A normalized answer as an Anthropic SSE stream, in event order. */
export function renderAnthropicStream(answer: NormAnswer, model: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: `msg_${AGGREGATE_ID}`,
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.input ?? 0, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (let at = 0; at < answer.text.length; at += SSE_DELTA_CHARS) {
    frames.push(
      sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index: 0,
        delta: { type: 'text_delta', text: answer.text.slice(at, at + SSE_DELTA_CHARS) },
      }),
    );
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStopOf(answer.stop), stop_sequence: null },
      usage: { output_tokens: answer.output ?? 0 },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}
