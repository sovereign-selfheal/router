# Namespace policy: contract for agents (router v0.11.0)

An agent that investigates an application in a namespace tells the router which namespace it works on.
When a platform team marks that namespace as restricted, every request of the investigation stays on
the local model. This page is for the people who write agents. The router side is in the README,
section "Namespace policy".

> **Support status:** LiteLLM is community software, operated by the customer, **not** supported by
> Red Hat. OGX (the agent runtime of `triage-agent`) is the upstream project formerly named Llama Stack.

## What the agent sends

One field in the body of **every** chat completion request:

```json
{
  "model": "auto",
  "messages": [...],
  "selfheal_namespaces": ["payments"]
}
```

- A list of namespace names (a single string also works). Names are lowercase RFC 1123 labels; the
  router ignores invalid names. It checks every valid name of the first 500 entries (since v0.11.1;
  v0.11.0 kept only 20) and logs at most 20.
- Send the namespaces that the investigation is about, for example the `namespace` label of each alert
  of the Alertmanager group. Send the same list on every call of the investigation: the router decides
  each call on its own and keeps no state between calls.
- The router removes the field before it calls a model. A model, local or external, never sees it.
- The router reads the field only when its `hint` switch is on (`NAMESPACE_HINT_ENABLED`). With the
  switch off the field is removed and has no effect, so an agent can send it at any time.

The router also scans the request text for namespace names when its `scan` switch is on. The hint is
still useful: a name that never appears in the text (an alert without a `namespace` label, a log line
without it) is found only through the hint. A restricted namespace found by either way is enough.

Since v0.11.1 the scan also reads the names inside JSON strings (the arguments of a tool call, where
the quotes are escaped: `{"query": "up{namespace=\"payments\"}"}`), in JSON argv arrays
(`["oc", "get", "pods", "-n", "payments"]`) and in every alternative of a PromQL regex matcher:
`namespace=~"agentic-triage|payments"`, `"(payments)"`, `"^(?:a|b)$"`. A wildcard alternative
(`"pay.*"`) selects the labelled namespaces that start with `pay`. The router parses the value and
never runs it as a regex; a match-all value (`".*"`, `".+"`) names no namespace, like a query without
`namespace`.

## How to send it

**Through OGX** (Responses API, the agent loop runs in OGX). OGX copies the keys of a top-level
`extra_body` object of the Responses request into each chat completion it sends. With `ogx_client`,
that object goes inside the client's own `extra_body`:

```python
await client.responses.create(
    model="router/auto",
    input=prompt,
    instructions=instructions,
    tools=tools,
    stream=True,
    extra_body={
        "max_infer_iters": 18,                                   # read by OGX
        "extra_body": {"selfheal_namespaces": ["payments"]},     # copied into every LLM call
    },
)
```

**With the OpenAI SDK** (the agent calls the router directly):

```python
client.chat.completions.create(model="auto", messages=messages,
                               extra_body={"selfheal_namespaces": ["payments"]})
```

**With curl:**

```bash
curl -s https://router.<apps domain>/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"auto","selfheal_namespaces":["payments"],
       "messages":[{"role":"user","content":"Why does the orders API return 500?"}]}'
```

## What was checked (2026-10-05, cluster ocp.rw287)

- **OGX 1.4.0** (`docker.io/ogxai/distribution-starter@sha256:9d6afd54…`, the image of `triage-agent`),
  with the `triage-agent` run config, pointed at a fake OpenAI server that recorded every request.
  One Responses call with an MCP tool (Prometheus MCP), `stream=false` and `stream=true`: two LLM calls
  each (tool call, then the answer after the tool result), and **all four** had
  `"selfheal_namespaces": ["payments"]` at the top level of the chat completion body.
- **LiteLLM 1.102.0 with router v0.10.0** (before this release) received the field through the gateway:
  the local model (vLLM) and the external model (Gemini) both answered 200. This release removes the
  field anyway, so that an external provider never learns the namespace names.

## Things to avoid

- Do not put the name of a restricted namespace in a fixed system prompt or knowledge file when the
  `scan` switch is on: every request of the agent would stay local, also for other applications.
- Do not use the field to choose a model. Only the namespace labels decide, and only `restricted`
  changes the routing.
