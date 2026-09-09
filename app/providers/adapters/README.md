# Custom adapters

Most providers NOXIS ships with are OpenAI-compatible and are served by
`app.providers.openai_compatible.OpenAICompatibleProvider`. Drop a new
module here only when a provider needs bespoke request/response shaping
that the generic adapter can't express (e.g. a non-standard streaming
format). Subclass `app.providers.base.Provider` and register it in
`app/providers/registry.py`.
