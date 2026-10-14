# Coding and working style

Write straightforward research code that I can understand, explain, and modify.

- Prefer explicit steps, descriptive names, small functions, and ordinary
  loops and conditionals.
- Use lists and dictionaries unless another structure clearly helps.
- Reuse the repository's existing interfaces and dependencies.
- Add classes only when required by the existing framework or genuinely
  useful for managing state.
- Do not introduce generic frameworks, plugin systems, factories,
  deep inheritance, or speculative abstractions.
- Do not add command-line options for every setting.
  Keep experiment settings together in one small config file.
  Use environment variables for secrets and deployment-specific values.
- Preserve BALROG's existing configuration system; do not replace it.
- Prefer readable code over clever one-liners or compressed expressions.
- Comments should explain intent, assumptions, and non-obvious decisions.
  Avoid narrating obvious code or adding excessive section banners.
- Validate important boundaries: model responses, action parsing,
  episode resets, saved results, and configuration.
- Do not silently swallow exceptions or replace failed runs with fake results.
- Keep retries bounded and record when they happen.
- Add focused tests for behaviour that could invalidate experiments.
  Do not build an elaborate test framework.
- Keep changes scoped to the assignment. Do not refactor unrelated code.
- Before completing a milestone, inspect the diff and simplify unnecessary
  complexity without changing behaviour.
- Explain the implementation in plain language and identify the files
  I should read to understand it.

# Research integrity

- Never invent results, citations, execution status, or personal reflections.
- Separate pilot experiments from final evaluation.
- Preserve raw results and the configuration used to produce them.
- Disclose AI assistance according to the course policy.
- Do not describe generated code as unaided human work.
