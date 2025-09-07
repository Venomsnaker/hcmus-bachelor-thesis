#### How does this work?
- Break down the original response `r` into `f` atomic facts (by an LLM).
- Verify each fact in `f` (separately) by one of the following techniques.
    - No-context LM
    - Retrieve -> LM
    - Nonparametric probability (NP)
    - Retreieve -> LM + NP (ensemble)
