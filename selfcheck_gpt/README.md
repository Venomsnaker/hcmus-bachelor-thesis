#### How does this work?
- Break down the original response `r` into sentences.
- From the original prompt `p`, generate n sample responses `sp`.
- Verify each sentence of `r` with each sample response `sp_i` from `sp` to look for hallucination by one of the following techniques.
    - n-gram
    - BERTScore
    - NLI
    - QA
    - Prompt (LLM)

#### Main points
- Focus on checking individual sentences of the response.
- Output whether each sentence is hallucinated or not.
- Do not use the original prompt `r` directly in hallucination detection.