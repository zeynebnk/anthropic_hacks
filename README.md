# anthropic_hacks

## Language Models (can be) Few-Shot Fakers: Disentangling Robust Reasoning from Broad Pattern Matching

We present a series of explorations aiming to draw insights on the relevant yet understudies areas of chain-of-thought faithfulness, robustness and generalizability, and underlying systems of reasoning in language models. 

Our contributions are as follows: 

(a) we find that language models' chains of thought are not always reliable explanations of model decisions, identifying faked reasoning in language models. Faked and unfaithful reasoning is  often indicative of when they are doing the very opposite—relying on memorization over extrapolative reasoning. In-distribution examples may present lines of reasoning when reasoning was not used. This has implications on the usefulness of learned reasoning paths in more complex tasks.  

(b) models with corrupted chains of thought inconsistently affects the ability reach the correct answer, questioning the role of in-context information. Identifying where this degradation happens is relevant, and may be tied to similar topics of memorization. 

(c) we experiment with RL for robustness and error-correction by performing GRPO for math reasoning with random intervention and corruption by Claude in intermediate reasoning steps.
