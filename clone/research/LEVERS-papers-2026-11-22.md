# KV-cache translation: KV-Lingo, HeteroFold, related work, and levers for kvx

Written 2026-11-22 03:08 UTC. Web reading only, no compute rented. Sources: arXiv HTML of 2609.32610 and
2609.32259 (local copies are in this directory), plus arXiv and NeurIPS abstract pages. Where a paper
does not state something, this file says "not stated" rather than guessing.

## 1. KV-Lingo (arXiv 2609.32610; Castin, Sakamoto, Filippova, Monteiro, Cuturi, Ablin; Apple)

**Map form (Eq. 3.1).** Linear maps, "typically one per layer of the target model". For each target
layer l, K_hat = sum over l' in E_l of T_K^{l'->l} K_src^{l'}, and the same for V with separate maps.
- Head-wise (block-diagonal) or head-mixing (dense d_src x d_tgt). Head-wise "suffice[s]" on
  shape-matched Qwen3 pairs because "the heads of these released pairs are aligned one to one".
  On cross-layer pairs, "Mixing heads and adding source layers both improve the scores".
- Source layers (nu per target layer): relative depth, greedy ridge selection, or R2 scoring (top nu by
  single-layer R2, ranked separately for K and V). The cross-layer headline uses nu=2 with R2 scoring.
  Table 13 sweeps nu in {1,2,4,8}. Our k=16-24 is about 10x wider.
- **Where the map acts:** "pre-norm (before both [k_norm and RoPE])" is the default: translated keys
  go through the target's own k_norm and then RoPE at target positions. The alternatives are
  "post-norm (after key normalisation, before RoPE)" and post-RoPE. Pre-norm and post-RoPE match up to
  about 2k tokens. Pre-norm is "slightly better" on longer prefixes. **Post-norm is worse on long
  context.** Stripping RoPE from post-k_norm keys, as our recipe does, is their post-norm variant.
- **Native tokens:** positions 1..n-1 are translated. "The keys and values of token n come instead from
  a forward pass of the target on the last prefix token." Only one native token. No sink or BOS
  handling is described.

**Stage 1: closed form, which is also the init.** Each map minimizes E||M K_src - K_tgt||_F^2. "It is
solved in closed form on the fitting corpus, without gradient steps." Second moments are accumulated in
fp64 over 400 conversations (about 2.8M prefix positions). The solve is a truncated pseudo-inverse with
eigenvalues below 1e-8 relative dropped. Their diagnostic uses a ridge of 1e-3 times the mean Gram
diagonal, which is the same as our lam_rel=1e-3. An attention-weighted variant (Eq. 3.4) cuts
closed-form KL from 0.175 to 0.064. After stage 2 the two inits end at 0.0195 and 0.0186, so stage 2
erases most of the difference.

**Stage 2: self-distillation.**
- **Loss:** mean over continuation positions j=1..m of KL(p_tgt(.|native cache, y<j) || p_tgt(.|translated
  cache, y<j)). This is forward KL with the teacher first. The teacher is the target itself running on
  its native cache. "the supervision comes from the target; the examples only fix the positions at
  which the two distributions are compared." Continuations come from the dataset (teacher-forced), and
  the target computes them natively on top of the translated prefix cache. **No loss is taken on prefix
  positions.**
- **Gradients:** "Gradients flow through the target into the translator only". Source and target are
  frozen in bf16. "the translator's parameters are held in fp32".
- **Trainable:** the full linear map, initialized from stage 1. There is no low-rank residual.
- **Optimizer:** AdamW, 5,000 steps, effective batch of 8 conversations (gradient accumulation on one
  H100, or 8-way data parallel for wide maps), cosine schedule with 5% warmup, **no weight decay,
  gradient clip 1.0**.
- **LR** is "the only tuned hyperparameter". They try 5 rates per cell, half a decade apart, and widen
  the window if the best rate is on an edge. **The range searched is 3e-7 to 3e-2.** The run is chosen
  by **lowest final validation KL on 64 held-out conversations**. The chosen rates per cell are not
  given in the text we read.
- **Data:** Nemotron-SFT-Instruction-Following-Chat-v2 (multi-turn chat), split at the start of an
  assistant turn. Half are natural samples with prefixes of 10 to 10k tokens. Half are concatenated
  samples with prefixes of 12k to 16k tokens. Continuations are capped at 2,048 tokens (about 750 mean
  reasoning-off, about 1k reasoning-on). That is about 40k conversations seen in total and roughly
  6-8k KL positions per optimizer step. The from-scratch pairs use a 256-token prefix plus a 256-token
  continuation.
- **Instability they report:** "seed collapses". Five times, always on cross-layer pairs with reasoning
  off, one seed lost a third or more of a benchmark (for example XQuAD 0.018 vs 0.190 and 0.235).

**Headline numbers** (Qwen3 unless stated; reasoning-off; per-benchmark N not in the text we read):
- **Stage 1 to stage 2 (Table 8, L->S).** 0.6B/1.7B: RULER 0.261->0.782, XQuAD 0.087->0.256,
  CoQA 0.291->0.476, IFEval 0.416->0.530, MMLU-Pro 0.210->0.251. 4B/8B: MMLU-Pro 0.503->0.576,
  RULER 0.960->0.991.
- **Stage 1 to stage 2 (Table 9, S->L, 0.6B/1.7B).** CoQA 0.144->0.546, RepLiQA 0.200->0.874,
  XQuAD 0.007->0.304.
- **Closed form on long context.** RULER and LongMemEval cells are 0.000-0.013 (Table 11).
  Distillation is what makes long context work.
- **Shape-matched pairs:** distilled "significantly outperform closed-form ones" and reach "the level
  of the small model in both directions" (Fig. 3a, numbers not in the text).
- **Cross-layer 0.6B<->8B, mean of 9 benchmarks.** L->S: KV-Lingo 0.476, head-mixing MLP 0.466,
  small model 0.506, head-wise MLP 0.270. S->L: KV-Lingo 0.577, head-mixing MLP 0.597, head-wise MLP
  0.454.
- **LongMemEval, reasoning-on, S->L 0.6B->8B:** 0.531, against 0.383 for the 0.6B on the same items.
  Table 12 uses 389 questions, n=1556 question-rung pairs.
- **MLA pair** (Qwen3-4B/8B converted with CARE, 6-benchmark mean): 0.588 and 0.587, against native
  0.629 and 0.633.
- **Cross-family Gemma-3-4B/12B and Gemma-4-E2B/E4B:** L->S beats the small model. S->L only reaches
  the small model.
- **Qwen3-8B<->Mistral-7B-v0.3 (cross-tokenizer, 3 source layers):** beats the small model on 2 of 7
  benchmarks L->S and 4 of 7 S->L.
- **TTFT:** 9.6x on a 64-token prompt on an M3 Ultra. On an H100 with vLLM, 4.8-5.6x at 2k tokens and
  12.0x / 28.9x at 32k (into 0.6B / 8B). Translation adds 1.8 ms for 4B->30B-A3B. nu=8 translation
  takes about 25 ms against 49 ms for a 1.7B prefill.
- **Multi-turn QuAC:** stays within 0.05 of full re-prefill across turns.
- **Baseline:** the "head-wise linear closed-form translator" is Heo et al. 2608.03893, which is our
  stage 1.

## 2. HeteroFold (arXiv 2609.32259, "Prefill-Free Cross-Family KV Cache Transfer for Heterogeneous Multi-Agent LLMs")

**Map form.** At inference there is "one fixed affine map per receiver layer and role, with no separate
correction module": Phi(x) = (x/r) A_final + b_final, with A_final = A(I + BG) and b_final = b(I + BG).
The low-rank correction is folded into A after training.

**Init, called "Recolor" (whiten-then-color Procrustes, not ridge).**
1. Divide the sender features by one fixed scalar RMS r per layer and role.
2. Take the thin SVD Sigma_Z^-1/2 Sigma_ZY Sigma_Y^-1/2 = U Lambda Q^T and set R* = U Q^T.
3. Set A = Sigma_Z^-1/2 R* Sigma_Y^1/2 and b = mu_Y - mu_Z A.

The mapped features then have the receiver's mean and covariance exactly, so there is no ridge
shrinkage. The motivation is that token-wise reconstruction "can keep low error while altering
attention", and Recolor "better matches native K/V distributions than Ridge". "Numerical stabilization
makes covariance matching approximate in practice."

**Correction (training).** Rank rho=16 per role: y = y0 + (y0 B) G, with B ~ N(0, 0.02) and **G = 0**.
The map therefore starts exactly at Recolor. Only B and G train.

**Loss: local and per-layer, no logits.**
- Key correction: KL(P* || P_hat) between attention distributions computed with **native receiver
  queries**, plus ||Delta O^K||^2 / d_model (the change in attention output), averaged over layers.
- Value correction: normalized squared Frobenius error of the post-W_O attention output.
- The two are "optimized jointly, with each objective updating only its corresponding correction".
- Teacher signals are "native receiver queries, attention weights, values, and outputs". Gold answers
  and generated continuations are excluded.
- Positions: query tokens "overlapping the prompt question span", attending over the full causally
  accessible prompt.

**Optimizer and data.** AdamW, **lr 3e-4, weight decay 1e-4, gradient clip 1, 4 epochs**. The kept
checkpoint has the lowest combined held-out loss. Calibration uses 1,600 prompts (800 Open-R1 and
800 HotpotQA) plus 400 held out. Batch size, steps and sequence length are not stated. Construction
takes about 2 H100 GPU-hours. The mapper stores 201-252M parameters, against 537-671M for KV Ridge.

**Cross-family alignment.**
- **Tokens (TA):** "pairs nonempty sender and receiver tokens that end at the same character position
  in the original text." An unmatched receiver position uses "the sender state at the latest preceding
  shared boundary". Positions before the first match use the first sender token. Many-to-one cases are
  not averaged. Special and chat-template tokens are handled separately.
- **Layers:** proportional depth pi(l) = 1 + round((l-1)(L_S-1)/(L_R-1)). Each receiver layer reads
  3 sender layers at offsets (-4, 0, +4), clipped at the model boundaries. They are concatenated and
  flattened ("flattening allows cross-head mapping even when head counts differ").
- **RoPE:** keys are mapped before k_norm and RoPE, and values after projection. The receiver applies
  its own k_norm and RoPE at its own positions (BF16 storage, FP32 RoPE math). No native recompute of
  context tokens is described. The receiver appears to run its native suffix (question, options, chat
  formatting) over the mapped article cache.
- **What "calibrates it to preserve receiver behavior" means:** concretely, the attention-routing KL
  plus the attention-output error above, measured against the native receiver at question-span
  queries. The paper gives no further definition.

**Headline numbers** (Table 1; N: Qasper 200, HotpotQA 200, LoCoMo 1540, QuALITY 2086, ARC-C 1172,
MMLU 14042, WinoGrande 1267, HellaSwag 10042, GSM8K 1319). Models are Llama-3.1-8B-Instruct, Qwen3-4B
and Ministral-3-14B, in 6 directions. Each cell lists HeteroFold / KV Ridge+TA / TextMas (native
receiver prefill of the text):
- **Ministral->Llama:** Qasper 33.72/19.09/44.64, HotpotQA 50.29/37.48/57.54, QuALITY 79.00/74.74/75.07,
  GSM8K 63.46/53.30/85.67.
- **Qwen3-4B->Llama:** Qasper 29.12/16.06/44.64, GSM8K 55.12/43.21/85.67, MMLU 63.28/61.42/66.74.
- **Llama->Qwen3-4B:** HotpotQA 49.44/25.12/56.05, GSM8K 70.74/70.66/90.14.
- **Qwen3-4B->Ministral:** GSM8K 77.26/60.20/94.69. **Llama->Ministral:** GSM8K 72.33/47.99/94.69.
- **Ministral->Qwen3-4B:** close to native: ARC-C 88.99/88.14/87.80, MMLU 72.14/71.73/67.95,
  GSM8K 89.23/87.11/90.14.
- **Losses:** HellaSwag Llama->Qwen 41.74 vs 44.54 (KVR+TA). WinoGrande Ministral->Qwen 53.83 vs 54.78.
- **Without TA (same-index pairing),** KV Ridge and Dense Latent collapse to about chance (GSM8K 0.5-7,
  Qasper about 2-3).
- **HiddenBench (N=65 tasks):** HeteroFold 30.6%, TextMas 29.2%, KVR+TA 21.8%, Dense Latent+TA 18.5%.
- **Ablation, Ministral->Llama (ARC-C/GSM8K/Qasper/HotpotQA/LoCoMo/QuALITY).**
  TA+Ridge: 87.03/4.55/3.27/6.82/2.13/47.41. TA+Recolor: 90.19/62.77/26.52/40.71/40.81/79.34.
  Adding the correction: HotpotQA 40.71->50.29 and GSM8K 62.77->63.46. **The init causes most of the
  jump.**
- **Rank, Llama->Qwen (Table C):** rank 0 gives HotpotQA 30.47 and QuALITY 65.10. Rank 16 gives 49.44
  and 62.90.
- **Latency, 32K, Llama->Ministral, 2xH100, batch 2:** native prefill 5167 ms, KVR+TA 570 ms,
  HeteroFold 481 ms (10.74x).
- **Their KV Ridge baseline (Heo):** k=8 sender layers by R2, ridge 0.01, per-head K/V maps.

## 3. Why their distillation is stable and ours diverged

Our three failures: lr 1e-3 diverged; 100 docs overfit; 2000 docs with bs4, lr 1e-4 and standardized
inputs drifted upward. Each paper differs from our setup in these ways:

1. **LR is swept, not guessed.** KV-Lingo tries 5 rates half a decade apart across 3e-7..3e-2,
   chosen per cell by validation KL, with warmup and cosine decay. Our 1e-3 and 1e-4 sit high in that
   range. With a ridge init the loss starts low, so the useful lr is likely 1e-6..3e-5 for a dense map
   in raw units (inference from the paper, not stated). HeteroFold's 3e-4 is for B and G of a
   zero-init low-rank factor, where the effective step on A is far smaller.
2. **Start exactly at the closed form.** KV-Lingo trains the full map from stage 1. HeteroFold uses
   G=0, so step 0 equals the init. Check that our rank-16 residual has one factor at zero and that
   "standardized inputs" did not change the function at step 0. The step-0 validation KL must equal
   the ridge KL.
3. **Many KL positions per step.** KV-Lingo uses 8 conversations x about 750-2048 continuation tokens
   per step, over prefixes up to 16k. Our bs4 x 1024-token docs gives far fewer, noisier positions.
   KL is taken only on continuation positions, which the target computes natively on top of the
   translated prefix. Prefix positions never enter the loss.
4. **Clip 1.0, no weight decay, fp32 translator, bf16 frozen models, warmup 5%, cosine.** HeteroFold
   also clips at 1.0.
5. **Pick the checkpoint by held-out KL** (64 conversations in KV-Lingo, 400 prompts in HeteroFold),
   not the last step. "Drifted upward" is the signal to stop early or lower the lr.
6. **HeteroFold avoids end-to-end backprop entirely.** Its loss is per layer and local: attention KL
   under native receiver queries, plus attention-output error. Layers are independent, the gradient
   path is short and nothing compounds across layers. This is the most stable option and maps
   directly onto our per-layer ridge.
7. **Key scale goes through the target's own k_norm.** In both papers, keys are mapped pre-norm and
   pre-RoPE, and the target applies its own k_norm. That normalization is in the forward graph, so the
   gradient cannot change key scale arbitrarily. Our post-norm fit plus RMS re-imposition is a
   non-learned patch. If the restore step sits outside the graph during training, the trained map and
   the deployed map differ.

**Settings to copy** (KV-Lingo stage 2): AdamW(wd=0, betas default), lr grid {1e-6, 3e-6, 1e-5,
3e-5, 1e-4}, warmup 5%, cosine, clip 1.0, fp32 map, init = our ridge solution (full map trainable, or
residual with G=0), effective batch 8 docs, loss = forward KL(native || translated) over teacher-forced
continuation tokens only, about 1-2k-token continuations after 1-8k-token prefixes, and selection by
validation KL on 64 held-out docs. Run 300-500 steps per lr before the full 5k.

## 4. Ranked levers for our recipe

Costs are rough GPU estimates for the cheapest box that fits and need confirming with vast-guard.

1. **Local per-layer attention-KL residual (HeteroFold loss) on top of ridge.** (kvx-quality)
   - Change: for each target layer, capture native Q, K, V and attention outputs on calibration
     prompts. Train a rank-16 B·G residual (G=0) to minimize KL(attn_native || attn_mapped) under
     native queries plus ||Delta O||^2/d over the post-W_O output. Use lr 3e-4, wd 1e-4, clip 1,
     4 epochs, and keep the best held-out checkpoint. Fold BG into the map afterwards, so inference
     cost is zero.
   - Evidence: the correction adds HotpotQA +9.6 and GSM8K +0.7 over Recolor (Ministral->Llama) and
     HotpotQA +19 at rank 16 (Llama->Qwen).
   - Cost: no full-model backprop, about 2 H100-h in their setup, and fits a 24 GB card one layer at
     a time.
2. **Retry logit distillation with KV-Lingo's exact settings** (Section 3). (kvx-quality)
   - Evidence: stage 1 to stage 2 gives RULER 0.261->0.782 and RepLiQA 0.200->0.874. Closed-form long
     context is about 0. This is the largest gain in either paper.
   - Cost: an lr sweep of 5 x 500 steps on one GPU before any 5k-step run.
   - Prerequisite: the step-0 KL check in Section 3, item 2.
3. **Recolor (whitened Procrustes, mean and covariance matched) instead of ridge plus gain 1.1 for
   no-k_norm targets.** (kvx-pairs first, then kvx-quality)
   - Evidence: TA+Ridge to TA+Recolor gives GSM8K 4.55->62.77, Qasper 3.27->26.52, HotpotQA
     6.82->40.71 (Ministral->Llama).
   - Cost: closed form from the stats we already accumulate (Sigma_XX, Sigma_XY plus Sigma_YY),
     CPU minutes plus one eval.
   - Ridge shrinks low-R2 directions, which is why we needed the gain hack.
4. **Fit pre-k_norm and let the target's k_norm and RoPE run** (Qwen3). (kvx-quality)
   - Change: regress the target's pre-norm k_proj output, then apply the target's q/k_norm weights and
     RoPE. This replaces "re-impose RMS".
   - Evidence: KV-Lingo calls our post-norm setting worse on long context. Both papers use pre-norm.
   - Cost: one refit plus eval.
5. **Attention-weighted closed-form fit.** (kvx-quality)
   - Evidence: KL 0.175->0.064 with no gradient steps (KV-Lingo App. A).
   - Cost: near zero. `attention_received` and `Stats.add(w=)` already exist in kvx/core.py. Confirm
     the production recipe uses them, and if not, run an A/B.
6. **Longer and chat-shaped calibration.** (kvx-quality)
   - Change: add multi-turn chat (Nemotron-SFT-IF-Chat-v2 or similar) and prefixes of 4-16k, instead
     of only 1024-token FineWeb-Edu and PG19.
   - Evidence: Khare et al. show Heo-style S->L falls up to 14.76 pts below the small model on RULER
     at 4-8k. KV-Lingo fits on prefixes up to 16k.
   - Cost: a stats pass of about 1-2 GPU-h.
7. **Cross-family: character-end-boundary token alignment plus proportional depth with offsets
   (-4,0,+4).** (kvx-pairs)
   - Evidence: KVR to KVR+TA moves GSM8K 0.53->43.21 (Qwen->Llama). Same-index pairing is at chance.
   - Cost: CPU code plus one pair eval.
8. **Narrower source sets and fewer native tail tokens, for TTFT.** (kvx-speed)
   - Change: A/B k=16-24 against 3-8 (KV-Lingo nu=2 with R2, HeteroFold 3, Heo 8). KV-Lingo recomputes
     only 1 native token, so after distillation, test a 64-256 tail against 16 or 1.
   - Evidence: HeteroFold has 201-252M parameters against 537-671M for ridge and is 1.18x faster at
     32k.
   - Cost: eval only.
9. **Head-wise vs head-mixing check.** (kvx-pairs) Use head-mixing only for cross-layer or cross-family
   pairs. KV-Lingo found no gain on shape-matched Qwen3 pairs.

Do not try a value MLP as a general fix. KV-Lingo's head-mixing MLP is about level with linear
(0.597 vs 0.577 S->L), which matches our finding that it helps only failing pairs.

## 5. Related work (2026)

| Title | arXiv | Claim |
|---|---|---|
| Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse (Heo et al.) | 2608.03893 | Per-head ridge from top-k source layers. KV is largely linear within a family and stable over multi-turn handoff. This is our base recipe. |
| Can Cross-Model KV Cache Mappings Survive Longer Prompts? (Khare, Qian, Wan; NeurIPS'26 LCFM workshop) | NeurIPS 173793 (no arXiv id seen) | Heo map, Qwen3 pairs. S->L beats the source by up to 8.40 pts on ARC-C/HellaSwag but falls up to 14.76 pts below the small model on RULER SQuAD/HotPotQA at 4k/8k. L->S holds up. |
| Translating KV Caches Between Hybrid-Attention LMs: MLA Latents, Linear-Attention States, Sparse-Attention Indexers (Kvedaras, Rasymas) | 2611.12062 | Replicates KV-Lingo (Qwen3-8B<->4B, 2k of 5k steps). Extends it to MLA, GDN and sparse indexers. Decoding token ids from the layer-1 MLA latent lifts needle recall 0.09->0.99 at 2-32k. 18.7x/46.8x TTFT at 64k. About 6% refusals inherited from the source. |
| KV Cache Translation across Heterogeneous LLMs (Mixture-of-Translators) | 2607.28979 | Token-routed multiple translators plus a Context Correction Loss on Llama/Gemma/Qwen. 95.0% of native accuracy, 1.5x the best baseline. Analyses early- vs late-injection error. |
| A Universal Context-Reuse Layer for Cross-Model KV Sharing | 2608.30963 | Qwen2.5-7B->1.5B: LongBench2 27.59->34.48%. Llama3.1-70B->Qwen2.5-7B: 44.0% vs 45.7% native, 899->138 ms. |
| Swarm Telepathy (SwaT): Prefill-Free KV-Cache Transfer Across Families | 2610.15095 | Cross-family with frozen models. About +40% accuracy over text-to-text and about 4x faster, on 17 benchmarks, 12 LLMs and 6 families. Under 50M training tokens and under 1 GPU-day. |
| BridgeKV: Large-Model Prefill, Small-Model Decoding via KV Translation | 2611.11761 | Same-family L->S. Translator trained to match the large model's next tokens, plus LoRA. Closes up to 80% of the gap, 3.0x faster (Llama-8B->1B, A40). |
| KVPort: Zero-Prefill Cross-Family KV Transfer Across Tokenizers (NeurIPS'26 SLM-Agents workshop) | no arXiv id seen | UTF-8 byte-span token alignment, RoPE re-framing and per-head ridge, Qwen3-8B->Llama-3.1-8B. No numbers on the page. |
| On the Opportunities and Challenges of Prefill-to-Autoregressive Cross-Model Transfer (PLatour) | 2611.06976 | Linear bridges with a KV-Lingo-style KL. Some gains are reproducible with uninformative senders, so they propose PACT-Check. Lesson for us: always include a receiver-only or null-sender control row. |
