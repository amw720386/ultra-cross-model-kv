# Agents

τ2-bench (text, English) and Terminal-Bench 2.0. This is where the transfer is weakest: matching native needs a
setting that computes most of the prompt natively and is slower than native. Terms in
[method.md](method.md#glossary).

## Settings

Both compute the `<tools>` block natively (a native span).

| | **lighter agent setting** | **agent setting** |
|---|---|---|
| natively computed | tools + last 3 messages (≤ 4096 tokens), no repair | tools + last 10 messages (≤ 8192 tokens) + up to 4096 tokens of whole-message recall (`+rep4096m`) |
| share of prompt native (τ2) | telecom 23%, retail 58% | telecom 27–74%, retail 66–98% |
| map | agent map `4b-27b-fid-agent` | fidelity map `4b-27b-fid` |
| TTFT telecom (7.8K), graphed tail off | 1.31 s vs native 1.41–1.46 s; 1.09 s vs 1.28–1.31 s (2nd run) | 1.67 s vs 1.28–1.31 s, slower |
| TTFT retail (5.3K), graphed tail off | 1.13 s vs 1.00–1.01 s, slower | 1.44 s vs 1.00–1.01 s, slower |
| TTFT telecom, graphed tail on | 1.80 s vs 1.41–1.46 s, slower | 2.36 s vs 1.41 s, slower |
| TTFT retail, graphed tail on | 1.82 s vs 1.18 s, slower | 1.99 s vs 1.17 s, slower |

The lighter setting beats native only on telecom with the graphed tail off (~1.1–1.2×). The agent setting is slower
than native everywhere. Native timings vary ~10% between containers, so each comparison uses the same run's native
arm (`tau2_replay.json`).

## τ2-bench, official simulator

OpenAI `gpt-5.5` as simulated customer and judge, as in the τ2 release (OneTriangle also used the official setup).
50 tasks per domain for us (±0.13); 17 for OneTriangle's mapped arms (±0.23).

| domain | native 27B | lighter setting | agent setting (slower than native) | policy-deleted control | earlier: fid map, `+rep2048m`, 3 msgs native | OneTriangle mapped (native) |
|---|---:|---:|---:|---:|---:|---:|
| airline | 0.70 | 0.50 (in-sample) | 0.68 | 0.42 | 0.48 | 0.59 (0.86) |
| retail | 0.84 | **0.78** | **0.84** | 0.72 | 0.78 | 0.41 (0.80) |
| telecom | 0.68 | 0.22 | 0.68 | 0.44 | 0.10 → 0.20 (repair fix) | 0.47 (0.86) |

`tau2_fid.json`. **Policy-deleted control** (`target-drop`): native 27B sees only what the agent setting computes
natively; the policy and older history are deleted instead of transferred.

How each setting was chosen: the lighter setting is one pre-registered official run of the agent map, which was
trained with airline traces (airline in-sample; retail and telecom unseen). The agent setting was developed by
replaying failures from the same 50 telecom tasks, so its telecom 0.68 is a development-set result.

- **Transfer carries real information.** In the agent setting, transferring the policy instead of deleting it is
  worth +0.26 airline, +0.12 retail, +0.24 telecom. That setting is slower than native, so this shows quality
  preserved, not a speedup.
- **Lighter setting: retail holds, telecom fails.** Retail keeps 93% of native (vs OneTriangle 51%). On telecom the
  transferred policy does *worse* than no policy (0.22 vs 0.44): it pushes the agent to call customer-only tools, and
  τ2 ends conversations after repeated tool errors.
- **Why telecom is hard.** The ~23K-character policy is a long troubleshooting procedure plus rules on which tools
  belong to whom. On 160 replayed requests (next tool matches native): 4B alone 0.81, transfer without repair 0.66,
  transfer with the target's exact attention KV swapped in 0.84 (exact DeltaNet states instead: 0.69). The 4B reads
  the policy fine; the attention mapping loses the exact rules.

### What was tried

All in `tau2_fid.json` and `tau2_replay.json`.

1. **Repair overwrote exact cache.** Repair could re-encode blocks inside the native `<tools>` span. It now never
   selects native spans (`XKV_REPAIR_PROTECT`, default on). Telecom 0.10 → 0.20; RULER (2B, `+rep1024`) unchanged.
2. **More native context.** Last 10 messages native + 4096 tokens of whole-message recall cut invalid customer-tool
   calls on 42 replayed error requests from 98% to 7% (native 10%), and official telecom to 0.68. About 1.7× slower
   than native with the graphed tail on.
3. **Agent-trained map.** Fidelity map fine-tuned with native 27B airline traces in the mix (`agent_map.json`): KL on
   held-out airline requests 0.235 → 0.116, held-out gate unchanged. Replay: telecom 0.66 → 0.76, retail 0.87 → 0.94.
   Official run (no repair): 0.50 / 0.78 / 0.22, vs 0.48 / 0.78 / 0.20 for the earlier fidelity-map runs.

The replay (teacher-forced next action) is a weak proxy: it checks whether the next tool matches native, while τ2
fails whole conversations on policy violations. Deleting the policy scored close to the agent setting on replay but
0.24–0.26 lower on official airline and telecom.

## τ2-bench, native 27B as simulator and judge

No API needed; earlier settings, before the repair fix.

| domain | native 27B | no repair | `+rep2048` (blocks) | `+rep2048m` | mostly native [1] |
|---|---:|---:|---:|---:|---:|
| airline | 0.74 | 0.26 | 0.60 [2] | 0.60 | 0.35 |
| retail | 0.78 | 0.60 | 0.70 | 0.735 | 0.74 |
| telecom | 0.60 | not run | 0.18 | 0.32 (`+rep4096m`: 0.20) | not run |

Arms read the newest turn (≤ 4096 tokens) and `<tools>` natively; the `+rep2048m` arms the last three messages.
Later native runs gave telecom 0.50, retail 0.74, airline 0.70, so native varies ±0.1 between runs.

- **[1]** Only the system prompt and first user turn are mapped; later turns append natively. Not a transfer result.
- **[2]** Two runs with 1024 block-repair tokens gave 0.40 and 0.32.

## Terminal-Bench 2.0

Harbor, Terminus-2 agent, official task containers. Same 10 tasks for native and the agent setting
(`terminal_bench_tb2.json`):

| time limit | native 27B | agent setting |
|---|---:|---:|
| 2× Harbor default | **0.70** (35 min) | 0.10 (all 10 trials hit the limit, 2 h) |
| default | 0.20 | 0.00 (9 timeouts, 1 runtime error) |

Prompts grow to ~75K tokens and every turn re-reads them; each agent-setting turn pays the 4B prefill, mapping and
repair, so it completes far fewer turns. OneTriangle reports 0.10 (their 4B alone 0.09) with no native value, so
share of native cannot be compared.

## Notes on OneTriangle's agent numbers

Their page does not describe the agent setup, so this comparison rests on assumptions:

- **How much of each prompt they compute natively is not stated.** Ours ranges 23–98% by setting and domain, which
  moves τ2 scores more than the gap between us.
- **Reference point:** our policy-deleted control scores 0.42 / 0.72 / 0.44 (airline / retail / telecom); their
  mapped scores are 0.59 / 0.41 / 0.47. This does not say how much policy their transfer keeps, only that scores in
  this range are reachable without the policy.
- **17 tasks per domain** gives ±0.23 at 95%.
- **Their native 27B is stronger on τ2** (0.86 vs 0.68–0.70), suggesting a different harness or thinking mode.
  On telecom the raw scores (0.22 vs 0.47) overstate the gap: measured as the drop from each side's own native,
  ours is −0.46 (0.68 → 0.22) and theirs −0.39 (0.86 → 0.47), well inside their ±0.23.
- **Speed for agents** is not reported on their page.
