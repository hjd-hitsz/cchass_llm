# Finding the Move Is Not Winning the Game: XiangqiBench for Closed-Loop Evaluation of LLM Agents

Yekun Chai $^{1,2}$ , Qiwei Peng $^{1,3}$ , Haoyi Xiong $^{4}$

$^{1}$ FloatAI, $^{2}$ ETH Zurich, $^{3}$ University of Copenhagen, $^{4}$ Independent Researcher

Static evaluations credit a language model for naming the right move, but an agent must carry a plan through to a verified outcome while an opponent responds. We introduce XiangqiBench, an executable benchmark that measures this difference in Chinese chess: starting from 119 tactical endgames with forced mates supported by engine or checks-only search, an LLM agent must deliver checkmate against an engine defender. An interactive REPL interface separates real moves, state queries, and forward simulation, and we record 8,568 multi-turn trajectories from 12 frontier LLMs under two observation protocols. Three signals that look like competence each overstate closed-loop success. (i) The Conversion Gap: models play the stored reference first move in $26.1\%$ of Sighted trials, yet only $13.9\%$ of these trials end in a win. (ii) The Consistency Gap: the leading model reaches $38.7\%$ pass@3 but only $5.9\%$ pass^3, winning all three trials on 7 of the 46 positions it ever wins. (iii) The Simulation Gap: $32.3\%$ of accepted simulation calls stop on an illegal move, and in $49.3\%$ of comparable cases the real defender replies differently from the line the agent simulated; self-authored rollouts check legality but cannot anticipate the opponent. Finding the move is not winning the game: agent evaluations should score closed-loop outcomes and report reliability alongside coverage.

Date: October 5, 2026
Code: https://github.com/floatai/xiangqibench

![](images/1c83e8287714c7ff637a563e09a064b44bfe8aebfc37e2da5736a8baf47eb428.jpg)

## 1 Introduction

Static evaluations such as puzzle solving and move prediction credit a language model for naming a good action (Chen et al., 2025). Agents are held to a different standard: they must carry a plan through to a verified outcome while an active adversary responds (Kolasani et al., 2025; Liu et al., 2026). Success then hinges on conversion: navigating an unfolding contingency tree to a verified goal. Whether static competence predicts conversion is rarely measured directly, because existing benchmarks often confound heuristic proposal, mental planning, and closed-loop execution.

Board-game endgames provide a controlled testbed for this decision loop: state transitions are exact, legality is executable, and outcomes are determined by verifiable terminal states (Toshniwal et al., 2022; Ruoss et al., 2024; Schultz et al., 2025). We study endgame conversion in Chinese chess (Xiangqi): starting from a tactical position in which engine or checks-only search reports a forced mate for Red, an LLM agent plays Red against an engine defender. Agreement with reference moves receives no score; the agent must sustain a winning continuation across multiple turns until checkmate (Grace et al., 2026). XiangqiBench exposes this process through an interactive Read-Evaluate-Print Loop (REPL) interface (Fig. 1) that separates real moves, state queries of the current board and legal moves, and a forward-simulation sandbox on a cloned board. Simulation is not an external search engine: the agent must supply both its own candidate moves and the hypothesized opponent replies. The tool therefore shows what the agent expects: does it anticipate the defender's replies, or does it check lines the real defender does not play?

We evaluate 12 frontier LLMs (Gemini 3.1 Pro (Google DeepMind, 2026), GPT-5.5 (OpenAI, 2026), Qwen 3.7 Max (Qwen Team, 2026), Claude Opus 4.7 (Anthropic, 2026), DeepSeek V4 Pro/R1/V3.2 (Guo et al., 2025; Liu et al., 2025; Xu et al., 2026), Seed 2.0 Pro (Seed, 2026), Kimi K2.5 (Team et al., 2026), Mistral Large 3 (Mistral AI, 2025), MiniMax M2.5 (Chen et al., 2026), and GPT-OSS 120B (Agarwal et al., 2025); Appendix B) under two observation settings: Sighted, which provides board refreshes, legal-move feedback, and the simulation sandbox; and Restricted observation, which withholds board refreshes, state queries, and simulation to probe state tracking from move history. Our contributions are:

- A benchmark for adversarial conversion: 119 search-supported tactical endgames, an executable rules environment with an engine defender, and 8,568 recorded trajectories.

- An instrumented decision loop: we trace each trial from the first move through simulation legality and real-move adoption to defender divergence and the terminal outcome.

- Three gaps between apparent and closed-loop competence: a Conversion Gap (the reference first move rarely leads to a win), a Consistency Gap (pass@3 far exceeds pass^3), and a Simulation Gap (self-authored rollouts often break legality and mispredict the defender); a controlled ablation on the two strongest models suggests that much of the Restricted-observation drop reflects the setting's prompt and feedback rather than the missing board alone.

![](images/0591d1e129a1c2e22836cdf8a37e7156ef5aa1f4a3c5c42e81c177d109a6bf27.jpg)  
Figure 1 XiangqiBench at a glance. Left: a task instance with the real trajectory overlaid on the initial board (numbered plies; deep red for the agent, charcoal for the defender; curves separate annotations and are not piece paths), tool availability by setting, and the scale of the evaluation. Right: the second Red decision of Gemini 3.1 Pro in Sighted trial 3 on case SQYQ-251, after real plies 1–2 (Appendix G). The complete pre-simulation think response is shown; simulate then checks three model-specified plies, including an assumed Black reply, on a copy of the board. The defender's actual reply informs the next decision, and the terminal Red win determines task success.

## 2 Related Work

Interactive agent evaluation. AgentBench, SWE-bench, WebArena, and OSWorld evaluate sequential tool invocation and computer control (Liu et al., 2024; Jimenez et al., 2024; Zhou et al., 2024; Xie et al., 2024). In game domains, GTBench, GameBench, GAMEBoT, KORGym, and TextArena provide rulegrounded strategic environments (Duan et al., 2024; Costarelli et al., 2024; Lin et al., 2025; Shi et al., 2025; Guertler et al., 2025). XiangqiBench complements them by isolating adversarial conversion of search-supported winning positions against an engine defender, scoring only the terminal outcome while recording intermediate planning and real-move commitment (Grace et al., 2026).

Game planning, state tracking, and simulation. Board games serve as testbeds for neural planning and representation (Toshniwal et al., 2022; Feng et al., 2023; Ruoss et al., 2024; Schultz et al., 2025). In chess, ChessArena and LLM Chess evaluate full games against other LLMs, random players, and engines, and ChessArena adds a blindfold mode that supplies only the move history (Liu et al., 2026; Kolasani et al., 2025). LMRL-Gym uses theoretical endgames as a multi-turn RL testbed (Abdulhai et al., 2025), while Xiangqi-R1 trains a 7B model in multiple stages to improve move legality and position analysis (Chen et al., 2025). We instead study generalist frontier LLMs on multi-turn endgame conversion and separate the decision loop into scratchpad simulation, real-move adoption, and resilience against engine counterplay.

Reliability metrics. Following Chen et al. (2021) and Yao et al. (2025), we use unbiased finite-sample estimators for pass@k (at least one success) and pass^k (all k attempts succeed); contrasting them reveals whether an agent that can reach a win also reaches it reliably.

## 3 XiangqiBench: Task Construction

Corpus. XiangqiBench curates 119 tactical compositions from classical Chinese-chess treatises in which Red can force checkmate: 116 from Shi Qing Ya Qu $^{1}$ (case identifiers SQYQ-n) and 3 from the classical Jianghu collection (JH-n). Each entry specifies an initial position in Forsyth–Edwards Notation (FEN) with Red to move; historical solution text is withheld from prompts and scoring. During evaluation, an executable rules adapter verifies every proposed action. Xiangqi legality involves non-local constraints: pieces must respect palace boundaries and river crossings, cannons require an intervening screen to capture, horses are blocked by adjacent pieces (“hobbling the horse’s leg”), and generals cannot face each other across an open file.

Search-supported admission. Historical treatises can contain transcription errors or overlooked defenses, so source solutions are not trusted. Instead, each position is screened with Pikafish (Official Pikafish Contributors, 2026), a strong open-source Xiangqi engine derived from Stockfish that combines alpha-beta search with an NNUE evaluation network. For admission, a single-threaded, time-limited search is run from each position, and the position is admitted when the engine reports a forced mate for Red. The reported mate distance and the engine's preferred first move are stored as case metadata; the latter serves as the reference root move in Section 5. Composed positions that the engine does not confirm may instead be admitted by a checks-only mate search and are labelled separately (Appendix A). For defense, the same engine family plays Black during evaluation (Section 4). Because admission relies on bounded search rather than a complete proof tree, we describe the positions as search-supported rather than formally proved; all 119 positions were additionally checked manually to confirm that Red can force checkmate.

Why endgames. Like the chess-endgame task in LMRL-Gym (Abdulhai et al., 2025), we start from endgames rather than full games: the goal (checkmate) is fixed and reachable, so a failure to mate is attributable to the agent; opening recall and attrition are removed; and a few precisely calculated moves decide the game. Unlike LMRL-Gym's few-piece theoretical endgames against a 1200-Elo engine, ours are composed tactical endgames (median 14 pieces) built on consecutive checks, mating threats, and pins, and a strong engine defends them, so wins cannot rely on blunders.

Position categories and horizon. Red can force mate in every position; the two categories differ only in Black's counterplay. At admission, the same engine or solver is also run with Black to move. If Black then has a forced mate of its own, the position is a threat race: Red must keep forcing, because a slow move can let Black execute its mate first. Otherwise it is an unopposed mate: no counter-mate was found, so a slow move forfeits tempo but does not by itself hand Black a win. Stored mate distances range from 3 to 11 plies (median 9; Figure 2). The 40-ply cap is more than three times the longest stored distance, leaving room for detours and recovery, and success already falls steeply within this range (Section 5); longer compositions would push nearly every model to zero, whereas this range still separates the strongest models.

![](images/fe8c127e67e963367f383f01f865bbabdf643c60ed28eacf0d30b6e2f3d0c285.jpg)  
Figure 2 Stored mate distances of the 119 positions, by category. Distances are those reported at admission by Pikafish or the checks-only solver, stacked by category: 106 threat races (Black would also have a forced mate if it were to move) and 13 unopposed mates (no such counter-mate found). Distances are admission metadata, not limits on trajectory length during evaluation.

Table 1 REPL action space and observability contract. Only $A_{env}$ mutates the real board.

<table><tr><td>Class</td><td>Command &amp; Operational Semantics</td><td>Sighted</td><td>Restricted</td></tr><tr><td rowspan="2"> $\mathcal{A}_{env}$  (Transitions)</td><td>move(from_to): Validate and commit a real move; trigger defender</td><td>√</td><td>√</td></tr><tr><td>resign(): Surrender; terminate trial with defender win</td><td>√</td><td>√</td></tr><tr><td rowspan="3"> $\mathcal{A}_{query}$  (Perception)</td><td>view_board(): Render ASCII board, FEN, active color, and piece census</td><td>√</td><td>✕</td></tr><tr><td>get_legal_moves(): Enumerate legal moves in the current state</td><td>√</td><td>✕</td></tr><tr><td>get_history(): Return the move sequence with piece annotations</td><td>√</td><td>√</td></tr><tr><td rowspan="2"> $\mathcal{A}_{sim}$  (Hypothesis)</td><td>simulate(moves): Execute alternating line on cloned state; verify legality</td><td>√</td><td>✕</td></tr><tr><td>think(text): Record uncommitted deliberation</td><td>√</td><td>√</td></tr></table>

## 4 Interaction and Evaluation Protocol

REPL action space. The agent plays Red against an engine defender for at most 40 plies. At each turn it receives an observation $o_t$ (with interaction history $h_t$ ) and emits free text alongside a command from three classes (Table 1): transitions $\mathcal{A}_{\text{env}}$ , which commit a move in ICCS coordinates (e.g., move h2h9) and trigger the defender's reply, or resign; read-only queries $\mathcal{A}_{\text{query}}$ , which inspect the board, legal moves, or history without advancing the game; and hypotheses $\mathcal{A}_{\text{sim}}$ , where simulate plays an alternating line of up to 8 model-supplied plies on a cloned board and reports its legality, and think records deliberation.

Observation settings. Under Sighted observation, the agent receives the updated board, FEN, and legal moves after every environment step and may use all queries and the simulator; this tests the agent as a system with external perception. Under Restricted observation, the board and FEN are shown once at the start; afterwards the agent receives only move coordinates and captures, view\_board, get\_legal\_moves, and simulate are disabled, and it must track the position from the move history. The two settings also use different system prompts (Appendix A.3).

Engine defense and termination. After each accepted agent move, Black replies with a fixed-depth (depth-18), single-threaded Pikafish (Official Pikafish Contributors, 2026) search; environment fallbacks are described in Appendix A. A trial ends in an agent win if Red achieves checkmate or leaves Black without legal moves (stalemate is a loss for the stalemated player). It ends in failure upon Black checkmate, agent resignation, exhausting the 40-ply budget, exceeding 3 consecutive illegal moves, or threefold repetition.

![](images/b8637b27cb1d8d077c45ca883b201dda3e9dd79a512a3a691874241d4f32cbb7.jpg)  
Figure 3 Multi-turn endgame conversion across frontier LLMs. Bars report mean pass@1 (single-trial success), pass@3 (at least one success in 3 attempts), and pass^3 (all 3 attempts successful) across 119 positions (357 trials per model/setting). Whiskers denote 95% case-clustered bootstrap intervals.

Under official rules, perpetual checking loses; our replay adjudicates perpetual checks with a simplified long-check rule (Appendix A.5).

Metrics: exploration vs. consistency. Let $c_{i}$ denote the successes on case i out of $n_{i}=3$ trials, over $|I|=119$ positions. We report (Chen et al., 2021; Yao et al., 2025)

$$
\widehat{\text{pass@k}} = \frac{1}{|\mathcal{I}|} \sum_{i=1}^{|\mathcal{I}|} \left[1 - \frac{\binom{n_i-c_i}{k}}{\binom{n_i}{k}}\right], \qquad \widehat{\text{pass}^k} = \frac{1}{|\mathcal{I}|} \sum_{i=1}^{|\mathcal{I}|} \frac{\binom{c_i}{k}}{\binom{n_i}{k}} \quad (1)
$$

pass@k estimates the probability of at least one success in k attempts (exploratory reach), while pass^k estimates the probability that all k attempts succeed (consistency).

## 5 Evaluation Results

## 5.1 The Consistency Gap: Occasional Success vs. Reliable Conversion

Single-trial conversion (pass@1) is low for every model and observation setting (Figure 3). Under Sighted observation, Gemini 3.1 Pro leads at 22.1% pass@1 (79/357 trials), followed by GPT-5.5 at 9.0% (32/357) and Qwen 3.7 Max at 6.2% (22/357). Under Restricted observation every model declines: Gemini to 14.0%, GPT-5.5 to 3.4%, and Qwen to 2.8%, with nine models at most 0.8%.

Repeated attempts separate reaching a win from reproducing it. For Gemini 3.1 Pro under Sighted observation, pass@3 is $38.7\%$ (46/119 positions) but pass^3 is $5.9\%$ (7/119): of the 46 positions it wins at least once, it wins all three trials on only 7. For GPT-5.5 the corresponding counts are 21 and 3 positions ( $17.6\%$ and $2.5\%$ ). pass^3 is zero in both settings for five models, and zero under Restricted observation for nine. Wins are also not spread evenly over positions. If Gemini's $22.1\%$ per-trial rate held on every position independently, pass@3 would be $52.7\%$ and pass^3 $1.1\%$ ; the observed $38.7\%$ and $5.9\%$ indicate that wins concentrate on a subset of positions. Under Sighted observation, Mistral Large 3 and MiniMax M2.5 each win one position in all three trials and no other. Within the subset, outcomes vary from attempt to attempt: on the 46 positions Gemini wins at least once, it wins $57\%$ of trials (79/138), and its split of 20, 19, and 7 positions with one, two, and three wins is close to the 16, 21, and 9 expected from independent trials at that rate. Competence is thus position-specific, and even where it exists a single run does not reveal it reliably.

![](images/52386c24e5ad21dc1f7943a56353f7f62758258aa1cf08ac3ccd5f0075d8f998.jpg)

![](images/3a4a27d0165a60ab6d3fc53b16db44cea746434a843795fa4081f64a21742c61.jpg)  
Figure 4 Deconstructing the Conversion Gap. (a) Mean pass@1 across the 12 LLMs by stored mate distance, per observation setting (95% case-bootstrap intervals; the 3-ply bin contains a single position). (b) Win rate conditioned on whether the agent's first move matches the stored reference move.

## 5.2 The Conversion Gap: Finding the Move vs. Winning the Game

Success falls steeply with the stored mate distance (Figure 4a): Sighted pass@1, averaged over the 12 LLMs, is 11.1% at 5 plies, 6.1% at 7, 2.9% at 9, and 1.8% at 11. The 85 positions at 9 and 11 plies make up 71% of the benchmark.

To locate where conversion breaks down, we compare the agent's first real move with the engine's stored reference move (Figure 4b). Across all Sighted trials, the first move matches the reference in $26.1\%$ of cases, far above the $3.3\%$ expected from choosing uniformly among legal root moves; Gemini 3.1 Pro agrees in $45.1\%$ of trials and GPT-5.5 in $42.9\%$ . A correct first move, however, rarely leads to a win. Trials that begin with the reference move are won only $13.9\%$ of the time (155/1,117), so more than $86\%$ of them are lost later in the game. Trials that begin with any other accepted move are won in $0.8\%$ (Sighted) and $0.3\%$ (Restricted observation); a different first move is not necessarily losing, since some positions may admit more than one winning first move. The reference move is thus close to necessary in practice (155 of the 178 Sighted wins begin with it) but far from sufficient. The same gap holds within models: Gemini agrees with the reference move in $45.1\%$ of trials but converts $22.1\%$ ; GPT-5.5 agrees in $42.9\%$ but converts $9.0\%$ . We call this the Conversion Gap: an evaluation that scored only the first move would credit these models with roughly two to five times the success they achieve over the full game.

## 5.3 The Effect of Restricted Observation

Restricted observation lowers pass@1 for every model. Mean pass@1 drops from 4.15% in Sighted to 1.82% in Restricted observation, a reduction of 2.33 percentage points (95% case-bootstrap interval [1.42, 3.41]), or more than half. First-move agreement falls to 15.1%, and the mean rate of terminal invalid-action or no-command losses rises from 2.7% to 27.6% of trials (Section 6). Restricted observation, however, also changes the prompt. An ablation that fixes the prompt and removes the state refreshes and the commands separately (Appendix E) shrinks the drop: on positions they solved under Sighted, GPT-5.5 and Gemini without either win 38.1% and 41.7% of trials, against 39.7% and 48.3% with both and 9.5% and 20.0% under Restricted observation. The drop thus measures the full setting, not the missing board alone.

## 6 Behavioral Analysis

## 6.1 How Models Use the REPL Interface

Figure 5 shows how each model distributes its actions across the REPL commands; three regularities emerge. Simulation volume does not track success. Under Sighted observation, simulate accounts for 45% of Gemini 3.1 Pro's actions, but also for 44% of DeepSeek V3.2's and 43% of DeepSeek V4 Pro's, whose pass@1 is 0.3% and 2.8%. Seven of the 12 models request simulation in at least 290 of their 357 Sighted trajectories (Table 8); only DeepSeek R1 (1.1% of actions) and GPT-OSS 120B (5.0%) largely forgo it. Line length does not separate the models either: Qwen 3.7 Max and Mistral Large 3 request the longest lines (mean 4.2 and 3.9 plies), whereas GPT-5.5 simulates a single ply in 71.9% of accepted calls (Table 9). Line legality separates the extremes but not the middle: GPT-5.5 and Gemini obtain at least one fully legal line in 353/356 and 299/332 simulating trajectories, against 14/213 for Mistral Large 3, yet DeepSeek V3.2 reaches 292/352 and still converts almost nothing. Explicit state queries are rare and not concentrated in the strongest models. Because the Sighted interface already returns the board and legal moves after every step, the two strongest models almost never call view\_board or get\_legal\_moves (at most 2.2% of actions); the heaviest users are DeepSeek V3.2 (12% and 8.4%) and DeepSeek V4 Pro (8.6% and 5.8%). Rejected actions dominate the weakest models under Restricted observation. Without state refreshes, rejected actions remain at 5–9% of actions for Gemini, GPT-5.5, and Claude Opus 4.7, rise to 24–42% for the middle group, and reach 93% for Mistral Large 3 and 79% for MiniMax M2.5 (Figure 5). These two models contribute 2,101 of the 5,850 Restricted rejections and recover from under 5% of them, whereas Gemini 3.1 Pro, GPT-5.5, Claude Opus 4.7, and Qwen 3.7 Max recover from at least 88% in both settings (Appendix Figure 19; temporal patterns in Section 6.2).

![](images/724bb4e4f9c311df94ba24fe938d4fecc8f795c2bb52713a68cc8dea057384ae.jpg)  
Figure 5 Action mix by model. Each cell gives the share (\%) of all agent REPL calls over complete trajectories spent on each command of Table 1; simulate includes budget-rejected calls, move counts accepted moves, and rejected counts rejected moves and the other rejected calls of Figure 6. Rows sum to $100\%$ . Hatched cells are commands unavailable under Restricted observation. Red numbers give pass@1 (%).

## 6.2 REPL Use Along Trajectories

Figure 6 tracks the composition of REPL calls across relative trial progress [0, 1].

- Deliberation versus over-budget simulation (Sighted): Successful trials spend 14.9% of their calls on think (17.4% in the first quarter of the trajectory), against 7.6% in failed trials. Failed trials spend 14.2% of calls on simulation requests rejected for exceeding the per-turn budget, against 8.6% in successes; these requests do not advance the board. In the last quarter of failed trajectories, resignation rises to 9.3% of calls.

- Persistent illegality (Restricted): In failed Restricted trials, rejected moves make up 33.3% of all calls, against 6.7% in successful ones. The share is high from the start (29.1% in the first quarter, 33.1% in the last), indicating that state-tracking errors arise early rather than accumulating only as the position deteriorates.

![](images/2ebbb78ed43493c30d8e5ee061c950db4e306d4b5c6c492f5f7722420f029364.jpg)  
Figure 6 REPL-call composition along trajectories. Calls from all 12 LLMs are combined over normalized trajectory progress [0,1]; each agent call belongs to exactly one category. State query counts accepted read-only perception calls of Table 1 (view\_board, get\_legal\_moves, get\_history; only get\_history under Restricted observation); other rejected covers unknown or malformed commands, think calls without reasoning, and over-limit or disabled queries. Successful Sighted trials devote more calls to think and fewer to over-budget simulation than failed ones; failed Restricted trials contain a high share of rejected moves throughout.

Across the benchmark, the attacker being checkmated or stalemated remains the dominant failure mode (3,124 Sighted, 2,777 Restricted), while Restricted observation additionally produces 1,178 terminal invalid-action forfeits (Table 11).

Per-model breakdowns (Figures 14–17) show the Sighted contrast most clearly for Gemini 3.1 Pro: its wins spend 14.0% of calls on think and 8.7% on over-budget simulation, its losses 2.4% and 19.0%. GPT-5.5 is the exception, interleaving think and simulation up to its final move whether it wins or loses (think 24.4% versus 21.2%). Under Restricted observation, rejected moves account for 5.2% and 5.6% of calls for Gemini 3.1 Pro and GPT-5.5 but 77.6% for MiniMax M2.5 and 92.9% for Mistral Large 3. With at most 79 wins per model, these contrasts are descriptive.

## 6.3 The Simulation Gap: Checking Legality Is Not Anticipating the Opponent

A natural hypothesis is that a forward simulation sandbox $(\mathcal{A}_{\mathrm{sim}})$ lets models verify tactical lines before committing them. The 11,342 simulation requests issued under Sighted observation reveal two limits on what it verifies (Figure 7a,b); a controlled ablation shows that it nonetheless contributes to conversion (Figure 7c).

Legality falls with line length. Of 6,749 accepted simulation calls, 4,474 executed the complete candidate sequence legally, 2,178 (32.3%) stopped on an illegal move, and the remaining 97 ended early at a simulated terminal or the eight-ply cap. Single-ply requests are almost always legal (96.3% of 2,592 calls), but full-line legality falls to 47.6% for 3–4 plies (1,110/2,332) and 37.4% for 5–8 plies (357/955). Within their own scratchpads, models move pieces through palace boundaries, pins, and missing cannon screens.

Assumed opponent replies often differ from the defender's. Every multi-ply simulated line contains a defender reply chosen by the model itself. Among the 1,677 cases in which an agent adopted the first move of a fully legal line on the real board, the actual defender played a different reply in $49.3\%$ (826/1,677), so the model's continuation was no longer the line it had checked. These data record disagreement, not the quality of the assumed reply; the sandbox verifies only the model's own hypothesis about the opponent, which the defender often does not share.

The sandbox still helps the strongest models. These limits do not make simulation useless. We re-ran Gemini 3.1 Pro and GPT-5.5 under Sighted observation with and without simulate, using a shared prompt and a correct check field, on the 20 and 21 positions each had solved at least once in the main evaluation (Appendix E). The ablation also disables view\_board and get\_legal\_moves, but with the state refreshed after every step both models almost never call them (at most 0.05 calls per turn, against 0.75 and 0.98

(a) Simulation requests

Share (%)  
![](images/ade9b8752ec6aab6160d77b1a243d026fd4174f02bfd9110196feeca9621ae8d.jpg)

![](images/0ad9d8c1b25507adc2034a69ec3844a6f2cccad3eb13eed13deef5508b9e9196.jpg)  
(c) Without simulate

![](images/8486682b45479036f4fb9dde178e19fe433d883f7b998121be76d209bf2d1f92.jpg)  
Figure 7 Simulation checks legality but not the opponent, yet removing it costs wins. (a) Outcomes of all 11,342 Sighted simulation requests; “early stop / capped” comprises 93 legal simulated terminals and four legal prefixes truncated at eight plies. (b) For the 1,677 adopted candidate lines with a legal assumed defender reply, whether the actual defender chose that reply. (c) Controlled ablation on positions each model solved under Sighted observation, three trials per position (Appendix E): Sighted with and without simulate; the ablation also disables view\_board and get\_legal\_moves, which neither model uses when the state is refreshed. Bars: pass@1 with 95% case-bootstrap intervals; $\Delta$ : paired difference in points; p: exact sign test over positions.

simulate calls). Without simulate, pass@1 falls from 48.3% to 33.3% for Gemini and from 39.7% to 25.4% for GPT-5.5 (Figure 7c). The paired drop is significant for GPT-5.5 (p = 0.013) and has the same sign but is not significant for Gemini (p = 0.27). Simulation thus helps the strongest models find wins even though it verifies only their own hypothesis about the opponent. Without state refreshes, disabling the same commands does not lower pass@1 further (Appendix E).

Failing agents are out-raced by the defender. How games end shows the cost of this gap (Figure 8). Wins rarely wander: 142 of 178 Sighted wins and 62 of 78 Restricted-observation wins end exactly at the stored mate distance. Failures end early instead: $77.8\%$ of failed Sighted trials and $83.8\%$ of failed Restricted-observation trials stop before that distance. Only 80 of the 8,568 trajectories reach ply 40: 76 end at the turn cap (29 Sighted and 47 Restricted-observation), and four end in a repetition draw on that ply. Most failures are lost on the board, and the share tracks the defender's counter-threat: $81.8\%$ (Sighted) and $70.4\%$ (Restricted) of failures in the 106 threat-race positions, against $28.2\%$ and $29.8\%$ in the 13 unopposed-mate positions. In the 81 positions whose defender threatens mate in one, $23.8\%$ of Sighted and $30.0\%$ of Restricted-observation trials end at ply 2: the agent's first move does not stop the threat, and the defender mates at once. Failing agents are thus not stalled but overtaken by replies they did not anticipate. Stronger models survive longer even when they lose: across the 12 models, the median relative length of failed trials rises with the number of wins (Spearman $\rho = 0.71$ Sighted, 0.89 Restricted observation).

## 7 Discussion

The results suggest two explanations for the gap between finding a good move and converting a won position. Both are interpretations consistent with the data rather than established causes.

Amortized policies vs. adversarial search. Converting a won position requires each move to hold up against the defender's best reply, i.e., an approximation of $V^{*}(s_{t}) = \max_{a_{t}} \min_{a_{\mathrm{def}}} V^{*}(s_{t + 1})$ over board states $s_t$ . A language model instead samples each action from an amortized policy $P_{\theta}(a_t \mid o_t, h_t)$ shaped largely by pre-training. Such a policy can capture familiar motifs, as the $26.1\%$ first-move agreement shows, but nothing in it checks each continuation against counterplay. If a model errs with probability $\epsilon_t$ at its $t$ -th move, success on a line of $H$ moves shrinks roughly as $\prod_{t=1}^{H}(1 - \epsilon_t)$ . This simple picture is consistent with the steep decline in pass@1 with mate distance and with the gap between Gemini's $45.1\%$ first-move agreement and its $22.1\%$ conversion. It is incomplete: errors are not independent across moves, and wins concentrate on particular positions.

![](images/f7f39a58ec7ca8b95b45bbb57cb3af8f3b33fd69a5595a96cc658354d5b354ff.jpg)  
Figure 8 Failed trajectories end early; successful ones end at the reference length. Top: plies played divided by the stored mate distance of the case (dashed line: equal length). Boxes summarize failed trials (whiskers: 5th–95th percentiles), points show successful trials, and numbers above the panels give wins out of 357. Triangles mark the 15 wins longer than four times the mate distance, all in JH-084, where the defender loses by perpetual check. Bottom: how failed trials end; lost on board means the attacker was left without a legal move, and action forfeit covers invalid-action, no-command, and action-limit losses.

Simulation checks legality, not the opponent. The agent writes both sides of every simulated line. A third of accepted simulations stop on an illegal move, and when a legal line began with the move actually played, the real defender chose a different reply in about half of the cases (Section 6). This does not show that the imagined replies were bad, since several defenses can be equally strong, but it does show that the sandbox cannot supply the opponent's choices. Agents that rely on self-authored rollouts in adversarial settings still need an external source of opponent behavior.

Implications for agent evaluation. (i) Evaluate closed-loop conversion: agreement on the first move does not imply the ability to finish the game, so benchmarks should include play against an active opponent. (ii) Report reliability alongside coverage: pass@k and pass^k diverge sharply here and should be reported together. (iii) Vary one affordance at a time: Restricted observation more than halves pass@1, yet a controlled ablation attributes much of that drop to its prompt and feedback rather than the missing board, so observation settings should differ only in the affordance under test.

## 8 Conclusion

We introduced XiangqiBench, an executable benchmark in which language agents must convert search-supported winning Xiangqi endgames into checkmate against an engine defender. Across 8,568 trajectories from 12 frontier LLMs, finding the move is not winning the game: first-move agreement, pass@3, and self-authored simulation each overstate closed-loop success. Closing these gaps likely takes more than better single-step proposals; agents must win reliably and anticipate how the opponent responds. XiangqiBench provides a controlled testbed for measuring both.

## References

Marwa Abdulhai, Isadora White, Charlie Victor Snell, Charles Sun, Joey Hong, Yuexiang Zhai, Kelvin Xu, and Sergey Levine. LMRL Gym: Benchmarks for multi-turn reinforcement learning with language models. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 126–153. PMLR, 2025. URL https://proceedings.mlr.press/v267/abdulhai25a.html.

Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K Arora, Yu Bai, Bowen Baker, Haiming Bao, et al. gpt-oss-120b & gpt-oss-20b model card. arXiv preprint arXiv:2508.10925, 2025.

Anthropic. Claude Opus 4.7, 2026. URL https://www.anthropic.com/news/claude-opus-4-7.

Aili Chen, Aonian Li, Baichuan Zhou, Bangwei Gong, Binyang Jiang, Boji Dan, Changhao Zhang, Changqing Yu, Chao Wang, Cheng Ma, et al. The minimax-m2 series: Mini activations unleashing max real-world intelligence. arXiv preprint arXiv:2605.26494, 2026.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021. URL https://arxiv.org/abs/2107.03374.

Yuhao Chen, Shuochen Liu, Yuanjie Lyu, Chao Zhang, Jiayao Shi, and Tong Xu. Xiangqi-R1: Enhancing spatial strategic reasoning in LLMs for chinese chess via reinforcement learning. arXiv preprint arXiv:2507.12215, 2025. URL https://arxiv.org/abs/2507.12215.

Anthony Costarelli, Mat Allen, Roman Hauksson, Grace Sodunke, Suhas Hariharan, Carlson Cheng, Wenjie Li, Joshua M. Clymer, and Arjun Yadav. GameBench: Evaluating strategic reasoning abilities of LLM agents. arXiv preprint arXiv:2406.06613, 2024. URL https://arxiv.org/abs/2406.06613.

Jinhao Duan, Renming Zhang, James Diffenderfer, Bhavya Kailkhura, Lichao Sun, Elias Stengel-Eskin, Mohit Bansal, Tianlong Chen, and Kaidi Xu. GTBench: Uncovering the strategic reasoning capabilities of LLMs via game-theoretic evaluations. In Advances in Neural Information Processing Systems, volume 37, 2024. doi: 10.52202/079017-0885. URL https://proceedings.neurips.cc/paper\_files/paper/2024/hash/3191170938b6102e5c203b036b7c16dd-Abstract-Conference.html.

Xidong Feng, Yicheng Luo, Ziyan Wang, Hongrui Tang, Mengyue Yang, Kun Shao, David Mguni, Yali Du, and Jun Wang. ChessGPT: Bridging policy learning and language modeling. In Advances in Neural Information Processing Systems, volume 36, pp. 7216–7262, 2023.

Google DeepMind. Gemini 3.1 Pro, 2026. URL https://deepmind.google/models/gemini/pro/.

Mikaela Grace, Jeremy Hadfield, Rodrigo Olivares, and Jiri De Jonghe. Demystifying evals for AI agents. Anthropic Engineering, January 2026. URL https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents. Published January 9, 2026.

Leon Guertler, Bobby Cheng, Simon Yu, Bo Liu, Leshem Choshen, and Cheston Tan. TextArena. arXiv preprint arXiv:2504.11442, 2025. URL https://arxiv.org/abs/2504.11442.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. SWE-bench: Can language models resolve real-world GitHub issues? In International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=VTF8yNQM66.

Sai Kolasani, Maxim Saplin, Nicholas Crispino, Kyle Montgomery, Jared Quincy Davis, Matei Zaharia, Chi Wang, and Chenguang Wang. LLM Chess: Benchmarking reasoning and instruction-following in LLMs through chess. arXiv preprint arXiv:2512.01992, 2025. URL https://arxiv.org/abs/2512.01992.

Wenye Lin, Jonathan Roberts, Yunhan Yang, Samuel Albanie, Zongqing Lu, and Kai Han. GAMEBoT: Transparent assessment of LLM reasoning in games. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics, pp. 7656–7682, 2025. doi: 10.18653/v1/2025.acl-long.378. URL https://aclanthology.org/2025.acl-long.378/.

Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, et al. Deepseek-v3. 2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556, 2025.

Jincheng Liu, Sijun He, Jingjing Wu, Xiangsen Wang, Yang Chen, Zhaoqi Kuang, Siqi Bao, and Yuan Yao. ChessArena: A chess testbed for evaluating strategic reasoning capabilities of large language models. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics, pp. 7901–7954, 2026. doi:10.18653/v1/2026.acl-long.360. URL https://aclanthology.org/2026.acl-long.360/.

Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, and Jie Tang. AgentBench: Evaluating LLMs as agents. In International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=zAdUBOaCTQ.

Mistral AI. Introducing Mistral 3, 2025. URL https://mistral.ai/news/mistral-3/.

Official Pikafish Contributors. Pikafish: A free and strong UCI xiangqi engine. https://github.com/official-pikafish/Pikafish, 2026. Accessed 2026-07-31.

OpenAI. Introducing GPT-5.5, 2026. URL https://openai.com/index/introducing-gpt-5-5/.

Qwen Team. Qwen3.7: The agent frontier, May 2026. URL https://qwen.ai/blog?id=qwen3.7.

Anian Ruoss, Grégoire Delétang, Sourabh Medapati, Jordi Grau-Moya, Li Kevin Wenliang, Elliot Catt, John Reid, Cannada A. Lewis, Joel Veness, and Tim Genewein. Amortized planning with large-scale transformers: A case study on chess. In Advances in Neural Information Processing Systems, volume 37, 2024. doi: 10.52202/079017-2102. URL https://proceedings.neurips.cc/paper\_files/paper/2024/hash/78f0db30c39c850de728c769f42fc903-Abstract-Conference.html.

Philipp Schmid. Pass@k vs. Pass^k: Understanding agent reliability. Technical blog post, March 2025. URL https://www.philschmid.de/agents-pass-at-k-pass-power-k. Published March 24, 2025.

John Schultz, Jakub Adamek, Matej Jusup, Marc Lanctot, Michael Kaisers, Sarah Perrin, Daniel Hennes, Jeremy Shar, Cannada A. Lewis, Anian Ruoss, Tom Zahavy, Petar Veličković, Laurel Prince, Satinder Singh, Eric Malmi, and Nenad Tomasev. Mastering board games by external and internal planning with language models. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 53581–53644. PMLR, 2025. URL https://proceedings.mlr.press/v267/schultz25a.html.

Bytedance Seed. Seed2. 0 model card: Towards intelligence frontier for real-world complexity. arXiv preprint arXiv:2607.00248, 2026.

Jiajun Shi, Jian Yang, Jiaheng Liu, Xingyuan Bu, Jiangjie Chen, Junting Zhou, Kaijing Ma, Zhoufutu Wen, Bingli Wang, Yancheng He, Liang Song, Hualei Zhu, Shilong Li, Xingjian Wang, Wei Zhang, Ruibin Yuan, Yifan Yao, Wenjun Yang, Yunli Wang, Siyuan Fang, Siyu Yuan, Qianyu He, Robert Tang, Yingshui Tan, Wangchunshu Zhou, Zhao-Xiang Zhang, Zhoujun Li, Wenhao Huang, and Ge Zhang. KORGym: A dynamic game platform for LLM reasoning evaluation. In Advances in Neural Information Processing Systems, volume 38, 2025. doi: 10.52202/085713-5384. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ebfa4297cd6419f64efe86f657ba49d0-Abstract-Conference.html.

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, SH Cai, Yuan Cao, Ziwei Chai, Y Charles, HS Che, Cheng Chen, et al. Kimi k2. 5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276, 2026.

Shubham Toshniwal, Sam Wiseman, Karen Livescu, and Kevin Gimpel. Chess as a testbed for language model state tracking. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pp. 11385–11393, 2022. doi:10.1609/aaai.v36i10.21390.

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. OSWorld: Benchmarking multimodal agents for open-ended tasks in real computer environments. In Advances in Neural Information Processing Systems, 2024.

Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chenchen Ling, et al. Deepseek-v4: Towards highly efficient million-token context intelligence. arXiv preprint arXiv:2606.19348, 2026.

Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik R. Narasimhan. $\tau$ -bench: A benchmark for tool-agent-user interaction in real-world domains. In International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=roNSXZpUDN.

Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. WebArena: A realistic web environment for building autonomous agents. In International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=oKn9c6ytLx.

## Appendix

## A Benchmark and Protocol

## A.1 Positions

Each case stores an identifier and source, the normalized FEN, the attacking side, the admission method, one reference root move, the reported mate distance in plies, a position category, and piece tags. Model outputs play no role in admission. Engine-admitted cases come from a single-threaded, fixed-time Pikafish search (800 ms by default) that reports a forced mate; its final preferred move is stored as the reference move, and every reference move is legal under rules replay. Positions that Pikafish cannot load are admitted by a checks-only mate search and labelled separately. Search logs, principal variations, and per-case search settings were not archived, so exact reconstruction would also require the engine build and network.

The category comes from a second search with Black to move. A threat race (106 cases) is a position in which Black also has a forced mate; an unopposed mate (13 cases) is one in which none is found within the search limit, which is time-limited (300 ms by default) for engine-admitted cases and depth-limited for solver-admitted ones.

## A.2 Defender

On every Black turn the environment sends the current FEN to Pikafish and plays its depth-18, single-threaded choice. If the engine cannot load the position or returns an error, a rule-based defender answers instead: it scores each legal Black move with a depth-5 forced-mate search for Red and prefers moves with no mate found, then the longest mate, then checking moves, breaking ties by move string. The two solver-admitted cases, JH-084 and JH-327, place a Black pawn on its own back rank, a composed placement that Pikafish rejects, so the rule-based defender plays almost every Black move in them. These two cases contribute 39 of the 178 Sighted wins and 17 of the 78 Restricted-observation wins, including 18 of the 20 perpetual-check wins (Appendix A.5). Records log the defender type at initialization only; per-move provenance is inferred from these engine rejections.

## A.3 Interface, Budgets, and Termination

Both settings share the command interface of Table 1; a real move is a coordinate command such as move h2h9. The Sighted prompt asks for a forcing checkmate. Its first message gives the board, FEN, and legal moves, and every later message repeats them together with captures and check status. The Restricted-observation prompt is a generic blindfold prompt that describes the standard opening position, although the first message shows the actual endgame board and FEN. Later messages report only the move, capture, check status, and material; board views, legal-move queries, and simulation are disabled, while get\_history remains available. The prompt mismatch is visible in behavior: $70.5\%$ (2,330/3,304) of rejected first-turn moves under Restricted observation are legal only from the standard opening, against $2.6\%$ under Sighted observation. Because prompt and observation change together, we treat the setting comparison as descriptive and isolate observation in Appendix E. Both prompts are reproduced in Appendix C.

Every trial is capped at 40 plies. In the dominant configuration, a turn allows eight actions, one board view, one simulation of at most eight model-supplied plies, three rejected actions (unknown commands included), and five consecutive responses without a command. Exhausting any budget forfeits the trial, and reaching the cap without a win counts as a failure. The Sighted prompt instead states that exceeding the action limit forces a move; 25 Sighted trials end by action-limit forfeit. Seventy-five Sighted records use a four-action limit and two a five-rejection limit. Records ending in API errors, timeouts, or exceptions, and a small number of undetermined or action-budget outcomes, are excluded from scoring.

## A.4 Trajectory Records and Selection

Each record stores the initial board and FEN, every move with the positions before and after it, the legal moves at each turn, every prompt, the model's text and commands, provider token usage, environment feedback, and the terminal result. Simulation results are stored as rendered feedback, from which we recover per-ply validity. Within each of the 2,856 model-case-setting cells we keep the first three scored records by start time, giving 8,568 trajectories. This omits eight further records, all failures, in three Sighted cells of Mistral Large 3 whose retained trials also fail. Requests that exhausted their API retries leave no record, so the number of attempted requests is unknown.

## A.5 Rules Replay and Repetition

Check flag. The collection environment set the check flag with the rules library's is\_checking() predicate after each move. The predicate tests whether the side to move attacks the opposing general, which never holds after a legal move. All 57,000 Check fields shown to agents, and every per-ply field in simulation output, therefore read No. The defect leaves board transitions, legality, and winners unchanged, but it labelled every checkmate as a stalemate and disabled the collection-time perpetual-check rule.

Replay. We replay all accepted actions with cchess 1.25.5 after discarding 423 synthetic forfeit entries that repeat the prior move. The replay splits the 6,137 archived stalemates into 6,034 checkmates and 103 stalemates without changing any winner.

Repetition. With the flag always false, collection scored every threefold repetition as a draw. We instead apply the released environment's long-check rule: at threefold repetition, a side that gave check on each of its moves since the first occurrence loses, and any other repetition is a draw. Replaying the 138 repetition trajectories reproduces every archived position; only terminal labels change. Of the 43 perpetual checks, the 23 by the attacker remain failures (16 Sighted, seven Restricted) and the 20 by the defender become wins (14 and six), 18 of them in JH-084 under the rule-based defender. Scoring every repetition as a draw would give 164 Sighted and 72 Restricted-observation wins, mean pass@1 of $3.83\%$ and $1.68\%$ , and a setting difference of 2.15 points ([1.38, 2.99]). The rule simplifies official long-check and long-chase adjudication. Because agents never received a correct check signal, their play may differ from what it would have been with one; no relabelling can recover that counterfactual.

## A.6 Metrics and Uncertainty

Pass@1 is the mean of 357 binary outcomes per model and setting. With three trials per case, pass@k uses the estimator of Chen et al. (2021) and pass^k that of Yao et al. (2025): pass@3 is the fraction of the 119 cases with at least one win and pass^3 the fraction with three. We report $k \leq 3$ only. The plug-in estimate $(c_i / n_i)^k$ (Schmid, 2025) is a different finite-sample estimator and is not used. Intervals for the setting comparison come from a case bootstrap that resamples the 119 cases with all models, settings, and trials attached (4,000 resamples, seed 2027).

## B Models

Table 2 lists the 12 models, each with three scored trials per case in both settings. Names follow the providers; deployment versions were not independently verified. Kimi K2.6, GLM 5.1, and Seed 2.1 Pro were also attempted but did not complete collection because of credential failures or request timeouts; they are omitted rather than scored as losses.

Table 2 Evaluated models. Model IDs are the names under which trajectories are stored; Weights indicates whether weights are publicly released. Each model contributes 357 scored trajectories per setting.

<table><tr><td>Model</td><td>Developer</td><td>Weights</td><td>Model ID</td></tr><tr><td>Gemini 3.1 Pro</td><td>Google</td><td>Closed</td><td>gemini-3.1-pro</td></tr><tr><td>GPT-5.5</td><td>OpenAI</td><td>Closed</td><td>gpt-5.5</td></tr><tr><td>Qwen 3.7 Max</td><td>Alibaba</td><td>Closed</td><td>qwen3.7-max</td></tr><tr><td>Claude Opus 4.7</td><td>Anthropic</td><td>Closed</td><td>claude-opus-4.7</td></tr><tr><td>DeepSeek V4 Pro</td><td>DeepSeek</td><td>Open</td><td>deepseek-v4-pro</td></tr><tr><td>Seed 2.0 Pro</td><td>ByteDance</td><td>Closed</td><td>seed-2.0-pro</td></tr><tr><td>Kimi K2.5</td><td>Moonshot AI</td><td>Open</td><td>kimi-k2.5</td></tr><tr><td>Mistral Large 3</td><td>Mistral AI</td><td>Open</td><td>mistral-large-3</td></tr><tr><td>MiniMax M2.5</td><td>MiniMax</td><td>Open</td><td>minimax-m2.5</td></tr><tr><td>DeepSeek R1</td><td>DeepSeek</td><td>Open</td><td>deepseek-r1</td></tr><tr><td>GPT-OSS 120B</td><td>OpenAI</td><td>Open</td><td>gpt-oss-120b</td></tr><tr><td>DeepSeek V3.2</td><td>DeepSeek</td><td>Open</td><td>deepseek-v3.2</td></tr></table>

## C Prompt Templates

This appendix reproduces the agent's prompts from the trajectory records; the release includes the script that generates it. The Restricted-observation prompt is identical across all 4,284 trajectories. The Sighted prompt shown is used in 4,209; the other 75 state a four-action rather than an eight-action limit. Per-turn messages appear as templates with $\langle$ $placeholder$ $\rangle$ for content that changes between turns. For typesetting we omit the Chinese glosses that accompany English terms, replace the board diagram (drawn with Chinese glyphs) with a placeholder, and render Unicode punctuation in ASCII; the wording is otherwise unchanged.

The prompts depart from the evaluated configuration in four places. The Sighted prompt says that exceeding the action limit forces a move, whereas the configuration forfeits the trial. Both prompts call the 40-ply limit a draw, which is scored as a failure to convert. The Restricted-observation prompt says the board is never shown, although the first message shows it once, and states that three consecutive checks lose, a rule that never fired and is not used in scoring. Finally, every Check field reads No because of the defect in Appendix A.5.

## C.1 System Prompts

````text
Sighted system prompt
You are solving a Chinese Chess (Xiangqi) ENDGAME.
This is NOT the opening -- it is a sharp endgame position and it is YOUR move. You are the side to move. Your task is to find the precise forcing sequence that CHECKMATES the opponent. Play exactly: in these positions a single slack, non-forcing move can let the opponent escape or launch a decisive counterattack. There is no draw available -- play to win by checkmate.
================
1. GAME RULES =================
Board: 9 columns (a-i) x 10 rows (0-9), divided by a River.
Red occupies rows 0-4 (bottom), Black occupies rows 5-9 (top).
Each side has a Palace -- the 3x3 area around the King.
Pieces and Movement:
K/k King -- One step orthogonally, confined to the Palace. The two Kings may NEVER face each other on the same column with no piece between them (Flying General).
A/a Advisor -- One step diagonally, confined to the Palace.
B/b Elephant -- Two steps diagonally. Cannot cross the River. Blocked if the midpoint is occupied.
N/n Horse -- "L" shape. Blocked if the first orthogonal square is occupied.
R/r Rook -- Any distance orthogonally; cannot jump.
C/c Cannon -- Moves like a Rook; to CAPTURE it must jump exactly one piece (the "cannon platform").
P/p Pawn -- Forward one step; after crossing the River also sideways.
Never backward.

2. WIN / LOSS

- Checkmate: opponent's King is in check with no legal escape -> you WIN.

- Stalemate: opponent has no legal move (even if not in check) -> opponent LOSES (unlike Western chess).

- Move limit: if the game runs past 40 half-moves it is scored as a draw (i.e. you failed to convert) -- so do not dawdle.

```txt
3. MOVE FORMAT
Each move is exactly 4 characters: <from_col><from_row><to_col><to_row>
Columns: a b c d e f g h i (left -> right)
Rows: 0 1 2 3 4 5 6 7 8 9 (bottom -> top, Red's back rank = 0)
Examples: h2e2 (cannon h2->e2), b0c2 (horse b0->c2), a0a1 (rook a0->a1).
```

4. BOARD STATE -- WHAT TO TRUST

You get THREE views; trust them in this order:

1. PIECE POSITIONS census -- coordinate-keyed ground truth of every piece. This is authoritative; do NOT rely on memory (a common failure is misreading a glyph, e.g. confusing $\langle King\ glyph\rangle$ with $\langle Elephant\ glyph\rangle$ ).

2. FEN string -- compact full-board encoding (ranks listed row 9 -> row 0; N=Horse, B=Elephant).

3. ASCII board -- Chinese glyphs; useful for intuition but easy to misread. Reconcile your mental model with the census/FEN before every move.

5. AVAILABLE TOOLS

Each turn you MUST eventually call `move` (or `resign`) to end your turn. Before that you SHOULD `think`, and you MAY use auxiliary tools:

think(reasoning) -- \*\*ALWAYS call this first.\*\* Record your analysis of the mating idea, candidate forcing moves, and the lines you calculated. Does not end your turn.

move(from\_to) -- Play a move. Ends your turn.

resign() -- Give up (you lose). Only if the position is truly lost.
view\_board() -- Show the board with FEN (max 1/turn).

simulate(moves) -- Test a hypothetical line WITHOUT affecting the real game. YOU supply both sides: [your\_move, assumed\_reply, your\_move, ...]. Each entry is one half-move. Returns per ply: what moved/captured, check, FEN, and the number of legal replies. Max 8 half-moves, 1/turn. Use it to verify a forced mate before committing.

get\_legal\_moves() -- List all legal moves in the current position.
get\_history() -- Show the full move history.

Recommended flow: think -> (simulate to verify the mate) -> move

After each \`move\` the environment returns objective facts only (it never tells you who is winning -- assessing the position is YOUR job): what moved/captured, whether it is check, material balance, the updated FEN, and the legal moves for the next position.

6. CONSTRAINTS

- You can ONLY act via the provided commands/tools; plain text is rejected.

- You MUST call `think` BEFORE `move`.

- A turn ends only on `move` or `resign`. Auxiliary tools do not end the turn.

- If you exceed 8 actions in a turn without moving, a move is forced on your behalf.

- Illegal moves are rejected with the legal-move list; choose a legal move.

- Keep the attack forcing. Quiet moves usually throw away the win.

7. INTERACTION PROTOCOL (read carefully)

You control the game through a command-line interface, like a terminal.
This is the ONLY way you may act. Ignore any mention elsewhere of JSON tool
calls or `function(args)` syntax -- in this game you act by typing commands.

On each of your turns:
    1. Think out loud in plain text (your private reasoning -- the opponent never
    sees it).
    2. Then issue EXACTLY ONE command inside a fenced code block:

    ```bash
    move h2e2
    ```

The environment runs your command and replies with terminal output (prefixed
with `$ <command>`), then it is your turn again until you end it with a move.

Available commands (one per block):
    move <from><to>    Play a move in ICCS format (e.g. `move h2e2`). ENDS your turn.
    simulate <m1> <m2> ...    Forward-test a hypothetical line you supply for BOTH
    sides: [your move, assumed opponent move, ...]
    (e.g. `simulate h2e2 h9g7 b0c2`)
    view_board    Show the board, FEN, and the ground-truth piece list.
    legal_moves    List all legal moves right now.
    think <reasoning>    Record private reasoning (does NOT end your turn).
    history    Show the move history.
    resign    Forfeit the game. ENDS your turn.

Rules:
    - Put your command in a ``bash`` block. Only the LAST block is executed.
    - One command per turn-step. To look before you leap, send `view_board` or
    `simulate ...` first, read the output, then send `move ...`
    - If you emit no command block, you will be asked again.
    - Illegal or unknown commands are rejected with the list of legal moves; keep
    trying -- you do NOT lose your turn for one mistake, but repeated invalid
    commands in a single turn will be resolved against you.
````

## Restricted-observation system prompt

````text
You are playing Chinese Chess (Xiangqi) in BLINDFOLD mode.  
You interact with the game environment by typing commands in a terminal (see INTERACTION PROTOCOL below).  
You CANNOT see the board, FEN, or piece census at any point. You must track the entire position in your own reasoning, using only the move history and the per-move diffs you are given. Your memory and mental board-tracking ARE the test.
```

```txt
Pieces and Movement:
King -- One step orthogonally, confined to the Palace. The two Kings may NEVER face on the same file with nothing between them (Flying General).
Advisor -- One step diagonally, confined to the Palace.
Elephant -- Exactly two steps diagonally. CANNOT cross the River. Blocked if the midpoint is occupied.
Horse -- "L" shape. Blocked if the first orthogonal step is occupied.
Rook -- Any distance orthogonally. Cannot jump.
Cannon -- Moves like a Rook; to CAPTURE it must jump exactly one piece (the platform).
Pawn -- Before the River: one step forward only. After crossing: one step forward OR sideways. Never backward.
```

```rst
Columns a-i (left->right), rows 0-9 (bottom->top). Red's back rank is row 0,
Black's back rank is row 9. Moves are 4-char ICCS, e.g. 'h2e2'.

Red: King e0 | Advisors d0 f0 | Elephants c0 g0 | Horses b0 h0 | Rooks a0 i0
Cannons b2 h2 | Pawns a3 c3 e3 g3 i3

Black: King e9 | Advisors d9 f9 | Elephants c9 g9 | Horses b9 h9 | Rooks a9 i9
Cannons b7 h7 | Pawns a6 c6 e6 g6 i6

================

3. CONSTRAINTS
================

- You have NO board viewer, NO simulator, and NO legal-move list. The ONLY
game-changing action is `move`. (You may also `think` to record reasoning.)
- After each move you get only a diff: what moved/was captured, check, material.
After the opponent moves you get the same diff for their move -- never a board.
- If you make 3 illegal moves in a single turn, you FORFEIT.
- You cannot see your opponent's reasoning -- only their moves.

================

INTERACTION PROTOCOL (read carefully)
================

You control the game through a command-line interface, like a terminal.
This is the ONLY way you may act. Ignore any mention elsewhere of JSON tool
calls or `function(args)` syntax -- in this game you act by typing commands.

On each of your turns:
    1. Think out loud in plain text (your private reasoning -- the opponent never
    sees it).
    2. Then issue EXACTLY ONE command inside a fenced code block:

```bash
move h2e2
```

The environment runs your command and replies with terminal output (prefixed
with `$ <command>`), then it is your turn again until you end it with a move.

BLINDFOLD: there is no board viewer, simulator, or legal-move list. Track the position from the move history and per-move
diffs alone. Repeated illegal moves in one turn forfeit the game.

Available commands (one per block):
    move <from><to>    Play a move in ICCS format (e.g. `move h2e2`.). ENDS your turn.
    think <reasoning>    Record private reasoning (does NOT end your turn).
    history    Show the move history.
    resign    Forfeit the game. ENDS your turn.

Rules:
    - Put your command in a ``bash`` block. Only the LAST block is executed.
    - If you emit no command block, you will be asked again.
    - Illegal or unknown commands are rejected; repeated invalid commands in a single turn will be resolved against you.
````

## C.2 Per-Turn Messages

In both settings the first user message shows the initial position. Sighted feedback then repeats the full state after every ply; Restricted-observation feedback reports only the move.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Sighted | first user message
Game started. You are playing as RED.
Initial board:
&lt;board diagram with file letters a-i and rank numbers 0-9&gt;
Current turn: Red (Uppercase)
Move format: &lt;gloss of the ICCS format, e.g. h2e2 moves the piece on h2 to e2&gt;
FEN: &lt;FEN&gt; w
Legal moves ( $\langle n\rangle$ ): &lt;comma-separated ICCS moves&gt;
Type one command inside a \`\`\`bash\`\`\` block to act (e.g. `move h2e2'). You may `view_board`, `simulate ...`, or `legal_moves` first, then `move` to end your turn.
</div>

```txt
Sighted | after the agent's move, then after the defender's reply
Move executed: <move>
Result: Red <piece> <from)->(to) <no capture / captures piece)
Capture: you captured <piece>. <only on a capture>
Check: <Yes - you are giving check! | No>
Material -- Red: R(n) N(n) B(n) A(n) C(n) P(n) | Black: <same fields> | Balance: <side> <value> (nominal)
Board after your move:
<board diagram>
Current turn: Black (Lowercase)
Move format: <gloss of the ICCS format>
FEN: <FEN> b
Legal moves (<n>: <Black's legal moves>
Opponent (black) played: <move> -- Black <piece> <from)->(to) <no capture / captures piece>
Check: <Yes / No>
You LOST your <piece> on this move. <only on a capture>
Move <k> -- FEN: <FEN> w
Current board:
<board diagram>
Current turn: Red (Uppercase)
Move format: <gloss of the ICCS format>
Piece positions (ground truth -- trust this over your memory):
    Red : <piece-type list with coordinates, e.g. King e0 | Rook i5 | Cannon g9,h5)
    Black: <piece-type list with coordinates>
Legal moves (<n>: <comma-separated ICCS moves>
It is now your turn. Call a tool to proceed.
```

```txt
Restricted observation | first user message
Game started. You are playing as RED.
Initial board:
<board diagram with file letters a-i and rank numbers 0-9>
Current turn: Red (Uppercase)
Move format: <gloss of the ICCS format, e.g. h2e2 moves the piece on h2 to e2>
FEN: <FEN> w
BLINDFOLD: this is the ONLY time you are shown the board. From now on you get only per-move diffs -- no board, FEN, or
legal moves. Track the position yourself.
Type one command inside a ``bash`` block to act (e.g. `move h2e2'). You may `think ...` first, then `move` to end your
turn. There is no board viewer or legal-move list.
```

Restricted observation | after the agent's move, then after the defender's reply

```txt
Move executed: <move>
Result: Red <piece> <from)-><to> <no capture | captures piece>
Capture: you captured <piece>. <only on a capture>
Check: <Yes - you are giving check! | No>
Material -- Red: R(n) N(n) B(n) A(n) C(n) P(n) | Black: <same fields> | Balance: <side> <value> (nominal)

Opponent (black) played: <move> -- Black <piece> <from)-><to> <no capture | captures piece>
Check: <Yes - you are in check! | No>
<No capture. | You LOST your piece on this move.)
It is now your turn.
```

## D Additional Results

Table 3 gives pass@1, pass@3, and pass^3 for every model and setting, and Table 4 the full pass@k and pass^k for k = 1, 2, 3.

Table 3 Three-trial success and consistency (%). Pass@3 is the fraction of cases with at least one success; pass^3 is the fraction for which all three trials succeed. S/R denote Sighted/Restricted observation.

<table><tr><td rowspan="2">Model</td><td colspan="2">Pass@1</td><td colspan="2">Pass@3</td><td colspan="2"> $pass^3$ </td></tr><tr><td>S</td><td>R</td><td>S</td><td>R</td><td>S</td><td>R</td></tr><tr><td>Gemini 3.1 Pro</td><td>22.1</td><td>14.0</td><td>38.7</td><td>32.8</td><td>5.9</td><td>1.7</td></tr><tr><td>GPT-5.5</td><td>9.0</td><td>3.4</td><td>17.6</td><td>6.7</td><td>2.5</td><td>0.8</td></tr><tr><td>Qwen 3.7 Max</td><td>6.2</td><td>2.8</td><td>13.4</td><td>5.9</td><td>0.8</td><td>0.8</td></tr><tr><td>Claude Opus 4.7</td><td>5.3</td><td>0.8</td><td>10.9</td><td>1.7</td><td>0.8</td><td>0.0</td></tr><tr><td>DeepSeek V4 Pro</td><td>2.8</td><td>0.0</td><td>5.0</td><td>0.0</td><td>1.7</td><td>0.0</td></tr><tr><td>Seed 2.0 Pro</td><td>0.8</td><td>0.6</td><td>2.5</td><td>1.7</td><td>0.0</td><td>0.0</td></tr><tr><td>Kimi K2.5</td><td>0.8</td><td>0.3</td><td>1.7</td><td>0.8</td><td>0.0</td><td>0.0</td></tr><tr><td>Mistral Large 3</td><td>0.8</td><td>0.0</td><td>0.8</td><td>0.0</td><td>0.8</td><td>0.0</td></tr><tr><td>MiniMax M2.5</td><td>0.8</td><td>0.0</td><td>0.8</td><td>0.0</td><td>0.8</td><td>0.0</td></tr><tr><td>DeepSeek R1</td><td>0.6</td><td>0.0</td><td>1.7</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>GPT-OSS 120B</td><td>0.3</td><td>0.0</td><td>0.8</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>DeepSeek V3.2</td><td>0.3</td><td>0.0</td><td>0.8</td><td>0.0</td><td>0.0</td><td>0.0</td></tr></table>

Table 4 Full finite-sample consistency metrics (%). For each case with n = 3 trials and c successes, Pass@k = $1 - \binom{3-c}{k}/\binom{3}{k}$ and pass^k = $\binom{c}{k}/\binom{3}{k}$ ; entries average the case values. No metric with k > 3 is estimated.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Prot.</td><td colspan="3">Pass@k</td><td colspan="3">pass^k</td></tr><tr><td>k=1</td><td>k=2</td><td>k=3</td><td>k=1</td><td>k=2</td><td>k=3</td></tr><tr><td>Gemini 3.1 Pro</td><td>S</td><td>22.1</td><td>33.1</td><td>38.7</td><td>22.1</td><td>11.2</td><td>5.9</td></tr><tr><td>Gemini 3.1 Pro</td><td>R</td><td>14.0</td><td>24.4</td><td>32.8</td><td>14.0</td><td>3.6</td><td>1.7</td></tr><tr><td>GPT-5.5</td><td>S</td><td>9.0</td><td>14.0</td><td>17.6</td><td>9.0</td><td>3.9</td><td>2.5</td></tr><tr><td>GPT-5.5</td><td>R</td><td>3.4</td><td>5.3</td><td>6.7</td><td>3.4</td><td>1.4</td><td>0.8</td></tr><tr><td>Qwen 3.7 Max</td><td>S</td><td>6.2</td><td>10.4</td><td>13.4</td><td>6.2</td><td>2.0</td><td>0.8</td></tr><tr><td>Qwen 3.7 Max</td><td>R</td><td>2.8</td><td>4.5</td><td>5.9</td><td>2.8</td><td>1.1</td><td>0.8</td></tr><tr><td>Claude Opus 4.7</td><td>S</td><td>5.3</td><td>8.7</td><td>10.9</td><td>5.3</td><td>2.0</td><td>0.8</td></tr><tr><td>Claude Opus 4.7</td><td>R</td><td>0.8</td><td>1.4</td><td>1.7</td><td>0.8</td><td>0.3</td><td>0.0</td></tr><tr><td>DeepSeek V4 Pro</td><td>S</td><td>2.8</td><td>3.9</td><td>5.0</td><td>2.8</td><td>1.7</td><td>1.7</td></tr><tr><td>DeepSeek V4 Pro</td><td>R</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Seed 2.0 Pro</td><td>S</td><td>0.8</td><td>1.7</td><td>2.5</td><td>0.8</td><td>0.0</td><td>0.0</td></tr><tr><td>Seed 2.0 Pro</td><td>R</td><td>0.6</td><td>1.1</td><td>1.7</td><td>0.6</td><td>0.0</td><td>0.0</td></tr><tr><td>Kimi K2.5</td><td>S</td><td>0.8</td><td>1.4</td><td>1.7</td><td>0.8</td><td>0.3</td><td>0.0</td></tr><tr><td>Kimi K2.5</td><td>R</td><td>0.3</td><td>0.6</td><td>0.8</td><td>0.3</td><td>0.0</td><td>0.0</td></tr><tr><td>Mistral Large 3</td><td>S</td><td>0.8</td><td>0.8</td><td>0.8</td><td>0.8</td><td>0.8</td><td>0.8</td></tr><tr><td>Mistral Large 3</td><td>R</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>MiniMax M2.5</td><td>S</td><td>0.8</td><td>0.8</td><td>0.8</td><td>0.8</td><td>0.8</td><td>0.8</td></tr><tr><td>MiniMax M2.5</td><td>R</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>DeepSeek R1</td><td>S</td><td>0.6</td><td>1.1</td><td>1.7</td><td>0.6</td><td>0.0</td><td>0.0</td></tr><tr><td>DeepSeek R1</td><td>R</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>GPT-OSS 120B</td><td>S</td><td>0.3</td><td>0.6</td><td>0.8</td><td>0.3</td><td>0.0</td><td>0.0</td></tr><tr><td>GPT-OSS 120B</td><td>R</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>DeepSeek V3.2</td><td>S</td><td>0.3</td><td>0.6</td><td>0.8</td><td>0.3</td><td>0.0</td><td>0.0</td></tr><tr><td>DeepSeek V3.2</td><td>R</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr></table>

## D.1 Invalid-Action Losses

Table 5 reports the share of trials that end in an invalid action or a missing command. These forfeits do not track strength in either direction: Qwen 3.7 Max wins more Sighted trials than most models yet loses 25.5% of them this way, whereas several models rarely forfeit and also rarely win.

Table 5 Invalid/no-command loss rate (%) by observation setting. Check feedback is affected as described in Appendix A.5. Records classified as aborted or unfinished are excluded before selecting three scored records per cell; each denominator is 357 scored trials.

<table><tr><td>Model</td><td>Sighted</td><td>Restricted-observation</td></tr><tr><td>Gemini 3.1 Pro</td><td>2.2</td><td>0.3</td></tr><tr><td>GPT-5.5</td><td>0.0</td><td>0.6</td></tr><tr><td>Qwen 3.7 Max</td><td>25.5</td><td>3.4</td></tr><tr><td>Claude Opus 4.7</td><td>0.0</td><td>1.1</td></tr><tr><td>DeepSeek V4 Pro</td><td>0.0</td><td>13.2</td></tr><tr><td>Seed 2.0 Pro</td><td>0.0</td><td>25.2</td></tr><tr><td>Kimi K2.5</td><td>0.0</td><td>19.6</td></tr><tr><td>Mistral Large 3</td><td>1.1</td><td>93.8</td></tr><tr><td>MiniMax M2.5</td><td>0.8</td><td>93.8</td></tr><tr><td>DeepSeek R1</td><td>2.2</td><td>25.2</td></tr><tr><td>GPT-OSS 120B</td><td>0.8</td><td>31.9</td></tr><tr><td>DeepSeek V3.2</td><td>0.0</td><td>23.5</td></tr></table>

## D.2 Repeated Trials

![](images/bfbf7112060f8728b6ee0188f286e225a05847fa7c635a33e850c406d5774375.jpg)  
Figure 9 Reach and reliability diverge with k. Pass@k (solid) and pass^k (dashed) for k = 1, 2, 3, for the five models with the highest Sighted pass@1, computed from the same three trials per case.

For Gemini 3.1 Pro under Sighted observation, pass@k rises from 22.1% to 33.1% and 38.7% for k = 1, 2, 3, while pass^3 is 5.9% (Figure 9). Three trials do not support extrapolation to larger budgets.

## D.3 Evaluation Set Size

We recompute all metrics on nested subsets of m cases, keeping three trials per case (Figure 10). Across 1,000 random case orderings, the 2.5th–97.5th percentile range of Gemini 3.1 Pro's Sighted pass@1 narrows from 3.3–43.3% at m = 10 to 19.7–24.3% at m = 100, against 22.1% on all 119 cases. pass $^{3}$ is the most fragile metric: at m = 10 it is zero in 50.6% of Gemini's and 75.5% of GPT-5.5's Sighted subsets. These ranges describe variation within the corpus, not confidence intervals for new positions.

![](images/c53db5c29dd5fb20d221ce44b4e719502c8699b42020fa2bbb2f193b4a20df9c.jpg)  
Figure 10 Estimates on nested case subsets. Pass@1 (solid), pass@3 (dashed), and pass^3 (dotted) for $m = 10, 20, 30, 40, 60, 80, 100, 119$ cases under one shared random ordering; the full set reproduces the main results.

## E Observation Ablation

Restricted observation changes more than what the agent sees: it removes state refreshes and the query commands, but it also uses a different system prompt (Appendix A.3). To separate these factors, we ran a $2 \times 2$ ablation after the main evaluation, with a correct check field and a single endgame prompt whose rules and move-format sections match the Sighted prompt; only the paragraphs describing observation and commands vary. The factors are state refreshes (board, FEN, and legal moves attached to every message) and query commands (view\_board, legal\_moves, simulate). The Full arm has both, No queries keeps only the refreshes, No refresh keeps only the commands, and Neither removes both. Without refreshes, the board and FEN are shown once and later messages report only the move, capture, check status, and material. Defender, budgets, and scoring match the main evaluation.

We evaluate Gemini 3.1 Pro and GPT-5.5 on positions each won at least once under Sighted observation, since a never-solved position scores zero in every arm: 20 of Gemini's 46, drawn by mate distance with a fixed seed, and all 21 of GPT-5.5's. Each model-position-arm cell has three scored trials, with trials lost to API errors rerun. GPT-5.5 was served through a different API gateway than in the main evaluation, with the same model version. Because positions were selected on main-evaluation wins, we compare arms only within the ablation.

Table 6 Observation ablation. A $2 \times 2$ design crossing per-step state refreshes with the query commands (view\_board, legal\_moves, simulate); all arms share one system prompt and a correct check field. Pass@1 is over three trials per position; Solved counts positions won at least once; Rej. is the share of move attempts rejected; the last three columns give query calls per agent turn. For reference, the main evaluation on the same positions gives 51.7% (Sighted) and 20.0% (Restricted observation) for Gemini 3.1 Pro and 50.8% and 9.5% for GPT-5.5; the Sighted rates are biased upward because positions were selected on Sighted wins.

<table><tr><td rowspan="2">Arm</td><td colspan="2">Factors</td><td rowspan="2">Pass@1 (%)</td><td rowspan="2">Solved</td><td rowspan="2">Rej. (%)</td><td colspan="3">Queries per turn</td></tr><tr><td>Refresh</td><td>Queries</td><td>View</td><td>Legal</td><td>Sim.</td></tr><tr><td colspan="9">Gemini 3.1 Pro, 20 positions</td></tr><tr><td>Full</td><td>✓</td><td>✓</td><td>48.3</td><td>17/20</td><td>1.7</td><td>0.00</td><td>0.00</td><td>0.75</td></tr><tr><td>No queries</td><td>✓</td><td>✘</td><td>33.3</td><td>11/20</td><td>0.7</td><td>-</td><td>-</td><td>-</td></tr><tr><td>No refresh</td><td>✘</td><td>✓</td><td>35.0</td><td>14/20</td><td>4.0</td><td>0.32</td><td>0.19</td><td>1.00</td></tr><tr><td>Neither</td><td>✘</td><td>✘</td><td>41.7</td><td>15/20</td><td>6.9</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="9">GPT-5.5, 21 positions</td></tr><tr><td>Full</td><td>✓</td><td>✓</td><td>39.7</td><td>15/21</td><td>0.3</td><td>0.01</td><td>0.05</td><td>0.98</td></tr><tr><td>No queries</td><td>✓</td><td>✘</td><td>25.4</td><td>10/21</td><td>1.9</td><td>-</td><td>-</td><td>-</td></tr><tr><td>No refresh</td><td>✘</td><td>✓</td><td>34.9</td><td>13/21</td><td>2.0</td><td>0.38</td><td>0.67</td><td>1.03</td></tr><tr><td>Neither</td><td>✘</td><td>✘</td><td>38.1</td><td>15/21</td><td>3.6</td><td>-</td><td>-</td><td>-</td></tr></table>

Intervals below are 95% case-bootstrap intervals and p-values are exact sign tests over positions. (i) Removing the query commands costs both models about 15 points (Full minus No queries: 15.0 for Gemini, [0.0, 30.0]; 14.3 for GPT-5.5, [1.6, 23.8], p = 0.013); the agents lose simulation, which they use on most turns. (ii) Removing the refreshes while keeping the commands costs Gemini 13.3 points ([0.0, 25.0], p = 0.039) and GPT-5.5 4.8 ( $[-11.1, 19.0]$ ); both then query the state, which they almost never do when it is refreshed. (iii) With neither, the two models win 41.7% and 38.1% of trials, close to Full (differences of 6.7 and 1.6 points, intervals spanning zero) and far above the 20.0% and 9.5% they reach on the same positions under Restricted observation. Without refreshes, removing the commands does not lower pass@1 further, so the command effect in (i) appears only when the state is refreshed; with three trials per position we do not interpret this interaction further. Most of the Restricted drop in Section 5 therefore reflects the setting as a whole, including its prompt and feedback, rather than the withheld board alone.

## F Behavioral Diagnostics

The analyses in this section are descriptive. Models, positions, and decision states vary together, so the contrasts below characterize behavior without identifying its causes.

## F.1 Setting and Failure Profiles

![](images/57d116fcc3a3eea8a093baf9e64b70bda0e8c08eff558b1195c9ad4d35cb9c26.jpg)

(b) Invalid/no-command loss  
![](images/a6bff141e49cb7e2cbb48812e46aa702bb91d1fea1d7348a2dc13065e6bc1990.jpg)  
Figure 11 The setting difference varies across models. (a) Pass@1; (b) rate of terminal invalid-action or no-command losses, each over 357 trials per model and setting.

Figure 8 in Section 6.3 shows game length. No win ends before the stored mate distance, and the median failed trial lasts six plies under Sighted and four under Restricted observation. Most failures are lost on the board (76.1% Sighted, 66.0% Restricted). Resignation ends 18.4% of failed Sighted trials (40.3% for Gemini 3.1 Pro) but only 3.2% of Restricted ones, where action forfeits end 28.2%; Mistral Large 3 and MiniMax M2.5 forfeit 310 and 313 of their 357 Restricted trials before playing a move.

![](images/ee6ad9bb3e4a7dfef7d92b2416aeba61cae46e1b7a456c4b68eeb3dd7ec3b192.jpg)  
Figure 12 Action-level failures concentrate under Restricted observation. Left: terminal action losses (invalid action, no command, or action limit). Right: trajectories with any rejected move, which may be followed by recovery.

## F.2 Simulation and First-Move Choices

Tables 7–10 break down simulation by setting, model, request length, and turn pattern, and Table 11 separates terminal action losses from recoverable rejections. GPT-5.5 mostly submits one-ply simulations, whereas several other models submit multi-ply lines. Complete-line legality falls with length, from 96.3% (2,496/2,592) for one ply to 47.6% (1,110/2,332) for three or four plies and 37.4% (357/955) for five to eight, although length, model, and position vary together. Tables 12–15 profile the first real move against the uniform baseline over legal moves; for models that often forfeit early under Restricted observation, these profiles condition on trajectories with an accepted first move.

![](images/81dd9a0b4e188e5e46237fadf8b3843c1ab1f879bef25941727ad4de0f0f642e.jpg)

![](images/432d5bc7e579c737f44959f18651d78487c45ada77d8faacbbcb58130bb74dec.jpg)  
Figure 13 Simulation use varies widely across models. (a) Trajectories that request simulation on the first turn (357 per model). (b) Simulation requests, accepted calls, and complete legal lines per agent turn over 15,318 agent turns; requests include budget-rejected calls.

Table 7 Explicit simulation use by protocol. All entries count trajectories. Behavior rows overlap; acceptance of a simulation call does not imply full-line legality.

<table><tr><td>Trajectory measure</td><td>Sighted</td><td>Restricted-observation</td></tr><tr><td>Trajectory denominator</td><td>4284</td><td>4284</td></tr><tr><td>Requested simulation</td><td>3098</td><td>0</td></tr><tr><td>At least one accepted simulation call</td><td>3098</td><td>0</td></tr><tr><td>At least one full legal requested line</td><td>2155</td><td>0</td></tr><tr><td>At least one mid-line invalid step</td><td>1639</td><td>0</td></tr><tr><td>At least one capped request</td><td>32</td><td>0</td></tr><tr><td>Only budget/protocol rejected</td><td>0</td><td>0</td></tr></table>

Table 8 Sighted explicit simulation use by model. Any and Full count trajectories out of 357: any request and at least one full legal original request, respectively. Accepted and Rejected count calls; all rejections here are budget rejections.

<table><tr><td>Model</td><td>Any</td><td>Full</td><td>Accepted</td><td>Rejected</td></tr><tr><td>Gemini 3.1 Pro</td><td>332</td><td>299</td><td>1081</td><td>675</td></tr><tr><td>GPT-5.5</td><td>356</td><td>353</td><td>1727</td><td>191</td></tr><tr><td>Qwen 3.7 Max</td><td>290</td><td>188</td><td>360</td><td>284</td></tr><tr><td>Claude Opus 4.7</td><td>357</td><td>241</td><td>595</td><td>529</td></tr><tr><td>DeepSeek V4 Pro</td><td>354</td><td>266</td><td>810</td><td>1006</td></tr><tr><td>Seed 2.0 Pro</td><td>328</td><td>222</td><td>426</td><td>359</td></tr><tr><td>Kimi K2.5</td><td>156</td><td>76</td><td>180</td><td>148</td></tr><tr><td>GPT-OSS 120B</td><td>63</td><td>33</td><td>70</td><td>7</td></tr><tr><td>DeepSeek R1</td><td>8</td><td>2</td><td>9</td><td>2</td></tr><tr><td>DeepSeek V3.2</td><td>352</td><td>292</td><td>890</td><td>835</td></tr><tr><td>Mistral Large 3</td><td>213</td><td>14</td><td>242</td><td>212</td></tr><tr><td>MiniMax M2.5</td><td>289</td><td>169</td><td>359</td><td>345</td></tr></table>

Table 9 Original request length among accepted Sighted simulation calls. Length is measured in plies. Two or more plies include model-supplied assumed opponent replies, not engine search. Length and position differences prevent interpreting legality rates as planning ability.

<table><tr><td>Model</td><td>Median</td><td>Mean</td><td>1 ply (%)</td><td>≥2 plies (%)</td></tr><tr><td>Gemini 3.1 Pro</td><td>3</td><td>3.57</td><td>12.5</td><td>87.5</td></tr><tr><td>GPT-5.5</td><td>1</td><td>1.52</td><td>71.9</td><td>28.1</td></tr><tr><td>Qwen 3.7 Max</td><td>4</td><td>4.20</td><td>9.2</td><td>90.8</td></tr><tr><td>Claude Opus 4.7</td><td>3</td><td>3.38</td><td>5.5</td><td>94.5</td></tr><tr><td>DeepSeek V4 Pro</td><td>3</td><td>3.12</td><td>27.0</td><td>73.0</td></tr><tr><td>Seed 2.0 Pro</td><td>2</td><td>2.20</td><td>46.7</td><td>53.3</td></tr><tr><td>Kimi K2.5</td><td>3</td><td>3.33</td><td>24.4</td><td>75.6</td></tr><tr><td>GPT-OSS 120B</td><td>1.5</td><td>1.86</td><td>50.0</td><td>50.0</td></tr><tr><td>DeepSeek R1</td><td>3</td><td>3.44</td><td>22.2</td><td>77.8</td></tr><tr><td>DeepSeek V3.2</td><td>2</td><td>1.92</td><td>48.4</td><td>51.6</td></tr><tr><td>Mistral Large 3</td><td>3</td><td>3.93</td><td>3.7</td><td>96.3</td></tr><tr><td>MiniMax M2.5</td><td>1</td><td>1.68</td><td>58.5</td><td>41.5</td></tr></table>

Table 10 Mutually exclusive agent-turn patterns, assigned by the first applicable row in the displayed priority order. Entries are counts (percent), with 15,318 Sighted and 12,387 Restricted-observation agent turns. Recovery means only that a real move followed a rejected or aborted simulation, not that the strategy improved. Direct execution excludes explicit auxiliary actions but does not imply the absence of automatically supplied state information.

<table><tr><td>Turn pattern (priority order)</td><td>Sighted n (%)</td><td>Restricted-observation n (%)</td></tr><tr><td>No executed move</td><td>898 (5.9)</td><td>1,319 (10.6)</td></tr><tr><td>Simulation rejected/aborted, then recovery</td><td>4,386 (28.6)</td><td>0 (0.0)</td></tr><tr><td>Simulate, then adopt tested root</td><td>1,447 (9.4)</td><td>0 (0.0)</td></tr><tr><td>Simulate, then choose different root</td><td>774 (5.1)</td><td>0 (0.0)</td></tr><tr><td>Observe, then move</td><td>234 (1.5)</td><td>15 (0.1)</td></tr><tr><td>Direct execution</td><td>6,988 (45.6)</td><td>10,581 (85.4)</td></tr><tr><td>Other explicit action, then move</td><td>591 (3.9)</td><td>472 (3.8)</td></tr></table>

Table 11 Mutually exclusive observed failure classes. Opponent terminal collects the remaining failures; on replay, each ends with the attacker checkmated or stalemated. Classes follow the released long-check rule (Appendix A.5).

<table><tr><td>Observed failure class</td><td>Sighted</td><td>Restricted-observation</td></tr><tr><td>All observed failures</td><td>4106</td><td>4206</td></tr><tr><td>Invalid-action limit</td><td>12</td><td>1178</td></tr><tr><td>No command/tool call</td><td>105</td><td>6</td></tr><tr><td>Perpetual check by the attacker</td><td>16</td><td>7</td></tr><tr><td>Repetition draw</td><td>39</td><td>56</td></tr><tr><td>Action limit</td><td>25</td><td>0</td></tr><tr><td>Turn cap</td><td>29</td><td>47</td></tr><tr><td>Resignation</td><td>756</td><td>135</td></tr><tr><td>Opponent terminal (attacker cannot move)</td><td>3124</td><td>2777</td></tr></table>

Table 12 Root-action composition in the Sighted setting. Entries are observed percent (percentage-point difference from uniform choice among true legal root moves). For each model, the reference uses exactly the cases with an accepted first move. Check and capture overlap; quiet means neither. These are action patterns, not personality labels.

<table><tr><td>Model</td><td>Accepted</td><td>Check</td><td>Capture</td><td>Quiet</td></tr><tr><td>Gemini 3.1 Pro</td><td>353</td><td>95.5 (+84.8)</td><td>25.5 (+18.7)</td><td>4.2 (-80.8)</td></tr><tr><td>GPT-5.5</td><td>357</td><td>95.8 (+85.1)</td><td>25.5 (+18.7)</td><td>3.6 (-81.3)</td></tr><tr><td>Qwen 3.7 Max</td><td>287</td><td>88.9 (+77.9)</td><td>27.2 (+20.3)</td><td>8.7 (-76.0)</td></tr><tr><td>Claude Opus 4.7</td><td>356</td><td>84.6 (+73.8)</td><td>29.8 (+23.0)</td><td>10.7 (-74.3)</td></tr><tr><td>DeepSeek V4 Pro</td><td>350</td><td>81.4 (+70.7)</td><td>27.7 (+21.0)</td><td>14.6 (-70.4)</td></tr><tr><td>Seed 2.0 Pro</td><td>357</td><td>88.0 (+77.2)</td><td>33.3 (+26.6)</td><td>7.3 (-77.7)</td></tr><tr><td>Kimi K2.5</td><td>357</td><td>72.8 (+62.1)</td><td>26.3 (+19.5)</td><td>19.9 (-65.1)</td></tr><tr><td>GPT-OSS 120B</td><td>352</td><td>71.9 (+61.2)</td><td>20.7 (+14.0)</td><td>22.2 (-62.8)</td></tr><tr><td>DeepSeek R1</td><td>352</td><td>51.4 (+40.7)</td><td>34.7 (+27.8)</td><td>34.7 (-50.3)</td></tr><tr><td>DeepSeek V3.2</td><td>355</td><td>70.4 (+59.7)</td><td>41.1 (+34.3)</td><td>15.8 (-69.2)</td></tr><tr><td>Mistral Large 3</td><td>355</td><td>37.5 (+26.7)</td><td>25.6 (+18.8)</td><td>46.5 (-38.5)</td></tr><tr><td>MiniMax M2.5</td><td>344</td><td>34.9 (+24.2)</td><td>27.0 (+20.3)</td><td>52.6 (-32.4)</td></tr></table>

Table 13 Root piece-choice differences (percentage points) in the Sighted setting. Differences are relative to uniform legal choice on the same accepted-first-move case distribution. Other comprises general, advisor, and elephant. Intervals are omitted for readability, and no multiple-comparison correction is applied. These exploratory contrasts are not evidence of piece-specific playing strength.

<table><tr><td>Model</td><td>Rook</td><td>Cannon</td><td>Horse</td><td>Pawn</td><td>Other</td></tr><tr><td>Gemini 3.1 Pro</td><td>+10.9</td><td>-22.5</td><td>+3.2</td><td>+10.8</td><td>-2.5</td></tr><tr><td>GPT-5.5</td><td>+16.1</td><td>-17.5</td><td>-5.7</td><td>+8.7</td><td>-1.6</td></tr><tr><td>Qwen 3.7 Max</td><td>+4.7</td><td>-17.8</td><td>+6.2</td><td>+8.8</td><td>-1.9</td></tr><tr><td>Claude Opus 4.7</td><td>+4.7</td><td>-22.2</td><td>+11.2</td><td>+9.3</td><td>-3.0</td></tr><tr><td>DeepSeek V4 Pro</td><td>+11.7</td><td>-21.9</td><td>+7.7</td><td>+4.5</td><td>-1.9</td></tr><tr><td>Seed 2.0 Pro</td><td>+24.2</td><td>-27.3</td><td>-4.0</td><td>+9.9</td><td>-2.7</td></tr><tr><td>Kimi K2.5</td><td>+4.3</td><td>-16.9</td><td>+6.9</td><td>+8.7</td><td>-3.0</td></tr><tr><td>GPT-OSS 120B</td><td>+16.5</td><td>-16.6</td><td>+5.4</td><td>-2.7</td><td>-2.6</td></tr><tr><td>DeepSeek R1</td><td>+17.8</td><td>-22.3</td><td>+2.1</td><td>+3.5</td><td>-1.1</td></tr><tr><td>DeepSeek V3.2</td><td>+11.8</td><td>-21.6</td><td>+2.5</td><td>+9.9</td><td>-2.7</td></tr><tr><td>Mistral Large 3</td><td>+1.7</td><td>-12.0</td><td>+14.7</td><td>-1.6</td><td>-2.8</td></tr><tr><td>MiniMax M2.5</td><td>+12.0</td><td>-25.8</td><td>+11.9</td><td>+3.2</td><td>-1.4</td></tr></table>

Table 14 Root-action composition in the Restricted-observation setting. Entries are observed percent (percentage-point difference from uniform choice among true legal root moves). For each model, the reference uses exactly the cases with an accepted first move. Check and capture overlap; quiet means neither. These are action patterns, not personality labels.

<table><tr><td>Model</td><td>Accepted</td><td>Check</td><td>Capture</td><td>Quiet</td></tr><tr><td>Gemini 3.1 Pro</td><td>356</td><td>82.6 (+71.9)</td><td>26.7 (+19.9)</td><td>10.4 (-74.6)</td></tr><tr><td>GPT-5.5</td><td>357</td><td>72.5 (+61.8)</td><td>37.8 (+31.0)</td><td>13.7 (-71.3)</td></tr><tr><td>Qwen 3.7 Max</td><td>352</td><td>69.9 (+59.1)</td><td>26.7 (+19.9)</td><td>18.8 (-66.2)</td></tr><tr><td>Claude Opus 4.7</td><td>355</td><td>58.9 (+48.1)</td><td>45.1 (+38.3)</td><td>19.7 (-65.2)</td></tr><tr><td>DeepSeek V4 Pro</td><td>341</td><td>43.4 (+32.7)</td><td>36.1 (+29.3)</td><td>34.0 (-50.9)</td></tr><tr><td>Seed 2.0 Pro</td><td>334</td><td>55.7 (+45.0)</td><td>35.3 (+28.7)</td><td>32.0 (-53.0)</td></tr><tr><td>Kimi K2.5</td><td>340</td><td>52.1 (+41.4)</td><td>46.5 (+39.6)</td><td>25.3 (-59.7)</td></tr><tr><td>GPT-OSS 120B</td><td>320</td><td>61.3 (+50.7)</td><td>41.9 (+35.0)</td><td>24.1 (-61.0)</td></tr><tr><td>DeepSeek R1</td><td>325</td><td>51.7 (+40.8)</td><td>42.5 (+35.7)</td><td>28.3 (-56.5)</td></tr><tr><td>DeepSeek V3.2</td><td>303</td><td>36.0 (+25.3)</td><td>29.4 (+22.9)</td><td>50.2 (-35.1)</td></tr><tr><td>Mistral Large 3</td><td>47</td><td>12.8 (+5.1)</td><td>0.0 (-3.7)</td><td>87.2 (-2.7)</td></tr><tr><td>MiniMax M2.5</td><td>44</td><td>27.3 (+18.9)</td><td>4.5 (-0.5)</td><td>72.7 (-15.7)</td></tr></table>

Table 15 Root piece-choice differences (percentage points) in the Restricted-observation setting. Differences are relative to uniform legal choice on the same accepted-first-move case distribution. Other comprises general, advisor, and elephant. Intervals are omitted for readability, and no multiple-comparison correction is applied. These exploratory contrasts are not evidence of piece-specific playing strength.

<table><tr><td>Model</td><td>Rook</td><td>Cannon</td><td>Horse</td><td>Pawn</td><td>Other</td></tr><tr><td>Gemini 3.1 Pro</td><td>+11.4</td><td>-19.4</td><td>+2.5</td><td>+7.6</td><td>-2.2</td></tr><tr><td>GPT-5.5</td><td>+11.6</td><td>-17.5</td><td>-1.2</td><td>+8.2</td><td>-1.1</td></tr><tr><td>Qwen 3.7 Max</td><td>+0.5</td><td>-15.5</td><td>+5.2</td><td>+9.2</td><td>+0.6</td></tr><tr><td>Claude Opus 4.7</td><td>+5.4</td><td>-27.8</td><td>+10.4</td><td>+13.0</td><td>-1.1</td></tr><tr><td>DeepSeek V4 Pro</td><td>+7.7</td><td>-19.5</td><td>+4.4</td><td>+6.3</td><td>+1.1</td></tr><tr><td>Seed 2.0 Pro</td><td>+18.7</td><td>-16.1</td><td>-6.8</td><td>+7.0</td><td>-2.8</td></tr><tr><td>Kimi K2.5</td><td>+4.9</td><td>-21.5</td><td>+4.3</td><td>+13.8</td><td>-1.6</td></tr><tr><td>GPT-OSS 120B</td><td>+18.9</td><td>-24.1</td><td>-0.5</td><td>+7.6</td><td>-1.8</td></tr><tr><td>DeepSeek R1</td><td>+25.2</td><td>-25.6</td><td>-1.3</td><td>+2.8</td><td>-1.1</td></tr><tr><td>DeepSeek V3.2</td><td>+7.0</td><td>-22.6</td><td>+8.4</td><td>+2.4</td><td>+4.8</td></tr><tr><td>Mistral Large 3</td><td>-2.3</td><td>+5.9</td><td>-5.5</td><td>+3.9</td><td>-1.9</td></tr><tr><td>MiniMax M2.5</td><td>+1.2</td><td>-4.4</td><td>-1.0</td><td>+2.2</td><td>+2.0</td></tr></table>

## F.3 REPL-Call Composition by Setting, Outcome, and Model

Figures 14–17 apply the construction of Figure 6 by setting and outcome, to the four Sighted models with the most wins, and to every model in each setting, excluding environment-generated forfeit records.

![](images/48e11635451c7e383981f0427c9f7b83fd0be3371a15fbd5353ab2f5ee1202c4.jpg)  
(d) Restricted observation, all $n = 4,284$ ; 4.2 calls/trial  
(b) Sighted, successful $n = 178$ ; 9.8 calls/trial

![](images/3586c5161097d10f83d6015284f875fe755e35e5eb3ee310862fabe696faab22.jpg)  
(c) Sighted, failed $n = 4,106$ ; 7.6 calls/trial

![](images/773d00a348981176fef1c78a20cd10e27f6492a07b686ac5ac29501743447d7b.jpg)

(e) Restricted observation, successful
n=78; 5.6 calls/trial  
(f) Restricted observation, failed n=4,206; 4.2 calls/trial  
![](images/64d65fd8d859a5bb7b256de8f8bfbe0b12b9c7cede2b4d5defa32b752ec843bc.jpg)  
Figure 14 REPL-call composition by setting and outcome, pooling all 12 models. Columns show all, successful, and failed trials.

![](images/37b2dd6530c6eaf55cb752c7e4558ec381035c19b13f04dc8ee997b6e0378a52.jpg)  
Figure 15 Sighted REPL-call composition by outcome for the four models with the most wins. Successful panels contain 19–79 trials.

![](images/ba6078f8b455c6511b344280d528457c6b8af65763446d1c14691cfc8b01638f.jpg)  
Figure 16 Sighted REPL-call composition for each model over all 357 trials.

![](images/b8e512f3d0ea7371e56761f01074a85537cc7b58da94a913026cf8d051a1ae82.jpg)  
Figure 17 Restricted-observation REPL-call composition for each model over all 357 trials. State queries here are get\_history calls, the only query available.

## F.4 Feedback and Error Recovery

![](images/eaf4653899899ec21fb34bfb32d4c5c3015e2f2220637a565d251cf9de95e287.jpg)  
Figure 18 Next action after selected feedback events. Each row conditions on one event and shows the next interaction in the same turn; for defender-reply mismatch, the next decision after the real reply. Requests include rejected ones, and percentages are rounded.

Figure 18 shows how agents respond to feedback. For recovery (Figure 19), a rejected real move counts as recovered if rules replay confirms a legal real move within the next one, two, or three action interactions of the same turn; reads, simulations, and think calls occupy window positions, and resignation and forfeit compete with recovery. Under Sighted observation, 69.4%, 83.1%, and 87.9% of the 758 rejections are recovered within one, two, and three interactions; under Restricted observation, 31.2%, 38.6%, and 38.9% of the 5,850 rejections are (three-interaction intervals [84.6, 90.9]% and [36.0, 42.1]%). The Restricted total is dominated by Mistral Large 3 and MiniMax M2.5, which contribute 1,052 and 1,049 rejections and recover from 4.3% and 4.6% of them. The four strongest models recover from at least 88% in both settings (Gemini 3.1 Pro 90.0% Sighted and 95.8% Restricted; GPT-5.5 100% and 93.3%; Claude Opus 4.7 100% and 88.1%; Qwen 3.7 Max 96.4% and 91.0%), though each has only 11–28 Sighted rejections.

![](images/a635bb9a1264dc5e1f13060d54731886f295a695e159d79c3482e4a266bdb2a0.jpg)  
Figure 19 Recovery after a rejected move, by model. Cumulative fraction of rejected moves followed by a legal real move within one, two, or three action interactions of the same turn; n gives Sighted/Restricted rejections. Shading: 95% case-bootstrap intervals (1,000 resamples).

## F.5 Piece Composition

Figure 20 and Tables 16–18 group cases by initial attacking material. Presence groups overlap, combination groups are mutually exclusive, and first-move counts reflect piece availability and trajectory length rather than piece preference.

![](images/7d684a6eb39ac51feb215ff43f9db7d32dc53021c980197c15a1a70919bbcc67.jpg)  
Figure 20 Outcomes by initial attacking pieces. Left: overlapping rook, cannon, horse, and pawn presence groups. Right: mutually exclusive combinations with at least three cases. Labels give case counts; points pool 12 models and three trials per case.

Table 16 Observed passes by initial red-piece presence. Rows overlap and must not be added. Position structure and difficulty confound comparisons; these are not causal piece effects.

<table><tr><td>Initial red piece</td><td>Cases</td><td>Sighted pass/ traj.</td><td>Restricted-observation pass/ traj.</td></tr><tr><td>Rook</td><td>99</td><td>158/3564</td><td>66/3564</td></tr><tr><td>Cannon</td><td>103</td><td>144/3708</td><td>62/3708</td></tr><tr><td>Horse</td><td>91</td><td>140/3276</td><td>52/3276</td></tr><tr><td>Pawn</td><td>71</td><td>107/2556</td><td>53/2556</td></tr></table>

Table 17 Mutually exclusive initial red-piece presence combinations. Groups with fewer than five cases are retained for coverage, not model rankings.

<table><tr><td>Combination</td><td>Cases</td></tr><tr><td>rook+cannon+horse+pawn</td><td>34</td></tr><tr><td>rook+cannon+horse</td><td>33</td></tr><tr><td>rook+cannon+pawn</td><td>16</td></tr><tr><td>cannon+horse+pawn</td><td>7</td></tr><tr><td>rook+cannon</td><td>6</td></tr><tr><td>horse+pawn</td><td>5</td></tr><tr><td>rook+horse</td><td>5</td></tr><tr><td>rook+horse+pawn</td><td>4</td></tr><tr><td>cannon+horse</td><td>3</td></tr><tr><td>cannon+pawn</td><td>3</td></tr><tr><td>cannon</td><td>1</td></tr><tr><td>pawn</td><td>1</td></tr><tr><td>rook+pawn</td><td>1</td></tr></table>

Table 18 Piece type of the first accepted non-synthetic agent move, over 4,284 trajectories per setting. Other comprises general, advisor, and elephant; unknown denotes unparsed feedback. This is action composition, not position-independent preference.

<table><tr><td>First moved piece</td><td>Sighted</td><td>Restricted-observation</td></tr><tr><td>Rook</td><td>2032</td><td>1665</td></tr><tr><td>Cannon</td><td>705</td><td>588</td></tr><tr><td>Horse</td><td>851</td><td>608</td></tr><tr><td>Pawn</td><td>556</td><td>525</td></tr><tr><td>Other</td><td>31</td><td>88</td></tr><tr><td>Unknown</td><td>0</td><td>0</td></tr><tr><td>No Accepted First Move</td><td>109</td><td>810</td></tr></table>

## G Representative Trajectories

The examples below are selected by fixed, deterministic rules over the retained trajectories. They illustrate how play unfolds; they are not estimates of model tendencies.

![](images/6e3c266ecfa059b593cbd05770f71b61e3a0ef6243f76b20dc316f04f701ef35.jpg)  
Figure 21 Forced mate on SQYQ-251. Panels A–C show the real position before each of Red's three moves (solid arrow: move played; dashed outline: its destination). Black's replies e7f7 and h9f8 are forced. Panel D shows the final position: the cannon on f9 mates through the Black horse on f8 while the Red horse on g6 covers e7. The archived feedback labelled this terminal a stalemate; rules replay confirms checkmate.

SQYQ-251: two routes to the same mate. The solution is a forced mate in five plies (Figure 21). After h4g6, the Red horse controls e7 and protects f8. The rook sacrifice b8f8 forces h9f8; the capturing horse opens the cannon's line from i9 and becomes the screen for i9f9. Qwen 3.7 Max (Sighted, trial 2) states this plan, simulates the full line h4g6 e7f7 b8f8 h9f8 i9f9 at its first decision, and the real game follows it exactly. Gemini 3.1 Pro (Sighted, trial 3; Figure 1) simulates only after seeing the defender's first reply, checking b8f8 h9f8 i9f9 at its second decision before playing it. On the same position, DeepSeek V4 Pro reaches the prefix h4g6 e7f7 and then plays b8e8, stating that the move gives check. It does not; Black answers h7h0 and mates. The false claim is checkable against the rules, although we do not identify b8e8 as

the engine-determined losing move.

![](images/c47ed77d4a743d7adb61a17da231da5422fed53b5d33e5da6577470784dbeb20.jpg)  
Figure 22 Same model and position, different first moves. Two Sighted Gemini 3.1 Pro trials on SQYQ-012. Trial 1 simulates a legal five-ply line, plays g7e7, and wins; trial 2 simulates a legal four-ply line, plays d8e8, and resigns. Solid arrows mark first moves; dashed outlines mark their destinations.

SQYQ-012: legal lines, different outcomes. Both Gemini 3.1 Pro trials simulate a fully legal line before moving (Figure 22). Trial 1 checks g7e7 c5e7 d8e8 e9e8 c0c8, plays g7e7, and later wins through d8e8 and d7b8; trial 2 checks d8e8 e9e8 g8f8 e8d8, plays d8e8, and eventually resigns. Legality of the simulated line does not separate the two outcomes.

![](images/78a03969eaab91e28a457044ea02afaa116d25755cb129578dd3fef095f215db.jpg)  
Figure 23 Different models from the same state on JH-327. After the shared prefix h2h9 g9g8, DeepSeek V4 Pro (trial 3) simulates a fully legal three-ply line and plays i2d2 (win); Kimi K2.5 (trial 2) plays i2i9 without simulating (loss). Arrows and outlines follow Figure 22; simulation details appear in each panel.

JH-327: same state, different models. From an identical second decision, the model that simulates wins and the one that does not lose (Figure 23). A single pair cannot attribute the outcome to simulation;

Table 19 lists the selected comparisons.

Table 19 Same-case trajectory contrasts selected by fixed rules. C1, largest same-case success spread between models; C2, same model and first move, different outcomes; C3, Restricted-observation invalid-action losses versus a peer; C4, explicit first-turn tool use versus none; C5, rejected moves recovered within the turn versus none rejected; C6, different first-moved piece, different outcomes. Trial k is the k-th of the three trials in a model-case-setting cell. Root move is the stored reference first move; Played lists up to four agent moves. B/L count successful/attempted explicit board and legal-move calls; S counts accepted/attempted simulation calls, not fully legal lines. Terminal reasons follow the rules replay of Appendix A.5. These purposefully selected examples do not estimate representative effects or internal reasoning.

<table><tr><td>Contrast; model; setting</td><td>Root move; agent moves</td><td>Outcome; actions</td></tr><tr><td>C1a JH-084</td><td>Root move: g5g9.</td><td>Pass; checkmate.</td></tr><tr><td>Gemini 3.1 Pro, trial 1 Sighted</td><td>Played: g5g9, h5h9</td><td>Invalid: 0. B/L/S: 0/0, 0/0, 0/0.</td></tr><tr><td>C1b JH-084</td><td>Root move: g5g9.</td><td>Fail; resignation.</td></tr><tr><td>GPT-OSS 120B, trial 1 Sighted</td><td>Played: g5d5, e0d0, d0e0, e0d0, ...</td><td>Invalid: 3. B/L/S: 1/1, 1/1, 0/0.</td></tr><tr><td>C2a SQYQ-066</td><td>Root move: g8g9.</td><td>Pass; checkmate.</td></tr><tr><td>Gemini 3.1 Pro, trial 2 Sighted</td><td>Played: g8g9, d8e8, g9g8, i5g6, ...</td><td>Invalid: 0. B/L/S: 0/0, 0/0, 4/7.</td></tr><tr><td>C2b SQYQ-066</td><td>Root move: g8g9.</td><td>Fail; attacker checkmated.</td></tr><tr><td>Gemini 3.1 Pro, trial 1 Sighted</td><td>Played: g8g9, d8d9, g9g8, i5g6, ...</td><td>Invalid: 0. B/L/S: 0/0, 0/0, 8/18.</td></tr><tr><td>C3a SQYQ-062</td><td>Root move: f8e8.</td><td>Fail; invalid-action limit.</td></tr><tr><td>DeepSeek R1, trial 2 Restricted observation</td><td>Played: a5e5, e5e8, b8c6</td><td>Invalid: 5. B/L/S: 0/0, 0/0, 0/0.</td></tr><tr><td>C3b SQYQ-062</td><td>Root move: f8e8.</td><td>Fail; attacker checkmated.</td></tr><tr><td>Gemini 3.1 Pro, trial 1 Restricted observation</td><td>Played: c8c9, a5e5, e2g0</td><td>Invalid: 1. B/L/S: 0/0, 0/0, 0/0.</td></tr><tr><td>C4a JH-327</td><td>Root move: h2h9.</td><td>Pass; checkmate.</td></tr><tr><td>DeepSeek V4 Pro, trial 3 Sighted</td><td>Played: h2h9, i2d2, d2d4, d4d5</td><td>Invalid: 0. B/L/S: 1/1, 1/1, 2/3.</td></tr><tr><td>C4b JH-327</td><td>Root move: h2h9.</td><td>Fail; attacker checkmated.</td></tr><tr><td>Kimi K2.5, trial 2 Sighted</td><td>Played: h2h9, i2i9</td><td>Invalid: 0. B/L/S: 0/0, 0/0, 0/0.</td></tr><tr><td>C5a SQYQ-098</td><td>Root move: b2b9.</td><td>Pass; defender perpetual check.</td></tr><tr><td>Qwen 3.7 Max, trial 2 Restricted observation</td><td>Played: b2b9, g7g8, d0d1, d1d2, ...</td><td>Invalid: 5. B/L/S: 0/0, 0/0, 0/0.</td></tr><tr><td>C5b SQYQ-098</td><td>Root move: b2b9.</td><td>Pass; checkmate.</td></tr><tr><td>Gemini 3.1 Pro, trial 2 Restricted observation</td><td>Played: b2b9, h6h9, g7g8, h9f9, ...</td><td>Invalid: 0. B/L/S: 0/0, 0/0, 0/0.</td></tr><tr><td>C6a SQYQ-012</td><td>Root move: g7e7.</td><td>Pass; checkmate.</td></tr><tr><td>Gemini 3.1 Pro, trial 1 Sighted</td><td>Played: g7e7, d8e8, d7b8</td><td>Invalid: 0. B/L/S: 0/0, 0/0, 1/1.</td></tr><tr><td>C6b SQYQ-012</td><td>Root move: g7e7.</td><td>Fail; resignation.</td></tr><tr><td>Gemini 3.1 Pro, trial 2 Sighted</td><td>Played: d8e8, d7c9, g8f8, f8e8, ...</td><td>Invalid: 1. B/L/S: 0/0, 0/0, 6/10.</td></tr></table>
