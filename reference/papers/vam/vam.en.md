# VAM: Verbalized Action Masking for Controllable Exploration in RL Post-Training – A Chess Case Study

Zhicheng Zhang <sup>1</sup> Ziyan Wang <sup>2</sup> Yali Du <sup>2</sup> Fei Fang <sup>1</sup>

## Abstract

Exploration remains a key bottleneck for reinforcement learning (RL) post-training of large language models (LLMs), where sparse feedback and large action spaces can lead to premature collapse into repetitive behaviors. We propose Verbalized Action Masking (VAM), which verbalizes an action mask in the prompt and enforces that the model outputs an action from the masked set. Building on this interface, we introduce iterative action-space pruning: if the target action is not sampled, we remove valid sampled actions from the mask and resample under the reduced candidate set, repeating until the target is sampled or a fixed budget is exhausted. We study VAM in chess and evaluate it under two training regimes: an engine-play regime that generates states via play against an engine opponent and a fixed-dataset regime that trains from a fixed dataset of positions with verifier scores. Across held-out chess puzzles and full-game play measured by average centipawn loss (ACPL), VAM improves learning efficiency and final performance over strong baselines, highlighting verbalized masking as a practical mechanism for controllable exploration in LLM RL post-training.

## 1. Introduction

Reinforcement learning with verifiable rewards (RLVR) offers a scalable recipe for post-training large language models (LLMs) (Cobbe et al., 2021; Lightman et al., 2023; OpenAI et al., 2024; Team et al., 2025), where the model is trained against automatically checkable signals, such as exact correctness or strict output-format compliance. This “verifier-first” framing enables fully automated, on-policy RL loops and has driven strong improvements on reasoningheavy tasks in recent work (Guo et al., 2025).

RLVR, however, does not remove RL’s central difficulty: exploration. When the reward is sparse and the effective action space is large, on-policy updates stall if the policy does not sample informative behaviors, even with deterministic verification. In many LLM post-training settings, this failure mode is easy to miss because the “environment” is often just an evaluator. The learning problem is unchanged: the policy must discover high-reward actions (valid outputs under a strict interface) in an immense space of possible strings.

Classical RL often explores by visiting new states through environment interaction. In contrast, many RL post-training setups sample prompts from a fixed distribution and perform multiple rollouts per prompt, for example with grouped on-policy objectives such as Group Relative Policy Optimization (GRPO) (Guo et al., 2025). In this regime, the key exploration bottleneck is action selection within a single state. Given the same prompt, does the model actually try different plausible actions, or does it repeatedly sample the same high-probability output? When a rollout group collapses to near-identical actions, group-relative updates provide little learning signal, and additional samples largely waste compute.

We propose Verbalized Action Masking (VAM) to make action-space exploration controllable. The prompt explicitly lists an admissible action set, and a deterministic verifier checks that the model outputs a valid action from this set. Choosing an out-of-mask action receives a fixed penalty and terminates, which turns action validity into a verifiable signal suitable for RLVR. This is a prompt-level interface rather than logit-level masking; each rollout is sampled on-policy from the model conditioned on the presented candidate set. We then use this interface to enforce exploration via iterative action-space pruning. For a fixed state, we run multiple rounds of grouped sampling, remove alreadysampled valid actions from the candidate set, and resample under the pruned set until a target action is found or a fixed budget is exhausted. This procedure converts repeated sampling into controlled coverage of the action space, rather than repetition of the same few actions.

We study these ideas in chess, which provides an unusually clean testbed. Each position admits a finite, enumerable action set, the legal moves, and legality checking is exact. Moreover, strong engine evaluators can supply permove value estimates that serve as dense, deterministic rewards. At the same time, chess exposes practical failures of language-model policies. Prior work, including Hwang et al. (2025) and LLM CHESS (Kolasani et al., 2025), has found that providing legal-move lists and enforcing structured outputs are crucial for learning and evaluation, and that interface choices can dominate performance. We instantiate VAM for chess move selection, where the model must output a single move from a provided candidate list in a strict format, and we evaluate both one-step move selection on held-out puzzles and full-game play against an engine opponent using average centipawn loss (ACPL).

![](images/862d1a4c56c64fa22ac990ada38503cf0b1235df3705202c788ae86588053893.jpg)  
Figure 1. Overview of Verbalized Action Masking (VAM) with iterative action-space pruning. Given a state and a provided set of allowed actions, we prompt the LLM and sample rollout groups for Group Relative Policy Optimization (GRPO). Each sampled output is parsed into an action and then evaluated by a verifier to produce reward feedback. If the target condition is not met in the current round (for example, the target action is not sampled), we mask the distinct valid sampled actions out of the allowed action set and resample under the updated prompt. This iterative pruning procedure increases within-state action coverage and yields more informative grouped on-policy updates.

In summary, our contributions are the following:

1. We formalize action masking as an MDP augmentation that turns action validity into a verifiable reward and termination signal;

2. We propose VAM with iterative action-space pruning, enabling controllable action-space exploration under grouped on-policy RL;

3. We present two practical training pipelines, fixeddataset training and engine-play position generation, that share the same pruning mechanism;

4. We conduct extensive experiments in chess, demonstrating that VAM improves learning efficiency and final performance over strong baselines on both puzzle selection and full-game play.

## 2. Related Work

RL with verifiable rewards (RLVR) Recent work has shown that reinforcement learning with automatically checkable outcome rewards can improve LLM reasoning without relying on human preference labels. This aligns with approaches in code generation that use execution feedback as a reward signal (Le et al., 2022; Yang et al., 2023). DeepSeek-R1 (Guo et al., 2025) uses GRPO-style post-training with rule-based correctness signals and explicit output-format constraints, demonstrating that verifiable rewards and structured generation can meaningfully shape model behavior. Because our setting also admits deterministic verification (output parsing and legality checks in chess), we adopt a similar “verifier-first” philosophy for reward design. Wen et al. (2025) caution that commonly reported pass@k improvements can mask whether reasoning itself improved; they propose CoT-aware variants of pass@k that require both a correct final answer and a correct chain-of-thought.

Exploration in RL post-training RL post-training with verifiable rewards can suffer from exploration failures, where on-policy updates collapse diversity even when an outcome verifier is available. Recent work addresses this challenge with a variety of mechanisms, including outcome-level exploration incentives, trial-and-error interaction for agentic environments, and partial supervision that scaffolds rare long-horizon solutions. Outcomebased exploration (Song et al., 2025) adds bandit-inspired bonuses or within-batch repetition penalties over final outcomes to encourage diverse solution attempts under GRPOstyle training. Exploration-Based Trajectory Optimization (ETO) (Song et al., 2024) alternates between rolling out an LLM agent to collect failure trajectories and updating the policy with trajectory-level preference optimization that contrasts failures against higher-reward trajectories. Complementary analyses study how RLVR can sharpen pass@1 while reducing pass@k diversity, and propose algorithmic knobs that preserve exploration, such as clipping variants for entropy control and reference-policy resets under KL regularization (Deng et al., 2025; Park et al., 2025; Liu et al., 2025). Within this landscape, hint-based and off-policy guidance methods inject partial rationales, solution prefixes, or expert traces during training to scaffold rare long-horizon solutions, including StepHint (Zhang et al., 2025a), AdaBack (Amani et al., 2025), privileged on-policy exploration (POPE) (Qu et al., 2026), and branched rollouts from expert anchors (BREAD) (Zhang et al., 2025b). Related approaches include off-policy guidance (LUFFY) (Yan et al., 2025) and self-improvement methods such as STaR (Zelikman et al., 2022) and Self-Refine (Madaan et al., 2023). Our approach targets the same bottleneck, but rather than revealing rationales or shaping outcome rewards, we intervene on the action space: we present an explicit candidate set via VAM and enforce coverage by pruning already-sampled actions across rounds.

Chess-playing LLMs Several recent works study whether LLMs can acquire chess competence through either supervised learning or RL (Veeriah et al., 2025; Feng et al., 2025). Hwang et al. (2025) perform GRPO post-training of LLMs on chess puzzles and report that providing explicit legalmove lists is important for learning; their analysis suggests that performance plateaus and that limited chess-rule understanding of chess rules is a bottleneck. In contrast, Chess-LLM (Zhang et al., 2025c) studies large-scale supervised training on complete games and reports strong engine-based evaluations, highlighting the role of long-horizon data and evaluation in full-game settings. ChessGPT and Chess-CLIP (Feng et al., 2023) treat chess as a language modeling and alignment problem, pairing game records with natural language annotations and proposing a broad evaluation suite spanning state tracking, value judgment, and tactical policy probes. Finally, LLM CHESS (Kolasani et al., 2025) benchmarks chess play through an agentic tool interface and finds that instruction-following and tool-use failures, not only illegal moves, are a major source of breakdown, and that interface choices can materially affect performance. These findings motivate our focus on candidate-restricted move selection with strict output parsing and explicit action masks.

## 3. Preliminaries

Chess notation and legal moves We represent chess positions as Forsyth–Edwards Notation (FEN) strings and moves as Universal Chess Interface (UCI) strings (e.g., e2e4, $\mathfrak { g l f } 3 )$ . We also refer to Standard Algebraic Notation (SAN) for compact, human-readable moves (e.g., e4, Nf3, O-O) when discussing prior work. SAN can also include annotations such $\mathrm { a s ~ + }$ (check) and # (checkmate), which are convenient for humans but can make tactical consequences explicit and thus act as an additional move-quality cue; we therefore use strict UCI for model outputs and treat SAN only as auxiliary notation. For a position $s ,$ let $\boldsymbol { \mathcal { A } } ( \boldsymbol { s } )$ denote the set of legal moves. In our experiments and evaluation, models are required to output a single move in strict UCI format, and malformed outputs (or illegal moves) are treated as invalid. To quantify move quality during full-game play, we use average centipawn loss (ACPL) as an engine-based metric.

Group Relative Policy Optimization (GRPO) We posttrain the policy using Group Relative Policy Optimization (GRPO) (Guo et al., 2025), an on-policy method that estimates advantages from within-prompt groups of sampled responses, avoiding a learned critic/value function. Given prompts $q \sim P ( Q )$ , GRPO samples a group of $G$ outputs $\{ o _ { i } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( O \mid q )$ , computes scalar rewards $r _ { i }$ and group-relative advantages

$$
A _ {i} = \frac {r _ {i} - \mathrm{mean} (\{r _ {1} , \ldots , r _ {G} \})}{\mathrm{std} (\{r _ {1} , \ldots , r _ {G} \})},\tag{1}
$$

and optimizes the clipped objective

$$
\begin{array}{l}J _ {\mathrm{GRPO}} (\theta) = \mathbb {E} _ {\underset {\{o _ {i} \} _ {i = 1} ^ {G} \sim \pi_ {\theta_ {\mathrm{old}}} (O | q)} {q \sim P (Q)}} \left[ \frac {1}{G} \sum_ {i = 1} ^ {G} \Big (\min \big (\rho_ {i} (\theta) A _ {i}, \right.\left. \right.\\\left. \operatorname{clip} \left(\rho_ {i} (\theta), 1 - \epsilon , 1 + \epsilon\right) A _ {i}\right)\\\left. - \beta D _ {\mathrm{KL}} \left(\pi_ {\theta} \| \pi_ {\text {ref}}\right)\right) \Bigg ],\end{array}\tag{2}
$$

where $\rho _ { i } ( \theta ) = \pi _ { \theta } ( o _ { i } \mid q ) / \pi _ { \theta _ { \mathrm { o l d } } } ( o _ { i } \mid q ) , \epsilon > 0$ is the PPOstyle clipping parameter, and $\beta \geq 0$ weights a KL penalty to a reference policy $\pi _ { \mathrm { r e f } }$ (computed per sampled output).

## 4. Methods

Action-space exploration has been a central challenge in RL post-training with verifiable rewards. In grouped onpolicy objectives such as Group Relative Policy Optimization (GRPO) (Shao et al., 2024), each prompt is paired with a group of sampled outputs, and learning depends on reward differences within the group. When the per-state action space is large, the policy can collapse to a small subset of high-probability actions, yielding low diversity and weak learning signals within each group. We propose Verbalized Action Masking (VAM), which makes the admissible action set an explicit part of the RL interface. Specifically, we (i) formalize an action-masking MDP that assigns a fixed penalty when the policy selects an out-of-mask action, (ii) implement this interface for LLM policies by verbalizing the mask in the prompt and enforcing it with a deterministic parser and verifier, and (iii) improve exploration via iterative action-space pruning, which repeatedly samples GRPO-sized rollout groups for the same base prompt while shrinking the mask to avoid resampling previously produced valid actions. VAM can be trained under two regimes that differ only in how states and verifier scores are obtained: a fixed-dataset regime that samples states from a fixed dataset with precomputed verifier scores, and an engine-play regime that generates states on-policy, for example by playing against an engine opponent and querying the verifier on encountered positions. We first present VAM in a generic action-selection setting and then describe the chess instantiation used in our experiments.

## 4.1. Action-masking MDP

Intuitively, an action mask specifies which actions are currently admissible. If the agent selects an action outside this mask, the environment treats it as a constraint violation, assigns a fixed penalty, and terminates the rollout. This turns action validity into a deterministic verifier signal that can be used by an on-policy RL algorithm, and it provides a clean handle for manipulating the available action space across repeated sampling.

Formally, consider a Markov decision process (MDP) $( S , { \mathcal { A } } , P , R , \gamma )$ with states $s \in S$ , actions $a \in { \mathcal { A } } .$ , transition kernel $P ( s ^ { \prime } \mid s , a )$ , reward function $R ( s , a )$ , and discount factor $\gamma \in [ 0 , 1 )$ . Let 1[·] denote the indicator function. We assume that at each state s the environment can enumerate a finite admissible action set ${ \mathcal { A } } ( s ) \subseteq A .$ . An action mask is a subset ${ \mathcal { M } } \subseteq A ( s )$ of actions that are currently allowed. An action-masking MDP augments the state with an explici mask variable. Specifically, define an augmented state space

$$
\tilde {\mathcal {S}} = \{(s, \mathcal {M}): s \in \mathcal {S}, \mathcal {M} \subseteq \mathcal {A} (s) \},\tag{3}
$$

and an action-masking MDP $( \tilde { S } , \mathcal { A } , \tilde { P } , \tilde { R } , \gamma )$ . The mask may depend on the underlying state and can be updated as a function of the chosen action; we write $U ( \mathcal { M } ^ { \prime } \mid s , a , \mathcal { M } )$ for a (possibly deterministic) mask-update kernel. Then for in-mask actions the augmented transition factorizes as

$$
\tilde {P} \big ((s ^ {\prime}, \mathcal {M} ^ {\prime}) \mid (s, \mathcal {M}), a \big) = P (s ^ {\prime} \mid s, a) U (\mathcal {M} ^ {\prime} \mid s, a, \mathcal {M}).\tag{4}
$$

If a $\notin \mathcal { M }$ , we treat the transition as terminal (equivalently, a transition to an absorbing terminal state), so the next-state dynamics and mask update are not applied.

The augmented reward assigns the base reward for in-mask actions and a fixed penalty otherwise:

$$
\begin{array}{c} \tilde {R} \big ((s, \mathcal {M}), a \big) = \mathbf {1} [ a \in \mathcal {M} ] \cdot R (s, a) \\ + \big (1 - \mathbf {1} [ a \in \mathcal {M} ] \big) \cdot (- \text {Penalty}), \end{array}\tag{5}
$$

where Penalty $> 0$ is a constant. We terminate the rollout immediately when an out-of-mask action is chosen. Equivalently, define a done indicator

$$
d (s, a, \mathcal {M}) = \mathbf {1} [ a \notin \mathcal {M} ],\tag{6}
$$

and terminate whenever $d ( s , a , { \mathcal { M } } ) = 1$ . The penalty controls whether the mask is a soft constraint or a hard constraint. When Penalty is finite, the objective discourages out-of-mask actions while still defining a reward value for all actions. In the limiting case Penalty $ \infty .$ , any outof-mask action incurs an unbounded loss, and the optimal policy assigns zero probability mass to actions outside M, recovering a hard version of action masking.

## 4.2. Verbalized Action Masking (VAM)

The action-masking MDP specifies how a mask should affect reward and termination, but it does not dictate how the mask is represented to the policy. In the LLM setting, we implement action masking by verbalizing the mask in the prompt. Each prompt contains a description of the state s and an explicit action mask ${ \mathcal { M } } \subseteq A ( s )$ The model then outputs a single action, and a deterministic parser and verifier map the output to a predicted action aˆ and check membership in M. If parsing fails or if ${ \hat { a } } \notin { \mathcal { M } } .$ , we treat the outcome as an out-of-mask action, triggering termination and the penalty as defined by the action-masking MDP.

```txt
Verbalized Action Masking (VAM)
State description {task prompt}
Format specification <think>...</think>
<action>...</action>
Allowed action [act1, ..., actk]
```  
Figure 2. Prompt interface for VAM. Prompt inputs include the state description, format specification, and allowed action list; placeholders indicate where each appears.

VAM preserves on-policy data collection. The mask is part of the observation, and we sample from the current policy conditioned on the prompt, without applying logit-level masking or post-hoc filtering. As a result, each sampled group is on-policy for its corresponding mask-conditioned prompt, which makes the approach compatible with standard grouped on-policy RL implementations.

```txt
Algorithm 1 VAM with iterative action-space pruning
1: Procedure PRUNEANDSAMPLE(π, s, a*)  
2: Parameters: group size G, max rounds Rmax, reward \(\tilde{R}\)  
3: \(\mathcal{M} \leftarrow \mathcal{A}(s)\), \(\mathcal{D}_s \leftarrow \emptyset\)  
4: for r = 1 to Rmax do  
5: Sample {o_i}_{i=1}^G \sim \pi(\cdot | s, \mathcal{M})\) and parse to {a_i}_{i=1}^G  
6: Compute r_i \(\leftarrow \tilde{R}((s, \mathcal{M}), a_i)\) for i = 1, ..., G  
7: \(\mathcal{D}_s \leftarrow \mathcal{D}_s \cup \{(\mathcal{M}, \{(o_i, r_i)\}_{i=1}^G)\}\)  
8: V \(\leftarrow \{a_i : a_i \in \mathcal{M}\}\)  
9: if a* ∈ V then  
10: break  
11: \(\mathcal{M} \leftarrow \mathcal{M} \setminus V\)  
12: end for  
13: return \(\mathcal{D}_s\)  
14: Procedure TRAINVAM(π, D, T, B)  
15: Inputs: policy π, dataset D, total steps T, batch size B  
16: for t = 1 to T do  
17: Sample {s_j}_{j=1}^B from D and set B \(\leftarrow \emptyset\)  
18: for j = 1 to B do  
19: Read A(s_j) and set a* ∈ arg max_{a∈A(s_j)} μ(s_j, a)  
20: D_{s_j} \(\leftarrow\) PRUNEANDSAMPLE(π, s_j, a*)  
21: B \(\leftarrow\) B ∪ D_{s_j}  
22: end for  
23: Update π with GRPO on B (§3)  
24: end for  
25: return π
```

## 4.3. Iterative action-space pruning

The action mask defines which actions are admissible, but it does not by itself ensure diverse exploration within a single state. To avoid repeatedly sampling the same few actions, we introduce an iterative pruning procedure that progressively shrinks the mask across rounds of grouped sampling.

Verifier-based rewards and target actions In RL posttraining with verifiable rewards, a task reward is computed from the model output using an external verifier. We assume that for each state s and admissible action $a \in { \mathcal { A } } ( s )$ , the verifier provides a scalar score $\mu ( s , a ) \ \in \ \mathbb { R }$ This score may be binary (for example, an exact correctness check) or graded (for example, a test score or a heuristic quality metric). We use it to define the base reward $R ( s , a ) = \mu ( s , a )$ for in-mask actions. To make the interface verifiable, we also define a format reward over the raw output string o: if o does not match the required tag format (for example, <action>...</action> or <uci\_move> $\cdots ^ { < / \mu \mathrm { c } \dot { \mathrm { 1 } } }$ \_move>), or if the parsed action aˆ is not in $\mathcal { M } ,$ then $r _ { \mathrm { f o r m a t } } ( o , { \mathcal { M } } ) = - \mathrm { P e n a l t y }$ , and otherwise $r _ { \mathrm { f o r m a t } } ( o , { \mathcal { M } } ) = 0$ This corresponds to treating unparsable outputs and out-of-mask actions as constraint violations, assigned a fixed penalty via ${ \tilde { R } } ,$ and terminated as in the action-masking MDP.

Given a state s and a mask ${ \mathcal { M } } \subseteq A ( s )$ , we define the target action as

$$
a ^ {\star} (s, \mathcal {M}) \in \arg \max _ {a \in \mathcal {M}} \mu (s, a),\tag{7}
$$

using a deterministic tie-break when multiple actions share the maximum score.

For each base state s, the pruning procedure starts from an initial mask $\mathcal { M } _ { 0 } \subseteq \mathcal { A } ( s )$ . In the engine-play regime we use the full legal set $\mathcal { M } _ { 0 } = \mathcal { A } ( s )$ , while in the fixeddataset regime $\mathcal { M } _ { 0 }$ can be either the full legal set or a dataset-provided candidate subset. We set the target to $a ^ { \star } = a ^ { \star } ( s , \mathcal { M } _ { 0 } )$ . Each round uses a new prompt that differs only in the masked action list, and samples a rollout group of $G$ candidates, matching the GRPO group size (§3). If the target action is not present, we remove the sampled in-mask actions from the candidate set and resample, up to a maximum number of rounds $R _ { \mathrm { m a x } }$ . Each pruning round defines a separate mask-conditioned prompt for GRPO, so the procedure naturally produces a sequence of GRPO groups per base state. We apply GRPO to all collected groups without additional weighting across rounds. Algorithm 1 summarizes the full VAM training loop, including iterative pruning, resampling, and the GRPO update. Algorithm 2 shows an optional engine-play procedure for generating training states on-policy; when engine-play position generation is not used, we assume a provided fixed dataset of states with verifier scores.

## 4.4. Chess implementation details

Our experiments instantiate VAM in chess move selection.

State, actions, and prompt interface We represent states s as chess positions (FEN strings), and actions a as moves in strict UCI format. The admissible action set $\boldsymbol { \mathcal { A } } ( \boldsymbol { s } )$ is the set of legal moves. To implement VAM, each prompt includes an explicit candidate list corresponding to a mask ${ \mathcal { M } } \subseteq A ( s )$ (either the full legal set or a restricted subset), together with the ordered legal move list.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 2 Engine-play position generation (optional)
1: Procedure COLLECTBYPLAY(π, E, Verifier, B)
2: Inputs: environment E, target data size B, pool size N
3: Initialize or reuse  $E = \{e_1, \ldots, e_N\}$  from E, and set  $D_{play} \leftarrow \emptyset$ 
4: while  $|D_{play}| &lt; B$  do
5: Retrieve current states  $\{s_i\}_{i=1}^N$  from E
6: for i = 1 to N do
7: Read  $\mathcal{A}(s_i)$  and query Verifier for  $\mu(s_i, a)$  for all  $a \in \mathcal{A}(s_i)$ 
8: Append  $(s_i, \mathcal{A}(s_i), \mu(s_i, \cdot))$  to  $D_{play}$ 
9: if  $|D_{play}| \geq B$  then
10: break
11: end for
12: Step all instances in E
13: end while
14: return  $D_{play}$ 

Chess Prompt with Verbalized Mask
Original chess prompt {original chess prompt}
Current FEN string {FEN}
Legal moves (UCI) {UCI move list}
Allowed action {allowed UCI move list}
</div>

Figure 3. Chess prompt with verbalized mask. Prompt inputs include the original prompt, FEN, legal moves, and allowed actions; placeholders indicate where each appears.

The model must output a single move in a <uci\_move> tag. A deterministic parser and verifier checks that the output is well-formed, that it is legal in the underlying position, and that it lies in the provided mask. Appendix A.5 specifies the exact parsing and validity rules.

Engine-based verifier scores We instantiate $\mu ( s , a )$ using the Stockfish chess engine (The Stockfish Developers, 2024). Given fixed analysis settings, Stockfish can provide different deterministic per-move evaluation signals, including centipawn scores, win-draw-loss probabilities, and expected-score proxies. Any chosen scalar signal can be used as the verifier score $\mu ( s , a )$ to define rewards and targets. In our experiments, training positions are paired with engine-derived per-move value maps (e.g., expected-score or win-probability proxies) that serve as $\mu ( s , a )$

Fixed-dataset training In the fixed-dataset regime, training states s are sampled from a fixed dataset, and each state comes with a legal-move list and an engine-derived value map $\mu ( s , a )$ over legal actions. The initial mask $\mathcal { M } _ { 0 }$ can either be the full legal set $\boldsymbol { \mathcal { A } } ( \boldsymbol { s } )$ or a candidate subset provided by the dataset. Training follows the VAM loop in Algorithm 1, which performs iterative pruning and applies a GRPO update over all collected mask-conditioned groups.

Engine-play position generation For engine-play training data, we generate a stream of chess positions by playing games against a fixed engine opponent. Because the learning problem only requires a state, we record positions regardless of which side is to move. For each recorded state s, we enumerate legal moves $\boldsymbol { \mathcal { A } } ( \boldsymbol { s } )$ , query the engine-derived verifier score $\mu ( s , a )$ over legal actions, and add the resulting state records to a training buffer. Algorithm 2 summarizes this optional engine-play position generation procedure; these states can then be consumed by the same VAM training loop in Algorithm 1.

## 5. Experiments

We evaluate VAM in a text-only chess setting, in which the model must select a move from a provided list without access to search or external tools. We report (i) one-step move selection on the Searchless Chess puzzle positions (Ruoss et al., 2024) and (ii) full-game play against a fixed engine opponent, measured by average centipawn loss (ACPL).

## 5.1. Research questions

We organize our empirical study around three questions:

1. RQ1: Does VAM improve exploration compared to the baselines?

2. RQ2: Does VAM improve full-game play compared to the baselines?

3. RQ3: What are the relative strengths of engine-play position generation and a fixed dataset?

## 5.2. Experimental setup

Models We run experiments with Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct (Qwen et al., 2025).

Verbalized-masking interface Each decision prompt provides the position as a FEN string, the side to move, and an ordered list of legal moves in strict lowercase UCI. In the verbalized masking setting, we additionally provide a candidate subset, allowed\_moves, and require the model to choose from this subset. Models output a single move in strict lowercase UCI inside a <uci\_move> tag, and we evaluate with a deterministic parser and legality checker. We use the same prompt template as Hwang et al. (2025) (Appendix A). Appendix A.5 specifies the exact parsing and validity rules.

Training regimes and data We focus on post-training methods that learn from deterministic, engine-provided signals. Each training position is paired with an engine-derived per-move value map $\mu ( s , a ) \ \in \ [ 0 , 1 ]$ over legal moves, which we use to define reward signals (§4.3). We consider two sources of training positions: a fixed dataset aligned with Chess-R1 (Hwang et al., 2025) containing 100,000 positions, and an on-policy stream of positions generated via engine-play against a fixed engine opponent. These correspond to the fixed-dataset and engine-play regimes in §4. In both cases, training follows the same VAM loop in Algorithm 1, which applies GRPO to all mask-conditioned groups produced by iterative pruning. In the engine-play regime, training states are generated on-policy via play against an engine opponent using Algorithm 2, while in the fixed-dataset regime we assume a provided dataset. In RQ1 we use fixed-dataset training to isolate the effect of pruning on move-selection exploration, while in RQ2 we evaluate VAM with engine-play position generation and compare against baselines under the same prompt interface and engine verifier. RQ3 directly contrasts engine-play versus fixed-dataset training for VAM.

Puzzle benchmark We evaluate one-step move selection on the Lichess puzzle positions in Searchless Chess (Ruoss et al., 2024). Each position provides a FEN string, an ordered UCI legal-move list, and a provided puzzle solution move. We use all 10,000 positions as a held-out test set for puzzle evaluation. We treat the provided solution move as the target action for puzzle accuracy.

Full-game benchmark We evaluate end-to-end play by playing complete games from the standard initial position against a fixed Stockfish opponent (Skill Level 0), reported separately for depth 1 and depth 5. Appendix A.7 specifies game termination rules, invalid-move handling, and ACPL computation details.

Metrics For puzzles, we report pass@1 accuracy for selecting the provided puzzle solution under the candidate set. For full games, we report ACPL computed by a separate Stockfish analyzer that runs at depth 20, serving as an expert evaluator. Appendix A.7 specifies the exact ACPL computation and aggregation used by the evaluation harness.

## 5.3. Baselines

We compare VAM against GRPO baselines that use the same engine verifier and strict UCI output contract but omit iterative pruning. This setting follows Chess-R1 (Hwang et al., 2025), with the modification that we use strict UCI (instead of SAN) to avoid SAN annotations such as check or checkmate acting as additional tactical cues. To probe how reward design affects exploration under GRPO, we evaluate three reward definitions derived from the engine signal: expected score, win rate, and ranking among moves. We include a rejection-sampled supervised fine-tuning (SFT) baseline trained on the same fixed dataset as a supervised reference point. Appendix A provides additional training details.

Reward definitions from the engine signal. For each position s and legal move $a \in { \mathcal { A } } ( s )$ , Stockfish (The Stockfish Developers, 2024) provides an evaluation that we convert into a scalar verifier score $\mu ( s , a ) \in [ 0 , 1 ]$ ]. In our main experiments we use a win/draw/loss (WDL) based expectedscore proxy: if the engine reports win/draw/loss likelihoods $\left( p _ { W } ( s , a ) , p _ { D } ( s , a ) , p _ { L } ( s , a ) \right)$ (with $p _ { W } + p _ { D } + p _ { L } = 1 )$ , we define

$$
\mu_ {\mathrm{exp}} (s, a) = p _ {W} (s, a) + \frac {1}{2} p _ {D} (s, a).\tag{8}
$$

We consider two additional reward variants for the GRPO baselines: (i) win-rate reward $\mu _ { \mathrm { w i n } } ( s , a ) = p _ { W } ( s , a )$ , and (ii) rank-based reward computed from the within-position ordering of $\mu _ { \mathrm { e x p } } ( s , a )$

$$
\mu_ {\mathrm{rank}} (s, a) = 1 - \frac {\mathrm{rank} _ {\downarrow} (\mu_ {\mathrm{exp}} (s , a)) - 1}{| \mathcal {A} (s) | - 1},\tag{9}
$$

where rank<sub>↓</sub> assigns rank 1 to the highest-scoring move (ties broken deterministically). Unless stated otherwise, VAM and GRPO baselines use µ<sub>exp</sub>.

## 5.4. Results

RQ1: exploration on puzzles Figure 4 studies learning dynamics on the move-selection puzzle task. In the left panel, we vary only the reward definition used by GRPO and observe clear differences in learning curves, with the expected-score signal producing the strongest curve among the reward variants we tested. For consistency in subsequent comparisons, we use expected score as the enginederived reward for GRPO baselines and VAM. In the right panel, we compare VAM against GRPO under the same fixed dataset and find that VAM reaches higher puzzle accuracy for both model sizes. Both GRPO and VAM outperform the rejection-sampled SFT baseline on held-out puzzles, suggesting that the SFT setup does not generalize well to unseen puzzle positions. Interestingly, the smaller Qwen2.5-3B-Instruct model trained with VAM surpasses the larger Qwen2.5-7B-Instruct model trained with GRPO.

RQ2: full-game play Figure 5 reports ACPL during fullgame evaluation over the course of training. Across both evaluation opponents (depth 1 and depth 5), VAM achieves lower ACPL than the GRPO baselines, indicating stronger move quality in complete games. The rejection-sampled SFT baseline yields substantially higher ACPL, suggesting that the SFT setup does not generalize to full-game play under our evaluation protocol. Among RL-based methods, iterative pruning provides a further improvement over standard GRPO. Because ACPL becomes harder to reduce as it gets smaller, persistent gaps at lower ACPL correspond to meaningful differences in play.

![](images/ee0e388d19327561fad007846cce6592b2da79ad9a06685318e3451889b26629.jpg)

Figure 4. Performance on held-out chess puzzles. Pass@1 accuracy for selecting the dataset-provided solution move on the 10,000- position test set. The left panel compares fixed-dataset GRPO baselines under three engine-derived reward signals (expected-score proxy µ<sub>exp</sub>, win-rate $\mu _ { \mathrm { w i n } } ,$ and rank-based $\mu _ { \mathrm { { r a n k } } } ;$ Eqs. 8 and 9). The right panel compares VAM with iterative action-space pruning against GRPO under the same fixed dataset and matched rollout budget for Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct. The horizontal axis counts GRPO gradient updates, and each point evaluates the current policy with fixed decoding settings over the full test set. Dashed horizontal lines indicate rejection-sampled SFT baselines.  
![](images/62591c41d3f29f31a73c7d9f512b49cd465ab7abb6a69eee097137c8ba13cc23.jpg)  
Figure 5. Move quality in full games against Stockfish. Average centipawn loss (ACPL; lower is better) during evaluation against fixed Stockfish opponents at depth 1 and depth 5, shown for Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct. We report overall $\mathsf { A C P I }$ L as well as per-opponent ACPL trajectories over training. VAM is trained with engine-play position generation and iterative action-space pruning, while the GRPO baseline uses the same verifier and prompt interface but omits pruning. Dashed horizontal lines indicate rejection-sampled SFT baselines. VAM consistently reaches lower ACPL across opponents and model sizes, indicating stronger move quality under full-game play.

RQ3: engine-play versus fixed-dataset regimes Figure 6 contrasts VAM trained on a provided fixed dataset with VAM trained on states generated via games against a fixed engine opponent. We observe a clear trade-off between onestep selection and long-horizon play: fixed-dataset training yields stronger puzzle accuracy, while engine-play position generation yields better full-game ACPL. Beyond these headline metrics, the engine-play regime is associated with longer responses and smaller KL loss, which is consistent with training on a state distribution induced by the current policy.

## 6. Discussion

When VAM is a good fit VAM is most appropriate when the environment can enumerate a finite action set per state and when membership in the admissible set can be checked deterministically. In such settings, mask compliance becomes a reliable signal that can be used for training (as a hard penalty) and for evaluation (as a format and legality metric), while preserving on-policy sampling because we do not apply logit-level masking or post-hoc filtering.

Exploration as a controllable interface choice Iterative action-space pruning turns repeated sampling into a controlled exploration procedure: across rounds, the model is forced to propose different actions because previously sampled valid actions are removed from the prompt-specified candidate set. This perspective helps disentangle two sources of failure in RL post-training, limited policy entropy and limited action-space coverage under repeated sampling.

![](images/dd296cd15337d301de55e1bb6f76950e4bf746aed5e13d0f418274bf69bb35bc.jpg)  
Figure 6. Fixed-dataset versus engine-play training regimes for VAM. Radar plots summarize performance and training statistics for VAM trained either on a fixed dataset or with engine-play position generation via games against a fixed engine opponent, shown for Qwen2.5-7B-Instruct (left) and Qwen2.5-3B-Instruct (right). Axes report pass@1 puzzle accuracy, KL loss to the reference policy, average response length, mean verifier reward, and full-game move quality measured by ACPL (overall and per-move). Within each model, each metric is linearly normalized across the two regimes for visualization, and numeric annotations indicate the raw values. The profiles highlight a trade-off between one-step puzzle performance and full-game play quality across regimes.

## 7. Conclusion

We introduced Verbalized Action Masking (VAM), a prompt-level action-space interface for RL post-training with verifiable rewards, along with an iterative action-space pruning procedure that improves exploration by preventing repeated sampling of the same actions within a state. We instantiated these ideas in text-only chess move selection with strict UCI outputs and evaluated them on both one-step puzzle selection and full-game play against an engine opponent. Empirically, we found that expected score is the strongest reward choice for GRPO baselines on puzzles, and that VAM substantially improves puzzle accuracy and full-game ACPL over those baselines across model sizes. We also observed complementary strengths between fixed-dataset and engine-play regimes, suggesting that state distribution is a key design choice for RL post-training in structured action spaces.

Limitations and future work. Chess is an ideal testbed because legal moves are enumerable and both action validity (strict parsing and move legality) and move quality can be checked deterministically with an engine. Many structured decision problems do not offer these properties, so applying VAM in domains with very large action spaces or without strong deterministic evaluators will require dynamic candidate generation and approximate verifiers.

## Acknowledgements

This work was supported in part by NSF grant IIS-2046640 (CAREER). This work was supported by the Engineering and Physical Sciences Research Council [grant number EP/Y003187/1 and UKRI849].

## Impact Statement

This paper presents work whose goal is to advance the field of machine learning, specifically methods for reinforcement learning post-training of large language models under verifiable constraints. There are many potential societal consequences of improved LLM capabilities and of broader adoption of RL post-training, none of which we feel must be specifically highlighted here.

## References

Amani, M. H., Lotfi, A., Baldwin, N. M., Bengio, S., Farajtabar, M., Abbe, E., and West, R. Rl for reasoning by adaptively revealing rationales. arXiv preprint arXiv:2506.18110, 2025.

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Deng, J., Chen, J., Chen, Z., Cheng, D., Bai, F., Zhang, B., Min, Y., Gao, Y., Zhao, W. X., and Wen, J.-R. From trial-and-error to improvement: A systematic analysis of llm exploration mechanisms in rlvr. arXiv preprint arXiv:2508.07534, 2025.

Feng, X., Luo, Y., Wang, Z., Tang, H., Yang, M., Shao, K.,

Mguni, D., Du, Y., and Wang, J. Chessgpt: Bridging policy learning and language modeling. Advances in Neural Information Processing Systems, 36:7216–7262, 2023.

Feng, X., Veeriah, V., Chiam, M., Dennis, M., Pachauri, R., Tumiel, T., Barbero, F., Obando-Ceron, J., Shi, J., Singh, S., et al. Generating creative chess puzzles. arXiv preprint arXiv:2510.23881, 2025.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Hwang, D., Lee, H., Choo, J., Park, D., and Park, J. Can large language models develop strategic reasoning? posttraining insights from learning chess. arXiv preprint arXiv:2507.00726, 2025.

Kolasani, S., Saplin, M., Crispino, N., Montgomery, K., Davis, J. Q., Zaharia, M., Wang, C., and Wang, C. Llm chess: Benchmarking reasoning and instructionfollowing in llms through chess, 2025. URL https: //arxiv.org/abs/2512.01992.

Le, H., Wang, Y., Gotmare, A. D., Savarese, S., and Hoi, S. C. H. Coderl: Mastering code generation through pretrained models and deep reinforcement learning. Advances in Neural Information Processing Systems, 35: 21314–21328, 2022.

Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and Cobbe, K. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023.

Liu, M., Diao, S., Lu, X., Hu, J., Dong, X., Choi, Y., Kautz, J., and Dong, Y. Prorl: Prolonged reinforcement learning expands reasoning boundaries in large language models. arXiv preprint arXiv:2505.24864, 2025.

Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., Alon, U., Dziri, N., Prabhumoye, S., Yang, Y., et al. Self-refine: Iterative refinement with selffeedback. Advances in Neural Information Processing Systems, 36:46534–46594, 2023.

OpenAI, :, Jaech, A., Kalai, A., Lerer, A., Richardson, A., El-Kishky, A., Low, A., Helyar, A., Madry, A., Beutel, A., Carney, A., Iftimie, A., Karpenko, A., Passos, A. T., Neitz, A., Prokofiev, A., Wei, A., Tam, A., Bennett, A., Kumar, A., Saraiva, A., Vallone, A., Duberstein, A., Kondrich, A., Mishchenko, A., Applebaum, A., Jiang, A., Nair, A., Zoph, B., Ghorbani, B., Rossen, B., Sokolowsky, B., Barak, B., McGrew, B., Minaiev, B., Hao, B., Baker,

B., Houghton, B., McKinzie, B., Eastman, B., Lugaresi, C., Bassin, C., Hudson, C., Li, C. M., de Bourcy, C., Voss, C., Shen, C., Zhang, C., Koch, C., Orsinger, C., Hesse, C., Fischer, C., Chan, C., Roberts, D., Kappler, D., Levy, D., Selsam, D., Dohan, D., Farhi, D., Mely, D., Robinson, D., Tsipras, D., Li, D., Oprica, D., Freeman, E., Zhang, E., Wong, E., Proehl, E., Cheung, E., Mitchell, E., Wal lace, E., Ritter, E., Mays, E., Wang, F., Such, F. P., Raso, F., Leoni, F., Tsimpourlas, F., Song, F., von Lohmann, F., Sulit, F., Salmon, G., Parascandolo, G., Chabot, G., Zhao, G., Brockman, G., Leclerc, G., Salman, H., Bao, H., Sheng, H., Andrin, H., Bagherinezhad, H., Ren, H., Lightman, H., Chung, H. W., Kivlichan, I., O’Connell, I., Osband, I., Gilaberte, I. C., Akkaya, I., Kostrikov, I., Sutskever, I., Kofman, I., Pachocki, J., Lennon, J., Wei, J., Harb, J., Twore, J., Feng, J., Yu, J., Weng, J., Tang, J., Yu, J., Candela, J. Q., Palermo, J., Parish, J., Heidecke, J., Hallman, J., Rizzo, J., Gordon, J., Uesato, J., Ward, J., Huizinga, J., Wang, J., Chen, K., Xiao, K., Singhal, K., Nguyen, K., Cobbe, K., Shi, K., Wood, K., Rimbach, K., Gu-Lemberg, K., Liu, K., Lu, K., Stone, K., Yu, K., Ahmad, L., Yang, L., Liu, L., Maksin, L., Ho, L., Fedus, L., Weng, L., Li, L., McCallum, L., Held, L., Kuhn, L., Kondraciuk, L., Kaiser, L., Metz, L., Boyd, M., Trebacz, M., Joglekar, M., Chen, M., Tintor, M., Meyer, M., Jones, M., Kaufer, M., Schwarzer, M., Shah, M., Yatbaz, M., Guan, M. Y., Xu, M., Yan, M., Glaese, M., Chen, M., Lampe, M., Malek, M., Wang, M., Fradin, M., McClay, M., Pavlov, M., Wang, M., Wang, M., Murati, M., Bavarian, M., Rohaninejad, M., McAleese, N., Chowdhury, N., Chowdhury, N., Ryder, N., Tezak, N., Brown, N., Nachum, O., Boiko, O., Murk, O., Watkins, O., Chao, P., Ashbourne, P., Izmailov, P., Zhokhov, P., Dias, R., Arora, R., Lin, R., Lopes, R. G., Gaon, R., Miyara, R., Leike, R., Hwang, R., Garg, R., Brown, R., James, R., Shu, R., Cheu, R., Greene, R., Jain, S., Altman, S., Toizer, S., Toyer, S., Miserendino, S., Agarwal, S., Hernandez, S., Baker, S., McKinney, S., Yan, S., Zhao, S., Hu, S., Santurkar, S., Chaudhuri, S. R., Zhang, S., Fu, S., Papay, S., Lin, S., Balaji, S., Sanjeev, S., Sidor, S., Broda, T., Clark, A., Wang, T., Gordon, T., Sanders, T., Patwardhan, T., Sottiaux, T., Degry, T., Dimson, T., Zheng, T., Garipov, T., Stasi, T., Bansal, T., Creech, T., Peterson, T., Eloundou, T., Qi, V., Kosaraju, V., Monaco, V., Pong, V., Fomenko, V., Zheng, W., Zhou, W., McCabe, W., Zaremba, W., Dubois, Y., Lu, Y., Chen, Y., Cha, Y., Bai, Y., He, Y., Zhang, Y., Wang, Y., Shao, Z., and Li, Z. Openai o1 system card, 2024. URL https://arxiv.org/abs/2412.16720. Park, J. R., Kim, J., Kim, G., Jo, J., Choi, S., Cho, J., and Ryu, E. K. Clip-low increases entropy and clip-high decreases entropy in reinforcement learning of large lan guage models. arXiv preprint arXiv:2509.26114, 2025.

Qu, Y., Setlur, A., Smith, V., Salakhutdinov, R., and Ku-

mar, A. Pope: Learning to reason on hard problems via privileged on-policy exploration. arXiv preprint arXiv:2601.18779, 2026.

Qwen, :, Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., Lin, H., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Lin, J., Dang, K., Lu, K., Bao, K., Yang, K., Yu, L., Li, M., Xue, M., Zhang, P., Zhu, Q., Men, R., Lin, R., Li, T., Tang, T., Xia, T., Ren, X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Wan, Y., Liu, Y., Cui, Z., Zhang, Z., and Qiu, Z. Qwen2.5 technical report, 2025. URL https: //arxiv.org/abs/2412.15115.

Ruoss, A., Delétang, G., Medapati, S., Grau-Moya, J., Wenliang, L. K., Catt, E., Reid, J., and Genewein, T. Grandmaster-level chess without search. CoRR, 2024.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Song, Y., Yin, D., Yue, X., Huang, J., Li, S., and Lin, B. Y. Trial and error: Exploration-based trajectory optimization for llm agents. arXiv preprint arXiv:2403.02502, 2024.

Song, Y., Kempe, J., and Munos, R. Outcomebased exploration for llm reasoning. arXiv preprint arXiv:2509.06941, 2025.

Team, G., Kamath, A., Ferret, J., Pathak, S., Vieillard, N., Merhej, R., Perrin, S., Matejovicova, T., Ramé, A., Rivière, M., Rouillard, L., Mesnard, T., Cideron, G., bastien Grill, J., Ramos, S., Yvinec, E., Casbon, M., Pot, E., Penchev, I., Liu, G., Visin, F., Kenealy, K., Beyer, L., Zhai, X., Tsitsulin, A., Busa-Fekete, R., Feng, A., Sachdeva, N., Coleman, B., Gao, Y., Mustafa, B., Barr, I., Parisotto, E., Tian, D., Eyal, M., Cherry, C., Peter, J.-T., Sinopalnikov, D., Bhupatiraju, S., Agarwal, R., Kazemi, M., Malkin, D., Kumar, R., Vilar, D., Brusilovsky, I., Luo, J., Steiner, A., Friesen, A., Sharma, A., Sharma, A., Gilady, A. M., Goedeckemeyer, A., Saade, A., Feng, A., Kolesnikov, A., Bendebury, A., Abdagic, A., Vadi, A., György, A., Pinto, A. S., Das, A., Bapna, A., Miech, A., Yang, A., Paterson, A., Shenoy, A., Chakrabarti, A., Piot, B., Wu, B., Shahriari, B., Petrini, B., Chen, C., Lan, C. L., Choquette-Choo, C. A., Carey, C., Brick, C., Deutsch, D., Eisenbud, D., Cattle, D., Cheng, D., Paparas, D., Sreepathihalli, D. S., Reid, D., Tran, D., Zelle, D., Noland, E., Huizenga, E., Kharitonov, E., Liu, F., Amirkhanyan, G., Cameron, G., Hashemi, H., Klimczak-Plucinska, H., Singh, H., Mehta, H., Lehri,´ H. T., Hazimeh, H., Ballantyne, I., Szpektor, I., Nardini, I., Pouget-Abadie, J., Chan, J., Stanton, J., Wieting, J., Lai, J., Orbay, J., Fernandez, J., Newlan, J., yeong Ji,

J., Singh, J., Black, K., Yu, K., Hui, K., Vodrahalli, K., Greff, K., Qiu, L., Valentine, M., Coelho, M., Ritter, M., Hoffman, M., Watson, M., Chaturvedi, M., Moynihan, M., Ma, M., Babar, N., Noy, N., Byrd, N., Roy, N., Momchev, N., Chauhan, N., Sachdeva, N., Bunyan, O., Botarda, P., Caron, P., Rubenstein, P. K., Culliton, P., Schmid, P., Sessa, P. G., Xu, P., Stanczyk, P., Tafti, P., Shivanna, R., Wu, R., Pan, R., Rokni, R., Willoughby, R., Vallu, R., Mullins, R., Jerome, S., Smoot, S., Girgin, S., Iqbal, S., Reddy, S., Sheth, S., Põder, S., Bhatnagar, S., Panyam, S. R., Eiger, S., Zhang, S., Liu, T., Yacovone, T., Liechty, T., Kalra, U., Evci, U., Misra, V., Roseberry, V., Feinberg, V., Kolesnikov, V., Han, W., Kwon, W., Chen, X., Chow, Y., Zhu, Y., Wei, Z., Egyed, Z., Cotruta, V., Giang, M., Kirk, P., Rao, A., Black, K., Babar, N., Lo, J., Moreira, E., Martins, L. G., Sanseviero, O., Gonzalez, L., Gleicher, Z., Warkentin, T., Mirrokni, V., Senter, E., Collins, E., Barral, J., Ghahramani, Z., Hadsell, R., Matias, Y., Sculley, D., Petrov, S., Fiedel, N., Shazeer, N., Vinyals, O., Dean, J., Hassabis, D., Kavukcuoglu, K., Farabet, C., Buchatskaya, E., Alayrac, J.-B., Anil, R., Dmitry, Lepikhin, Borgeaud, S., Bachem, O., Joulin, A., Andreev, A., Hardin, C., Dadashi, R., and Hussenot, L. Gemma 3 technical report, 2025. URL https://arxiv.org/abs/2503.19786.

The Stockfish Developers. Stockfish, 2024. URL https: //stockfishchess.org/. Source code available at https://github.com/official-stockfish/Stockfish.

Veeriah, V., Barbero, F., Chiam, M., Feng, X., Dennis, M., Pachauri, R., Tumiel, T., Obando-Ceron, J., Shi, J., Hou, S., et al. Evaluating in silico creativity: An expert review of ai chess compositions. arXiv preprint arXiv:2510.23772, 2025.

Wen, X., Liu, Z., Zheng, S., Ye, S., Wu, Z., Wang, Y., Xu, Z., Liang, X., Li, J., Miao, Z., et al. Reinforcement learning with verifiable rewards implicitly incentivizes correct reasoning in base llms. arXiv preprint arXiv:2506.14245, 2025.

Yan, J., Li, Y., Hu, Z., Wang, Z., Cui, G., Qu, X., Cheng, Y., and Zhang, Y. Learning to reason under off-policy guidance. arXiv preprint arXiv:2504.14945, 2025.

Yang, J., Prabhakar, A., Narasimhan, K., and Yao, S. Intercode: Standardizing and benchmarking interactive coding with execution feedback. Advances in Neural Information Processing Systems, 36:23826–23854, 2023.

Zelikman, E., Wu, Y., Mu, J., and Goodman, N. Star: Bootstrapping reasoning with reasoning. Advances in Neural Information Processing Systems, 35:15476–15488, 2022.

Zhang, K., Lv, A., Li, J., Wang, Y., Wang, F., Hu, H., and Yan, R. Stephint: Multi-level stepwise hints en-

hance reinforcement learning to reason. arXiv preprint arXiv:2507.02841, 2025a.

Zhang, X., Huang, Z., Li, Y., Ni, C., Chen, J., and Oymak, S. Bread: Branched rollouts from expert anchors bridge sft & rl for reasoning. arXiv preprint arXiv:2506.17211, 2025b.

Zhang, Y., Han, X., Li, H., Chen, K., and Lin, S. Complete chess games enable llm become a chess master. arXiv preprint arXiv:2501.17186, 2025c.

## A. Additional experimental details

This appendix specifies evaluation details that are important for reproducibility, but are too low-level to include in the main experimental narrative.

## A.1. Training data

Training position sources We consider two sources of training positions. First, a fixed dataset aligned with Chess-R1 (Hwang et al., 2025), containing 100,000 chess positions. Second, an on-policy dataset generated via engineplay against an engine opponent, where we record encountered game states (from both sides) and query the engine to obtain $\mu ( s , a )$ over the legal move list (Algorithm 2).

Fields per training position Each training example includes the position (FEN), an ordered legal-move list in strict UCI, and an engine-provided per-move value map $\mu ( s , a )$ used for reward computation. For the fixed dataset, each example also includes a dataset-provided solution move in UCI. For VAM-style selection prompts, we additionally specify an ordered candidate subset allowed\_moves that serves as the initial mask $\mathcal { M } _ { 0 } ;$ when a candidate subset is not provided by the dataset, $\mathcal { M } _ { 0 }$ can be set to the full legal set A(s) (§4.3).

## A.2. Rejection-sampled supervised fine-tuning (SFT) baseline

We construct the supervised baseline dataset from the same fixed 100,000-position dataset described in §A.1. Each example provides a dataset-provided solution move in UCI. We use the baseline prompt interface without a candidate subset and apply rejection sampling over model generations, keeping only generations whose parsed strict UCI move matches the dataset-provided solution (rejecting malformed or illegal outputs). This procedure yields 45,806 accepted examples for Qwen2.5-3B-Instruct (5.73% acceptance) and 71,181 accepted examples for Qwen2.5-7B-Instruct (8.90% acceptance). Table 1 lists the settings used for this baseline.

## A.3. Engine-derived verifier scores

Stockfish-based scoring We instantiate the verifier score $\mu ( s , a )$ using Stockfish (The Stockfish Developers, 2024). In the main experiments, we use the expected-score proxy $\mu _ { \mathrm { { e x p } } } \left( \mathrm { E q . } \ 8 \right)$ derived from the engine’s win/draw/loss probabilities, and define the target action within a candidate set as the $\mu \cdot$ -best move with a deterministic tie-break $( \ S 4 . 3 )$ . Section 5.3 defines additional reward variants $( \mu _ { \mathrm { w i n } }$ and $\mu _ { \mathrm { r a n k } } )$ used in the GRPO reward-function comparison.

Reward assignment under the candidate mask For both VAM and GRPO baselines, the reward is computed from the decoded move and the engine metadata for the current position. If the decoded move is well-formed, legal, and (for selection prompts) inside allowed\_moves, it receives the engine-derived score $\mu ( s , a )$ . Otherwise (missing tag, malformed UCI, illegal move, or out-of-candidate move), we treat the output as invalid and assign a fixed negative penalty, terminating the rollout (§4).

## A.4. Hyperparameters

Table 1 summarizes the main hyperparameters used across experiments. Shared optimization settings are matched across RL methods for fair comparisons, and VAM-specific controls are listed in the VAM column. The last two columns report the rejection-sampled SFT baseline settings.

## A.5. Prompt interface and parsing

Inputs per decision Each decision includes the FEN string, an explicit side-to-move field, and an ordered list of legal moves in strict lowercase UCI. In the candidate-restricted (selection) setting we additionally provide an ordered subset allowed\_moves (also called considered\_moves\_uci) and require the model to choose from that subset.

Required output The model must output a single move inside a <uci\_move> tag, with a strict lowercase UCI string as payload (for example, e2e4). Promotions include the piece letter (for example, e7e8q). Castling is represented in UCI as the king move (for example, e1g1 for white kingside castling).

Validity checks We treat any unparsable output, illegal move, or (for selection prompts) out-of-candidate move as invalid. For selection prompts, the decoded move must match one of the provided candidate strings exactly.

## A.6. Puzzle evaluation details

Data fields Each puzzle row provides fen, an ordered legal\_moves\_uci list, and a provided solution move in UCI. Some rows additionally include per-move value maps defined over the legal move list, stored as JSON keyed by UCI strings. The value maps include an engine-derived expected score or win-probability proxy in [0, 1].

Targets and metrics We treat the dataset-provided solution move as the target action $a ^ { \star }$ for evaluation. The main paper reports pass@1 accuracy (success if $\hat { a } = a ^ { \star } )$ , where invalid outputs count as failures. We additionally track format and legality compliance and, when value maps are available, the mean selected-move value $\mathbb { E } [ \mu ( s , \hat { a } ) ]$ ].

Table 1. Hyperparameters for GRPO, VAM, and SFT.

<table><tr><td>Parameter</td><td>GRPO</td><td>VAM</td><td>Parameter</td><td>SFT</td></tr><tr><td>Batch size</td><td>128</td><td>128</td><td>Global batch size</td><td>32</td></tr><tr><td>Mini batch size</td><td>128</td><td>128</td><td>Total source prompts</td><td>100,000</td></tr><tr><td>GRPO group size</td><td>8</td><td>8</td><td>Samples per prompt</td><td>8</td></tr><tr><td>Sampling temperature</td><td>1.0</td><td>1.0</td><td>Sampling temperature</td><td>0.6</td></tr><tr><td>Sampling top-p</td><td>1.0</td><td>1.0</td><td>Sampling top-p</td><td>0.9</td></tr><tr><td>Max response tokens</td><td>2000</td><td>2000</td><td>Max response tokens</td><td>2000</td></tr><tr><td>Epochs</td><td>1</td><td>1</td><td>Epochs</td><td>2</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td>1e-6</td><td>1e-6</td><td>Base learning rate</td><td>1e-5</td></tr><tr><td>LR schedule</td><td>constant</td><td>constant</td><td>LR schedule</td><td>cosine</td></tr><tr><td>Warmup ratio</td><td>10 steps</td><td>10 steps</td><td>Warmup ratio</td><td>0.1</td></tr><tr><td>Weight decay</td><td>0.01</td><td>0.01</td><td>Weight decay</td><td>0.01</td></tr><tr><td>Gradient clipping</td><td>1</td><td>1</td><td>Gradient clipping</td><td>1</td></tr><tr><td>KL loss coefficient</td><td>1e-3</td><td>1e-3</td><td>#SFT examples</td><td>Qwen2.5-3B: 45,806</td></tr><tr><td>Entropy coefficient</td><td>0</td><td>0</td><td></td><td>Qwen2.5-7B: 71,181</td></tr><tr><td>Clip ratio</td><td>0.2</td><td>0.2</td><td></td><td></td></tr><tr><td>Reward function</td><td> $\mu_{exp}/\mu_{win}/\mu_{rank}$ </td><td> $\mu_{exp}$ </td><td></td><td></td></tr><tr><td> $R_{max}$ </td><td>/</td><td>4</td><td></td><td></td></tr><tr><td>Online opponent</td><td>/</td><td>stockfish (skill 0, depth 1)</td><td></td><td></td></tr></table>

## A.7. Full-game evaluation details

Match setup Games start from the standard initial position and are capped at 200 plies. We do not automatically claim draws. Unless stated otherwise, we play 50 games per evaluation depth (depth 1 and depth 5), with per-depth color balancing and a deterministic shuffle.

Invalid moves If the model output is unparsable or illegal, we retry up to three attempts. Persistent failure is treated as resignation or forfeit, which affects win rate and can yield worst-case ACPL when no valid moves are played.

ACPL computation We compute centipawn loss using a separate Stockfish analyzer at depth 20, with approximately 1 second of analysis per move. Mate scores are mapped to 1000, and centipawn values are clamped to a fixed cap, matching the evaluation harness semantics. Per-move centipawn loss (CPL) uses the before versus after evaluations, with evaluations converted to the mover’s point of view and clipped at 0:

$$
\mathrm{CPL} = \max \left(0, E _ {\text { before }} - E _ {\text { after }}\right).\tag{10}
$$

We first compute per-game ACPL as the mean CPL over the model’s moves within that game. We then report two aggregates across a batch of games: (i) overall ACPL, the mean of per-game ACPL values (each game equally weighted), and (ii) ACPL per move, a move-weighted mean over all model moves in the batch.

## A.8. Prompt

We include the exact prompt templates used in our experiments in Figures 7 and 8.

![](images/873e7aeff60f5110489d7f6f3f41ccdf3c8ad3a1458520b914d3264fb224321c.jpg)  
Figure 7. Prompt used for Verbalized Action Masking (VAM).

![](images/ca4f8773b59b8c23656bce8785fdf7a847a9d2f8c91a81a3577f779de03aad48.jpg)  
Figure 8. Prompt used for baseline methods.