# Doing full derivations well

The user explicitly asked for **heavy — full derivations** in every Math
section. That makes the Math section the differentiator: a paper-explainer
that can do real derivations is qualitatively more useful than one that
narrates equations. This file is the craft guide.

Read this once before drafting your first Math section, then refer back when
you're stuck.

## What "full derivation" actually means

Not: "we re-state every line from the paper."
Yes: "we fill in the steps the paper omitted."

Authors omit steps because their reviewers don't need them. Your reader does.
The standard is:

> A careful graduate student should be able to reproduce every line on a
> whiteboard given only the prompts you've written, without referring back
> to the paper.

If you're writing "by the chain rule" and that's the whole justification, ask
yourself whether a chain-rule application here has any subtle indexing or
substitution that a reader would slip on. If yes, write the intermediate
expression. If no, the one-line label is fine.

## Structure of a full derivation

1. **Pre-amble.** Before the first equation, name (a) the variables, (b) the
   assumptions, (c) what you're trying to derive. Two or three lines.
2. **Goal equation, stated.** Write the result you're about to derive. Don't
   make the reader guess where you're going.
3. **The derivation, line by line.** Each line is one algebraic manipulation
   with a one-line justification on the right (or below, if it's longer).
4. **Result re-stated, in plain words.** Close the loop: "this says that the
   loss decomposes into a data-fit term and a Fisher-weighted distance to the
   previous parameters."

## The one-line justification per step

Every derivation step deserves a label. Common labels:

| Step type | Label |
|---|---|
| Algebraic manipulation | `(rearrange)`, `(distribute)`, `(complete the square)` |
| Calculus | `(chain rule)`, `(product rule)`, `(integration by parts)` |
| Probability | `(Bayes)`, `(law of total prob.)`, `(linearity of E)` |
| Inequalities | `(Jensen)`, `(Cauchy–Schwarz)`, `(triangle ineq.)` |
| Limits / approximations | `(Taylor to 2nd order around θ*)`, `(o(h²) dropped)` |
| Substitution | `(plug in defn of L)`, `(use update rule)` |

When the step is a non-obvious algebraic move (e.g. "multiply numerator and
denominator by 1/N to expose the empirical mean"), spell it out.

## Worked micro-example: deriving the EWC loss

Goal: from the assumption that the posterior over parameters is Gaussian
around the previous task's optimum θ*, derive the EWC regularised loss.

**Pre-amble.** Let θ ∈ ℝᵈ be model parameters, $L_{\text{new}}(\theta)$ the
log-likelihood of the current task, and $L_{\text{old}}(\theta)$ that of the
previous task. We want to maximise $L_{\text{old}}(\theta) + L_{\text{new}}(\theta)$
under the constraint that θ stays close to the previous task's optimum θ*.

**Step 1.** Approximate the posterior over θ given the old task as a
Gaussian centred on θ*:

$$
p(\theta \mid \mathcal{D}_{\text{old}}) \approx \mathcal{N}(\theta;\, \theta^*,\, F^{-1})
\qquad (\text{Laplace approx.\ at the MAP})
$$

where $F$ is the Fisher information matrix at θ*.

**Step 2.** Take logs:

$$
\log p(\theta \mid \mathcal{D}_{\text{old}})
= -\tfrac{1}{2}(\theta - \theta^*)^\top F (\theta - \theta^*) + \text{const.}
\qquad (\text{Gaussian log-density})
$$

**Step 3.** Diagonal Fisher: assume $F$ is diagonal with entries $F_i$, so
$(\theta - \theta^*)^\top F (\theta - \theta^*) = \sum_i F_i (\theta_i - \theta^*_i)^2$.
This is the assumption EWC actually makes; flag it because the off-diagonal
terms it discards are precisely what later methods (SI, OWM) try to recover.

**Step 4.** Write the joint objective. Maximising
$\log p(\mathcal{D}_{\text{new}} \mid \theta) + \log p(\theta \mid \mathcal{D}_{\text{old}})$
is equivalent (up to a sign and a scaling constant $\lambda$) to minimising:

$$
\mathcal{L}_{\text{EWC}}(\theta)
= \mathcal{L}_{\text{new}}(\theta) + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2
\qquad (\text{negate, define } \mathcal{L} \equiv -\log p)
$$

**Result re-stated.** The EWC loss is the new task's loss plus a quadratic
penalty that pulls each parameter back toward its old-task optimum, with the
strength of the pull set by that parameter's Fisher information — the higher
the Fisher (i.e. the more the old task's loss curved at θ*), the harder it is
to move that parameter.

That's a full derivation: every algebraic move has a label, every assumption
is named (Laplace, diagonal Fisher), and the closing paragraph tells the
reader what they just learned.

## Common pitfalls

- **Skipping over a "simple" linear-algebra move.** Things like "by symmetry
  of $F$..." trip readers who haven't internalised that $F$ is symmetric
  positive semi-definite. Say it.
- **Mixing notation with the paper.** If the paper uses $\theta_t$ for "task
  $t$ parameters" and your derivation uses $\theta_{\text{new}}$, the reader
  can't cross-reference. Pick one and explain when you switch.
- **Index shenanigans.** Derivations involving sums over data points or
  parameters often have subtle index errors. Use distinct letters: $i$ for
  parameters, $n$ for data points, $t$ for tasks. Don't reuse.
- **Pretending an approximation isn't an approximation.** Laplace, Taylor,
  diagonal Fisher, mean-field — every one of these is a real assumption with
  real failure modes. Name them at the moment you use them, not at the end.

## When the paper's central result is empirical, not analytical

Some papers' central contribution is an empirical observation, not a theorem.
For these, "Math" becomes "the formal apparatus that lets us state and
measure the observation". Derive:

- The metric the paper uses (e.g. linear probe accuracy, CKA, BWT).
- Why that metric is the right thing to measure for the claim.
- Any back-of-envelope analysis the paper does (e.g. capacity scaling).

Don't fabricate a derivation when there isn't one. An honest "this paper's
contribution is empirical; the formal apparatus is just BWT and ACC" is
better than a forced theorem.
