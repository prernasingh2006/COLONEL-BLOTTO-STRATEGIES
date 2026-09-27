# Colonel Blotto: Evolving Strategies

## Why this project

Colonel Blotto has no clean pure-strategy optimum — for any fixed allocation
across battlefields, some other allocation beats it. Instead of trying to
find "the answer," I wanted to actually watch that fact play out: build a
population of strategies, let them compete, and see whether anything
stabilizes or whether it just keeps cycling. It doesn't stabilize, and
figuring out *why*, precisely, turned into the actual point of the project.

## Approach

I derived the math before writing simulation code, and framed it in physics
terms rather than pure game-theory language, since that's the lens I
actually think in.

**The game:** two players split a budget across N labeled battlefields;
whoever allocates more wins a field; score is fields won minus fields lost
— zero-sum, symmetric.

**The dynamics:** replicator dynamics over a population of strategies,
$\dot x_i = x_i(f_i(x) - \bar\phi(x))$, where $f_i(x) = (Mx)_i$ is strategy
$i$'s fitness against the current population mix.

**Key derivation:** since the payoff matrix $M$ is antisymmetric
($M_{ij} = -M_{ji}$, zero-sum), the mean fitness term $\bar\phi(x) = x^TMx$
vanishes identically — the quadratic form of an antisymmetric matrix is
always zero. That collapses the dynamics to $\dot x_i = x_i(Mx)_i$, which is
Lotka-Volterra in disguise.

**Conserved quantity:** an interior fixed point $x^*$ satisfies $Mx^*=0$
(a genuine null vector of $M$ — guaranteed to exist when the strategy count
is odd, since $\det M = \det(-M^T) = (-1)^K\det M$ forces $\det M = 0$ for
odd $K$). Around that fixed point, $V(x) = \sum_i x_i^*\ln(x_i/x_i^*)$ is
conserved along trajectories — it's $-D_{KL}(x^*\|x)$, maximized only at
$x^*$. This is the exact same structure as energy conservation for a
pendulum or an orbit: trajectories sit on level sets of $V$ instead of
converging.

**The integrator matters:** simulating this with a plain Euler step
($x_i \leftarrow x_i(1+\eta f_i)$, renormalized) leaks $V$ every step —
same failure mode as Euler-integrating a Hamiltonian system, where
numerical error pumps energy into the orbit over time. RK4 preserves $V$
far better. I validated both against 3-strategy rock-paper-scissors
(where $x^* = (1/3,1/3,1/3)$ is known analytically) before trusting either
integrator on anything else.

## What's built and confirmed (`dynamics_core.py`)

- `payoff_matrix`, with an explicit antisymmetry + zero-diagonal check
  (`assert_valid_payoff_matrix`) run before anything downstream trusts it
- `find_interior_fixed_point`, solving $Mx^*=0$ via null space, validated
  against RPS
- `V(x, x_star)`, the conserved quantity
- `step_euler` and `step_rk4`

Tracking $V$ over 3000 steps on RPS: Euler's drift **accelerates** (the
gap between generations 0–1500 and 1500–3000 grows ~3.5x), while RK4's
drift stays linear and is roughly 5–6 orders of magnitude smaller over the
same run. That's not just "RK4 is better" as a general claim — it's a
specific, measured confirmation of the physics prediction.

## In progress (`strategies.py`)

Building a 7-archetype strategy pool to replace the RPS toy case with a
real Blotto payoff matrix:

- [x] `normalize_to_budget` — clip negatives, rescale to sum to budget,
      falls back to uniform if everything gets clipped to ~0
- [x] `uniform`
- [x] `sniper` — bare-majority concentration (win just enough fields
      outright, abandon the rest)
- [x] `focus_heavy` — heavier concentration on fewer fields
- [ ] `geometric_decay`
- [ ] `geometric_incay`
- [ ] `dirichlet_random` + `archetype_pool`

**Finding so far, worth flagging:** `sniper` beats `uniform` (margin +1),
but `focus_heavy` **loses** to `uniform` (margin −1) despite dominating
the fields it stacks. Two strategies that both "concentrate the budget"
land on opposite sides depending on whether the concentration count
clears the majority threshold. That's a real, derivable fact about this
game, not a coding artifact — worth remembering when picking the final
archetype menu, since it means the pool isn't just "spread vs. concentrate,"
there's a sharp threshold effect inside "concentrate."

## Next

1. Finish the remaining archetypes, build the real 7-strategy $M$
2. Check whether an interior fixed point actually exists for that specific
   set of 7 (odd dimension guarantees a null vector exists, not that it's
   positive — this is a genuinely open question, not assumed)
3. If it exists: rerun the Euler vs. RK4 comparison and phase-portrait
   plot on the real matrix instead of RPS
4. If it doesn't: that's a result too — means this particular menu has no
   coexistence equilibrium, worth checking which subset does
