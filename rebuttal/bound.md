# False adjacencies at a positive threshold (answer to Kmr8-Q2, d2sj-W1)

Setting: skeleton mode, λ = 0, subsets S_1, …, S_m of size k, K = k(k−1)/2.
Ĝ_{S_j} is the training local graph on S_j, and for a candidate G,

  A(G) = (1/m) Σ_j (1/K) · #{pairs in S_j whose adjacency differs between π_{S_j}(G) and Ĝ_{S_j}}.

For a pair e = {x, y}, let

  c_e = #{ j : x, y ∈ S_j and x, y are non-adjacent in Ĝ_{S_j} },

the number of training local graphs that contain the pair and separate it.
c_e is computed from the data alone.

## Lemma (budget)

For every candidate G,

  Σ_{e ∈ adj(G)} c_e ≤ m · K · A(G).

Hence every G ∈ C_α satisfies Σ_{e ∈ adj(G)} c_e ≤ B_α := m · K · τ_α.

*Proof.* If x and y are adjacent in G, no set separates them in G, so they are adjacent in the marginal M_{S_j}(G) of every S_j that contains both (Def. 1(i)).
Each j counted in c_e is a subset where π_{S_j}(G) has the edge and Ĝ_{S_j} does not, so it adds 1/K to the j-th term of A(G).
Distinct pairs, or distinct subsets, are distinct terms, and all other terms are non-negative. Summing over e ∈ adj(G) gives the claim. ∎

## Proposition (false adjacencies)

Let F(G) be the set of false adjacencies of G: pairs adjacent in G but non-adjacent in the true marginal M_V(G*).
Let c_e* count only the subsets in which the training local graph is *correct* on e (non-adjacent, as in M_{S_j}(G*)).
If c_e* ≥ c for every non-adjacent pair e of M_V(G*), then every G ∈ C_α has

  |F(G)| ≤ m · K · τ_α / c.

*Proof.* c_e ≥ c_e* ≥ c for every e ∈ F(G), so the Lemma gives c·|F(G)| ≤ B_α. ∎

At τ_α = 0 with c ≥ 1, this is Proposition 2: separation cover plus correct local graphs on the covering subsets gives no false adjacency.
The bound degrades linearly in τ_α.

## Data-driven corollary (no knowledge of G* needed)

Any pair with c_e > B_α is non-adjacent in every admitted graph. More generally, the pairs that admitted graphs can share are limited by the budget B_α.
The rejected-adjacency claims this corollary certifies are correct whenever the local graphs are correct on those pairs.

## What to check numerically (rebuttal/bound_eval.py)

Per run, for the main setting:

- B_α, the distribution of c_e, and c_min* = min over non-adjacent e of c_e*;
- the bound m K τ_α / c_min* (when c_min* ≥ 1);
- the tight version given the c_e: the largest number of false adjacencies whose c_e sum fits within B_α;
- the observed maximum |F(G)| over G ∈ C_α;
- how many pairs the data-driven corollary certifies (c_e > B_α).

We expect the bound to be loose at the operating threshold. τ_α is an average over m·K pair slots, so B_α allows many disagreements anywhere, while one false adjacency costs only c_e ≈ (number of subsets containing the pair). The response should say this plainly. The value of the result is that it is the first finite-threshold statement, and that it reduces to Proposition 2 as τ_α → 0.
