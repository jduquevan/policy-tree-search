import hydra
import jax
import jax.numpy as jnp
import optax
import wandb

from flax import linen as nn
from functools import partial
from omegaconf import DictConfig
from typing import Dict


def compute_entropy_from_logits(logits: jnp.ndarray) -> jnp.ndarray:
    """
    logits shape: (..., action_dim)
    returns entropy shape: (...,)
    """
    policy = jax.nn.softmax(logits, axis=-1)
    log_policy = jnp.log(policy + 1e-10)
    return -jnp.sum(policy * log_policy, axis=-1)

def categorical_sample(rng_key, logits):
    """
    Sample an action from the categorical distribution
    defined by `logits` (1D).
    Return: (action, log_prob_of_that_action)
    """
    action = jax.random.categorical(rng_key, logits)
    log_probs = jax.nn.log_softmax(logits)
    log_prob = log_probs[action]
    return action, log_prob

def compute_gae_and_returns(
    rewards: jnp.ndarray,    # shape (T,)
    values: jnp.ndarray,     # shape (T,)
    gamma: float = 0.99,
    lam: float = 0.95,
    last_value: float = 0.0,
    mask: jnp.ndarray = None
):
    """
    Returns: (advantages, returns)
      advantages: shape (T,)
      returns:    shape (T,)
    """
    T = rewards.shape[0]
    if mask is None:
        mask = jnp.ones_like(rewards)

    def scan_fn(carry, t):
        gae, next_value = carry
        idx = T - 1 - t
        discount = gamma * mask[idx]
        delta = rewards[idx] + discount * next_value - values[idx]
        gae = delta + discount * lam * gae
        return (gae, values[idx]), gae

    init = (0.0, last_value)
    _, rev_advs = jax.lax.scan(scan_fn, init, jnp.arange(T))
    advantages = jnp.flip(rev_advs, axis=0)
    returns = advantages + values
    return advantages, returns


class MLP(nn.Module):
    num_layers: int
    features: int

    @nn.compact
    def __call__(self, x):
        for i in range(self.num_layers - 1):
            x = nn.Dense(self.features)(x)
            x = nn.relu(x)
        # Final layer
        x = nn.Dense(self.features)(x)
        return x


class GRUModule(nn.Module):
    """
    Combines:
    1) An input MLP
    2) A GRUCell
    3) A final MLP -> splits into value & policy heads
    """
    mlp_num_layers: int
    mlp_features: int
    gru_hidden_dim: int
    final_mlp_num_layers: int
    final_mlp_features: int
    action_dim: int

    @nn.compact
    def __call__(self, x, h):
        # 1) MLP
        x = MLP(num_layers=self.mlp_num_layers, features=self.mlp_features)(x)
        # 2) GRU cell
        new_h, gru_out = nn.GRUCell(name="gru_cell", features=self.gru_hidden_dim)(h, x)
        # 3) Final MLP
        hidden = MLP(num_layers=self.final_mlp_num_layers, features=self.final_mlp_features)(gru_out)
        # Value head
        value = nn.Dense(1)(hidden)  # shape (1,)
        # Policy head
        policy_logits = nn.Dense(self.action_dim)(hidden)  # shape (action_dim,)
        return value.squeeze(-1), policy_logits, new_h

def ipd(a1_single, a2_single):
    mat = jnp.array([[-1., -3.],
                    [ 0., -2.]])
    oh1 = jax.nn.one_hot(a1_single, 2)
    oh2 = jax.nn.one_hot(a2_single, 2)
    r1 = oh1 @ mat @ oh2
    r2 = oh2 @ mat @ oh1
    return r1, r2

def ipd_step(state, a1, a2, rollout_length=5):
    """
    state: shape (3,) => (old_a1, old_a2, t) [unbatched version]
    a1, a2: new actions
    returns (next_state, r1, r2, done)
    """
    old_a1, old_a2, t = state

    # Compute IPD rewards from new (a1, a2)
    r1, r2 = ipd(a1, a2)

    new_t = t + 1
    done = (new_t == rollout_length)  # <-- single boolean now

    reset_state = jnp.array([-1, -1, 0], dtype=jnp.int32)
    candidate_state = jnp.array([a1, a2, new_t], dtype=jnp.int32)

    # OLD (batched): next_state = jnp.where(done[:, None], reset_state, candidate_state)
    next_state = jnp.where(done, reset_state, candidate_state)
    return next_state, r1, r2, done

# ------------------------------------------------------------------------
# Batched IPD step, used by policy tree search
# ------------------------------------------------------------------------
def ipd_step_batch(states, a1, a2, rollout_length=5):
    """
    states: (N,3) => the last element is the step counter
    a1, a2: (N,) new actions
    returns next_states, r1, r2, done
      next_states: shape (N,3)
      r1, r2: (N,)
      done: (N,) boolean
    """
    # Parse
    old_a1 = states[:, 0]
    old_a2 = states[:, 1]
    t      = states[:, 2]

    # Compute IPD payoff
    r1, r2 = jax.vmap(ipd)(a1, a2)

    new_t = t + 1
    done = (new_t == rollout_length)  # shape (N,)

    # next_state if not done => [a1, a2, new_t], else => [-1, -1, 0]
    reset_state = jnp.array([-1, -1, 0], dtype=jnp.int32)  # shape (3,)
    candidate   = jnp.stack([a1, a2, new_t], axis=-1)      # shape (N,3)

    # We'll broadcast reset_state up to (N,3) so we can do a where
    reset_state_broadcast = jnp.tile(reset_state[None, :], (states.shape[0], 1))  # shape (N,3)

    # jnp.where(condition, x, y) => picks x where cond is True, else y
    next_states = jnp.where(done[:, None], reset_state_broadcast, candidate)

    return next_states, r1, r2, done

def always_cooperate_action(_rng_key, _obs):
    """
    Returns action=0 (Cooperate).
    """
    return 0

def always_defect_action(_rng_key, _obs):
    """
    Returns action=1 (Defect).
    """
    return 1

@partial(jax.jit, static_argnums=(2,))
def single_step_reinforce_update(
    params1,
    opt_state1,
    agent1,
    s,
    h,
    a1,
    r1,
    next_s,
    next_h,
    gamma=0.99,
    lr=1e-2
):
    """
    We'll compute advantage per example, then do a per-example update:
      grads[i], opt_state[i], params[i].
    Return (updated_params[i], updated_opt_state[i]) for i in [0..N-1].
    """
    def loss_fn(p, s_i, h_i, a_i, r_i, ns_i, nh_i):
        val_s, logits_s, _ = agent1.apply({"params": p}, s_i[0:2], h_i)
        val_ns, _, _       = agent1.apply({"params": p}, ns_i[0:2], nh_i)

        advantage = r_i + gamma * val_ns - val_s
        log_probs = jax.nn.log_softmax(logits_s)
        logp_a    = log_probs[a_i]

        policy_loss = -logp_a * advantage
        target = r_i + gamma * val_ns
        value_loss = (val_s - target) ** 2

        return policy_loss + 0.5 * value_loss

    def per_example_grad(p, s_i, h_i, a_i, r_i, ns_i, nh_i):
        return jax.grad(lambda pp: loss_fn(pp, s_i, h_i, a_i, r_i, ns_i, nh_i))(p)

    # Compute grads for each example => shape (N, ...)
    grads_all = jax.vmap(per_example_grad)(params1, s, h, a1, r1, next_s, next_h)

    # We'll do a vmap for the actual update using each example's (param, grad, opt_state).
    optimizer = optax.sgd(lr)

    def per_example_update(p, g, st):
        updates, new_st = optimizer.update(g, st, p)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_st

    updated_params, updated_states = jax.vmap(per_example_update)(params1, grads_all, opt_state1)
    return updated_params, updated_states

@partial(jax.jit, static_argnums=(2,3))
def expand_frontier_once(
    frontier,
    agent_params2,  # unbatched param for agent2
    agent1, agent2,
    gamma=0.99, lr=1e-2
):
    """
    frontier keys:
      - "N": int
      - "params1": shape (N, ...) 
      - "opt_state1": shape (N, ...)  # NEW: each node has an optimizer state
      - "h1": (N, hdim)
      - "state": (N,2)
      - "h2": (N, hdim)
    """
    N = frontier["N"]

    # --- (A) agent2 forward => a2
    def agent2_forward(state_i, h2_i):
        # Flip so agent2 sees [a2, a1] instead of [a1, a2]
        obs2 = jnp.array([state_i[1], state_i[0]], dtype=state_i.dtype)

        val2, logits2, new_h2 = agent2.apply(
            {"params": agent_params2}, obs2, h2_i
        )
        a2 = jnp.argmax(logits2)
        return a2, new_h2


    a2_all, next_h2_all = jax.vmap(agent2_forward)(
        frontier["state"], frontier["h2"]
    )

    # --- (B) agent1 forward => top-2
    def agent1_forward(params1_i, state_i, h1_i):
        val1, logits1, new_h1 = agent1.apply({"params": params1_i}, state_i[0:2], h1_i)
        top_vals, top_actions = jax.lax.top_k(logits1, k=2)
        return top_actions, new_h1

    top2_actions_all, next_h1_all = jax.vmap(agent1_forward)(
        frontier["params1"], frontier["state"], frontier["h1"]
    )

    # --- (C) replicate everything 2x => shape (2N)
    def repeat_2(x):
        return jnp.repeat(x, 2, axis=0)
    # For PyTrees (like params1 and opt_state1), we do the same tree_map trick:
    old_params1_2N   = jax.tree_util.tree_map(repeat_2, frontier["params1"])
    old_opt_state_2N = jax.tree_util.tree_map(repeat_2, frontier["opt_state1"])
    old_h1_2N        = repeat_2(frontier["h1"])
    old_state_2N     = repeat_2(frontier["state"])
    old_h2_2N        = repeat_2(frontier["h2"])
    next_h1_2N       = repeat_2(next_h1_all)
    next_h2_2N       = repeat_2(next_h2_all)
    a2_2N            = repeat_2(a2_all)

    # Flatten top-2 => shape(2N,)
    a1_2N = top2_actions_all.reshape(-1)

    # --- (D) ipd step => (2N)
    next_state_2N, r1_2N, r2_2N, done_2N = ipd_step_batch(
        old_state_2N, a1_2N, a2_2N
    )

    # --- (E) single-step update => returns (updated_params, updated_opt_state)
    updated_params1_2N, updated_opt_state1_2N = single_step_reinforce_update(
        old_params1_2N,      # param1
        old_opt_state_2N,    # opt_state1
        agent1,              # agent1
        old_state_2N,        # s
        old_h1_2N,           # h
        a1_2N,               # a1
        r1_2N,               # r1
        next_state_2N,       # next_s
        next_h1_2N,          # next_h
        gamma=gamma,
        lr=lr
    )

    # --- (F) new hidden states
    def forward_agent1(params1_i, s_i, h_i):
        _, _, new_h = agent1.apply({"params": params1_i}, s_i[0:2], h_i)
        return new_h

    def forward_agent2(params2, s_i, h_i):
        val, logits, new_h = agent2.apply({"params": params2}, s_i[0:2], h_i)
        return new_h

    new_h1_2N = jax.vmap(forward_agent1)(updated_params1_2N, next_state_2N, next_h1_2N)
    new_h2_2N = jax.vmap(
        forward_agent2,
        in_axes=(None, 0, 0)
    )(agent_params2, next_state_2N, next_h2_2N)

    # Build new frontier
    new_frontier = {
        "N": 2*N,
        "params1": updated_params1_2N,
        "opt_state1": updated_opt_state1_2N,  # keep new states
        "h1": new_h1_2N,
        "state": next_state_2N,
        "h2": new_h2_2N
    }
    return new_frontier

def get_batch_size(param_pytree):
    """Extract the leading batch dimension F from a PyTree of shape (F, ...)."""
    leaves = jax.tree_util.tree_leaves(param_pytree)
    if not leaves:
        raise ValueError("No leaves found in param pytree.")
    first_leaf = leaves[0]
    return first_leaf.shape[0]

@partial(jax.jit, static_argnums=(2,3,5))
def evaluate_frontier_batched(
    frontier_params: Dict,
    br_params: Dict,
    agent: GRUModule,
    best_response: GRUModule,
    rng_key: jax.random.PRNGKey,
    num_steps: int = 5
):
    F = get_batch_size(frontier_params)
    rng_keys = jax.random.split(rng_key, F)
    init_states = jnp.tile(jnp.array([0.0, 0.0, 0.0]), (F, 1))
    init_h1 = jnp.zeros((F, agent.gru_hidden_dim))
    init_h2 = jnp.zeros((F, best_response.gru_hidden_dim))

    def run_one(agent_p, rng_k, s0, h1_0, h2_0):
        _, rollout = generate_trajectory_functional(
            env_state=s0,
            h1=h1_0,
            h2=h2_0,
            agent1=agent,
            agent2=best_response,
            agent1_params=agent_p,
            agent2_params=br_params,
            rng_key=rng_k,
            num_steps=num_steps
        )
        return rollout  # shape (num_steps, ...)

    # shape => (F,) of rollouts, each a PyTree with shape (num_steps, ...)
    all_rollouts = jax.vmap(run_one)(frontier_params, rng_keys, init_states, init_h1, init_h2)

    # --- CHANGE: We'll compute returns and entropies from the "all_rollouts"
    # all_rollouts["r1"].shape => (F, num_steps)
    # all_rollouts["logits1"].shape => (F, num_steps, action_dim)

    def summarize_rollout(rollout):
        # rollout is shape (num_steps, ...)
        # sum of r1 => float
        total_r1 = jnp.sum(rollout["r1"])
        total_r2 = jnp.sum(rollout["r2"])

        # compute mean entropy for each agent
        ent1 = compute_entropy_from_logits(rollout["logits1"])  # shape (num_steps,)
        ent2 = compute_entropy_from_logits(rollout["logits2"])  # shape (num_steps,)
        avg_ent1 = jnp.mean(ent1)
        avg_ent2 = jnp.mean(ent2)

        return total_r1, total_r2, avg_ent1, avg_ent2

    (returns, br_returns, ent1_all, ent2_all) = jax.vmap(summarize_rollout)(all_rollouts)

    # You can choose to return multiple arrays or just returns. 
    # For selection, we pick "returns" (the agent's return).
    # But also let's return the other arrays for debugging/logging.
    return returns, br_returns, ent1_all, ent2_all

@partial(jax.jit, static_argnums=(2, 3, 5, 7))
def train_best_response_batched(
    br_params: Dict,         # best-response parameters
    frontier_params: Dict,   # shape (F, ...) frontier policies
    best_response: GRUModule,
    agent: GRUModule,
    opt_state_br,
    optimizer,
    rng_key,
    num_steps: int,
    gamma: float,
    lam: float,
    vf_coef: float,
):
    """
    1) Generate 1 rollout per frontier policy vs. current BR.
    2) Convert each rollout to the BR's perspective => single-agent trajectory.
    3) Combine all rollouts and do one gradient update on br_params.
    """
    # 1) generate all rollouts in parallel
    F = get_batch_size(frontier_params)
    rng_keys = jax.random.split(rng_key, F)

    init_states = jnp.tile(jnp.array([0.0, 0.0, 0.0]), (F, 1))
    init_h1 = jnp.zeros((F, agent.gru_hidden_dim))
    init_h2 = jnp.zeros((F, best_response.gru_hidden_dim))

    def run_one(agent_p, rng_k, s0, h1_0, h2_0):
        final_carry, rollout = generate_trajectory_functional(
            env_state=s0,
            h1=h1_0,
            h2=h2_0,
            agent1=agent,
            agent2=best_response,
            agent1_params=agent_p,
            agent2_params=br_params,
            rng_key=rng_k,
            num_steps=num_steps
        )
        return rollout  # each “rollout” is dict of arrays with shape (num_steps,)

    # shape => F of rollouts, each a PyTree of shape (num_steps, ...)
    all_rollouts = jax.vmap(run_one)(frontier_params, rng_keys, init_states, init_h1, init_h2)

    # 2) Flatten all rollouts so that we can do one big batch update.
    #    We combine them along the time dimension: total = F * num_steps
    #    For example, all_rollouts["r2"].shape => (F, num_steps)
    #    We'll reshape to (F*num_steps,).
    def stack_rollouts(x):
        # x shape => (F, num_steps, ...)
        # flatten => (F*num_steps, ...)
        return x.reshape((F * num_steps,) + x.shape[2:])

    # Build a single-agent trajectory from the perspective of BR => it uses actions “a2”, rewards “r2”
    # But states can remain the same or be re-labeled if needed.
    #   all_rollouts["env_state"].shape => (F, num_steps, 2)
    #   all_rollouts["a2"].shape       => (F, num_steps)
    #   all_rollouts["r2"].shape       => (F, num_steps)
    br_traj = {
        "states":  stack_rollouts(all_rollouts["env_state"]),
        "actions": stack_rollouts(all_rollouts["a2"]),
        "rewards": stack_rollouts(all_rollouts["r2"]),
    }

    # 3) Single gradient step for br_params
    def loss_fn(p):
        total_loss, metrics = actor_critic_loss(
            p,                 # best-response params
            best_response,     # best-response model
            br_traj,
            gamma=gamma,
            lam=lam,
            vf_coef=vf_coef
        )
        return total_loss, metrics

    (loss_val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(br_params)
    updates, new_opt_state_br = optimizer.update(grads, opt_state_br, br_params)
    new_br_params = optax.apply_updates(br_params, updates)

    return new_br_params, new_opt_state_br, loss_val, metrics

# @partial(jax.jit, static_argnums=(2,3,4))
def policy_tree_batched(
    frontier,
    agent_params2,
    agent1, agent2,
    max_depth=1,
    gamma=0.99,
    lr=1e-2
):
    """
    Repeats expand_frontier_once for max_depth steps.
    No Python for-loops over the frontier dimension, only a for-loop over depth.
    Each step doubles the frontier size => from N to 2N.
    """
    for _ in range(max_depth):
        frontier = expand_frontier_once(
            frontier, agent_params2,
            agent1, agent2,
            gamma=gamma, lr=lr
        )
    return frontier

def policy_tree_search(
    agent: GRUModule,
    best_response: GRUModule,
    agent_params: Dict,        
    br_params: Dict,           
    opt_state_agent,           
    opt_state_br,              
    optimizer,                 
    rng_key: jax.random.PRNGKey,
    cfg: DictConfig,
):
    frontier_size = cfg.frontier_size
    max_depth     = cfg.max_tree_depth
    num_iters     = cfg.num_iterations
    num_steps     = cfg.rollout_length

    gamma   = cfg.gamma
    lam     = cfg.lam
    vf_coef = cfg.vf_coef

    for iteration in range(num_iters):
        # A) Create the initial frontier
        frontier = {
            "N": frontier_size,
            "params1": replicate_params(agent_params, frontier_size),
            "opt_state1": replicate_opt_state(opt_state_agent, frontier_size),
            "h1": jnp.zeros((frontier_size, agent.gru_hidden_dim)),
            "state": jnp.zeros((frontier_size, 2)),
            "h2": jnp.zeros((frontier_size, best_response.gru_hidden_dim)),
        }

        # B) Expand the frontier
        frontier = policy_tree_batched(
            frontier,
            br_params,
            agent, best_response,
            max_depth=max_depth,
            gamma=gamma,
            lr=cfg.tree_lr
        )

        # C) Multiple best-response updates
        rng_key, br_rng = jax.random.split(rng_key)
        current_br_params = br_params
        current_opt_state_br = opt_state_br
        current_key = br_rng

        for br_iter in range(cfg.num_br_updates):
            current_key, subkey = jax.random.split(current_key)
            (current_br_params,
             current_opt_state_br,
             br_loss,
             br_metrics
            ) = train_best_response_batched(
                current_br_params,
                frontier["params1"],
                best_response,
                agent,
                current_opt_state_br,
                optimizer,
                subkey,
                num_steps=num_steps,
                gamma=gamma,
                lam=lam,
                vf_coef=vf_coef
            )
            # optional: wandb.log({"br_iter_loss": float(br_loss), "br_iter": br_iter})

        # Update global br_params & opt_state_br
        br_params = current_br_params
        opt_state_br = current_opt_state_br

        # D) Evaluate each frontier policy
        rng_key, eval_key = jax.random.split(rng_key)
        (returns,
         br_returns,
         ent1_all,
         ent2_all) = evaluate_frontier_batched(
            frontier["params1"],
            br_params,
            agent,
            best_response,
            eval_key,
            num_steps
        )

        # E) Pick best policy
        best_idx = jnp.argmax(returns)
        best_return = returns[best_idx]
        best_response_return = br_returns[best_idx]
        agent_entropy = ent1_all[best_idx]
        best_response_entropy = ent2_all[best_idx]
        best_policy = jax.tree_util.tree_map(lambda x: x[best_idx], frontier["params1"])

        # F) Update agent_params => best policy
        agent_params = best_policy

        # Logging
        print(f"[Iter {iteration+1}/{num_iters}] BR loss = {float(br_loss):.4f}, best frontier return = {float(best_return):.4f}")
        wandb.log({
            "iteration": iteration + 1,
            "br_loss": float(br_loss),
            "best_agent_return": float(best_return),
            "best_response_return": float(best_response_return),
            "agent_entropy": float(agent_entropy),
            "best_response_entropy": float(best_response_entropy),
        })

        # Evaluation
        if (iteration + 1) % cfg.eval_every == 0:
            evaluate_against_fixed_opponents(
                agent_params=agent_params,
                br_params=br_params,
                agent_module=agent,
                br_module=best_response,
                rng_key=rng_key,
                cfg=cfg
            )

    return agent_params, br_params

def actor_critic_loss(
    agent_params,
    agent: GRUModule, 
    trajectory: Dict[str, jnp.ndarray],
    gamma: float = 0.99, 
    lam: float = 0.95, 
    vf_coef: float = 0.5
):
    """Actor-critic loss function for a single GRU-based agent.

    Args:
        agent_params: Parameters of the GRU-based policy-value network.
        agent: A GRUModule or similar Flax module representing the agent.
        trajectory: A dictionary containing keys 'states', 'actions', 'rewards'.
            - trajectory["states"]  has shape (T, state_dim)
            - trajectory["actions"] has shape (T,)
            - trajectory["rewards"] has shape (T,)
        gamma: Discount factor.
        lam: GAE (Generalized Advantage Estimation) lambda parameter.
        vf_coef: Coefficient for the value loss term.

    Returns:
        total_loss: A scalar representing the combined policy and value loss.
        metrics: A dictionary containing separate loss components for diagnostics.
    """
    # Unpack the trajectory
    states  = trajectory["states"]   # shape (T, state_dim)
    actions = trajectory["actions"]  # shape (T,)
    rewards = trajectory["rewards"]  # shape (T,)

    T = states.shape[0]

    # Initialize GRU hidden state (assuming shape (T, hidden_dim))
    h_zeros = jnp.zeros((T, agent.gru_hidden_dim))

    # Define forward pass for the agent
    def agent_forward(carry, s):
        h = carry
        value, logits, new_h = agent.apply({"params": agent_params}, s[0:2], h)
        return new_h, (value, logits)

    # Run the agent forward over the entire trajectory
    _, (vals, logs) = jax.lax.scan(agent_forward, h_zeros[0], states)

    # Get final value estimate for last time step
    last_val = vals[-1]

    # -- Compute GAE and returns --
    adv, ret = compute_gae_and_returns(rewards, vals, gamma, lam, last_val)

    # Compute log probabilities of the taken actions
    def get_logp(logits, a):
        return jax.nn.log_softmax(logits)[a]

    logp = jax.vmap(get_logp)(logs, actions)

    # Policy loss: negative of advantage-weighted log probabilities
    policy_loss = -jnp.mean(adv * logp)

    # Value loss: MSE between predicted values and returns
    value_loss = jnp.mean((ret - vals)**2)

    # Combine losses with value loss scaled by vf_coef
    total_loss = policy_loss + vf_coef * value_loss

    # Metrics for logging/monitoring
    metrics = {
        "loss_total": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
    }

    return total_loss, metrics

def generate_trajectory_functional(
    env_state,
    h1,
    h2,
    agent1,
    agent2,
    agent1_params,
    agent2_params,
    rng_key,
    num_steps=5
):
    """
    env_state: shape (2,) or (...,2)
    h1, h2: agent hidden states
    We'll store extra info (logits1, logits2) for debugging/entropy.
    """

    def step_fn(carry, t):
        (env_s, h1, h2, rng_key) = carry

        def agent_forward(agent, params, state, hidden_state):
            val, logits, new_h = agent.apply({"params": params}, state[0:2], hidden_state)
            return  val, logits, new_h

        obs1 = env_s
        obs1 = env_s[0:2]
        # For agent2: flip => [env_s[1], env_s[0]]
        obs2 = jnp.array([obs1[..., 1], obs1[..., 0]])

        value1, logits1, new_h1 = agent_forward(agent1, agent1_params, obs1, h1)
        value2, logits2, new_h2 = agent_forward(agent2, agent2_params, obs2, h2)

        # Sample actions
        rng_key, subkey1, subkey2 = jax.random.split(rng_key, 3)
        (a1, logp1) = categorical_sample(subkey1, logits1)
        (a2, logp2) = categorical_sample(subkey2, logits2)
        a1_f = a1.astype(jnp.float32)
        a2_f = a2.astype(jnp.float32)

        # Step environment => [a1,a2]
        next_env_s, r1, r2, done = ipd_step(env_s, a1, a2, rollout_length=num_steps)
        t_f = next_env_s[-1].astype(jnp.float32)
        next_env_s = jnp.stack([a1_f, a2_f, t_f], axis=-1)

        # --- CHANGE: Store logits1 and logits2 in the transition
        transition = {
            "env_state": env_s,
            "a1": a1,
            "a2": a2,
            "r1": r1,
            "r2": r2,
            "value1": value1,
            "value2": value2,
            "logits1": logits1,   # NEW
            "logits2": logits2,   # NEW
        }
        new_carry = (next_env_s, new_h1, new_h2, rng_key)
        return new_carry, transition

    carry_0 = (env_state, h1, h2, rng_key)
    final_carry, rollout = jax.lax.scan(step_fn, carry_0, jnp.arange(num_steps))
    # rollout: shape (num_steps, ...) PyTree

    return final_carry, rollout

def replicate_opt_state(opt_state, N):
    """
    Replicate an Optax optimizer state PyTree N times along a new leading axis.
    We'll do the same strategy as replicate_params, but for the state.
    """
    def replicate_leaf(leaf):
        # leaf is an ndarray. We add a new axis of size N at the front.
        return jnp.stack([leaf] * N, axis=0)

    return jax.tree_util.tree_map(replicate_leaf, opt_state)

def replicate_params(params, N):
    """
    Replicate a parameter PyTree `params` N times along a new leading axis (size=N).
    """
    def replicate_leaf(leaf):
        # leaf is an ndarray. We add a new axis of size N at the front.
        # e.g. if leaf.shape == (10, 10), result becomes (N, 10, 10).
        return jnp.stack([leaf] * N, axis=0)
    
    return jax.tree_map(replicate_leaf, params)

def evaluate_agent_vs_fixed_action(
    agent_params,
    agent_module,
    fixed_action_fn,
    rng_key,
    num_steps=5
):
    """
    Runs a single rollout of length num_steps, with (agent vs. fixed_action_fn).
    Returns (agent_return, fixed_opponent_return).
    """
    env_s = jnp.array([0.0, 0.0, 0.0])  # IPD initial state
    h_agent = jnp.zeros((agent_module.gru_hidden_dim,))

    # We'll collect the cumulative reward for each side
    agent_return = 0.0
    fixed_return = 0.0

    def agent_forward(params, obs, hidden):
        val, logits, new_h = agent_module.apply({"params": params}, obs[0:2], hidden)
        return val, logits, new_h

    carry_rng = rng_key
    for _ in range(num_steps):
        # (1) Agent picks action
        carry_rng, subkey = jax.random.split(carry_rng)
        val_agent, logits_agent, new_h_agent = agent_forward(agent_params, env_s, h_agent)
        a_agent = jax.random.categorical(subkey, logits_agent)

        # (2) Fixed action is always 0 or 1:
        a_fixed = fixed_action_fn(carry_rng, env_s)

        # (3) Step the IPD environment
        next_s, rA, rF, done = ipd_step(env_s, a_agent, a_fixed)

        # Update cumulative returns
        agent_return += rA
        fixed_return += rF

        # Update agent hidden state and env state
        env_s = next_s
        h_agent = new_h_agent

    return agent_return, fixed_return

def evaluate_against_fixed_opponents(agent_params, br_params, agent_module, br_module, rng_key, cfg):
    """
    Evaluate both agent_module and br_module vs. Always-Cooperate and Always-Defect.
    Log the results to wandb.
    """
    # Split RNGs for each matchup
    rng_key, ac_agent_key = jax.random.split(rng_key)
    rng_key, ad_agent_key = jax.random.split(rng_key)
    rng_key, ac_br_key    = jax.random.split(rng_key)
    rng_key, ad_br_key    = jax.random.split(rng_key)

    # 1) Agent vs. Always-Cooperate
    agent_vs_coop_rA, coop_r  = evaluate_agent_vs_fixed_action(
        agent_params, agent_module, always_cooperate_action,
        ac_agent_key, cfg.rollout_length
    )

    # 2) Agent vs. Always-Defect
    agent_vs_def_rA, def_r    = evaluate_agent_vs_fixed_action(
        agent_params, agent_module, always_defect_action,
        ad_agent_key, cfg.rollout_length
    )

    # 3) Best-Response vs. Always-Cooperate
    br_vs_coop_rB, coop_r2    = evaluate_agent_vs_fixed_action(
        br_params, br_module, always_cooperate_action,
        ac_br_key, cfg.rollout_length
    )

    # 4) Best-Response vs. Always-Defect
    br_vs_def_rB, def_r2      = evaluate_agent_vs_fixed_action(
        br_params, br_module, always_defect_action,
        ad_br_key, cfg.rollout_length
    )

    # Log these to wandb
    wandb.log({
        "agent_vs_AC": float(agent_vs_coop_rA),
        "AC_return_vs_agent": float(coop_r),
        "agent_vs_AD": float(agent_vs_def_rA),
        "AD_return_vs_agent": float(def_r),

        "br_vs_AC": float(br_vs_coop_rB),
        "AC_return_vs_br": float(coop_r2),
        "br_vs_AD": float(br_vs_def_rB),
        "AD_return_vs_br": float(def_r2),
    })

@hydra.main(
    version_base=None,
    config_path="/home/mila/j/juan.duque/projects/policy-tree-search/configs",
    config_name="ipd.yaml"
)
def main(cfg: DictConfig):
    # 1) Initialize wandb
    wandb.init(project="policy-tree-search")

    # 2) Instantiate agent and best_response (same GRUModule architecture)
    agent = GRUModule(
        mlp_num_layers=cfg.mlp.num_layers,
        mlp_features=cfg.mlp.features,
        gru_hidden_dim=cfg.gru.hidden_dim,
        final_mlp_num_layers=cfg.output_mlp.num_layers,
        final_mlp_features=cfg.output_mlp.features,
        action_dim=2
    )
    best_response = GRUModule(
        mlp_num_layers=cfg.mlp.num_layers,
        mlp_features=cfg.mlp.features,
        gru_hidden_dim=cfg.gru.hidden_dim,
        final_mlp_num_layers=cfg.output_mlp.num_layers,
        final_mlp_features=cfg.output_mlp.features,
        action_dim=2
    )

    # 3) Initialize parameters
    rng = jax.random.PRNGKey(cfg.get("seed", 42))
    x_init = jnp.array([0.0, 0.0]) 
    h_init = jnp.zeros((cfg.gru.hidden_dim,))
    agent_params = agent.init(rng, x_init, h_init)["params"]
    br_params    = best_response.init(rng, x_init, h_init)["params"]

    # 4) Create an optimizer & opt states for both agent and br
    learning_rate = cfg.get("learning_rate", 1e-3)
    optimizer = optax.adam(learning_rate)
    opt_state_agent = optimizer.init(agent_params)
    opt_state_br    = optimizer.init(br_params)

    # 5) Run policy_tree_search
    agent_params, br_params = policy_tree_search(
        agent=agent,
        best_response=best_response,
        agent_params=agent_params,
        br_params=br_params,
        opt_state_agent=opt_state_agent,
        opt_state_br=opt_state_br,
        optimizer=optimizer,
        rng_key=rng,
        cfg=cfg
    )

    # 7) Wrap up
    wandb.finish()

if __name__ == "__main__":
    main()
