import dynamiqs as dq
import jax.numpy as jnp
import optax
import qontrol as ql

dq.set_precision('double')
dq.set_progress_meter(False)

# hyper parameters
n = 5  # system size
K = -0.2 * 2.0 * jnp.pi  # Kerr nonlinearity
time = 40.0  # total simulation time
dt = 2.0  # control dt
seed_amplitude = 1e-3  # pulse seed amplitude
learning_rate = 1e-4  # learning rate for optimizer

# define model to optimize
a = dq.destroy(5)
H0 = 0.5 * K * dq.dag(a) @ dq.dag(a) @ a @ a
H1s = [a + dq.dag(a), 1j * (a - dq.dag(a))]

def H_pwc(drive_params):
    H = H0
    for idx, _H1 in enumerate(H1s):
        H += dq.pwc(tsave, drive_params[idx], _H1)
    return H

initial_states = [dq.basis(n, 0), dq.basis(n, 1)]
# We can track the behavior of observables by passing them to the model. Here we track
# the state populations
exp_ops = [dq.basis(n, idx) @ dq.dag(dq.basis(n, idx)) for idx in range(n)]
ntimes = int(time // dt) + 1
tsave = jnp.linspace(0, time, ntimes)
model = ql.sesolve_model(H_pwc, initial_states, tsave, exp_ops=exp_ops)

# define optimization
parameters = seed_amplitude * jnp.ones((len(H1s), ntimes - 1))
target_states = [-1j * dq.basis(n, 1), 1j * dq.basis(n, 0)]
cost = ql.cost.coherent_infidelity(target_states=target_states, target_cost=0.001)
cost += ql.cost.coherent_infidelity(target_states=target_states, target_cost=0.001)
optimizer = optax.adam(learning_rate=0.0001)
opt_options = {"verbose": True, "plot": False, "save_period": 1}
dq_options = dq.Options(save_states=False, progress_meter=None)
filepath = f'save_every_{opt_options['save_period']}_epochs2'

# run optimization
opt_params = ql.optimize(
    parameters,
    cost,
    model,
    optimizer=optimizer,
    opt_options=opt_options,
    dq_options=dq_options,
    filepath=filepath
)