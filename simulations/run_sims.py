import numpy as np
from simulate import SimulationDataset

SEED = 42
ns = 5

# Standard simulation sets:
n_set = [4, 8]
sim_sets = [
    {"sim": "r1", "dt": [5e-3], "nt": [1000], "n": n_set, "dim": [2, 3]},
    {"sim": "r2", "dt": [1e-3], "nt": [1000], "n": n_set, "dim": [2, 3]},
    {"sim": "spring", "dt": [1e-2], "nt": [1000], "n": n_set, "dim": [2, 3]},
    {"sim": "string", "dt": [1e-2], "nt": [1000], "n": [30], "dim": [2]},
    {"sim": "charge", "dt": [1e-3], "nt": [1000], "n": n_set, "dim": [2, 3]},
    {
        "sim": "superposition",
        "dt": [1e-3],
        "nt": [1000],
        "n": n_set,
        "dim": [2, 3],
    },
    {"sim": "damped", "dt": [2e-2], "nt": [1000], "n": n_set, "dim": [2, 3]},
    {
        "sim": "discontinuous",
        "dt": [1e-2],
        "nt": [1000],
        "n": n_set,
        "dim": [2, 3],
    },
]


for ss in sim_sets:
    sim = ss["sim"]
    if sim in ["r1", "r2", "charge", "spring", "damped", "discontinuous"]:
        for n in ss["n"]:
            for dim in ss["dim"]:
                dt = ss["dt"][0]
                nt = ss["nt"][0]
                s = SimulationDataset(sim, n=n, dim=dim, nt=nt, dt=dt)
                dt_formatted = "{:.0e}".format(dt)
                title = "sim={}_key={}_ns={}_n={}_dim={}_nt={}_dt={}".format(
                    sim, SEED, ns, n, dim, nt, dt_formatted
                )
                print("Running on", title)
                s.simulate(ns, key=SEED)
                np.save(f"test_docker_sim_data/docker_{title}.npy", s.data)
                np.save(
                    f"test_docker_sim_data/docker_accel_{title}.npy",
                    s.get_acceleration(),
                )
                print("Saved run!")
