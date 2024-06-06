import numpy as np


def spring_force(df, eps=1e-2):
    pos_cols = ["dx", "dy"] + (["dz"] if "dz" in df.columns else [])
    dr = df[pos_cols].to_numpy()
    r = (df.r + eps).to_numpy()[:, None]
    return (r - 1) * (dr / r)


def r1_force(df, eps=1e-2):
    pos_cols = ["dx", "dy"] + (["dz"] if "dz" in df.columns else [])
    dr = np.array(df[pos_cols])
    r = (df.r + eps).to_numpy()[:, None]
    m1 = df.m1.to_numpy()[:, None]
    m2 = df.m2.to_numpy()[:, None]
    return ((m1 * m2) / r) * (dr / r)


def r2_force(df, eps=1e-2):
    pos_cols = ["dx", "dy"] + (["dz"] if "dz" in df.columns else [])
    dr = np.array(df[pos_cols])
    r = (df.r + eps).to_numpy()[:, None]
    m1 = df.m1.to_numpy()[:, None]
    m2 = df.m2.to_numpy()[:, None]
    return ((m1 * m2) / r**2) * (dr / r)


def charge_force(df, eps=1e-2):
    pos_cols = ["dx", "dy"] + (["dz"] if "dz" in df.columns else [])
    dr = np.array(df[pos_cols])
    r = (df.r + eps).to_numpy()[:, None]
    q1 = df.q1.to_numpy()[:, None]
    q2 = df.q2.to_numpy()[:, None]
    return ((q1 * q2) / r**2) * (dr / r)
