import tensorflow as tf

independent_variable_names = ["t", "x"]
independent_variable_index = {}
for (i, s) in enumerate(independent_variable_names):
    independent_variable_index[s] = i
it = independent_variable_index["t"]
ix = independent_variable_index["x"]
independent_variable_labels = ["$t$", "$x$"]
n_dim = len(independent_variable_names)

dependent_variable_names = ["n", "P", "ux"]
dependent_variable_index = {}
for (i, s) in enumerate(dependent_variable_names):
    dependent_variable_index[s] = i
i_n = dependent_variable_index["n"]
iP = dependent_variable_index["P"]
iux = dependent_variable_index["ux"]
dependent_variable_labels = ["$n$", "$P$", "$u_x$"]
n_var = len(dependent_variable_names)

ɣ = 1.4    # Adiabatic index = Cp/Cv
μ0 = 1.0   # Normalized vacuum permeability
m = 1.0    # Plasma article mass


def pde_rho(X, Y, del_Y):
    nX = X.shape[0]
    t = tf.reshape(X[:, it], (nX, 1))
    x = tf.reshape(X[:, ix], (nX, 1))
    (rho, P, ux) = Y
    (del_rho, del_momden, del_E) = del_Y
    drho_dt = tf.reshape(del_rho[:, it], (nX, 1))
    drho_dx = tf.reshape(del_rho[:, ix], (nX, 1))
    dmomden_dt = tf.reshape(del_momden[:, it], (nX, 1))
    dmomden_dx = tf.reshape(del_momdem[:, ix], (nX, 1))
    dE_dt = tf.reshape(del_E[:, it], (nX, 1))
    dE_dx = tf.reshape(del_E[:, ix], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = drho_dt + dmomden_dx
    return G


def pde_P(X, Y, del_Y):
    nX = X.shape[0]
    t = tf.reshape(X[:, it], (nX, 1))
    x = tf.reshape(X[:, ix], (nX, 1))
    (rho, P, ux) = Y
    (del_rho, del_momden, del_E) = del_Y
    drho_dt = tf.reshape(del_rho[:, it], (nX, 1))
    drho_dx = tf.reshape(del_rho[:, ix], (nX, 1))
    dmomden_dt = tf.reshape(del_momden[:, it], (nX, 1))
    dmomden_dx = tf.reshape(del_momdem[:, ix], (nX, 1))
    dE_dt = tf.reshape(del_E[:, it], (nX, 1))
    dE_dx = tf.reshape(del_E[:, ix], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = dmomden_dt 
    return G


def pde_ux(X, Y, del_Y):
    nX = X.shape[0]
    # t = tf.reshape(X[:, it], (nX, 1))
    # x = tf.reshape(X[:, ix], (nX, 1))
    (n, P, ux) = Y
    (del_n, del_P, del_ux) = del_Y
    # (del2_n, del2_P, del2_ux) = del2_Y
    # dn_dt = tf.reshape(del_n[:, it], (nX, 1))
    # dn_dx = tf.reshape(del_n[:, ix], (nX, 1))
    # dP_dt = tf.reshape(del_P[:, it], (nX, 1))
    dP_dx = tf.reshape(del_P[:, ix], (nX, 1))
    dux_dt = tf.reshape(del_ux[:, it], (nX, 1))
    dux_dx = tf.reshape(del_ux[:, ix], (nX, 1))

    # G is a Tensor of shape (n, 1).
    G = n*(dux_dt + ux*dux_dx) + dP_dx/m
    return G


# Make a list of all of the differential equations.
de = [
    pde_n,
    pde_P,
    pde_ux,
]


if __name__ == "__main__":
    print(f"independent_variable_names = {independent_variable_names}")
    print(f"independent_variable_index = {independent_variable_index}")
    print(f"it = {it}")
    print(f"ix = {ix}")
    print(f"independent_variable_labels = {independent_variable_labels}")
    print(f"n_dim = {n_dim}")

    print(f"dependent_variable_names = {dependent_variable_names}")
    print(f"dependent_variable_index = {dependent_variable_index}")
    print(f"i_n = {i_n}")
    print(f"iP = {iP}")
    print(f"iux = {iux}")
    print(f"dependent_variable_labels = {dependent_variable_labels}")
    print(f"n_var = {n_var}")

    print(f"ɣ = {ɣ}")
    print(f"μ0 = {μ0}")
    print(f"m = {m}")
