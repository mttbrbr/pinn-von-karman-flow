import os
# Specifichiamo il backend PRIMA di importare deepxde
os.environ["DDE_BACKEND"] = "pytorch"

import deepxde as dde
import numpy as np
import torch

# Creazione cartella per i checkpoint
if not os.path.exists("model"):
    os.makedirs("model")

# --- 1. PARAMETRI FISICI ---
Re = 20
nu = 1 / Re
T_max = 10.0

# --- 2. GEOMETRIA E DOMINIO TEMPORALE ---
rect = dde.geometry.Rectangle([-5, -2], [15, 2])
cyl = dde.geometry.Disk([0, 0], 0.5)
geom = dde.geometry.CSGDifference(rect, cyl)
timedomain = dde.geometry.TimeDomain(0, T_max)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# --- 3. DEFINIZIONE FISICA (Navier-Stokes 2D) ---
def navier_stokes(x, y):
    # x = [x, y, t], y = [u, v, p]
    u, v, p = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    
    # Derivate del primo ordine
    du_x = dde.grad.jacobian(y, x, i=0, j=0)
    du_y = dde.grad.jacobian(y, x, i=0, j=1)
    du_t = dde.grad.jacobian(y, x, i=0, j=2)
    
    dv_x = dde.grad.jacobian(y, x, i=1, j=0)
    dv_y = dde.grad.jacobian(y, x, i=1, j=1)
    dv_t = dde.grad.jacobian(y, x, i=1, j=2)
    
    dp_x = dde.grad.jacobian(y, x, i=2, j=0)
    dp_y = dde.grad.jacobian(y, x, i=2, j=1)
    
    # Derivate del secondo ordine
    du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    du_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    dv_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    dv_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)

    # Equazioni di Navier-Stokes
    continuity = du_x + dv_y
    mom_u = du_t + u*du_x + v*du_y + dp_x - nu*(du_xx + du_yy)
    mom_v = dv_t + u*dv_x + v*dv_y + dp_y - nu*(dv_xx + dv_yy)
    
    return [continuity, mom_u, mom_v]

# --- 4. CONDIZIONI AL CONTORNO E INIZIALI ---
# Funzioni di supporto per identificare i bordi
def boundary_inlet(x, on_boundary):
    return on_boundary and np.isclose(x[0], -5)

def boundary_outlet(x, on_boundary):
    return on_boundary and np.isclose(x[0], 15)

def boundary_cylinder(x, on_boundary):
    return on_boundary and cyl.on_boundary(x[:2])

# BCs: Inlet (u=1, v=0)
bc_u_in = dde.icbc.DirichletBC(geomtime, lambda x: 1.0, boundary_inlet, component=0)
bc_v_in = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, boundary_inlet, component=1)

# BCs: Cylinder No-slip (u=0, v=0)
bc_u_cyl = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, boundary_cylinder, component=0)
bc_v_cyl = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, boundary_cylinder, component=1)

# BCs: Outlet (p=0 per stabilizzare la pressione)
bc_p_out = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, boundary_outlet, component=2)

# IC: Initial Condition a t=0 (velocità uniforme)
ic_u = dde.icbc.IC(geomtime, lambda x: 1.0, lambda _, on_initial: on_initial, component=0)
ic_v = dde.icbc.IC(geomtime, lambda x: 0.0, lambda _, on_initial: on_initial, component=1)

# --- 5. DATASET E ARCHITETTURA RETE ---
data = dde.data.TimePDE(
    geomtime, navier_stokes,
    [bc_u_in, bc_v_in, bc_u_cyl, bc_v_cyl, bc_p_out, ic_u, ic_v],
    num_domain=10000, num_boundary=3000, num_initial=2000
)

# Rete Neurale: 3 input -> 6 layer da 64 neuroni -> 3 output
net = dde.nn.FNN([3] + [64] * 6 + [3], "tanh", "Glorot normal")
model = dde.Model(data, net)

# Callback per salvataggio pesi
checkpointer = dde.callbacks.ModelCheckpoint(
    "model/pinn_karman", period=1000, verbose=1
)

# --- 6. TRAINING STRATEGY ---
# Fase 1: Adam (Global search)
print("Starting Adam training...")
model.compile("adam", lr=1e-3)
model.train(iterations=20000, callbacks=[checkpointer])

# Fase 2: L-BFGS (Fine-tuning)
print("Starting L-BFGS refinement...")
model.compile("L-BFGS")
model.train()

# --- 7. ESPORTAZIONE DATI ---
# Generazione griglia di punti per visualizzazione
x_pts = np.linspace(-5, 15, 200)
y_pts = np.linspace(-2, 2, 100)
t_pts = np.linspace(0, T_max, 20) # 20 frame temporali
X, Y, T = np.meshgrid(x_pts, y_pts, t_pts)
test_points = np.vstack((X.flatten(), Y.flatten(), T.flatten())).T

predictions = model.predict(test_points)
u_pred, v_pred, p_pred = predictions[:, 0:1], predictions[:, 1:2], predictions[:, 2:3]

# Salvataggio CSV
output_data = np.hstack((test_points, u_pred, v_pred, p_pred))
np.savetxt("results_karman.csv", output_data, header="x,y,t,u,v,p", delimiter=",", comments='')

print("Training finished. File 'results_karman.csv' ready for ParaView.")
