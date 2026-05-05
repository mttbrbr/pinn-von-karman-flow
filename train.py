import deepxde as dde
import numpy as np
import torch
import os

# Creazione cartella per i pesi se non esiste
if not os.path.exists("model"):
    os.makedirs("model")

# --- 1. PARAMETRI ---
Re = 200
nu = 1 / Re

# --- 2. GEOMETRIA ---
rect = dde.geometry.Rectangle([-5, -2], [15, 2])
cyl = dde.geometry.Disk([0, 0], 0.5)
geom = dde.geometry.CSGDifference(rect, cyl)
timedomain = dde.geometry.TimeDomain(0, 10)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# --- 3. FISICA (Navier-Stokes) ---
def navier_stokes(x, y):
    u, v, p = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    
    du_x = dde.grad.jacobian(y, x, i=0, j=0)
    du_y = dde.grad.jacobian(y, x, i=0, j=1)
    du_t = dde.grad.jacobian(y, x, i=0, j=2)
    dv_x = dde.grad.jacobian(y, x, i=1, j=0)
    dv_y = dde.grad.jacobian(y, x, i=1, j=1)
    dv_t = dde.grad.jacobian(y, x, i=1, j=2)
    dp_x = dde.grad.jacobian(y, x, i=2, j=0)
    dp_y = dde.grad.jacobian(y, x, i=2, j=1)
    du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    du_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    dv_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    dv_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)

    continuity = du_x + dv_y
    mom_u = du_t + u*du_x + v*du_y + dp_x - nu*(du_xx + du_yy)
    mom_v = dv_t + u*dv_x + v*dv_y + dp_y - nu*(dv_xx + dv_yy)
    
    return [continuity, mom_u, mom_v]

# --- 4. CONDIZIONI AL CONTORNO ---
def boundary_inlet(x, on_boundary):
    return on_boundary and np.isclose(x[0], -5)

def boundary_cylinder(x, on_boundary):
    return on_boundary and cyl.on_boundary(x[:2])

bc_u_in = dde.icbc.DirichletBC(geomtime, lambda x: 1.0, boundary_inlet, component=0)
bc_v_in = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, boundary_inlet, component=1)
bc_u_cyl = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, boundary_cylinder, component=0)
bc_v_cyl = dde.icbc.DirichletBC(geomtime, lambda x: 0.0, boundary_cylinder, component=1)

# --- 5. SETUP MODELLO ---
data = dde.data.TimePDE(
    geomtime, navier_stokes,
    [bc_u_in, bc_v_in, bc_u_cyl, bc_v_cyl],
    num_domain=8000, num_boundary=2000, num_initial=1000
)

net = dde.nn.FNN([3] + [50] * 6 + [3], "tanh", "Glorot normal")
model = dde.Model(data, net)

# Callback per salvare i pesi ogni 1000 iterazioni
checkpointer = dde.callbacks.ModelCheckpoint(
    "model/pinn_karman", save_every=1000, verbose=1
)

# --- 6. TRAINING ---
model.compile("adam", lr=1e-3)
model.train(iterations=15000, callbacks=[checkpointer])

# --- 7. POST-PROCESSING (Per Paraview) ---
# Creiamo una griglia di punti per la predizione finale
x = np.linspace(-5, 15, 200)
y = np.linspace(-2, 2, 100)
t = np.linspace(0, 10, 50)
X, Y, T = np.meshgrid(x, y, t)
test_points = np.vstack((X.flatten(), Y.flatten(), T.flatten())).T

predictions = model.predict(test_points)
u_pred, v_pred, p_pred = predictions[:, 0:1], predictions[:, 1:2], predictions[:, 2:3]

# Salvataggio in formato CSV (facilmente leggibile da Paraview)
output_data = np.hstack((test_points, u_pred, v_pred, p_pred))
np.savetxt("results_karman.csv", output_data, header="x,y,t,u,v,p", delimiter=",", comments='')

print("Training completato. File 'results_karman.csv' generato per Paraview.")
