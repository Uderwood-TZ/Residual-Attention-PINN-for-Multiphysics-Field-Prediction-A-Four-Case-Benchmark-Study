import os
import time
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

MODEL_TYPE = "resattn"
CASE_NAME = "case4_v2_phifront"
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 50000
LEARNING_RATE = 1e-3

N_INTERIOR_TRAIN = 3600
N_INTERIOR_VAL = 1600
N_BOUNDARY_EACH_TRAIN = 700
N_BOUNDARY_EACH_VAL = 300

NX_PLOT = 201
VAL_EVERY = 100
PRINT_EVERY = 200
SAVE_EVERY = 5000

NU = 0.020
ALPHA_T = 0.018
DIFF_PHI = 0.022
ETA_UV = 0.045
C_T = 0.070
PHI_COUP = 0.100
JOULE = 0.040

W_CONT = 1.0
W_MX = 1.0
W_MY = 1.0
W_T = 1.6
W_PHI = 2.5
W_BC = 8.0

HIDDEN = 128
NUM_BLOCKS = 8

OUTPUT_DIR = f"outputs_{CASE_NAME}_{MODEL_TYPE}_e{EPOCHS}_seed{SEED}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)
torch.set_default_dtype(torch.float32)


def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
    return gradients(gradients(u, x, order=1), x, order=order - 1)


def grad_wrt_xy(u, xy):
    g = torch.autograd.grad(
        u, xy,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    return g[:, 0:1], g[:, 1:2]


def to_numpy(x):
    return x.detach().cpu().numpy()


def psi_true(x, y):
    shear_center = 0.26 + 0.54 * x - 0.06 * torch.sin(1.8 * math.pi * x + 0.10)
    return (
        0.075 * torch.sin(1.10 * math.pi * x + 0.20) * torch.sin(0.95 * math.pi * y + 0.10)
        + 0.030 * torch.exp(-26.0 * ((x - 0.70) ** 2 + (y - 0.78) ** 2))
        + 0.040 * torch.tanh(12.0 * (y - shear_center))
    )


def p_true_func(x, y):
    return (
        0.095 * torch.cos(1.05 * math.pi * x + 0.10) * torch.sin(0.90 * math.pi * y + 0.20)
        + 0.105 * torch.exp(-30.0 * ((x - 0.77) ** 2 + (y - 0.36) ** 2))
        - 0.070 * torch.exp(-28.0 * ((x - 0.24) ** 2 + (y - 0.64) ** 2))
    )


def T_true_func(x, y):
    return (
        0.96
        + 0.13 * x - 0.09 * y
        + 0.060 * torch.sin(1.0 * math.pi * x + 0.25) * torch.sin(0.82 * math.pi * y + 0.10)
        + 0.055 * torch.exp(-28.0 * ((x - 0.34) ** 2 + (y - 0.30) ** 2))
        + 0.040 * torch.exp(-24.0 * ((x - 0.78) ** 2 + (y - 0.80) ** 2))
    )


def phi_true_func(x, y):
    front_center = 0.56 - 0.14 * torch.exp(-16.0 * (x - 0.52) ** 2) + 0.06 * torch.sin(1.8 * math.pi * x + 0.10)
    return (
        0.42 * torch.tanh(10.0 * (y - front_center))
        + 0.045 * torch.exp(-32.0 * ((x - 0.67) ** 2 + (y - 0.74) ** 2))
        + 0.035 * torch.sin(1.45 * math.pi * x + 0.10) * torch.sin(0.92 * math.pi * y + 0.05)
        + 0.040 * x
    )


def exact_fields_from_xy(xy, need_grad=True):
    with torch.enable_grad():
        xy_local = xy.clone().detach().requires_grad_(True)
        x = xy_local[:, 0:1]
        y = xy_local[:, 1:2]

        psi = psi_true(x, y)
        u = gradients(psi, y, order=1)
        v = -gradients(psi, x, order=1)
        p = p_true_func(x, y)
        T = T_true_func(x, y)
        phi = phi_true_func(x, y)

    if need_grad:
        return xy_local, x, y, u, v, p, T, phi
    return (
        xy_local.detach(), x.detach(), y.detach(),
        u.detach(), v.detach(), p.detach(), T.detach(), phi.detach()
    )


class ResAttnBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.attn1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.attn2 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.res_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.act = nn.Tanh()

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.fc2(h)
        a = self.act(self.attn1(x))
        a = torch.sigmoid(self.attn2(a))
        return x + self.res_scale * (h * a)


class ResAttnPINN(nn.Module):
    def __init__(self, in_dim=2, out_dim=5, hidden_dim=128, num_blocks=8):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResAttnBlock(hidden_dim) for _ in range(num_blocks)])
        self.mid = nn.Linear(hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, out_dim)
        self.act = nn.Tanh()

    def forward(self, x):
        h = self.act(self.in_layer(x))
        for blk in self.blocks:
            h = blk(h)
        h = self.act(self.mid(h))
        return self.out_layer(h)


def sample_interior(n):
    pts = np.random.rand(n, 2).astype(np.float32)
    return torch.tensor(pts, dtype=torch.float32, device=DEVICE)


def sample_boundary_side(n, side):
    s = np.random.rand(n, 1).astype(np.float32)
    if side == "left":
        pts = np.concatenate([np.zeros_like(s), s], axis=1)
    elif side == "right":
        pts = np.concatenate([np.ones_like(s), s], axis=1)
    elif side == "bottom":
        pts = np.concatenate([s, np.zeros_like(s)], axis=1)
    elif side == "top":
        pts = np.concatenate([s, np.ones_like(s)], axis=1)
    else:
        raise ValueError("Unknown side")
    return torch.tensor(pts, dtype=torch.float32, device=DEVICE)


@dataclass
class DataPack:
    interior_train: torch.Tensor
    interior_val: torch.Tensor
    boundary_train: torch.Tensor
    boundary_val: torch.Tensor


def build_dataset():
    interior_train = sample_interior(N_INTERIOR_TRAIN)
    interior_val = sample_interior(N_INTERIOR_VAL)
    b_train = torch.cat([
        sample_boundary_side(N_BOUNDARY_EACH_TRAIN, "left"),
        sample_boundary_side(N_BOUNDARY_EACH_TRAIN, "right"),
        sample_boundary_side(N_BOUNDARY_EACH_TRAIN, "bottom"),
        sample_boundary_side(N_BOUNDARY_EACH_TRAIN, "top"),
    ], dim=0)
    b_val = torch.cat([
        sample_boundary_side(N_BOUNDARY_EACH_VAL, "left"),
        sample_boundary_side(N_BOUNDARY_EACH_VAL, "right"),
        sample_boundary_side(N_BOUNDARY_EACH_VAL, "bottom"),
        sample_boundary_side(N_BOUNDARY_EACH_VAL, "top"),
    ], dim=0)
    return DataPack(interior_train, interior_val, b_train, b_val)


def compute_sources_from_exact(xy):
    _, x, y, u, v, p, T, phi = exact_fields_from_xy(xy, need_grad=True)

    ux = gradients(u, x, 1)
    uy = gradients(u, y, 1)
    vx = gradients(v, x, 1)
    vy = gradients(v, y, 1)
    px = gradients(p, x, 1)
    py = gradients(p, y, 1)

    Tx = gradients(T, x, 1)
    Ty = gradients(T, y, 1)
    phix = gradients(phi, x, 1)
    phiy = gradients(phi, y, 1)

    uxx = gradients(ux, x, 1)
    uyy = gradients(uy, y, 1)
    vxx = gradients(vx, x, 1)
    vyy = gradients(vy, y, 1)
    Txx = gradients(Tx, x, 1)
    Tyy = gradients(Ty, y, 1)
    phixx = gradients(phix, x, 1)
    phiyy = gradients(phiy, y, 1)

    s_cont = ux + vy
    s_mx = u * ux + v * uy + px - NU * (uxx + uyy) - ETA_UV * phix + C_T * Tx
    s_my = u * vx + v * vy + py - NU * (vxx + vyy) - ETA_UV * phiy + C_T * Ty
    s_T = u * Tx + v * Ty - ALPHA_T * (Txx + Tyy) + JOULE * (phix ** 2 + phiy ** 2)
    s_phi = -DIFF_PHI * (phixx + phiyy) + PHI_COUP * (u * phix + v * phiy) - 0.05 * T * phi
    return s_cont.detach(), s_mx.detach(), s_my.detach(), s_T.detach(), s_phi.detach()


def loss_pde(model, xy):
    xy = xy.clone().detach().requires_grad_(True)

    pred = model(xy)
    u = pred[:, 0:1]
    v = pred[:, 1:2]
    p = pred[:, 2:3]
    T = pred[:, 3:4]
    phi = pred[:, 4:5]

    ux, uy = grad_wrt_xy(u, xy)
    vx, vy = grad_wrt_xy(v, xy)
    px, py = grad_wrt_xy(p, xy)
    Tx, Ty = grad_wrt_xy(T, xy)
    phix, phiy = grad_wrt_xy(phi, xy)

    uxx, _ = grad_wrt_xy(ux, xy)
    _, uyy = grad_wrt_xy(uy, xy)
    vxx, _ = grad_wrt_xy(vx, xy)
    _, vyy = grad_wrt_xy(vy, xy)
    Txx, _ = grad_wrt_xy(Tx, xy)
    _, Tyy = grad_wrt_xy(Ty, xy)
    phixx, _ = grad_wrt_xy(phix, xy)
    _, phiyy = grad_wrt_xy(phiy, xy)

    s_cont, s_mx, s_my, s_T, s_phi = compute_sources_from_exact(xy)

    r_cont = ux + vy - s_cont
    r_mx = u * ux + v * uy + px - NU * (uxx + uyy) - ETA_UV * phix + C_T * Tx - s_mx
    r_my = u * vx + v * vy + py - NU * (vxx + vyy) - ETA_UV * phiy + C_T * Ty - s_my
    r_T = u * Tx + v * Ty - ALPHA_T * (Txx + Tyy) + JOULE * (phix ** 2 + phiy ** 2) - s_T
    r_phi = -DIFF_PHI * (phixx + phiyy) + PHI_COUP * (u * phix + v * phiy) - 0.05 * T * phi - s_phi

    l_cont = torch.mean(r_cont ** 2)
    l_mx = torch.mean(r_mx ** 2)
    l_my = torch.mean(r_my ** 2)
    l_T = torch.mean(r_T ** 2)
    l_phi = torch.mean(r_phi ** 2)

    total = W_CONT * l_cont + W_MX * l_mx + W_MY * l_my + W_T * l_T + W_PHI * l_phi
    parts = {
        "cont": float(l_cont.detach().cpu()),
        "mx": float(l_mx.detach().cpu()),
        "my": float(l_my.detach().cpu()),
        "T": float(l_T.detach().cpu()),
        "phi": float(l_phi.detach().cpu()),
    }
    return total, parts


def loss_bc(model, xy_bc):
    pred = model(xy_bc)
    _, _, _, u_t, v_t, p_t, T_t, phi_t = exact_fields_from_xy(xy_bc, need_grad=False)

    loss_u = torch.mean((pred[:, 0:1] - u_t) ** 2)
    loss_v = torch.mean((pred[:, 1:2] - v_t) ** 2)
    loss_p = torch.mean((pred[:, 2:3] - p_t) ** 2)
    loss_T = torch.mean((pred[:, 3:4] - T_t) ** 2)
    loss_phi = torch.mean((pred[:, 4:5] - phi_t) ** 2)
    return loss_u + loss_v + loss_p + loss_T + loss_phi


def save_checkpoint(epoch, model, optimizer, scheduler, history):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
    }
    torch.save(ckpt, os.path.join(OUTPUT_DIR, f"checkpoint_epoch_{epoch}.pt"))


def train_model(model, data_pack):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[5000, 15000, 30000, 40000],
        gamma=0.35
    )

    history = {
        "epochs": [],
        "train_total": [],
        "train_pde": [],
        "train_bc": [],
        "val_total_raw": [],
        "val_total": [],
        "best_val": float("inf"),
        "best_epoch": -1,
    }

    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        train_pde, _ = loss_pde(model, data_pack.interior_train)
        train_bc = loss_bc(model, data_pack.boundary_train)
        train_total = train_pde + W_BC * train_bc

        train_total.backward()
        optimizer.step()
        scheduler.step()

        history["epochs"].append(epoch)
        history["train_total"].append(float(train_total.detach().cpu()))
        history["train_pde"].append(float(train_pde.detach().cpu()))
        history["train_bc"].append(float(train_bc.detach().cpu()))

        if epoch % VAL_EVERY == 0 or epoch == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                val_bc = loss_bc(model, data_pack.boundary_val)

            val_pde, _ = loss_pde(model, data_pack.interior_val)
            val_total = val_pde + W_BC * val_bc
            raw_val = float(val_total.detach().cpu())
            history["val_total_raw"].append(raw_val)

            smoothed = raw_val if len(history["val_total_raw"]) < 7 else float(np.mean(history["val_total_raw"][-7:]))
            history["val_total"].append(smoothed)

            if raw_val < history["best_val"]:
                history["best_val"] = raw_val
                history["best_epoch"] = epoch
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))

        if epoch % PRINT_EVERY == 0 or epoch == EPOCHS - 1:
            lr_now = optimizer.param_groups[0]["lr"]
            msg = (
                f"Epoch {epoch:5d} | LR: {lr_now:.2e} | "
                f"Train Total: {float(train_total.detach().cpu()):.6e} | "
                f"Train PDE: {float(train_pde.detach().cpu()):.6e} | "
                f"Train BC: {float(train_bc.detach().cpu()):.6e}"
            )
            if history["val_total_raw"]:
                msg += f" | Val Total: {history['val_total_raw'][-1]:.6e}"
            print(msg)

        if epoch > 0 and (epoch % SAVE_EVERY == 0):
            save_checkpoint(epoch, model, optimizer, scheduler, history)

    elapsed = time.time() - start_time
    return history, elapsed


def save_loss_plots(history, save_dir):
    epochs = np.array(history["epochs"], dtype=np.int32)
    val_epochs = np.array([e for e in history["epochs"] if e % VAL_EVERY == 0 or e == EPOCHS - 1], dtype=np.int32)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history["train_total"], label="Train Total")
    plt.plot(epochs, history["train_pde"], label="Train PDE")
    plt.plot(epochs, history["train_bc"], label="Train BC")
    plt.plot(val_epochs, history["val_total_raw"], alpha=0.35, label="Val Total (raw)")
    plt.plot(val_epochs, history["val_total"], linewidth=3.0, label="Val Total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.semilogy(epochs, history["train_total"], label="Train Total")
    plt.semilogy(epochs, history["train_pde"], label="Train PDE")
    plt.semilogy(epochs, history["train_bc"], label="Train BC")
    plt.semilogy(val_epochs, history["val_total_raw"], alpha=0.35, label="Val Total (raw)")
    plt.semilogy(val_epochs, history["val_total"], linewidth=3.0, label="Val Total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log)")
    plt.title("Log-loss curves")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve_log.png"), dpi=300)
    plt.close()

    np.savez(
        os.path.join(save_dir, "history_raw.npz"),
        epochs=epochs,
        train_total=np.array(history["train_total"]),
        train_pde=np.array(history["train_pde"]),
        train_bc=np.array(history["train_bc"]),
        val_epochs=val_epochs,
        val_total_raw=np.array(history["val_total_raw"]),
        val_total=np.array(history["val_total"]),
    )


def build_plot_grid(nx=NX_PLOT):
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    y = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    pts = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
    return X, Y, torch.tensor(pts, dtype=torch.float32, device=DEVICE)


def metric_dict(pred, true):
    err = pred - true
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    l2 = float(np.linalg.norm(err) / (np.linalg.norm(true) + 1e-12))
    max_abs = float(np.max(np.abs(err)))
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "L2": l2, "MAX_ABS": max_abs}


def save_field_txt(path, X, Y, Z):
    arr = np.column_stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)])
    np.savetxt(path, arr, fmt="%.8e", header="x y value", comments="")


def save_triplet(field_name, X, Y, pred, true, save_dir):
    err = np.abs(pred - true)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    items = [
        (pred, f"{field_name} prediction"),
        (true, f"{field_name} exact"),
        (err, f"{field_name} abs error"),
    ]
    for ax, (Z, title) in zip(axes, items):
        im = ax.contourf(X, Y, Z, levels=120)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{field_name}_triplet.png"), dpi=300)
    plt.close()

    save_field_txt(os.path.join(save_dir, f"{field_name}_pred.txt"), X, Y, pred)
    save_field_txt(os.path.join(save_dir, f"{field_name}_true.txt"), X, Y, true)
    save_field_txt(os.path.join(save_dir, f"{field_name}_abs_err.txt"), X, Y, err)


def post_process_and_save(model, elapsed, history):
    best_path = os.path.join(OUTPUT_DIR, "best_model.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        print(f"[Info] 已恢复 best_model.pt (epoch={history['best_epoch']}).")

    model.eval()

    X, Y, pts = build_plot_grid()
    with torch.no_grad():
        pred = model(pts).cpu().numpy()

    _, _, _, u_t, v_t, p_t, T_t, phi_t = exact_fields_from_xy(pts, need_grad=False)

    u_true = to_numpy(u_t).reshape(X.shape)
    v_true = to_numpy(v_t).reshape(X.shape)
    p_true = to_numpy(p_t).reshape(X.shape)
    T_true = to_numpy(T_t).reshape(X.shape)
    phi_true = to_numpy(phi_t).reshape(X.shape)

    u_pred = pred[:, 0].reshape(X.shape)
    v_pred = pred[:, 1].reshape(X.shape)
    p_pred = pred[:, 2].reshape(X.shape)
    T_pred = pred[:, 3].reshape(X.shape)
    phi_pred = pred[:, 4].reshape(X.shape)

    metrics_all = {
        "u": metric_dict(u_pred, u_true),
        "v": metric_dict(v_pred, v_true),
        "p": metric_dict(p_pred, p_true),
        "T": metric_dict(T_pred, T_true),
        "phi": metric_dict(phi_pred, phi_true),
    }

    save_triplet("u", X, Y, u_pred, u_true, OUTPUT_DIR)
    save_triplet("v", X, Y, v_pred, v_true, OUTPUT_DIR)
    save_triplet("p", X, Y, p_pred, p_true, OUTPUT_DIR)
    save_triplet("T", X, Y, T_pred, T_true, OUTPUT_DIR)
    save_triplet("phi", X, Y, phi_pred, phi_true, OUTPUT_DIR)

    save_loss_plots(history, OUTPUT_DIR)

    with open(os.path.join(OUTPUT_DIR, "statistics.txt"), "w", encoding="utf-8") as f:
        f.write("[meta]\n")
        f.write(f"CASE_NAME = {CASE_NAME}\n")
        f.write(f"MODEL_TYPE = {MODEL_TYPE}\n")
        f.write(f"SEED = {SEED}\n")
        f.write(f"DEVICE = {DEVICE}\n")
        f.write(f"EPOCHS = {EPOCHS}\n")
        f.write(f"LEARNING_RATE = {LEARNING_RATE}\n")
        f.write(f"HIDDEN = {HIDDEN}\n")
        f.write(f"NUM_BLOCKS = {NUM_BLOCKS}\n")
        f.write(f"W_CONT = {W_CONT}\n")
        f.write(f"W_MX = {W_MX}\n")
        f.write(f"W_MY = {W_MY}\n")
        f.write(f"W_T = {W_T}\n")
        f.write(f"W_PHI = {W_PHI}\n")
        f.write(f"W_BC = {W_BC}\n")
        f.write(f"ELAPSED_SECONDS = {elapsed:.6f}\n")
        f.write(f"BEST_VAL_TOTAL = {history['best_val']:.8e}\n")
        f.write(f"BEST_EPOCH = {history['best_epoch']}\n\n")
        f.write("[error_analysis]\n")
        rmse_rank = sorted([(k, v["RMSE"]) for k, v in metrics_all.items()], key=lambda x: x[1], reverse=True)
        f.write("RMSE_rank = " + " > ".join([f"{k}:{v:.4e}" for k, v in rmse_rank]) + "\n\n")
        for name, md in metrics_all.items():
            f.write(f"[{name}]\n")
            for k, v in md.items():
                f.write(f"{k} = {v:.8e}\n")
            f.write("\n")

    print(f"\n训练完成，结果已保存到：{OUTPUT_DIR}")


def main():
    print("=" * 72)
    print("Case 4 剩余注意力 PINN | 50000 轮主文件")
    print(f"DEVICE = {DEVICE}")
    print(f"OUTPUT_DIR = {OUTPUT_DIR}")
    print("=" * 72)
    data_pack = build_dataset()
    model = ResAttnPINN(in_dim=2, out_dim=5, hidden_dim=HIDDEN, num_blocks=NUM_BLOCKS).to(DEVICE)
    history, elapsed = train_model(model, data_pack)
    post_process_and_save(model, elapsed, history)


if __name__ == "__main__":
    main()
