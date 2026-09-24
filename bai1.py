import numpy as np
import matplotlib.pyplot as plt

# --- Giữ nguyên cấu trúc của bạn ---
def cost(x):
    return x**2 - 2

def grad(x):
    return 2*x

def myGD1(eta, x0):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x, it)

(x1, it1) = myGD1(.1, 4.0)
print('Solution x1 = %f, cost = %f, obtained after %d iterations' % (x1[-1], cost(x1[-1]), it1))

# --- Phần vẽ đồ thị  ---
fig, ax = plt.subplots(figsize=(6, 5))
X = np.linspace(-5, 5, 200)
ax.plot(X, cost(X), 'b-')

# Vẽ điểm cực tiểu tìm được
ax.plot(x1[-1], cost(x1[-1]), 'ro', markeredgecolor='k')
ax.set_title(f'$f(x) = x^2 - 2; x_0 = 4; \\eta = 0.10$')

# Nhãn hiển thị bên dưới
txt = ax.set_xlabel(f'cost = {cost(x1[-1]):.2f}, grad = {grad(x1[-1]):.4f}')

# Rê chuột cập nhật cost và grad
def on_move(event):
    if event.inaxes == ax and event.xdata is not None:
        x_val = event.xdata
        txt.set_text(f'cost = {cost(x_val):.2f}, grad = {grad(x_val):.4f}')
        fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', on_move)
plt.show()