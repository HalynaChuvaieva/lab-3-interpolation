import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline

# задана функція
def f(x):
    return (x - 1.0)**3 + 0.5 * np.exp(x)

# інтервал локалізації кореня [a, b] та параметри сітки
a, b = 0.0, 0.5
m = 10          # кількість вузлів інтерполяції
n = m - 1       # степінь інтерполяційного полінома (n = 9)
h = (b - a) / n # крок рівновіддалених вузлів

# формування 10 рівновіддалених вузлів
x_nodes = np.linspace(a, b, m)
y_nodes = f(x_nodes)

# прямий інтерполяційний поліном Лагранжа L_n(x)
def lagrange_poly(x_val, x_k, y_k):
    total = 0.0
    num_nodes = len(x_k)
    for k in range(num_nodes):
        term = y_k[k]
        for j in range(num_nodes):
            if j != k:
                term *= (x_val - x_k[j]) / (x_k[k] - x_k[j])
        total += term
    return total

# Побудова природного кубічного сплайна (другі похідні на кінцях дорівнюють 0)
cs_natural = CubicSpline(x_nodes, y_nodes, bc_type='natural')

# Точний корінь за допомогою методу Брента
root_exact = brentq(f, a, b, xtol=1e-15)

# Корінь за допомогою прямої інтерполяції Лагранжа: розв'язуємо L_9(x) = 0
root_direct_lagrange = brentq(lambda x: lagrange_poly(x, x_nodes, y_nodes), a, b, xtol=1e-12)
# Корінь за допомогою оберненої інтерполяції Лагранжа: обчислюємо L_9^(-1)(0)
root_inverse_lagrange = lagrange_poly(0.0, y_nodes, x_nodes)
# Корінь за допомогою природного кубічного сплайна: розв'язуємо S_3(x) = 0
root_spline = brentq(cs_natural, a, b, xtol=1e-12)


print("=== ВУЗЛИ ІНТЕРПОЛЯЦІЇ (10 рівновіддалених вузлів) ===")
print(f"{'k':>2} | {'x_k':>10} | {'f(x_k)':>12}")
print("-" * 30)
for k in range(m):
    print(f"{k:2d} | {x_nodes[k]:10.6f} | {y_nodes[k]:12.6f}")

print("\n=== РЕЗУЛЬТАТИ ПОШУКУ КОРЕНЯ ===")
print(f"Точний корінь (еталон):                 x* = {root_exact:.10f}")
print(f"Пряма інтерполяція Лагранжа L_9(x)=0:   x  = {root_direct_lagrange:.10f} (похибка: {abs(root_exact - root_direct_lagrange):.2e})")
print(f"Обернена інтерполяція Лагранжа x(0):    x  = {root_inverse_lagrange:.10f} (похибка: {abs(root_exact - root_inverse_lagrange):.2e})")
print(f"Природний кубічний сплайн S_3(x)=0:     x  = {root_spline:.10f} (похибка: {abs(root_exact - root_spline):.2e})")


# Дослідження функції на широкому проміжку [-1, 1.5]
x_wide = np.linspace(-1.0, 1.5, 400)
plt.figure(figsize=(8, 5))
plt.plot(x_wide, f(x_wide), 'b-', linewidth=2, label=r'$f(x) = (x-1)^3 + 0.5e^x$')
plt.axhline(0, color='black', linewidth=1, linestyle='--')
plt.axvline(0, color='gray', linewidth=0.8, linestyle=':')
plt.scatter([root_exact], [0], color='red', s=70, zorder=5, label=f'Корінь $x^* \\approx {root_exact:.4f}$')
plt.axvspan(a, b, color='yellow', alpha=0.2, label=f'Обраний проміжок $[{a}; {b}]$')
plt.title('Графік функції та відокремлення кореня')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()

# Інтерполяційні поліноми Лагранжа, природний кубічний сплайн та похибки
x_dense = np.linspace(a, b, 500)
y_true = f(x_dense)
y_lagrange = np.array([lagrange_poly(val, x_nodes, y_nodes) for val in x_dense])
y_spline = cs_natural(x_dense)

# для оберненого полінома Лагранжа будуємо криву x = L_9^(-1)(y)
y_dense_inv = np.linspace(y_nodes[0], y_nodes[-1], 500)
x_inv_lagrange = np.array([lagrange_poly(val, y_nodes, x_nodes) for val in y_dense_inv])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Лівий графік: функція, поліноми, сплайн і корінь
ax1.plot(x_dense, y_true, 'k-', linewidth=3, alpha=0.6, label='Початкова функція $f(x)$')
ax1.plot(x_dense, y_lagrange, 'r--', linewidth=1.8, label='Прямий поліном Лагранжа $L_9(x)$')
ax1.plot(x_inv_lagrange, y_dense_inv, 'g-.', linewidth=1.8, label='Обернений поліном Лагранжа $L_9^{-1}(y)$')
ax1.plot(x_dense, y_spline, 'b:', linewidth=2.2, label='Природний кубічний сплайн $S_3(x)$')
ax1.scatter(x_nodes, y_nodes, color='darkorange', edgecolor='black', s=60, zorder=5, label='Вузли інтерполяції (10 точок)')
ax1.scatter([root_exact], [0], color='red', marker='*', s=160, zorder=6, label=f'Корінь $x^* \\approx {root_exact:.6f}$')
ax1.axhline(0, color='black', linewidth=1)
ax1.set_title('Інтерполяція функції та пошук кореня на [0; 0.5]')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.4)

# Правий графік: похибки інтерполяції у логарифмічній шкалі
err_lagrange = np.abs(y_true - y_lagrange)
err_spline = np.abs(y_true - y_spline)
ax2.semilogy(x_dense, err_lagrange + 1e-16, 'r-', linewidth=1.5, label='$|f(x) - L_9(x)|$ (Поліном Лагранжа)')
ax2.semilogy(x_dense, err_spline + 1e-16, 'b-', linewidth=1.5, label='$|f(x) - S_3(x)|$ (Кубічний сплайн)')
ax2.scatter(x_nodes, np.full_like(x_nodes, 1e-16), color='darkorange', s=40, label='Вузли інтерполяції')
ax2.set_title('Абсолютна похибка інтерполяції (лог. шкала)')
ax2.set_xlabel('x')
ax2.set_ylabel('Похибка $|f(x) - F(x)|$')
ax2.legend()
ax2.grid(True, which='both', alpha=0.4)

plt.tight_layout()
plt.show()