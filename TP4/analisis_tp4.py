"""
TP4 - Medición de la constante de Planck con LEDs.

Lee las curvas I(V) de la planilla, ajusta la zona lineal (I >= 10 mA),
obtiene la tensión umbral Vg como intersección de la recta con el eje V,
calcula h_e = e*Vg*lambda/c, propaga incertidumbres y genera la figura
figuras/curvas_IV.pdf.

Uso:  python3 analisis_tp4.py
Requiere: numpy, matplotlib, openpyxl
"""
from pathlib import Path

import numpy as np
import openpyxl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = Path(__file__).resolve().parent
XLSX = AQUI / "TPs 3 & 4 - F4.xlsx"
FIG = AQUI / "figuras"
FIG.mkdir(exist_ok=True)

# Constantes de referencia indicadas en la consigna
H_TEO = 6.62e-34      # J s
C = 2.99e8            # m/s
E = 1.602e-19         # C
# Valores exactos del SI (CODATA 2018), solo para comparación
H_SI, C_SI, E_SI = 6.62607015e-34, 299792458.0, 1.602176634e-19

I_MIN_AJUSTE = 10.0   # mA: zona lineal adoptada
DELTA_V_INST = 0.01   # V: resolución del voltímetro (escala de 20 V)
DELTA_LAMBDA = 2.0    # nm: resolución del espectrómetro (misma que en el TP3)

# Columnas de la hoja "TP 4": (nombre, lambda [nm], col V, col I)
LEDS = [("Rojo", 628, "B", "C"), ("Amarillo", 595, "F", "G"),
        ("Verde", 530, "J", "K"), ("Azul", 459, "N", "O")]
COLORES = {"Rojo": "#C0392B", "Amarillo": "#B8760B",
           "Verde": "#1E8449", "Azul": "#2167AC"}
MARCAS = {"Rojo": "o", "Amarillo": "s", "Verde": "^", "Azul": "D"}


def leer_datos():
    ws = openpyxl.load_workbook(XLSX, data_only=True)["TP 4"]
    datos = {}
    for nombre, lam, cv, ci in LEDS:
        V = np.array([ws[f"{cv}{r}"].value for r in range(3, 18)], float)
        I = np.array([ws[f"{ci}{r}"].value for r in range(3, 18)], float)
        datos[nombre] = (lam, V, I)
    return datos


def ajuste_lineal(x, y):
    """Mínimos cuadrados y = a + b x. Devuelve a, b, sigma_a, sigma_b, R2."""
    n = len(x)
    A = np.vstack([np.ones(n), x]).T
    (a, b), res, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - (a + b * x)
    s2 = np.sum(r**2) / (n - 2)
    Sxx = np.sum((x - x.mean())**2)
    sb = np.sqrt(s2 / Sxx)
    sa = np.sqrt(s2 * (1 / n + x.mean()**2 / Sxx))
    R2 = 1 - np.sum(r**2) / np.sum((y - y.mean())**2)
    return a, b, sa, sb, R2


def main():
    datos = leer_datos()
    filas = []
    print(f"{'LED':9s} {'lam':>4s} {'n':>2s} {'Vg[V]':>7s} {'sVg':>6s} {'dVg':>6s}"
          f" {'r_d[ohm]':>8s} {'R2':>7s} {'Vg(I-V)':>8s} {'h_e':>10s} {'dh':>9s}"
          f" {'err%':>6s} {'Eph[eV]':>7s} {'eVg-Eph':>7s} {'h_SI':>10s}")
    for nombre, (lam, V, I) in datos.items():
        m = I >= I_MIN_AJUSTE
        # La corriente fue la variable fijada (valores redondos) y la tensión la
        # leída: se ajusta V = Vg + r_d * I. La ordenada al origen es la
        # intersección de la recta tangente con el eje de tensiones (I = 0).
        Vg, rd, sVg, srd, R2 = ajuste_lineal(I[m], V[m])
        # Para comparación: recta de tendencia I = a + b V (como en Excel)
        a, b, *_ = ajuste_lineal(V[m], I[m])
        Vg_excel = -a / b
        dVg = np.hypot(sVg, DELTA_V_INST)
        h_e = E * Vg * lam * 1e-9 / C
        dh = h_e * np.hypot(dVg / Vg, DELTA_LAMBDA / lam)
        err = 100 * (h_e - H_TEO) / H_TEO
        E_ph = H_TEO * C / (lam * 1e-9) / E          # eV (constantes de la consigna)
        h_si = E_SI * Vg * lam * 1e-9 / C_SI
        filas.append(dict(nombre=nombre, lam=lam, Vg=Vg, dVg=dVg, sVg=sVg, rd=rd * 1e3,
                          srd=srd * 1e3, R2=R2, h=h_e, dh=dh, err=err, E_ph=E_ph,
                          V=V, I=I, m=m, h_si=h_si, Vg_excel=Vg_excel))
        print(f"{nombre:9s} {lam:4d} {m.sum():2d} {Vg:7.4f} {sVg:6.4f} {dVg:6.4f}"
              f" {rd*1e3:8.2f} {R2:7.4f} {Vg_excel:8.4f} {h_e:10.4e} {dh:9.2e}"
              f" {err:+6.2f} {E_ph:7.4f} {Vg-E_ph:+7.4f} {h_si:10.4e}")

    hs = np.array([f["h"] for f in filas])
    h_med = hs.mean()
    s = hs.std(ddof=1)
    print(f"\nPromedio h = {h_med:.4e} J s, desvío estándar s = {s:.3e},"
          f" error estándar = {s/np.sqrt(len(hs)):.3e},"
          f" error rel. = {100*(h_med-H_TEO)/H_TEO:+.2f} %")
    print(f"Factor C_SI/C = {C_SI/C:.5f};  E_SI/E = {E_SI/E:.6f}")

    # Sensibilidad a la elección de la zona lineal
    print("\nSensibilidad de Vg al umbral de corriente:")
    for f in filas:
        vals = []
        for imin in (8, 10, 15):
            mm = f["I"] >= imin
            vals.append(ajuste_lineal(f["I"][mm], f["V"][mm])[0])
        print(f"  {f['nombre']:9s} I>=8: {vals[0]:.3f}  I>=10: {vals[1]:.3f}  I>=15: {vals[2]:.3f}")

    # ---------------- Figura ----------------
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 4.0), sharey=True,
                            gridspec_kw={"width_ratios": [1, 1.6]})
    fmt = matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}".replace(".", ","))
    grupos = [(axs[0], ["Rojo", "Amarillo"], (1.62, 2.02)),
              (axs[1], ["Verde", "Azul"], (2.35, 3.42))]
    for ax, nombres, xlim in grupos:
        for k, f in enumerate([f for f in filas if f["nombre"] in nombres]):
            col, mk = COLORES[f["nombre"]], MARCAS[f["nombre"]]
            V, I, m = f["V"], f["I"], f["m"]
            pos = I > 0
            ax.plot(V[pos & ~m], I[pos & ~m], mk, ms=6, mfc="white", mec=col, mew=1.3)
            ax.plot(V[m], I[m], mk, ms=6.5, color=col, mec="white", mew=0.6,
                    label=f"{f['nombre']} ({f['lam']} nm)")
            vv = np.linspace(f["Vg"], V.max() + 0.02, 50)
            ax.plot(vv, (vv - f["Vg"]) / (f["rd"] / 1e3), "--", lw=1.3, color=col)
            ax.plot([f["Vg"]], [0], "|", ms=11, mew=2, color=col)
            ax.text(f["Vg"], -2.6 - 2.6 * k,
                    f"$V_g={f['Vg']:.3f}$ V".replace(".", "{,}"),
                    fontsize=8.5, ha="center", va="center", color="#333333")
        ax.axhline(0, color="#888888", lw=0.8)
        ax.set_xlim(*xlim)
        ax.set_ylim(-7, 32)
        ax.set_xlabel("Tensión $V$ (V)")
        ax.grid(True, color="#e3e3e3", lw=0.6)
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
    axs[0].set_ylabel("Corriente $I$ (mA)")
    fig.tight_layout()
    fig.savefig(FIG / "curvas_IV.pdf")
    print("\nFigura guardada en", FIG / "curvas_IV.pdf")


if __name__ == "__main__":
    main()
