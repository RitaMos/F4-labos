"""
TP3 - Espectrometría gaseosa.

1) Compara las líneas medidas de He, Kr y Hg (planilla) con NIST.
2) Digitaliza los espectros de los tubos incógnita (capturas en PDF del
   software del espectrómetro) y mide la posición de cada línea.
3) Calcula las series de Balmer y Lyman y una constante de Rydberg experimental.
4) Genera figuras/espectros_incognita.pdf y figuras/residuos_conocidos.pdf.

Uso:  python3 analisis_tp3.py      (requiere poppler-utils: pdfimages)
Requiere: numpy, matplotlib, pillow, openpyxl
"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import openpyxl
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = Path(__file__).resolve().parent
FIG = AQUI / "figuras"
DAT = AQUI / "datos"
FIG.mkdir(exist_ok=True)
DAT.mkdir(exist_ok=True)

DELTA_LAMBDA = 2.0  # nm, incertidumbre adoptada (ancho de banda / resolución)

# --------------------------------------------------------------------------
# Valores de referencia: NIST Handbook of Basic Atomic Spectroscopic Data,
# tablas de líneas intensas (longitudes de onda en aire, nm).
# --------------------------------------------------------------------------
NIST_CONOCIDOS = {  # (elemento medido, lambda medida) -> (lambda NIST, especie)
    ("Helio", 501.0): (501.568, "He I"),
    ("Helio", 587.0): (587.562, "He I"),     # triplete 587.5615/.5640/.5966 no resuelto
    ("Helio", 668.0): (667.815, "He I"),
    ("Helio", 707.0): (706.518, "He I"),     # triplete 706.5177/.5215/.5709
    ("Kripton", 761.0): (760.155, "Kr I"),
    ("Kripton", 770.0): (769.454, "Kr I"),
    ("Kripton", 812.0): (811.290, "Kr I"),   # (+ Kr I 810.437, no resuelta)
    ("Kripton", 830.5): (829.811, "Kr I"),
    ("Mercurio", 435.5): (435.833, "Hg I"),
    ("Mercurio", 546.5): (546.074, "Hg I"),
    ("Mercurio", 764.4): (763.511, "Ar I"),  # gas de llenado (Ar)
    ("Mercurio", 812.0): (811.531, "Ar I"),  # (+ Ar I 810.369, no resuelta)
}

NE_NIST = [585.24879, 594.48342, 602.99969, 614.30626, 621.72812, 626.64950,
           633.44278, 640.2248, 650.65281, 659.89529, 667.82762, 671.70430,
           692.94673, 703.24131, 717.39381, 724.51666, 743.88712]
NE_VENTANAS = [(580, 590), (590, 598), (600, 606), (612, 618), (618, 624),
               (624, 630), (630, 634.5), (637.6, 646), (646, 655), (655, 663),
               (663, 669.5), (669.5, 675), (688, 698), (698, 708), (712, 720),
               (720, 730), (740, 748)]
H_NIST = {3: 656.279, 4: 486.135, 5: 434.047, 6: 410.174}   # aire
NA_NIST = [588.995, 589.592, 818.326, 819.482]

# Constantes (CODATA 2018)
R_INF = 10973731.568160      # m^-1
ME_MP = 1 / 1836.15267343
R_H = R_INF / (1 + ME_MP)


def n_aire(lam_vac_nm):
    """Índice de refracción del aire estándar (Edlén/Morton 2000)."""
    s2 = (1e3 / lam_vac_nm) ** 2  # (1/um)^2
    return 1 + 8.34254e-5 + 2.406147e-2 / (130 - s2) + 1.5998e-4 / (38.9 - s2)


def serie(n_inf, n_sup):
    lam_vac = 1e9 / (R_H * (1 / n_inf**2 - 1 / n_sup**2))
    return lam_vac


# --------------------------------------------------------------------------
# Digitalización de las capturas
# --------------------------------------------------------------------------
# Calibración de ejes a partir de la grilla de la captura (página rotada a
# 600 ppi): líneas verticales en 400, 500, ..., 1000 nm y horizontales en
# 1000, ..., 4000 cuentas.
X_GRID = np.array([715.5, 1465.5, 2207.0, 2957.0, 3698.5, 4448.5, 5190.5])
L_GRID = np.array([400, 500, 600, 700, 800, 900, 1000.0])
Y_GRID = np.array([111.5, 861.5, 1603.5, 2345.0])
C_GRID = np.array([4000, 3000, 2000, 1000.0])
PX = np.polyfit(X_GRID, L_GRID, 1)
PY = np.polyfit(Y_GRID, C_GRID, 1)
COLS_CURSOR = slice(1274, 1286)   # cursor verde del software (~475 nm)


def pagina(pdf):
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["pdfimages", "-png", str(pdf), f"{td}/im"], check=True)
        ims = [Image.open(p).convert("RGB") for p in sorted(Path(td).glob("im-*.png"))]
        alto = sum(im.height for im in ims)
        pag = Image.new("RGB", (ims[0].width, alto), "white")
        y = 0
        for im in ims:
            pag.paste(im, (0, y))
            y += im.height
    return np.asarray(pag.rotate(-90, expand=True)).astype(int)


def mascara_curva(a):
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    blanco = (R > 215) & (G > 215) & (B > 215)
    gris = (abs(R - G) < 35) & (abs(G - B) < 35) & (abs(R - B) < 35) & (R > 70)
    m = ~(blanco | gris)
    m[:, COLS_CURSOR] = False
    return m


def digitalizar(a):
    m = mascara_curva(a)
    xs = np.arange(352, 5188)
    tope = np.full(xs.size, np.nan)
    for k, x in enumerate(xs):
        y, hueco, ultimo = 3078, 0, None
        while y > 105:
            if m[y, x]:
                hueco, ultimo = 0, y
            else:
                hueco += 1
                if hueco >= 8:
                    break
            y -= 1
        if ultimo is not None:
            tope[k] = ultimo
    ok = ~np.isnan(tope)
    tope = np.interp(xs, xs[ok], tope[ok])
    # Un píxel de pantalla original ocupa ~7 px de la captura: se toma el máximo
    # en una ventana de 5 px para eliminar huecos de un solo píxel entre el
    # contorno rojo y el relleno de color.
    ventana = np.lib.stride_tricks.sliding_window_view(np.pad(tope, 2, mode="edge"), 5)
    tope = ventana.min(axis=1)
    return np.polyval(PX, xs), np.polyval(PY, tope), m


def centro_linea(lam, cuentas, mask, lo, hi, fondo):
    """Centro = punto medio del ancho a media altura (corte horizontal)."""
    sel = (lam >= lo) & (lam <= hi)
    pico = cuentas[sel].max()
    nivel = fondo + (pico - fondo) / 2
    y = int(round((nivel - PY[1]) / PY[0]))
    x0, x1 = [int((v - PX[1]) / PX[0]) for v in (lo, hi)]
    idx = np.where(mask[y, x0:x1])[0]
    tramos = np.split(idx, np.where(np.diff(idx) > 2)[0] + 1)
    t = max(tramos, key=len)
    l_izq, l_der = np.polyval(PX, x0 + t[0]), np.polyval(PX, x0 + t[-1] + 1)
    return (l_izq + l_der) / 2, l_der - l_izq, pico


def main():
    # ---------------- 1) Gases conocidos ----------------
    ws = openpyxl.load_workbook(AQUI / "TPs 3 & 4 - F4.xlsx", data_only=True)["TP 3"]
    print("== Gases conocidos")
    res = []
    for r in range(3, 15):
        el, lam, dl = ws[f"B{r}"].value, ws[f"D{r}"].value, ws[f"E{r}"].value
        ref, esp = NIST_CONOCIDOS[(el, lam)]
        err = 100 * abs(lam - ref) / ref
        res.append((el, lam, dl, ref, esp, lam - ref, err))
        print(f"  {el:9s} {lam:6.1f} ± {dl:3.1f}  NIST {ref:8.3f} ({esp})"
              f"  dif {lam-ref:+5.2f} nm  error {err:.3f} %")
    d = np.array([x[5] for x in res])
    lam_m = np.array([x[1] for x in res])
    p = np.polyfit(lam_m, d, 1)
    print(f"  residuo medio {d.mean():+.2f} nm, desvío {d.std(ddof=1):.2f} nm,"
          f" |max| {abs(d).max():.2f} nm; pendiente {p[0]*100:.3f} nm/100nm")

    # ---------------- 2) Incógnitas ----------------
    espectros = {}
    for i in (1, 2, 3):
        a = pagina(AQUI / f"Gas-desconocidoN{i}.pdf")
        # Copia reducida de la captura original para el anexo del informe
        cap = Image.fromarray(a[:3480].astype(np.uint8))
        cap.resize((1600, round(1600 * cap.height / cap.width)),
                   Image.LANCZOS).save(FIG / f"captura_N{i}.png", optimize=True)
        lam, cts, mask = digitalizar(a)
        espectros[i] = (lam, cts, mask)
        np.savetxt(DAT / f"espectro_incognita{i}.csv", np.c_[lam, cts], fmt="%.3f",
                   delimiter=",", header="lambda_nm,cuentas", comments="")
    fondo = {i: np.median(espectros[i][1]) for i in espectros}
    print("\n== Fondo (cuentas):", {i: round(v, 1) for i, v in fondo.items()})

    lam, cts, mask = espectros[1]
    c1, w1, pk1 = centro_linea(lam, cts, mask, 540, 640, fondo[1])
    print(f"\n== Incógnita 1: centro {c1:.2f} nm, FWHM {w1:.2f} nm, pico {pk1:.0f} cuentas")
    kT = 8.617333e-5 * 298.15
    print(f"   FWHM teórica LED 1.8kT: {1.8*kT*1000:.1f} meV -> "
          f"{1.8*kT*c1**2/1239.842:.1f} nm")
    for l_na in NA_NIST:
        s = (lam > l_na - 3) & (lam < l_na + 3)
        print(f"   en Na {l_na}: máx {cts[s].max():.0f} cuentas (fondo {fondo[1]:.0f})")

    lam, cts, mask = espectros[2]
    print("\n== Incógnita 2 (Ne)")
    ne = []
    for (lo, hi), ref in zip(NE_VENTANAS, NE_NIST):
        c, w, pk = centro_linea(lam, cts, mask, lo, hi, fondo[2])
        ne.append((c, ref, pk))
        print(f"   {c:7.2f} nm  NIST {ref:7.3f}  dif {c-ref:+5.2f}  err {100*abs(c-ref)/ref:.3f} %"
              f"  pico {pk:.0f}")
    dn = np.array([x[0] - x[1] for x in ne])
    print(f"   residuo medio {dn.mean():+.2f} nm, desvío {dn.std(ddof=1):.2f} nm, |max| {abs(dn).max():.2f}")

    lam, cts, mask = espectros[3]
    print("\n== Incógnita 3 (H)")
    h_med = {}
    for n, (lo, hi) in {3: (650, 661), 4: (480, 490), 5: (428, 440)}.items():
        c, w, pk = centro_linea(lam, cts, mask, lo, hi, fondo[3])
        h_med[n] = (c, pk)
        print(f"   n={n}: {c:.2f} nm  pico {pk:.0f} cuentas (sobre fondo {pk-fondo[3]:.0f})")
    s = (lam > 500) & (lam < 560)
    print(f"   ruido (500-560 nm): desvío {cts[s].std():.1f} cuentas")
    s = (lam > 407) & (lam < 413)
    print(f"   máx en zona Hdelta (407-413): {cts[s].max():.0f}")

    # ---------------- 3) Balmer, Lyman, Rydberg ----------------
    print(f"\n== R_H = {R_H:.4f} m^-1")
    print("   Balmer (n -> 2):")
    for n in range(3, 8):
        lv = serie(2, n)
        la = lv / n_aire(lv)
        print(f"     n={n}: vacío {lv:8.3f}  aire {la:8.3f}  NIST {H_NIST.get(n,'-')}")
    print(f"     límite: vacío {1e9*4/R_H:.3f}  aire {1e9*4/R_H/n_aire(1e9*4/R_H):.3f}")
    print("   Lyman (n -> 1):")
    for n in range(2, 6):
        print(f"     n={n}: vacío {serie(1, n):8.3f}")
    print(f"     límite: vacío {1e9/R_H:.3f}")
    Rs, sR = [], []
    for n in (3, 4):
        la = h_med[n][0]
        lv = la * n_aire(la)
        R = 1 / (lv * 1e-9 * (1 / 4 - 1 / n**2))
        Rs.append(R)
        sR.append(R * DELTA_LAMBDA / la)
        print(f"   R desde n={n}: {R:.5e} ± {R*DELTA_LAMBDA/la:.2e}  ({100*(R-R_H)/R_H:+.3f} %)")
    w = 1 / np.array(sR) ** 2
    Rm = np.sum(w * Rs) / w.sum()
    print(f"   R ponderado: {Rm:.5e} ± {1/np.sqrt(w.sum()):.2e}  ({100*(Rm-R_H)/R_H:+.3f} %)")

    # ---------------- 4) Figuras ----------------
    plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False,
                         "axes.spines.right": False})
    fmt = matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}".replace(".", ","))
    TINTA, REF = "#222222", "#B03A2E"
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 8.4))

    ax = axs[0]
    lam, cts, _ = espectros[1]
    s = (lam > 520) & (lam < 660)
    ax.plot(lam[s], cts[s], color=TINTA, lw=1)
    for l_na in NA_NIST[:2]:
        ax.axvline(l_na, color=REF, lw=0.9, ls="--")
    ax.text(590.5, 2450, "Na I 589,0 / 589,6 nm\n(NIST)", color=REF, fontsize=8.5, va="top")
    yh = fondo[1] + (pk1 - fondo[1]) / 2
    ax.annotate("", xy=(c1 - w1 / 2, yh), xytext=(c1 + w1 / 2, yh),
                arrowprops=dict(arrowstyle="<->", color="#555555", lw=1))
    ax.text(c1 + w1 / 2 + 1.5, yh, f"FWHM ≈ {w1:.0f} nm".replace(".", ","),
            va="center", fontsize=8.5, color="#333333")
    ax.set_title(f"(a) Incógnita N.º 1: banda ancha centrada en {c1:.0f} nm",
                 loc="left", fontsize=9.5)

    ax = axs[1]
    lam, cts, _ = espectros[2]
    s = (lam > 575) & (lam < 755)
    ax.plot(lam[s], cts[s], color=TINTA, lw=1)
    ytop = 2350
    for k, ref in enumerate(NE_NIST):
        ax.plot([ref, ref], [ytop - 180, ytop], color=REF, lw=1.1,
                label="Líneas de Ne I (NIST)" if k == 0 else None)
    ax.legend(loc="center right", frameon=False, fontsize=8.5)
    for c, ref, pk in ne:
        if pk > 800:
            ax.text(c, pk + 40, f"{ref:.1f}".replace(".", ","), ha="center",
                    va="bottom", fontsize=7.5, color="#333333")
    ax.set_ylim(0, 2500)
    ax.set_title("(b) Incógnita N.º 2: espectro de líneas compatible con neón",
                 loc="left", fontsize=9.5)

    ax = axs[2]
    lam, cts, _ = espectros[3]
    s = (lam > 400) & (lam < 680)
    ax.plot(lam[s], cts[s], color=TINTA, lw=1)
    nombres = {3: r"H$\alpha$", 4: r"H$\beta$", 5: r"H$\gamma$", 6: r"H$\delta$"}
    for n in (3, 4, 5, 6):
        lv = serie(2, n)
        la = lv / n_aire(lv)
        ax.axvline(la, color=REF, lw=0.9, ls="--")
        ax.text(la + 2, 3150, nombres[n], color=REF, fontsize=9, va="top")
    ax.set_ylim(0, 3200)
    ax.set_title("(c) Incógnita N.º 3: líneas de Balmer del hidrógeno "
                 "(trazos: fórmula de Balmer)", loc="left", fontsize=9.5)
    axins = ax.inset_axes([0.42, 0.33, 0.3, 0.45])
    s2 = (lam > 420) & (lam < 450)
    axins.plot(lam[s2], cts[s2], color=TINTA, lw=1)
    lv = serie(2, 5)
    axins.axvline(lv / n_aire(lv), color=REF, lw=0.9, ls="--")
    axins.set_ylim(140, 200)
    axins.tick_params(labelsize=7)
    axins.set_title(r"zoom H$\gamma$", fontsize=7.5)
    axins.xaxis.set_major_formatter(fmt)

    for ax in axs:
        ax.set_ylabel("Intensidad (cuentas)")
        ax.grid(True, color="#e6e6e6", lw=0.6)
        ax.xaxis.set_major_formatter(fmt)
    axs[-1].set_xlabel("Longitud de onda $\\lambda$ (nm)")
    fig.tight_layout()
    fig.savefig(FIG / "espectros_incognita.pdf")

    # Residuos de los gases conocidos
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.axhspan(-DELTA_LAMBDA, DELTA_LAMBDA, color="#eeeeee", lw=0)
    ax.axhline(0, color="#888888", lw=0.8)
    estilos = {"Helio": ("o", "#2167AC"), "Kripton": ("s", "#B8760B"),
               "Mercurio": ("^", "#1E8449")}
    for el, (mk, col) in estilos.items():
        xs = [x[3] for x in res if x[0] == el and x[4] != "Ar I"]
        ys = [x[5] for x in res if x[0] == el and x[4] != "Ar I"]
        ax.plot(xs, ys, mk, color=col, ms=7, mec="white", mew=0.6, label=el)
    xs = [x[3] for x in res if x[4] == "Ar I"]
    ys = [x[5] for x in res if x[4] == "Ar I"]
    ax.plot(xs, ys, "^", ms=7, mfc="white", mec="#1E8449", mew=1.3,
            label="Tubo de Hg: líneas de Ar I")
    ax.text(832, 1.8, "$\\pm\\Delta\\lambda$", fontsize=8.5, va="top", color="#555555")
    ax.set_xlim(420, 860)
    ax.set_ylim(-2.4, 2.4)
    ax.set_xlabel("Longitud de onda de referencia NIST (nm)")
    ax.set_ylabel(r"$\lambda_\mathrm{med}-\lambda_\mathrm{NIST}$ (nm)")
    ax.grid(True, color="#e6e6e6", lw=0.6)
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    fig.tight_layout()
    fig.savefig(FIG / "residuos_conocidos.pdf")
    print("\nFiguras guardadas en", FIG)


if __name__ == "__main__":
    main()
