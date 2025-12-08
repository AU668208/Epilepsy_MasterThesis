import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


# --- Projektets farvepalette ---
myGreenDark  = "#4CAF50"
myGreenLight = "#A5D6A7"
grey_light   = "#E0E0E0"
grey_mid     = "#9E9E9E"
grey_dark    = "#616161"
rust         = "#8B2F1C"

palette = {
    "primary": myGreenDark,
    "secondary": myGreenLight,
    "accent": rust,
    "grey_light": grey_light,
    "grey_mid": grey_mid,
    "grey_dark": grey_dark,
}

def set_project_style():
    plt.rcParams.update({
        # -------------------------------
        # Fonts & text
        # -------------------------------
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,

        # -------------------------------
        # Backgrounds
        # -------------------------------
        "figure.facecolor": "white",
        "axes.facecolor": "white",

        # -------------------------------
        # Axes lines
        # -------------------------------
        "axes.edgecolor": grey_mid,

        # -------------------------------
        # Grid styling (high-quality, visible)
        # -------------------------------
        "axes.grid": True,
        "axes.grid.axis": "y",         # Only horizontal grid (cleanest)
        "grid.color": "#BDBDBD",       # Slightly darker grey for visibility
        "grid.linewidth": 0.8,
        "grid.alpha": 0.6,

        # -------------------------------
        # Lines & bar color cycle
        # -------------------------------
        "axes.prop_cycle": plt.cycler(color=[myGreenDark]),

        # -------------------------------
        # Histogram / patch styling
        # -------------------------------
        "patch.edgecolor": "black",
        "patch.linewidth": 0.3,

        # -------------------------------
        # Tick size
        # -------------------------------
        "ytick.major.size": 6,
        "ytick.minor.size": 4,
        "xtick.major.size": 6,
        "xtick.minor.size": 4,
    })


def set_integer_ticks(ax):
    """Force y-axis to show only integer ticks."""
    ax.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    return ax
