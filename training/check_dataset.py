import matplotlib.pyplot as plt

# =========================
# CREATE FIGURE
# =========================
fig, ax = plt.subplots(figsize=(12, 3))

# =========================
# FUNCTION DRAW BOX
# =========================
def draw_box(x, text, color):
    ax.text(
        x, 0, text,
        ha='center',
        va='center',
        fontsize=12,
        fontweight='bold',
        bbox=dict(
            boxstyle="round,pad=0.6",
            fc=color,
            ec="black",
            lw=2
        )
    )

# =========================
# POSITIONS (CÂN ĐỀU)
# =========================
positions = [0, 3, 6, 9]

# =========================
# DRAW PIPELINE
# =========================
draw_box(positions[0], "Input Image", "#f8d7da")
draw_box(positions[1], "Preprocessing\n(Resize, Normalize)", "#cfe2f3")
draw_box(positions[2], "EfficientNet Model", "#d4edda")
draw_box(positions[3], "Prediction\n(Label + Confidence)", "#d1ecf1")

# =========================
# DRAW ARROWS
# =========================
for i in range(len(positions) - 1):
    ax.annotate(
        "",
        xy=(positions[i+1] - 0.9, 0),
        xytext=(positions[i] + 0.9, 0),
        arrowprops=dict(
            arrowstyle="->",
            lw=2
        )
    )

# =========================
# STYLE
# =========================
ax.set_xlim(-1, 10)
ax.set_ylim(-1, 1)
ax.axis('off')

# TITLE
plt.title(
    "Banknote Recognition Pipeline",
    fontsize=15,
    fontweight='bold'
)

# SAVE + SHOW
plt.tight_layout()
plt.savefig("Hinh_5_2_pipeline_no_softmax.png", dpi=300)
plt.show()