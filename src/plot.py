#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# plt.style.use("ggplot")

rng = np.random.default_rng(42)

amplitude = 3
offset = 7
# Increase the time range to show 3 full cycles
t = np.linspace(0, 6 * np.pi, 300)  # 3 cycles
sinusoidal_signal = amplitude * np.sin(t) + offset

# Multiply the sinusoidal by 0.7
adjusted_signal = 0.7 * sinusoidal_signal

# Add random Gaussian noise with mean 0 and standard deviation 0.5 at each time step
noise = rng.normal(0, 1.2, t.shape)
# Recreate the noisy signal with the adjusted sinusoidal
noisy_signal = adjusted_signal + noise
noisy_signal2 = pd.Series(noisy_signal).rolling(10).mean().values

# Recalculate the rolling standard deviation with the new signal
rolling_std = np.array(
    [np.std(noisy_signal[max(0, i - 20) : i + 1]) for i in range(len(t))]
)
rolling_std = pd.Series(noisy_signal).rolling(10).std().values

# Recreate upper and lower bounds for the confidence interval
upper_bound = noisy_signal2 + rolling_std
lower_bound = noisy_signal2 - rolling_std

# Set the font size for better readability
plt.rcParams.update({"font.size": 14})

# Recreate the plot
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Plot the noisy sinusoidal signal
ax.plot(t, noisy_signal2, label="Noisy Sinusoidal Signal")
ax.fill_between(t, lower_bound, upper_bound, color="gray", alpha=0.3)

# Set the y-axis limits to accommodate the noise and the confidence interval
ax.set_ylim(0, 10)

# Remove y-ticks and x-ticks
ax.set_yticks([])
ax.set_xticks([])

# Add labels for x and y axis with increased font size
ax.set_xlabel("minute")
ax.set_ylabel("kW")

# Keep the grid to provide a reference for the variability in the data
# ax.grid(True, which="both", linestyle="--", linewidth=0.5)

# Revert the axes to the standard lines without arrow tips
ax.spines["bottom"].set_position(("data", 0))
ax.spines["left"].set_position(("data", 0))
ax.spines["right"].set_color("none")
ax.spines["top"].set_color("none")

# add horizontal line at 7 kW
ax.axhline(y=0.5, color="black", linestyle="--", linewidth=1)
ax.axhline(y=9.5, color="black", linestyle="--", linewidth=1)

# create arrows
ax.annotate(
    "",
    xy=(5, 9.5),
    xytext=(5, 3.7),
    arrowprops=dict(arrowstyle="->", color="black", linewidth=1),
)
ax.text(
    s=r"$p^{\mathrm{cap},\downarrow}$",
    x=5.2,
    y=8,
    verticalalignment="center",
    horizontalalignment="left",
    fontsize=16,
)
ax.annotate(
    "",
    xy=(5, 0.45),
    xytext=(5, 1.7),
    arrowprops=dict(arrowstyle="->", color="black", linewidth=1),
)
ax.text(
    s=r"$p^{\mathrm{cap},\uparrow}$",
    x=5.2,
    y=1.5,
    verticalalignment="center",
    horizontalalignment="left",
    fontsize=16,
)

# Add annotation for P^min and P^max to the y-axis with increased font size
ax.text(
    s=r"$P^{\mathrm{Min}}$",
    x=0.2,
    y=1,
    verticalalignment="center",
    horizontalalignment="left",
    fontsize=16,
)
ax.text(
    s=r"$P^{\mathrm{Max}}$",
    x=0.2,
    y=9,
    verticalalignment="center",
    horizontalalignment="left",
    fontsize=16,
)

# Save the plot with high resolution
high_res_image_path_final = "tex/figures/power_high_res_plot.png"
plt.savefig(high_res_image_path_final, dpi=300)

if True:
    import tikzplotlib

tikzplotlib.save("tex/figures/power_high_res_plot.tikz")

# Show the updated plot
plt.show()
