# %%
import matplotlib.pyplot as plt
import numpy as np

# Set parameters for the Poisson process
lambda_rate = 5  # average number of arrivals per unit time, e.g., 5 arrivals per hour
total_time = 10  # total time period for the simulation, e.g., 10 hours

# Generate arrival times
np.random.seed(0)  # for reproducibility
inter_arrival_times = np.random.exponential(
    scale=1 / lambda_rate, size=1000
)  # exponential inter-arrival times
arrival_times = np.cumsum(inter_arrival_times)  # cumulative sum to get arrival times

# Filter arrival times to fit within the total time
arrival_times = arrival_times[arrival_times <= total_time]

# Plotting
plt.figure(figsize=(10, 6))
plt.eventplot(arrival_times, orientation="horizontal", colors="blue")
plt.xlabel("Time")
plt.title("Poisson Arrival Process Simulation (Lambda = 5 arrivals/hour)")
plt.xlim(0, total_time)
plt.grid(True)
plt.show()

# Number of arrivals
num_arrivals = len(arrival_times)
num_arrivals


# %%
import matplotlib.pyplot as plt
import numpy as np

# Time series simulation parameters
A1, B1 = 1, 0.5  # Amplitudes for the daily pattern
A2, B2 = 0.0, 0.0  # Amplitudes for the weekly pattern
total_hours = 24 * 7 * 4  # Simulating for 4 weeks (in hours)

# Time points (in hours)
t = np.arange(0, total_hours, 1)  # hourly intervals

# Fourier components for daily and weekly patterns
daily_component = A1 * np.sin(2 * np.pi * t / 24) + B1 * np.cos(2 * np.pi * t / 24)
weekly_component = A2 * np.sin(2 * np.pi * t / (24 * 7)) + B2 * np.cos(
    2 * np.pi * t / (24 * 7)
)

# Combined time series
y = daily_component + weekly_component

# Adding random noise for realism
noise = np.random.normal(
    0, 0.2, total_hours
)  # noise with mean 0 and standard deviation 0.2
y_noisy = y + noise

# Plotting
plt.figure(figsize=(15, 6))
plt.plot(t, y_noisy, label="Simulated Time Series with Noise")
plt.plot(t, y, label="Underlying Pattern without Noise", alpha=0.7)
plt.xlabel("Time (hours)")
plt.ylabel("Value")
plt.title("Simulated Time Series Data with Daily and Weekly Patterns")
plt.legend()
plt.grid(True)
plt.show()

#%%
# Time simulation parameters
total_days = 4 * 7  # 4 weeks
hours_per_day = 24

# Time series simulation parameters
A1, B1 = 1, 0.5  # Amplitudes for the daily pattern
A2, B2 = 0.0, 0.0  # Amplitudes for the weekly pattern
A3, B3 = 0.0, 0.0  # Amplitudes for the monthly pattern
A0, B0 = 2, 1.8  # Amplitudes for weekend pattern

# Time array (in hours)
time_hours = np.arange(total_days * hours_per_day)

# Define simple patterns for demonstration
# Daily pattern
daily_pattern = A1 * np.sin(2 * np.pi * time_hours / 24) + B1 * np.cos(
    2 * np.pi * time_hours / 24
)
weekly_pattern = A2 * np.sin(2 * np.pi * time_hours / (24 * 7)) + B2 * np.cos(
    2 * np.pi * time_hours / (24 * 7)
)
# Weekend pattern
weekend_pattern = A0 * np.sin(2 * np.pi * time_hours / (24 * 7)) + B0 * np.cos(
    2 * np.pi * time_hours / (24 * 7)
)
# Monthly pattern
monthly_pattern = A3 * np.sin(2 * np.pi * time_hours // 24 / (30)) + B3 * np.cos(
    2 * np.pi * time_hours // 24 / (30)
)
# regular pattern
regular_pattern = daily_pattern + weekly_pattern + monthly_pattern

# Initialize the combined pattern array
combined_pattern = np.zeros_like(time_hours, dtype=float)

# Map the patterns to the corresponding days
for hour in time_hours:
    day_of_week = (
        hour // hours_per_day
    ) % 7  # Calculate day of the week (0 = Monday, 6 = Sunday)
    if day_of_week < 5:  # Weekdays
        combined_pattern[hour] = regular_pattern[hour]
    else:  # Weekends
        combined_pattern[hour] = regular_pattern[hour] + weekend_pattern[hour]
        combined_pattern[hour] = regular_pattern[hour]

# Adding random noise for realism
noise = np.random.normal(0, 0.1, total_days * hours_per_day)
combined_pattern_noisy = combined_pattern + noise

# Plotting
plt.figure(figsize=(15, 6))
plt.plot(time_hours, combined_pattern_noisy, label="Combined Pattern with Noise")
plt.plot(time_hours, combined_pattern, label="Underlying Combined Pattern", alpha=0.7)
plt.xlabel("Time (hours)")
plt.ylabel("Value")
plt.title("Simulated Time Series with Different Weekday and Weekend Patterns")
plt.legend()
plt.grid(True)
plt.show()
