from scipy import stats

# Create some data
x = [1, 2, 3, 4, 5]
y = [2, 1, 4, 3, 5]

# Calculate the slope, intercept, r_value, p_value, and std_err
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Print the result
print("slope: %f    intercept: %f" % (slope, intercept))