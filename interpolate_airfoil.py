import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from math import pi, cos, tanh, sinh, atan, cosh, log

def read_airfoil_dat(filename):
    """
    Reads airfoil coordinates from a .dat file.
    :param filename: Path to the .dat file.
    :return: Arrays of x and y coordinates.
    """
    x_coords = []
    y_coords = []
    
    with open(filename, 'r') as file:
        next(file)  # Skip header
        
        for line in file:
            try:
                x, y = map(float, line.split())
                x_coords.append(x)
                y_coords.append(y)
            except ValueError:
                continue

    return np.array(x_coords), np.array(y_coords)

def x_transformation(theta, K, region):
    """
    Applies x-axis transformation to theta values.
    :param theta: Array of theta values.
    :param K: Constant K for transformation.
    :param region: 'leading' or 'trailing' to distinguish the transformation type.
    :return: Transformed x values.
    """
    if region == 'leading':
        return K * (1 - np.cos(theta))
    elif region == 'trailing':
        return K * (np.arctan(np.sinh(theta - pi/2)) + 1)

def inverse_x_transformation(x_transformed, K, region):
    """
    Applies the inverse x-axis transformation.
    :param x_transformed: Transformed x values.
    :param K: Constant K for transformation.
    :param region: 'leading' or 'trailing'.
    :return: Inverse transformed theta values.
    """
    if region == 'leading':
        return np.arccos(1 - x_transformed / K)
    elif region == 'trailing':
        return pi/2 + np.arcsinh(np.tan((x_transformed / K) - 1))

def apply_transformation(x, K):
    """
    Apply the transformation for both leading and trailing edge based on x-values.
    :param x: Original x coordinates.
    :param K: Transformation constant.
    :return: Transformed x coordinates.
    """
    mid_idx = len(x) // 2
    theta_leading = np.linspace(0, pi/2, mid_idx)  # Theta for leading edge (0 to pi/2)
    theta_trailing = np.linspace(pi/2, pi, len(x) - mid_idx)  # Theta for trailing edge (pi/2 to pi)
    
    # Apply leading edge transformation
    x_leading = x_transformation(theta_leading, K, 'leading')
    
    # Apply trailing edge transformation
    x_trailing = x_transformation(theta_trailing, K, 'trailing')
    
    # Combine transformed x values
    return np.concatenate((x_leading, x_trailing))

def inverse_apply_transformation(x_transformed, K):
    """
    Apply the inverse transformation for both leading and trailing edge.
    :param x_transformed: Transformed x coordinates.
    :param K: Transformation constant.
    :return: Original x coordinates.
    """
    mid_idx = len(x_transformed) // 2
    
    # Apply inverse leading edge transformation
    theta_leading = inverse_x_transformation(x_transformed[:mid_idx], K, 'leading')
    
    # Apply inverse trailing edge transformation
    theta_trailing = inverse_x_transformation(x_transformed[mid_idx:], K, 'trailing')
    
    # Combine theta values
    return np.concatenate((np.cos(theta_leading), np.cos(theta_trailing)))

def interpolate_surface(x, y, num_points):
    """
    Interpolates a surface (upper or lower) to the specified number of points.
    :param x: Original x coordinates.
    :param y: Original y coordinates.
    :param num_points: Number of points for interpolation.
    :return: Interpolated x and y coordinates.
    """
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    interpolator = interp1d(x_sorted, y_sorted, kind='linear')
    x_new = np.linspace(np.min(x_sorted), np.max(x_sorted), num_points)
    y_new = interpolator(x_new)
    
    return x_new, y_new

def plot_theta_vs_x(theta_original, theta_interp, x_original, x_interp):
    """
    Plot theta vs x for both original and interpolated airfoils.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(theta_original, x_original, 'o-', label='Original Airfoil (theta vs x)')
    plt.plot(theta_interp, x_interp, 's-', label='Interpolated Airfoil (theta vs x)')
    plt.title('Theta vs X: Original and Interpolated Airfoil')
    plt.xlabel('Theta')
    plt.ylabel('X')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_airfoil_coordinates(x_original, y_original, x_interp, y_interp):
    """
    Plot original and interpolated airfoil coordinates (x vs y).
    """
    plt.figure(figsize=(10, 5))
    plt.plot(x_original, y_original, 'o-', label='Original Airfoil')
    plt.plot(x_interp, y_interp, 's-', label='Interpolated Airfoil')
    plt.title('Original and Interpolated Airfoil Coordinates')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_transformed_airfoil(x_transformed, y_original, x_transformed_interp, y_interp):
    """
    Plot transformed original and interpolated airfoil using x-transformation.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(x_transformed, y_original, 'o-', label='Transformed Original Airfoil')
    plt.plot(x_transformed_interp, y_interp, 's-', label='Transformed Interpolated Airfoil')
    plt.title('Transformed Original vs Transformed Interpolated Airfoil')
    plt.xlabel('Transformed X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

def main(filename, num_points, K):
    """
    Main function to read, transform, split, interpolate, and plot the airfoil data.
    :param filename: Path to the airfoil .dat file.
    :param num_points: Number of points for interpolation.
    :param K: Constant for x-axis transformation.
    """
    # Read original airfoil data
    x_original, y_original = read_airfoil_dat(filename)
    
    # Apply x-axis transformation to the entire airfoil
    x_transformed = apply_transformation(x_original, K)
    
    # Split the airfoil after applying the transformation
    mid_idx = len(x_transformed) // 2
    x_upper = x_transformed[:mid_idx+1]
    y_upper = y_original[:mid_idx+1]
    x_lower = x_transformed[mid_idx:]
    y_lower = y_original[mid_idx:]
    
    plt.plot(x_lower, y_lower, x_upper, y_upper)

    # Interpolate upper and lower surfaces after transformation
    x_upper_interp, y_upper_interp = interpolate_surface(x_upper, y_upper, num_points)
    x_lower_interp, y_lower_interp = interpolate_surface(x_lower, y_lower, num_points)
    
    # plt.plot(x_lower_interp, y_lower_interp, x_upper_interp, y_upper_interp, x_original, y_original)

    # Combine the interpolated x and y for both surfaces
    x_combined_interp = np.concatenate((x_upper_interp, x_lower_interp[::-1]))
    y_combined_interp = np.concatenate((y_upper_interp, y_lower_interp[::-1]))
    
    # Plot 1: Theta vs X for both original and interpolated airfoil
    theta_original = np.linspace(0, pi, len(x_original))
    theta_interp = np.linspace(0, pi, len(x_combined_interp))
    plot_theta_vs_x(theta_original, theta_interp, x_original, x_combined_interp)
    
    # Plot 2: Original and interpolated airfoil coordinates (x vs y)
    plot_airfoil_coordinates(x_original, y_original, x_combined_interp, y_combined_interp)
    
    # Plot 3: Transformed airfoil coordinates using x-transformation
    plot_transformed_airfoil(x_transformed, y_original, x_combined_interp, y_combined_interp)

# Example usage:
filename = 'GOE602.dat'
num_points = 100  # Example: interpolate to 100 points per surface
K = 0.46278  # Constant for the transformation based on the equations
main(filename, num_points, K)
