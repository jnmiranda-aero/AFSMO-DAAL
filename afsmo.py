import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from scipy.interpolate import interp1d

from airfoil_util import normalize_airfoil

template ="""\
{}
   80.       10.       1.        0.        1.        1.        1.        2.
{}
{}
{}
{}
{}
{} 
  1.         
"""

def clear_dir():
	"""
	Clear the output files from the previous run if exists
	"""
	files_to_be_removed = ["afsmo.out","afsmo.pch","afsmo.dat","afsmo.smr"]
	for file in files_to_be_removed:
		if os.path.exists(file):
		    os.remove(file)

def extract_afsmo_dat(n_inter, filename="afsmo.dat"):
	"""
	Reads the xy dump file created by AFSMO and spits out
	the coordinates in standard *.dat format
	"""
	x, y= np.loadtxt(filename).T
	x_upper = x[:n_inter][::-1]
	y_upper = y[:n_inter][::-1]
	x_lower = x[n_inter:][1:]
	y_lower = y[n_inter:][1:]
	return np.concatenate([x_upper,x_lower]), np.concatenate([y_upper,y_lower])

def extract_datfile_title(filename):
	"""
	Reads title and x,y data from a standard *.dat file 
	"""
	title = open(filename).readlines()[0].strip()
	return title

def extract_datfile_data(filename):
	"""
	returns x,y point array extracted from a standard XFoil *.dat file. 
	"""
	return np.loadtxt(filename, skiprows=1)

def gen_input(title, x,y, input_filename, n_inter):
	"""
	Creates an input file for AFSMO. 
	Standard *dat file x,y array is split from the LE to create 4 arrays on Upper and Lower XY data.
	Interpolation points were created with cosine spacing with given n_inter
	Once data is generated it will be dumped in to a standard template. (template.in file)
	"""
	le = x.argmin()

	xu = x[:le+1][::-1]
	yu = y[:le+1][::-1]

	xl = x[le:]
	yl = y[le:]

	nu = len(xu)
	nl = len(xl)

	xy_upper = "\n".join("{:12.6f}{:12.6f}".format(x,y) for x,y in zip(xu,yu))
	xy_lower = "\n".join("{:12.6f}{:12.6f}".format(x,y) for x,y in zip(xl,yl))

	x_inter = 0.5*(1 + np.cos(np.linspace(np.pi, 0, n_inter)))
	x_inter = "\n".join(["{:12.6f}".format(x) for x in x_inter])

	file_content = template.format(title, nu, xy_upper, nl, xy_lower, n_inter, x_inter)
	with open(input_filename, "w") as f:
		f.write(file_content)

def remove_consecutive_duplicates(x,y):
	"""
	Removes duplicate points by element-wise comparison between
	the given array and shifted array. 
	Indexing is done on	x-values [:,0]
	"""
	points = np.vstack((x,y)).T
	return points[(points != np.roll(points,1,0))[:,0]].T

def write2dat(title, x_new, y_new, filename):
	"""
	Writes xy data in to an airfoil *.dat file
	To keep the precision :.7f format was used.
	"""
	data = "\n".join(["{:.7f} {:.7f}".format(x,y) for x,y in zip(x_new, y_new)])
	with open(filename, "w") as f:
		f.write("{}\n{}".format(title,data))

def extract_afsmo_smr(filename="afsmo.smr"):
	"""
	Reads the summary dump file created by AFSMO and spits out
	the x,y and y_smoothed coordinates. Helpful in identifying how successful the 
	smoothing process was, without interpolation
	"""
	theta, x, y_smoo = np.loadtxt(filename, skiprows=5, usecols=(1,2,5)).T
	return theta, x, y_smoo

def interpolate_new_points(theta, x, y, n_inter):
	# Bug fix - 29-Jan-2023
	# Problem 
	# =======
	# AFSMO.f tend to produce a duplicate point when interpolation option is used.
	# Solution
	# ========
	# The new point generation through interpolation is now done through python

	theta_x_spline = interp1d(theta,x, kind='cubic', fill_value='extrapolate')
	theta_y_spline = interp1d(theta,y, kind='cubic', fill_value='extrapolate')

	# To resemble n_inter as closely as possible with how AFSMO
	# treats n_inter --n*x_values on one surface so that the total airfoil
	# will have 2*n - 1 points after removing duplicate LE
	# 180 to -180 such that the points are laid out counter-clockwise

	new_theta = np.linspace(180, -180, (n_inter*2)-1 )

	# new_theta = np.concatenate((new_theta[new_theta<0], [0], new_theta[new_theta>0]))
	# Adding 0 above is a bad idea. MSS proved the curvature discontinuity through Rhino
	# Keeping this commented out in case I forget in the future.

	new_x = theta_x_spline(new_theta)
	new_y = theta_y_spline(new_theta)

	return new_x, new_y


def compare_plot(x,y, x_new, y_new, title, fix_aspect=False):
	"""
	plots original and smoothed airfoil coordinates for comparison
	"""
	plt.title(title)
	plt.plot(x, y, lw=0.5, color='blue', label="Original")
	plt.plot(x_new, y_new, lw=0.5, color='red', label="Smoothed")
	plt.legend()
	if fix_aspect:
		plt.gca().set_aspect("equal", 'datalim')
	plt.show()

def afsmo(dat_file, interpolate, n_inter, plotter, aspect, scale, external):
	"""
	main program
	1. extracts data from standard dat file
	2. Generates an *.in file understandable by AFSMO
	3. Runs AFSMO
	4. Extract data from AFSMO output file
	5. Rearranges the extracted data applicable for *dat files
	6. Writes smoothed coordinates to a dat file
	7. Plots both airfoils for comparison
	"""
	clear_dir()

	# two interpolation schemes are supported on this program, internal and external
	# when internal interpolation is used (AFSMO built-in interpolation)
	# the maximum n_inter AFSMO supports is 100. 

	afsmo_n_inter = min(n_inter, 100)

	dat_basefilename = dat_file[:-4]
	in_file = "{}.in".format(dat_basefilename)
	out_file = dat_basefilename + "_sm.dat"

	title = extract_datfile_title(dat_file)

	if scale:
		x, y  = normalize_airfoil(extract_datfile_data(dat_file))
	else:
		x, y  = extract_datfile_data(dat_file).T
	
	gen_input(title, x,y, in_file, afsmo_n_inter)
	os.system("{} {}".format("afsmo.exe" if os.name == "nt" else "./afsmo", in_file))
	
	try:
		theta, x_new, y_new  = extract_afsmo_smr()
		if interpolate:
			if external:
				x_new, y_new = interpolate_new_points(theta, x_new, y_new, n_inter)
			else:
				x_new, y_new = extract_afsmo_dat(afsmo_n_inter)
				x_new, y_new = remove_consecutive_duplicates(x_new, y_new)
				
	except:
		print("Failure encountered")
		return

	write2dat(title, x_new, y_new, out_file)

	if plotter:
		compare_plot(x,y, x_new, y_new, title, aspect)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='A simple wrapper for NASA AFSMO airfoil smoothing program')

	parser.add_argument('-d', '--dat-file',    
						type=str,  
						help='Input *.dat file name')

	parser.add_argument('-i', '--interpolate',    
						help='*dat file with (n) interpolated x coordinates will be generated when set. False by default', 
						action='store_true')

	parser.add_argument('-n', '--n-inter',    
						type=int,  
						help='Number of cosine spaced points to be interpolated', 
						default=80)

	parser.add_argument('-p', '--plotter',    
						help='Plots the original and Smoothed airfoils', 
						action='store_true')

	parser.add_argument('-a', '--aspect',     
						help='Uses real aspect ratio for plotting', 
						action='store_true')

	parser.add_argument('-s', '--scale',     
						help='Scales and de-rotates the airfoil [Normalize]', 
						action='store_true')

	parser.add_argument('-e', '--external',     
						help='Use external interpolation instead of AFSMO built-in interpolation', 
						action='store_true')

	args = parser.parse_args()
	afsmo(args.dat_file,
		  args.interpolate, 
		  args.n_inter,
		  args.plotter, 
		  args.aspect, 
		  args.scale, 
		  args.external)
