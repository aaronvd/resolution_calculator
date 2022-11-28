#%%
import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
import types
import ast
import csv
import time
from tqdm import tqdm
import scipy.constants

C = scipy.constants.c

def load_list(filename):
    '''
        Loads in CSV file, supporting nested lists (i.e. CSVs exported from multidimensional arrays)
    '''
    list1 = []
    with open(filename, 'r', newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            list1.append(row)
    try:
        list1 = [[float(v) for v in r] for r in list1]
    except:
        list1 = [[ast.literal_eval(i) for i in v] for v in list1]
    return list1

def fft_resample(img, Nx, Ny, Nz):
    '''
    Resamples image img by padding or truncating in the Fourier domain.

    '''
    img = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(img)))
    scal = img.size
    img = padortruncate(img, Nx, Ny, Nz)
    scal = img.size / scal
    img = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(img)))*scal

    return img

def padortruncate(array, dx, dy, dz, val_bg=0):
    '''
    Pads (with value val_bg) or truncates array depending on whether array dimensions are great than or less than (dx, dy)

    '''
    dx = int(dx)
    dy = int(dy)
    dz = int(dz)
    nx = max((dx-array.shape[0])//2, 0)
    ny = max((dy-array.shape[1])//2, 0)
    nz = max((dz-array.shape[2])//2, 0)
    px = max((array.shape[0]-dx)//2, 0)
    py = max((array.shape[1]-dy)//2, 0)
    pz = max((array.shape[2]-dz)//2, 0)
    newarray = val_bg * np.ones((dx, dy, dz), dtype=array.dtype)
    cx = min(array.shape[0], dx)
    cy = min(array.shape[1], dy)
    cz = min(array.shape[2], dz)
    newarray[nx:nx+cx, ny:ny+cy, nz:nz+cz] = array[px:px+cx, py:py+cy, pz:pz+cz]

    return newarray

class ResolutionCalculator():
    '''
    Accepts a list of vectors describing element positions and generates a list of translated and rotated copies.
    Input array of points at which to evaluate resolution using calculate_bandpass().
    '''
    def __init__(self, f_list, array1_element_vectors, array2_element_vectors=None, points_array=None, measurement_type=[], rotation_list=[], translation_list=[], **kwargs):
        
        self.array1_element_vectors = array1_element_vectors
        self.array2_element_vectors = array2_element_vectors
        self.measurement_type = measurement_type
        if len(self.measurement_type) == 0:
            if self.array2_element_vectors is None:
                self.measurement_type.append('S11')
            else:
                self.measurement_type.append('S21')

        if self.array2_element_vectors is None:
            if len(list(filter(lambda x: x != 'S11', self.measurement_type))) != 0:
                raise Exception('Must provide Array 2 element position vectors.')

        self.N1_elements = self.array1_element_vectors.shape[0]
        if self.array2_element_vectors is not None:
            self.N2_elements = self.array2_element_vectors.shape[0]

        self.rotation_list = rotation_list
        if len(self.rotation_list) == 0:
            self.rotation_list.append(np.array([0, 0, 0]))
        self.translation_list = translation_list
        if len(self.translation_list) == 0:
            self.translation_list.append(np.array([0, 0, 0]))
        self.points_array = np.array(points_array)
        if self.points_array.ndim == 1:
            self.points_array = self.points_array[None,:]
        
        self.transformed_arrays1 = []
        self.transformed_arrays2 = []
        self.resolution_list = [None for i in range(self.points_array.shape[0])]
        self.k_extrema = [None for i in range(self.points_array.shape[0])]
        self.f = f_list
        self.lam = C/self.f
        self.k = 2*np.pi/self.lam
        
        self.transform()

    def rotation_matrix_x(self, theta):
        mtx = np.array([[1, 0, 0], 
                        [0, np.cos(theta), -np.sin(theta)], 
                        [0, np.sin(theta), np.cos(theta)]])
        return mtx

    def rotation_matrix_y(self, theta):
        mtx = np.array([[np.cos(theta), 0, np.sin(theta)], 
                        [0, 1, 0], 
                        [-np.sin(theta), 0, np.cos(theta)]])
        return mtx

    def rotation_matrix_z(self, theta):
        mtx = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])
        return mtx

    def transformation_matrix(self, rotation_vector, translation_vector):
        rotation_matrix = self.rotation_matrix_z(rotation_vector[2]) @ self.rotation_matrix_y(rotation_vector[1]) @ self.rotation_matrix_x(rotation_vector[0])
        return np.concatenate((np.concatenate((rotation_matrix, translation_vector[:,None]), axis=1), np.array([0, 0, 0, 1])[None,:]), axis=0)

    def transform(self):
        for i in range(len(self.translation_list)):
            self.transformed_arrays1.append(
                np.matmul(self.transformation_matrix(self.rotation_list[i], self.translation_list[i]),
                        np.concatenate((self.array1_element_vectors[:,:,None], np.ones((self.N1_elements, 1, 1))), axis=1))[:,:3,0]
                                            )
            if self.array2_element_vectors is not None:
                self.transformed_arrays2.append(
                    np.matmul(self.transformation_matrix(self.rotation_list[i], self.translation_list[i]),
                            np.concatenate((self.array2_element_vectors[:,:,None], np.ones((self.N2_elements, 1, 1))), axis=1))[:,:3,0]
                                                )
    
    @staticmethod
    def pick_extrema(k_array1, k_array2, k_extrema):
        if k_extrema is None:
            k_extrema = np.zeros((2, 3), dtype=np.float32)
            k_extrema[0,:] = np.amin(np.amin(k_array1, axis=0) + np.amin(k_array2, axis=0), axis=1)
            k_extrema[1,:] = np.amax(np.amax(k_array1, axis=0) + np.amax(k_array2, axis=0), axis=1)
        else:
            k_extrema[0,:] = np.minimum(np.amin(np.amin(k_array1, axis=0) + np.amin(k_array2, axis=0), axis=1), k_extrema[0,:])
            k_extrema[1,:] = np.maximum(np.amax(np.amax(k_array1, axis=0) + np.amax(k_array2, axis=0), axis=1), k_extrema[1,:])
        return k_extrema

    def calculate_resolution(self, points_index=None, quiet=True):

        if points_index is None:
            points_index = range(self.points_array.shape[0])
        elif type(points_index)==int:
            points_index = [points_index]

        for i in points_index:
            for j in range(len(self.translation_list)):

                theta1 = np.arccos((self.points_array[i,0] - self.transformed_arrays1[j][:,0]) / np.linalg.norm(self.points_array[i,None,:] - self.transformed_arrays1[j], ord=2, axis=1))
                phi1 = np.arctan2((self.points_array[i,2] - self.transformed_arrays1[j][:,2]), (self.points_array[i,1] - self.transformed_arrays1[j][:,1]))
                array1_k_list = np.stack((self.k[None,:] * np.cos(theta1[:,None]),
                                          self.k[None,:] * np.sin(theta1[:,None]) * np.cos(phi1[:,None]),
                                          self.k[None,:] * np.sin(theta1[:,None]) * np.sin(phi1[:,None])), axis=1)

                if self.array2_element_vectors is not None:
                    theta2 = np.arccos((self.points_array[i,0] - self.transformed_arrays2[j][:,0]) / np.linalg.norm(self.points_array[i,None,:] - self.transformed_arrays2[j], ord=2, axis=1))
                    phi2 = np.arctan2((self.points_array[i,2] - self.transformed_arrays2[j][:,2]), (self.points_array[i,1] - self.transformed_arrays2[j][:,1]))
                    array2_k_list = np.stack((self.k[None,:] * np.cos(theta2[:,None]),
                                            self.k[None,:] * np.sin(theta2[:,None]) * np.cos(phi2[:,None]),
                                            self.k[None,:] * np.sin(theta2[:,None]) * np.sin(phi2[:,None])), axis=1)

                if 'S11' in self.measurement_type:
                    self.k_extrema[i] = self.pick_extrema(array1_k_list, array1_k_list, self.k_extrema[i])
                if 'S22' in self.measurement_type:
                    self.k_extrema[i] = self.pick_extrema(array2_k_list, array2_k_list, self.k_extrema[i])
                if ('S21' in self.measurement_type) or ('S12' in self.measurement_type):
                    self.k_extrema[i] = self.pick_extrema(array1_k_list, array2_k_list, self.k_extrema[i])
            
            self.resolution_list[i] = np.maximum(2*np.pi/(self.k_extrema[i][1,:] - self.k_extrema[i][0,:]), 
                                                  np.array([np.amin(self.lam)/4, np.amin(self.lam)/4, np.amin(self.lam)/4]))
            
            resolution_table = 'POINT INDEX = {}\n'.format(points_index[i])
            resolution_table += '{:<25} {:<25}\n'.format('DIRECTION', 'RESOLUTION')
            resolution_table += '{:<25} {} cm\n'.format('x', np.around(self.resolution_list[i][0]*100, decimals=2))
            resolution_table += '{:<25} {} cm\n'.format('y', np.around(self.resolution_list[i][1]*100, decimals=2))
            resolution_table += '{:<25} {} cm\n'.format('z', np.around(self.resolution_list[i][2]*100, decimals=2))
            if not quiet:
                print(resolution_table)

    def plot_arrays(self, fig=None):
        data = []
        x = np.array(self.transformed_arrays1)[:,:,0].flatten()
        y = np.array(self.transformed_arrays1)[:,:,1].flatten()
        z = np.array(self.transformed_arrays1)[:,:,2].flatten()
        data.append(go.Scatter3d(x=x, y=y, z=z, mode='markers', name='Array 1',
                                marker=dict(
                                        size=3,
                                        color='red',                # set color to an array/list of desired values
                                        # colorscale='Viridis',   # choose a colorscale
                                        opacity=1)))
        if self.array2_element_vectors is not None:
            x = np.array(self.transformed_arrays2)[:,:,0].flatten()
            y = np.array(self.transformed_arrays2)[:,:,1].flatten()
            z = np.array(self.transformed_arrays2)[:,:,2].flatten()
            data.append(go.Scatter3d(x=x, y=y, z=z, mode='markers', name='Array 2',
                                    marker=dict(
                                            size=3,
                                            color='blue',                # set color to an array/list of desired values
                                            # colorscale='Viridis',   # choose a colorscale
                                            opacity=1)))
        if fig is None:
            self.fig = go.Figure(data=data)
            self.fig.update_layout(scene_aspectmode='data')
            self.fig.show()
        else:
            for trace in data:
                fig.add_trace(trace)

    def plot_scene(self, fig=None):
        data = go.Scatter3d(x=self.points_array[:,0], 
                            y=self.points_array[:,1],
                            z=self.points_array[:,2],
                            mode='markers',
                            name='Target',
                            marker=dict(size=3,
                                        color='orange',
                                        opacity=1))
        if fig is None:
            self.fig = go.Figure(data=data)
            self.fig.update_layout(scene_aspectmode='data')
            self.fig.show()
        else:
            fig.add_trace(data)

    def plot_system(self):
        self.fig = go.Figure()
        self.plot_arrays(fig=self.fig)
        self.plot_scene(fig=self.fig)
        self.fig.update_layout(scene_aspectmode='data')
        self.fig.show()

# #%% Test 1
# c = 3E8
# f = np.linspace(17.5E9, 26.5E9, 101)
# # f = np.array([24E9])
# lam = c/f
# k = 2*np.pi/lam

# L = 0.002

# y_tx = np.arange(-L/2, L/2+np.amin(lam)/4, np.amin(lam)/4)
# z_tx = np.arange(-L/2, L/2+np.amin(lam)/4, np.amin(lam)/4)
# Y_tx, Z_tx = np.meshgrid(y_tx, z_tx, indexing='ij')
# X_tx = np.zeros_like(Y_tx)
# array1_element_vectors = np.stack((X_tx.reshape(-1), Y_tx.reshape(-1), Z_tx.reshape(-1)), axis=1)

# points_array = np.array([[.8, 0, 0], [.8, .05, .05]])

# calculator = ResolutionCalculator(f,
#                                   array1_element_vectors, 
#                                   points_array=points_array)
# calculator.plot_arrays()
# calculator.plot_system()
# calculator.calculate_resolution(points_index=0, quiet=False)
# print('c/2B = {} cm'.format(np.around(c/(2*(np.amax(f) - np.amin(f)))*100, decimals=2)))


# #%% Test 2
# c = 3E8
# # f = np.linspace(17.5E9, 26.5E9, 101)
# f = np.array([24E9])
# lam = c/f
# k = 2*np.pi/lam

# L = 20*np.amin(lam)

# y_tx = np.arange(-L/2, L/2+np.amin(lam)/4, np.amin(lam)/4)
# z_tx = np.arange(-L/2, L/2+np.amin(lam)/4, np.amin(lam)/4)
# Y_tx, Z_tx = np.meshgrid(y_tx, z_tx, indexing='ij')
# X_tx = np.zeros_like(Y_tx)
# array1_element_vectors = np.array([0, 0, 0], dtype=np.float32)[None,:]
# translation_array = np.stack((X_tx.reshape(-1), Y_tx.reshape(-1), Z_tx.reshape(-1)), axis=1)
# translation_list = [translation_array[i,:] for i in range(translation_array.shape[0])]
# rotation_list = [np.array([0, 0, 0], dtype=np.float32) for i in range(translation_array.shape[0])]

# x_offset = 1
# points_array = np.array([[x_offset, 0, 0], [x_offset, .05, .05]])

# calculator = ResolutionCalculator(f,
#                                   array1_element_vectors, 
#                                   points_array=points_array,
#                                   translation_list=translation_list,
#                                   rotation_list=rotation_list)
# calculator.plot_arrays()
# calculator.plot_system()
# calculator.calculate_resolution(points_index=0, quiet=False)
# print('lambda*x/2L = {} cm'.format(np.around(np.amin(lam)*x_offset/(2*L)*100, decimals=2)))
