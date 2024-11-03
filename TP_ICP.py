# TP MAREVA Nuages de Points et Modelisation 3D - Python - FG 24/09/2020
# coding=utf8

# Import Numpy
import numpy as np

# Import library to plot in python
from matplotlib import pyplot as plt
from matplotlib import collections as mc
from mpl_toolkits.mplot3d import Axes3D

# Import functions from scikit-learn : KDTree
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply
# utils.ply est le chemin relatif utils/ply.py


def read_data_ply(path):
# Lecture de nuage de points sous format ply
    '''
    Lecture de nuage de points sous format ply
    Inputs :
        path = chemin d'acces au fichier
    Output :
        data = matrice (3 x n)
    '''
    data_ply = read_ply(path)
    data = np.vstack((data_ply['x'], data_ply['y'], data_ply['z']))
    return(data)

def write_data_ply(data,path):
    '''
    Ecriture de nuage de points sous format ply
    Inputs :
        data = matrice (3 x n)
        path = chemin d'acces au fichier
    '''
    write_ply(path, data.T, ['x', 'y', 'z'])
    
def show3D(data):
    '''
    Visualisation de nuages de points avec MatplotLib'
    Input :
        data = matrice (3 x n)
    '''
    #plt.cla()
    # Aide en ligne : help(plt)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(data[0], data[1], data[2], '.')
    #ax.plot(data_aligned[0], data_aligned[1], data_aligned[2], '.')
    #plt.axis('equal')
    plt.show()


def decimate(data,k_ech):
    '''
    Decimation
    # ----------
    Inputs :
        data = matrice (3 x n)
        k_ech : facteur de decimation
    Output :
        decimated = matrice (3 x (n/k_ech))
    '''

    if False:    
        # 1ere methode : boucle for
        n = data.shape[1]
        decimated = np.vstack(data[:, 0])  # Commence avec le premier point

        # Boucle pour ajouter chaque k_ech-ème point à decimated
        for i in range(1, n // k_ech):
            Xi = np.vstack(data[:, i * k_ech])
            decimated = np.hstack((decimated, Xi))  # Concaténation du point Xi


    else:
        # 2e methode : fonction de Numpy array
        decimated = data[:, ::k_ech]
        
    return decimated


def best_rigid_transform(data, ref):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs:
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
        ref = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns:
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''

    # Step 1: Compute barycenters
    data_center = np.mean(data, axis=1).reshape(3, 1)
    ref_center = np.mean(ref, axis=1).reshape(3, 1)

    # Step 2: Center the clouds
    data_c = data - data_center
    ref_c = ref - ref_center

    # Step 3: Compute the H matrix
    H = np.dot(data_c, ref_c.T)  # Using np.dot() for H matrix computation

    # Step 4: Perform Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)

    # Step 5: Checking the determinant of R
    R = np.dot(Vt.T, U.T)  # Using np.dot() for calculating R
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)  # Recalculate R if the determinant is negative

    # Step 6: Compute the translation vector
    T = ref_center - np.dot(R, data_center)  # Using np.dot() for calculating T

    return R, T


def icp_point_to_point(data, ref, max_iter, RMS_threshold):
    '''
    Iteratice closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N) matrix where "N" is the number of point and "d" the dimension
        ref = (d x N) matrix where "N" is the number of point and "d" the dimension
        max_iter = stop condition on the number of iteration
        RMS_threshold = stop condition on the distance
    Returns :
        R = (d x d) rotation matrix aligning data on ref
        T = (d x 1) translation vector aligning data on ref
        data_aligned = data aligned on ref
           
    '''

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Create a neighbor structure on ref
    search_tree = KDTree(ref.T)

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    RMS_list = []

    for i in range(max_iter):

        # Find the nearest neighbors
        distances, indices = search_tree.query(data_aligned.T, return_distance=True)

        # Compute average distance
        RMS = np.sqrt(np.mean(np.power(distances, 2)))

        # Distance criteria
        if RMS < RMS_threshold:
            break

        # Find best transform
        R, T = best_rigid_transform(data, ref[:, indices.ravel()])

        # Update lists
        R_list.append(R)
        T_list.append(T)
        neighbors_list.append(indices.ravel())
        RMS_list.append(RMS)

        # Aligned data
        data_aligned = R.dot(data) + T


    return data_aligned, R_list, T_list, neighbors_list, RMS_list


def show3D_superposed(data_original, data_decimated):
    '''
    Superpose deux nuages de points pour comparaison
    Inputs :
        data_original = nuage de points original (3 x n)
        data_decimated = nuage de points décimé (3 x m)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Afficher le nuage original en bleu
    ax.plot(data_original[0], data_original[1], data_original[2], '.', color='blue', label='Original')
    
    # Afficher le nuage décimé en rouge
    ax.plot(data_decimated[0], data_decimated[1], data_decimated[2], '.', color='red', label='Décimé')
    
    ax.legend()
    plt.show()


def show3D_superposed_transformed(data_original, data_transformed, title='Superposition'):
    '''
    Superpose le nuage de points original et le nuage transformé pour comparaison.
    Inputs :
        data_original = nuage de points original (3 x n)
        data_transformed = nuage de points transformé (3 x m)
        title = Titre de la visualisation
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Afficher le nuage original en bleu
    ax.plot(data_original[0], data_original[1], data_original[2], '.', color='blue', label='Original')
    
    # Afficher le nuage transformé en rouge
    ax.plot(data_transformed[0], data_transformed[1], data_transformed[2], '.', color='red', label='Transformé')
    
    ax.legend()
    plt.title(title)
    plt.show()


#
#           Main
#       \**********/
#

if __name__ == '__main__':


    # Fichiers de nuages de points
    bunny_o_path = 'data/bunny_original.ply'
    bunny_p_path = 'data/bunny_perturbed.ply'
    bunny_r_path = 'data/bunny_returned.ply'
    NDC_o_path = 'data/Notre_Dame_Des_Champs_1.ply'
    NDC_r_path = 'data/Notre_Dame_Des_Champs_returned.ply'

    # Lecture des fichiers
    bunny_o=read_data_ply(bunny_o_path)                    
    bunny_p=read_data_ply(bunny_p_path)
    NDC_o=read_data_ply(NDC_o_path)

    # Transformations : decimation, rotation, translation, echelle
    # ------------------------------------------------------------
    if False:
        # Visualisation du fichier d'origine
        show3D(bunny_o)

        # Decimation        
        k_ech=10
        decimated = decimate(bunny_o,k_ech)
        
        # Visualisation sous Python et par ecriture de fichier
        show3D(decimated)
        
        # Visualisation sous CloudCompare apres ecriture de fichier
        write_data_ply(decimated,bunny_r_path)
        # Puis ouvrir le fichier sous CloudCompare pour le visualiser

        # Visualisation superposée pour comparaison
        show3D_superposed(bunny_o, decimated)

    if False:
        # Visualisation du fichier d'origine
        show3D(NDC_o)

        # Decimation        
        k_ech=1000
        decimated = decimate(NDC_o, k_ech)

        # Visualisation sous Python et par ecriture de fichier
        show3D(decimated)

        # Visualisation sous CloudCompare apres ecriture de fichier
        write_data_ply(decimated,NDC_r_path)

        # Visualisation superposée pour comparaison
        show3D_superposed(NDC_o, decimated)

    if False:        
        # 1. Translation
        translation = np.array([0, -0.1, 0.1]).reshape(3, 1)  # Définir le vecteur de translation
        points_translated = bunny_o + translation  # Appliquer la translation
        show3D(points_translated)  # Visualiser le nuage de points après translation

        # Superpose original and translated
        show3D_superposed_transformed(bunny_o, points_translated, title='Original vs Translated')

        # Visualisation sous CloudCompare après écriture de fichier
        write_data_ply(points_translated, 'data/bunny_translated.ply')
        
        # 2. Centrage
        centroid = np.mean(bunny_o, axis=1).reshape(3, 1)  # Calculer le barycentre à partir de l'original
        points_centered = bunny_o - centroid  # Centrer le nuage de points
        show3D(points_centered)  # Visualiser le nuage de points centré

        # Superpose original and centered
        show3D_superposed_transformed(bunny_o, points_centered, title='Original vs Centered')

        # Visualisation sous CloudCompare après écriture de fichier
        write_data_ply(points_centered, 'data/bunny_centered.ply')
        
        # 3. Échelle
        points_scaled = points_centered / 2  # Diviser par 2
        show3D(points_scaled)  # Visualiser le nuage de points après échelle

        # Superpose original and scaled
        show3D_superposed_transformed(bunny_o, points_scaled, title='Original vs Scaled')

        # Visualisation sous CloudCompare après écriture de fichier
        write_data_ply(points_scaled, 'data/bunny_scaled.ply')
        
        # 4. Rotation
        theta = np.pi / 3  # Définir l'angle de rotation
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])  # Définir la matrice de rotation

        # Appliquer la rotation sur les points centré (pas sur le nuage précédemment transformé)
        points_rotated = R.dot(bunny_o)  # Appliquer la rotation directement sur l'original
        show3D(points_rotated)  # Visualiser le résultat final

        # Superpose original and rotated
        show3D_superposed_transformed(bunny_o, points_rotated, title='Original vs Rotated')

        # Visualisation sous CloudCompare après écriture de fichier
        write_data_ply(points_rotated, 'data/bunny_rotated.ply')


    # Meilleure transformation rigide (R, Tr) entre nuages de points
    # -------------------------------------------------------------
    if True:  # Ensure this section runs
        # show3D(bunny_p)  # Show the perturbed point cloud
        show3D(NDC_o)  # Show the perturbed point cloud

        # Create a translation or slight perturbation
        translation_vector = np.array([0.05, 0.02, -0.03]).reshape(3, 1)  # Example translation
        NDC_o_perturbed = NDC_o + translation_vector  # Perturb the original cloud

        # Find the best transformation
        # R, Tr = best_rigid_transform(bunny_p, bunny_o)
        R, Tr = best_rigid_transform(NDC_o_perturbed, NDC_o)
        
        # Apply the transformation
        # bunny_r_opt = np.dot(R, bunny_p) + Tr  # Use np.dot() for consistency
        NDC_r_opt = np.dot(R, NDC_o_perturbed) + Tr  # Use np.dot() for consistency
        
        # Show the transformed cloud
        # show3D(bunny_r_opt)
        show3D(NDC_r_opt)
        
        # Superpose original and perturbed clouds
        # show3D_superposed_transformed(bunny_o, bunny_p, title='Original vs Before Transformation')

        # Superpose original and transformed clouds
        # show3D_superposed_transformed(bunny_o, bunny_r_opt, title='Original vs After Transformation')
        
        # Superposition avant et après transformation
        show3D_superposed_transformed(NDC_o, NDC_r_opt, title='Original vs Transformé')

        # Écriture pour visualisation dans CloudCompare
        write_data_ply(NDC_r_opt, NDC_r_path)

        # Write both original and transformed clouds to PLY for CloudCompare
        # write_data_ply(bunny_o, 'data/bunny_original.ply')    # Original cloud
        # write_data_ply(bunny_p, 'data/bunny_perturbed.ply')   # Before transformation
        # write_data_ply(bunny_r_opt, 'data/bunny_transformed.ply')  # After transformation

        # Get average distances
        '''
        distances2_before = np.sum(np.power(bunny_p - bunny_o, 2), axis=0)
        RMS_before = np.sqrt(np.mean(distances2_before))
        distances2_after = np.sum(np.power(bunny_r_opt - bunny_o, 2), axis=0)
        RMS_after = np.sqrt(np.mean(distances2_after))
        
        print('Average RMS between points :')
        print('Before = {:.3f}'.format(RMS_before))
        print(' After = {:.3f}'.format(RMS_after))
        '''
   
    # Test ICP and visualize
    # **********************
    if True:

        # Parameters
        max_iter = 25
        RMS_threshold = 1e-4  # Stopping criteria based on RMS

        # data_aligned, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(bunny_p, bunny_o, max_iter, RMS_threshold)
        data_aligned, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(NDC_o_perturbed, NDC_o, max_iter, RMS_threshold)

        print("RMS List:", RMS_list)

        # Plot the error evolution
        plt.plot(RMS_list)
        plt.title("Evolution of RMS over iterations")
        plt.xlabel("Iteration")
        plt.ylabel("RMS error")
        plt.show()

        # Visualize original vs transformed clouds
        # show3D_superposed_transformed(bunny_o, data_aligned, title='Original vs Aligned (ICP)')
        show3D_superposed_transformed(NDC_o, data_aligned, title='Original vs Aligned (ICP)')




