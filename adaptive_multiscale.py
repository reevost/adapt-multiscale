import numpy as np
from scipy.sparse import csc_array
from scipy.sparse.linalg import cg, svds
import matplotlib.pyplot as plt
from utils import write_array_on_file, write_matrix_on_file  # to remove after data gathering


w11 = lambda r: np.where(1 - r < 0, 0, 1 - r) ** 3 * (3 * r + 1)
w31 = lambda r: np.where(1 - r < 0, 0, 1 - r) ** 4 * (4 * r + 1)


def generate_ngrid_set(dim, step_size, start=0.0, end=1.0):
    """
    Generate a centered grid of points in the hypercube [start, end]^dim.

    Parameters:
        dim (int): Dimension of the points in the grid. Must be a positive integer.
        step_size (float): Must be a positive number.
                           Represent the spacing between adjacent grid points along each axis.
        start (float): Starting point of the sequence.
        end (float): Ending point of the sequence.

    Returns:
        numpy.ndarray: Array of shape (N, dim), where
                       N = ceil((end - start) / step_size) ** dim,
                       containing all grid points in [start, end]^dim.
    """

    # --- Input validation ----------------------------------------------------
    if not isinstance(dim, int) or dim <= 0:
        raise ValueError("dim must be a positive integer")

    if not isinstance(step_size, (int, float)) or 0 >= step_size:
        raise ValueError("step_size must be a positive number")

    # --- Function body -------------------------------------------------------
    point_on_ax = np.arange(start + (end - step_size * np.floor((end - start) / step_size)) / 2, end + 1e-12,
                            step_size).reshape((-1, 1))

    # --- 1D case: return a simple vector -------------------------------------
    # Use a small epsilon to ensure inclusion of end despite floating‑point arithmetics.
    if dim == 1:
        return point_on_ax

    # --- Multi‑dimensional case ----------------------------------------------
    # Build coordinate arrays for each dimension.
    # np.meshgrid creates a full grid; indexing="ij" preserves matrix‑style ordering.
    grid = np.meshgrid(*([point_on_ax] * dim), indexing="ij")

    # Flatten each dimension and stack column‑wise to form (N, dim).
    return np.column_stack([g.ravel() for g in grid])


def generate_1d_nested_sequence(starting_step_size, mu, levels, start=0.0, end=1.0):
    """
    Generate a centered grid of points in the hypercube [start, end]^dim.

    Parameters:
        starting_step_size (float):
                Must be a positive number.
                Represent the spacing between adjacent grid points along each axis on the first level.
        mu (float):
                Must be a value in (0,0.5]. Other choices do not make sense for the purpose of this function.
                Scaling parameter of the nested sequence, every set will have a step size which is mu times the previous step.
        levels (int):
                Depth of the sequence, i.e. number of nested sets.
        start (float):
                Starting point of the sequence.
        end (float):
                Ending point of the sequence.

    Returns:
        X (np.ndarray of floats):
                Array of shape (N, ), containing all nested sets in [start, end].
                The n-th subset can be accessed via X[:N_list[n]]
        N_list (np.ndarray of ints):
                Array with integer entries and shape (levels,).
                Entries can be used to access the subsets of the nested sequence.
    """

    # --- Input validation ----------------------------------------------------
    if not isinstance(starting_step_size, (int, float)) or 0 >= starting_step_size:
        raise ValueError(f"starting_step_size must be a positive number! instead got {starting_step_size}.")

    if not isinstance(mu, (int, float)) or not 0.5 >= mu > 0:
        raise ValueError(f"mu must be in (0,0.5]! instead got {mu}.")

    # --- Function body -------------------------------------------------------
    centered_start = start + (end - start - starting_step_size * np.floor((end - start) / starting_step_size)) / 2
    # Use a small epsilon to ensure inclusion of end despite floating‑point arithmetics.
    # we will store every update to the set in a list and then concatenate the pieces instead of concatenate every update
    set1d_list = [np.arange(centered_start, end + 1e-12, starting_step_size)]
    # we keep a sorted set of the points to generate at every level a filling for each subsequent points
    sortedSet = set1d_list[0].copy()
    # define a list where the length of sets will be stored
    size_list = [len(sortedSet)]
    # Compute other levels
    for _ in range(1, levels):
        # define index to iterate over points
        ind = len(sortedSet) - 1
        # update the step
        starting_step_size *= mu
        # compute the filling in area prior to the first point of available in the sorted sequence
        tmp_increment = np.arange(sortedSet[0], start - 1e-12, -starting_step_size)[-1:0:-1]
        # insert the points at the end of the nested sequence and at the beginning in the sorted array
        set1d_list += [tmp_increment]
        sortedSet = np.insert(sortedSet, 0, tmp_increment)
        while ind != 0:
            # compute the filling between to subsequent points
            tmp_increment = np.arange((sortedSet[-ind - 1] + sortedSet[-ind] - starting_step_size * np.floor(
                (sortedSet[-ind] - sortedSet[-ind - 1]) / starting_step_size - 2)) / 2,
                                      sortedSet[-ind] - starting_step_size + 1e-12, starting_step_size)
            # insert the points at the end of the nested sequence at between the pre-existing points in the sorted array
            set1d_list += [tmp_increment]
            sortedSet = np.insert(sortedSet, -ind, tmp_increment)
            # move index
            ind -= 1
            # EndWhile
        # compute the filling in area after the last point of available in the sorted sequence
        tmp_increment = np.arange(sortedSet[-1], end + 1e-12, starting_step_size)[1:]
        # insert the points as in previous steps
        set1d_list += [tmp_increment]
        sortedSet = np.append(sortedSet, tmp_increment)
        # add the length of the computed level to the list
        size_list += [len(sortedSet)]
        # EndFor
    return np.concatenate(set1d_list), np.array(size_list)


def generate_ngrid_nested_sequence(dim, starting_step_size, mu, levels, start=0.0, end=1.0):
    """
    Generate a centered grid of points in the hypercube [start, end]^dim.

    Parameters:
        dim (int):
                Dimension of the points in the grid. Must be a positive integer greater than 1.
                Use the specific 1d version for dim == 1.
        starting_step_size (float):
                Must be a positive number.
                Represent the spacing between adjacent grid points along each axis on the first level.
        mu (float):
                Must be a value in (0,0.5]. Other choices do not make sense for the purpose of this function.
                Scaling parameter of the nested sequence, every set will have a step size which is mu times the previous step.
        levels (int):
                Depth of the sequence, i.e. number of nested sets.
        start (float):
                Starting point of the sequence.
        end (float):
                Ending point of the sequence.

     Returns:
        X (np.ndarray of floats):
                Array of shape (N, dim), containing all nested sets in [start, end]^dim.
                The n-th subset can be accessed via X[:N_list[n], :]
        N_list (np.ndarray of ints):
                Array with integer entries and shape (levels,).
                Entries can be used to access the subsets of the nested sequence.
    """

    # --- Input validation ----------------------------------------------------
    if not isinstance(dim, int) or dim <= 1:
        raise ValueError(f"dim must be an integer greater than 1! instead got {dim}.")

    if not isinstance(starting_step_size, (int, float)) or 0 >= starting_step_size:
        raise ValueError(f"starting_step_size must be a positive number! instead got {starting_step_size}.")

    if not isinstance(mu, (int, float)) or not 0.5 >= mu > 0:
        raise ValueError(f"mu must be in (0,0.5]! instead got {mu}.")

    # --- Function body -------------------------------------------------------
    # The idea is to work with 1d arrays to compute refinements and the tensorize them.
    # resulted grids are stored into a list then the uniqueness of set structure is exploited to achieve the nestedness.
    centered_start = start + (end - start - starting_step_size * np.floor((end - start) / starting_step_size)) / 2
    # Use a small epsilon to ensure inclusion of end despite floating‑point arithmetics.
    # we will compute a 1d array, refining over it and the tensorize it with meshgrid.
    set1d = np.arange(centered_start, end + 1e-12, starting_step_size)
    # Build coordinate arrays for each dimension. indexing="ij" preserves matrix‑style ordering.
    grid = np.meshgrid(*([set1d.reshape((-1, 1))] * dim), indexing="ij")
    set_nested_sequence = [np.column_stack([g.ravel() for g in grid])]
    # Compute other levels. Here, we compute a refined 1d array starting from the previous 1d array.
    for _ in range(1, levels):
        # update step size
        starting_step_size *= mu
        # generate the refined 1d array as in the 1d case
        # we start from including feasible points between the lower bound and the first point
        new_set1d_list = [np.arange(set1d[0], start - 1e-12, -starting_step_size)[-1::-1]]
        for i in range(len(set1d) - 1):
            # include feasible points between the couple of subsequent points
            new_set1d_list += [np.append(np.arange((set1d[i] + set1d[i + 1] - starting_step_size * np.floor(
                (set1d[i + 1] - set1d[i]) / starting_step_size - 2)) / 2,
                                                   set1d[i + 1] - starting_step_size + 1e-12, starting_step_size),
                                         set1d[i + 1]
                                         )
                               ]
            # EndFor
        # we finish with the inclusion of feasible points between the last point and the upper bound
        set1d = np.append(np.concatenate(new_set1d_list),
                          np.arange(set1d[-1], end + 1e-12, starting_step_size)[1:].reshape((-1, 1)))
        # Then the refined array is tensorized and add to the list of sets
        grid = np.meshgrid(*([set1d.reshape((-1, 1))] * dim), indexing="ij")
        set_nested_sequence += [np.column_stack([g.ravel() for g in grid])]
        # EndFor
    # Filtration to achieve nestedness
    # compute sizes of sets first
    size_list = [len(set_nested_sequence[l]) for l in range(levels)]
    for l in range(1, levels):
        # at every iteration we generate a set of tuple of the current set
        tmp_set = {tuple(point) for point in set_nested_sequence[0]}
        # then we compute the subset of points to include exploiting the set structure
        set_increment = np.array([point for point in set_nested_sequence[l] if tuple(point) not in tmp_set])
        # and append it to the first set updating it to the current level
        set_nested_sequence[0] = np.vstack([set_nested_sequence[0], set_increment])
        # EndFor
    return set_nested_sequence[0], np.array(size_list)


def point_selection_1d(nested_set, nested_size_level, selection_values, cut_mask, threshold, radius):
    """
    Selection algorithm from paper. Given the nested set, compute the boolean masks of cut points and selected points.
    Cut points are points for which every point in the ball centered on them with radius is associated
    with selection_values smaller than threshold. Selected points are all points that do not belong to balls centered
    of cut points with radius. The function returns also the number of selected points.

    Parameters:
        nested_set (np.ndarray):
                Array of nested points. It is assumed that the array stores the finest available set and
                through a list of lengths it is possible to take windows of the array to recover a
                nested level. Since this function is designed for 1d points, the array is assumed to
                have dimension 1 and shape (N,).
        nested_size_level (np.int):
                Must be a positive integer. Is associated with the subset of interest where
                the selection is performed.
        selection_values (np.ndarray):
                Array of values associated with the nested set with shape (N,).
        cut_mask (np.ndarray of bool):
                Array of boolean values associated with the nested set with shape (N,).
                Starting mask value for the cut points.
        threshold (float):
                Threshold value in the selection algorithm.
        radius (float):
                Radius of influence in the selection algorithm.

     Returns:
        cut_mask (np.ndarray of bool):
                Array with shape (N,). nested_set[selection_mask] represent the subset of cut points.
        selection_mask (np.ndarray of bool):
                Array with shape (nested_size_level,).
                Nested_set[:nested_size_level][selection_mask] represent the subset of selected points.
        N_check (int):
                Number of selected points.
    """
    # --- Input validation ----------------------------------------------------
    if not isinstance(nested_set, np.ndarray) or nested_set.ndim != 1:
        raise ValueError(f"nested_set must be a ndarray with ndim 1! instead got {nested_set.ndim = }.")

    if nested_size_level.dtype != int or nested_size_level < 1:
        raise ValueError(f"nested_size_level must be a positive integer! instead got {nested_size_level = }.")

    if not isinstance(selection_values, np.ndarray) or selection_values.shape != nested_set.shape:
        raise ValueError(f"selection_values must be a ndarray with the same shape of nested_set ({nested_set.shape})! "
                         f"instead got {selection_values.shape = }.")

    if not isinstance(cut_mask, np.ndarray) or cut_mask.shape != nested_set.shape or cut_mask.dtype != bool:
        raise ValueError(
            f"selection_values must be a bool ndarray with the same shape of nested_set ({nested_set.shape})!"
            f" instead got {cut_mask.shape = } and {cut_mask.dtype = }.")

    if not isinstance(threshold, (int, float)) or 0 >= threshold:
        raise ValueError(f"threshold must be a positive number! instead got {threshold = }.")

    if not isinstance(radius, (int, float)) or 0 >= radius:
        raise ValueError(f"radius must be a positive number! instead got {radius = }.")

    # --- Function body -------------------------------------------------------
    # fist we compute the mask of points associated with values larger than the threshold
    high_values = np.abs(selection_values) > threshold
    if not np.any(high_values):  # if there is none, means that we cut all the points, and select none.
        print(f"the error is smaller than the threshold in every evaluated point")
        return np.ones(len(nested_set), dtype=bool), np.zeros(nested_size_level, dtype=bool), 0
    else:  # otherwise we proceed with the selection
        # iterate over cut candidates, 7.0 points that are not yet cut and have associated value smaller than threshold
        for i in np.arange(nested_size_level)[~high_values[:nested_size_level] & ~cut_mask[:nested_size_level]]:
            cut_mask[i] = np.min(np.abs(nested_set[high_values] - nested_set[i])) > radius
        # now we have found the set X^{\prime}, i.e. the points that have a radius of negligible points around
        # we are left to remove the points and the one up to radius far, i.e. \bar_{X}
        # we pick as starting selection_mask all but cut points
        selection_mask = ~cut_mask[:nested_size_level]
        for i in np.arange(nested_size_level)[cut_mask[:nested_size_level]]:  # iterate over the cut points
            # and remove the one up to radius far
            selection_mask[selection_mask] &= np.abs(
                nested_set[:nested_size_level][selection_mask] - nested_set[i]) > radius
        print(
            f"selected {sum(selection_mask)} over {nested_size_level} points. With {threshold = :.6g} and {radius = :.6g}.")
        return cut_mask, selection_mask, sum(selection_mask)


def point_array_selection(nested_set, nested_size_level, selection_values, cut_mask, threshold, radius):
    """
    Selection algorithm from paper. Given the nested set, compute the boolean masks of cut points and selected points.
    Cut points are points for which every point in the ball centered on them with radius is associated
    with selection_values smaller than threshold. Selected points are all points that do not belong to balls centered
    of cut points with radius. The function returns also the number of selected points.

    Parameters:
        nested_set (np.ndarray):
                Array of nested points. It is assumed that the array stores the finest available set and
                through a list of lengths it is possible to take windows of the array to recover a
                nested level. The array is assumed to have dimension 2 and shape (N,dim) where N is the
                number of points, and dim is their dimension.
        nested_size_level (np.int):
                Must be a positive integer. Is associated with the subset of interest where
                the selection is performed.
        selection_values (np.ndarray):
                Array of values associated with the nested set with shape (N,).
        cut_mask (np.ndarray of bool):
                Array of boolean values associated with the nested set with shape (N,).
                Starting mask value for the cut points.
        threshold (float):
                Threshold value in the selection algorithm.
        radius (float):
                Radius of influence in the selection algorithm.

     Returns:
        cut_mask (np.ndarray of bool):
                Array with shape (N,). nested_set[selection_mask] represent the subset of cut points.
        selection_mask (np.ndarray of bool):
                Array with shape (nested_size_level,).
                Nested_set[:nested_size_level][selection_mask] represent the subset of selected points.
        N_check (int):
                Number of selected points.
    """
    # --- Input validation ----------------------------------------------------
    if not isinstance(nested_set, np.ndarray) or nested_set.ndim != 2:
        raise ValueError(f"nested_set must be a ndarray with ndim 2! instead got {nested_set.ndim = }.")

    if nested_size_level.dtype != int or nested_size_level < 1:
        raise ValueError(f"nested_size_level must be a positive integer! instead got {nested_size_level = }.")

    if not isinstance(selection_values, np.ndarray) or selection_values.shape[0] != nested_set.shape[0]:
        raise ValueError(f"selection_values must be a ndarray with the same length of nested_set "
                         f"({nested_set.shape[0]})! instead got {selection_values.shape[0] = }.")

    if not isinstance(cut_mask, np.ndarray) or cut_mask.shape == nested_set.shape or cut_mask.dtype != bool:
        raise ValueError(f"selection_values must be a bool ndarray with the same shape of selection_values"
                         f" ({selection_values.shape})! instead got {cut_mask.shape = } and {cut_mask.dtype = }.")

    if not isinstance(threshold, (int, float)) or 0 >= threshold:
        raise ValueError(f"threshold must be a positive number! instead got {threshold = }.")

    if not isinstance(radius, (int, float)) or 0 >= radius:
        raise ValueError(f"radius must be a positive number! instead got {radius = }.")

    # --- Function body -------------------------------------------------------
    # fist we compute the mask of points associated with values larger than the threshold
    high_values = np.abs(selection_values) > threshold
    if not np.any(high_values):  # if there is none, means that we cut all the points, and select none.
        print(f"the error is smaller than the threshold in every evaluated point")
        return np.ones(len(nested_set), dtype=bool), np.zeros(nested_size_level, dtype=bool), 0
    else:  # otherwise we proceed with the selection
        # iterate over cut candidates, i.e. points that are not yet cut and have associated value smaller than threshold
        for i in np.arange(nested_size_level)[~high_values[:nested_size_level] & ~cut_mask[:nested_size_level]]:
            cut_mask[i] = np.min(np.sqrt(np.sum((nested_set[high_values] - nested_set[i]) ** 2, axis=-1))) > radius
        # now we have found the set X^{\prime}, i.e. the points that have a radius of negligible points around
        # we are left to remove the points and the one up to radius far, i.e. \bar_{X}
        # we pick as starting selection_mask all but cut points
        selection_mask = ~cut_mask[:nested_size_level]
        for i in np.arange(nested_size_level)[cut_mask[:nested_size_level]]:  # iterate over the cut points
            # and remove the one up to radius far
            selection_mask[selection_mask] &= np.sqrt(
                np.sum((nested_set[:nested_size_level][selection_mask] - nested_set[i]) ** 2, axis=-1)) > radius
        print(
            f"selected {sum(selection_mask)} over {nested_size_level} points. With {threshold = :.6g} and {radius = :.6g}.")
        return cut_mask, selection_mask, sum(selection_mask)


def w11_kernel_matrix(row_points, col_points, delta, dim):
    """
    Compute the kernel matrix using the wendland function w11. The (i,j) entry of the matrix corresponds to
    w11(norm2(row_points[i]-col_points[j])/delta)/delta**dim.

    Parameters:
        row_points (np.ndarray):
                Array of points associated with rows of the kernel matrix with shape (N, dim).
        col_points (np.ndarray):
                Array of points associated with columns of the kernel matrix with shape (M, dim).
        delta (float):
                Scale of the kernel function.
        dim (int):
                Dimension of point sets (both row and col).

     Returns:
        kernel matrix (scipy.sparse.csc_array):
                Kernel matrix with shape (N,M).
    """
    # --- Input validation ----------------------------------------------------
    if not isinstance(dim, int) or dim < 1:
        raise ValueError(f"dim must be an integer greater than 0! instead got {dim}.")

    if not isinstance(delta, (int, float)) or 0 >= delta:
        raise ValueError(f"delta must be a positive number! instead got {delta}.")

    if not isinstance(row_points, np.ndarray) or row_points.ndim > 2:
        raise ValueError(f"row_points must be an array of dimension at most 2, instead got {row_points.ndim = }.")
    if row_points.ndim == 1:
        row_points = row_points.reshape((-1, 1))
    elif row_points.shape[-1] != dim:
        raise ValueError(f"row_points must be an array of shape (N, {dim}) instead got {row_points.shape = }.")

    if not isinstance(col_points, np.ndarray) or col_points.ndim > 2:
        raise ValueError(f"row_points must be an array of dimension at most 2, instead got {col_points.ndim = }.")
    if col_points.ndim == 1:
        col_points = col_points.reshape((-1, 1))
    elif col_points.shape[-1] != dim:
        raise ValueError(f"row_points must be an array of shape (N, {dim}) instead got {col_points.shape = }.")

    # --- Function body -------------------------------------------------------
    if dim != 1:
        # compute distance matrix
        dist_matrix = np.sum((row_points[:, np.newaxis, :] - col_points[np.newaxis, :, :])**2, axis=-1)**0.5
    else:
        # compute distance matrix
        dist_matrix = abs(row_points - col_points.T)
    # here phi_l = delta_l(\|.\|/delta_l)
    return csc_array(w11(dist_matrix / delta) / delta**dim)  # RBF


def w31_kernel_matrix(row_points, col_points, delta, dim):
    """
    Compute the kernel matrix using the wendland function w31. The (i,j) entry of the matrix corresponds to
    w31(norm2(row_points[i]-col_points[j])/delta)/delta**dim.

    Parameters:
        row_points (np.ndarray):
                Array of points associated with rows of the kernel matrix with shape (N, dim).
        col_points (np.ndarray):
                Array of points associated with columns of the kernel matrix with shape (M, dim).
        delta (float):
                Scale of the kernel function.
        dim (int):
                Dimension of point sets (both row and col).

     Returns:
        kernel matrix (scipy.sparse.csc_array):
                Kernel matrix with shape (N,M).
    """
    # --- Input validation ----------------------------------------------------
    if not isinstance(dim, int) or dim < 1:
        raise ValueError(f"dim must be an integer greater than 0! instead got {dim}.")

    if not isinstance(delta, (int, float)) or 0 >= delta:
        raise ValueError(f"delta must be a positive number! instead got {delta}.")

    if not isinstance(row_points, np.ndarray) or row_points.ndim > 2:
        raise ValueError(f"row_points must be an array of dimension at most 2, instead got {row_points.ndim = }.")
    if row_points.ndim == 1:
        row_points = row_points.reshape((-1, 1))
    elif row_points.shape[-1] != dim:
        raise ValueError(f"row_points must be an array of shape (N, {dim}) instead got {row_points.shape = }.")

    if not isinstance(col_points, np.ndarray) or col_points.ndim > 2:
        raise ValueError(f"row_points must be an array of dimension at most 2, instead got {col_points.ndim = }.")
    if col_points.ndim == 1:
        col_points = col_points.reshape((-1, 1))
    elif col_points.shape[-1] != dim:
        raise ValueError(f"row_points must be an array of shape (N, {dim}) instead got {col_points.shape = }.")

    # --- Function body -------------------------------------------------------
    if dim != 1:
        # compute distance matrix
        dist_matrix = np.sum((row_points[:, np.newaxis, :] - col_points[np.newaxis, :, :])**2, axis=-1)**0.5
    else:
        # compute distance matrix
        dist_matrix = abs(row_points - col_points.T)
    # here phi_l = delta_l(\|.\|/delta_l)
    return csc_array(w31(dist_matrix / delta) / delta ** dim)  # RBF

# interpolation functions ########################################################################
def multiscale_interp_adapt1d(nested_size, nested_set, target_rhs, mu, nu, k, eval_set, eval_target, h1, rad_const,
                              eps_0=1e-5):
    """
    Adapt algorithm from paper.

    Parameters:
        nested_size (np.ndarray of ints):
                List of increasing positive integers. Each entry should be associated with a nested subset of nested_set.
                Indeed, nested_set[:nested_size[i]] should have fill distance h1*mu**i.
        nested_set (np.ndarray of floats):
                Array of nested points. It is assumed that the array stores the finest available set and
                through a nested_size it is possible to take windows of the array to recover a
                nested level. The array is assumed to have dimension 1 and shape (N,) where N is the
                number of points.
        target_rhs (np.ndarray of floats):
                Array of values associated with the nested set with shape (N,).
        mu (float):
                Parameter associated to the nested_set. Must be positive and smaller than 1/k.
        nu (float):
                Parameter related to support of the kernel function. Must be positive.
        k (float):
                Parameter related to the nested set construction. Must be greater than 1.
        eval_set (np.ndarray of floats):
                Set of points where the approximation is evaluated and displayed. The array must have dimension 1
                and shape (M,).
        eval_target (np.ndarray of floats):
                Array of values associated with the evaluation points with shape (M,).
        h1 (float):
                Fill distance of the first set, i.e. nested_set[:nested_size[0]]. Must be positive.
        rad_const (float):
                Parameter involved in the selection process. Must be positive.
        eps_0 (float):
                Starting thresholding parameter. Must be non-negative. Default value: 1e-5.
    """
    # note: target eval is not necessary for the approximation itself, just to have error plots.
    # --- Input validation ----------------------------------------------------
    if not isinstance(nested_size, np.ndarray) or nested_size.dtype != int or not np.array_equal(nested_size,
                                                                                                 np.sort(nested_size)):
        raise ValueError(f"nested_size must be an array of increasing positive integers! "
                         f"instead got {nested_size.tolist() = } with {nested_size.dtype = }.")
    else:
        number_of_levels = len(nested_size)

    if not isinstance(nested_set, np.ndarray) or nested_set.ndim != 1 or nested_size[-1] != len(nested_set):
        raise ValueError(f"nested_set must be a ndarray with ndim 1 and length {nested_size[-1]}! "
                         f"instead got {nested_set.ndim = } and {len(nested_set) = }.")
    else:
        N = nested_size[-1]

    if not isinstance(target_rhs, np.ndarray) or target_rhs.shape != (N,):
        raise ValueError(f"target_rhs must be a ndarray with shape ({N},)! instead got {target_rhs.shape = }.")

    if not isinstance(k, (int, float)) or k <= 1:
        raise ValueError(f"k must be a number greater than 1, got {k = } instead.")

    if not isinstance(mu, (int, float)) or not 0 < mu < 1.0 / k:
        raise ValueError(f"mu must be in (0, 1/k), i.e, (0, {1 / k}), got {mu = } instead.")

    if not isinstance(nu, (int, float)) or 0 >= nu:
        raise ValueError(f"nu must be a positive number, got {nu = } instead.")

    if not isinstance(eval_set, np.ndarray) or eval_set.ndim != 1:
        raise ValueError(f"eval_set must be a ndarray with ndim 1!"
                         f" instead got {eval_set.ndim = }.")
    else:
        M = len(eval_set)
    if not isinstance(eval_target, np.ndarray) or eval_target.shape != (M,):
        raise ValueError(f"eval_target must be a ndarray with shape ({M},)! instead got {eval_target.shape = }.")

    if not isinstance(h1, (int, float)) or 0 >= h1:
        raise ValueError(f"h1 must be a positive number, got {h1 = } instead.")

    if not isinstance(rad_const, (int, float)) or 0 >= rad_const:
        raise ValueError(f"rad_const must be a positive number, got {rad_const = } instead.")

    if not isinstance(eps_0, (int, float)) or 0 > eps_0:
        raise ValueError(f"eps_0 must be a non-negative number, got {eps_0 = } instead.")

    L_star = np.ceil(k * mu / (1 - k * mu))
    print(f"Running 1d-multiscale interpolation with: {number_of_levels} levels, {L_star = }, "
          f"{mu = }, {nu = }, {k = }, {h1 = }, {rad_const = } and {eps_0 = }.\n"
          f"Displayed results computed over {M} equispaced points.\n")

    # --- Function body -------------------------------------------------------
    # setup lists for the evaluations
    approx_list = [np.zeros(M)]
    error_list = [eval_target]
    # setup of variables for routine
    removed_mask = np.zeros(N, dtype=np.bool)  # cut mask initialization
    local_set_cardinality_ratio = np.zeros(number_of_levels)  # store the percentage of used points at every level
    threshold = eps_0  # define starting threshold
    tmp_thresholding_list = np.zeros(number_of_levels)
    # iterate over the levels of the approximation
    for current_level in range(number_of_levels):
        # set up the cut_mask (for l < L_star we do not want to count already removed points)
        N_l = nested_size[current_level]
        if current_level < L_star:
            removed_mask = np.zeros(N, dtype=np.bool)
        # select the points to reduce computational cost
        removed_mask, local_mask, N_check = point_selection_1d(
            nested_set,
            N_l,
            target_rhs,
            removed_mask,
            threshold,
            k * rad_const * mu ** current_level * h1 * np.abs(current_level * np.log(mu) + np.log(h1))
        )
        # store the percentage of used point at the current level
        local_set_cardinality_ratio[current_level] = N_check / N_l
        # update rhs and point set
        selected_multiset = nested_set[:N_l][local_mask]
        # compute the scale and the kernel matrix
        delta = nu * h1 * mu ** current_level
        kernel_matrix = w11_kernel_matrix(nested_set[:N_l], nested_set[:N_l], delta, 1)
        # SVD decomposition
        U, S, Vt = svds(A=kernel_matrix, k=N_l, solver="propack")
        inv_kernel_matrix = Vt.T @ np.diag(1/S) @ U.T

        # update rhs on the finest grid, computing the cardinals
        # since the set are nested, we do not need to compute the cardinals on points contained at this level.
        # the local approximation on previous iteration points is equal to the rhs, due to the cardinal proprieties
        local_approx = np.copy(target_rhs)
        chi = inv_kernel_matrix @ w11_kernel_matrix(nested_set[:N_l], nested_set[N_l:], delta, 1)
        local_approx[N_l:] = target_rhs[:N_l][local_mask]@chi[local_mask, :]

        # update the threshold
        eps_mask = np.zeros(N, dtype=bool)
        for i in np.arange(N)[removed_mask]:
            eps_mask[~eps_mask] = np.abs(nested_set[~eps_mask] - nested_set[i]) < \
                                  rad_const * mu ** current_level * h1 * np.abs(current_level * np.log(mu) + np.log(h1))
        threshold += np.max(local_approx[eps_mask], initial=0)
        tmp_thresholding_list[current_level] = threshold
        # update evaluation of the solution and error
        # copy previous values
        approx_list += [approx_list[-1]]
        error_list += [error_list[-1]]
        # computing cardinals over evaluation point and update approximation and error
        chi = inv_kernel_matrix @ w11_kernel_matrix(nested_set[:N_l], eval_set, delta, 1)
        approx_list[-1] += target_rhs[:N_l][local_mask]@chi[local_mask, :]
        error_list[-1] -= target_rhs[:N_l][local_mask]@chi[local_mask, :]

        # update rhs
        target_rhs -= local_approx

        # plotting the solution
        approx_fig = plt.figure(num=current_level)
        approx_ax = approx_fig.add_subplot(1, 1, 1)
        approx_ax.scatter(nested_set[:N_l], np.zeros(N_l), color="red")
        approx_ax.scatter(selected_multiset, np.zeros(N_check), color="blue", label="selected points")
        approx_ax.plot(eval_points, error_list[-2], color="black")
        approx_ax.set_xlim(np.min(eval_set) - 0.1, np.max(eval_set) + 0.1)
        approx_ax.set_title(f"approximation error at {current_level = }", fontsize="small")
        plt.show()
        if current_level in []:
            write_array_on_file(selected_multiset, f"adapt_set_{current_level}.csv")
            write_matrix_on_file(
                np.concatenate([eval_points.reshape((-1, 1)), error_list[-2].reshape((-1, 1))], axis=-1),
                f"adapt_error_{current_level}.csv")
        if not N_check:
            break
    point_ratio_fig = plt.figure(num=number_of_levels)
    point_ratio_ax = point_ratio_fig.add_subplot(1, 1, 1)
    point_ratio_ax.plot(np.arange(number_of_levels), local_set_cardinality_ratio, label="#tilda_X over #X")
    plt.title("ratio of spared points w.r.t. level")
    plt.show()
    write_array_on_file(local_set_cardinality_ratio, f"point_ratio_{mu}.csv")
    write_array_on_file(tmp_thresholding_list, f"thresholding_{mu}.csv")

def multiscale_interp_adapt2d(nested_size, nested_set, target_rhs, mu, nu, k, eval_set, eval_target, eval_grid_shape,
                              h1, rad_const, eps_0=1e-5):
    """
    Adapt algorithm from paper.

    Parameters:
        nested_size (np.ndarray of ints):
                List of increasing positive integers. Each entry should be associated with a nested subset of nested_set.
                Indeed, nested_set[:nested_size[i]] should have fill distance h1*mu**i.
        nested_set (np.ndarray of floats):
                Array of nested points. It is assumed that the array stores the finest available set and
                through a nested_size it is possible to take windows of the array to recover a
                nested level. The array is assumed to have dimension 2 and shape (N,dim) where N is the
                number of points, and dim is their dimension.
        target_rhs (np.ndarray of floats):
                Array of values associated with the nested set with shape (N,).
        mu (float):
                Parameter associated to the nested_set. Must be positive and smaller than 1/k.
        nu (float):
                Parameter related to support of the kernel function. Must be positive.
        k (float):
                Parameter related to the nested set construction. Must be greater than 1.
        eval_set (np.ndarray of floats):
                Set of points where the approximation is evaluated and displayed. The array must have dimension 2
                and shape (M,dim).
        eval_target (np.ndarray of floats):
                Array of values associated with the evaluation points with shape (M,).
        eval_grid_shape (tuple):
                Tuple with the proper shape of the evaluation points to be displayed with 3d matplotlib.
                Most of the time this is (sqrt(M),sqrt(M)).
        h1 (float):
                Fill distance of the first set, i.e. nested_set[:nested_size[0]]. Must be positive.
        rad_const (float):
                Parameter involved in the selection process. Must be positive.
        eps_0 (float):
                Starting thresholding parameter. Must be non-negative. Default value: 1e-5.
    """
    # note: target eval is not necessary for the approximation itself, just to have error plots.
    # --- Input validation ----------------------------------------------------
    if not isinstance(nested_size, np.ndarray) or nested_size.dtype != int or not np.array_equal(nested_size,
                                                                                                 np.sort(nested_size)):
        raise ValueError(f"nested_size must be an array of increasing positive integers! "
                         f"instead got {nested_size.tolist() = } with {nested_size.dtype = }.")
    else:
        number_of_levels = len(nested_size)

    if not isinstance(nested_set, np.ndarray) or nested_set.ndim != 2 or nested_size[-1] != len(nested_set):
        raise ValueError(f"nested_set must be a ndarray with ndim 2 and length matching nested_size "
                         f"({nested_size[-1]})! instead got {nested_set.ndim = } and {len(nested_set) = }.")
    else:
        N, dim = nested_set.shape

    if not isinstance(target_rhs, np.ndarray) or target_rhs.shape != (N,):
        raise ValueError(f"target_rhs must be a ndarray with shape ({N},)! instead got {target_rhs.shape = }.")

    if not isinstance(k, (int, float)) or k <= 1:
        raise ValueError(f"k must be a number greater than 1, got {k = } instead.")

    if not isinstance(mu, (int, float)) or not 0 < mu < 1.0 / k:
        raise ValueError(f"mu must be in (0, 1/k), i.e, (0, {1 / k}), got {mu = } instead.")

    if not isinstance(nu, (int, float)) or 0 >= nu:
        raise ValueError(f"nu must be a positive number, got {nu = } instead.")

    if not isinstance(eval_set, np.ndarray) or eval_set.ndim != 2 or eval_set.shape[-1] != dim:
        raise ValueError(f"eval_set must be a ndarray with ndim 2 and shape (M, {dim}) with positive M!"
                         f" instead got {eval_set.shape = }.")
    else:
        M = len(eval_set)
    if not isinstance(eval_target, np.ndarray) or eval_target.shape != (M,):
        raise ValueError(f"eval_target must be a ndarray with shape ({M},)! instead got {eval_target.shape = }.")

    if not isinstance(h1, (int, float)) or 0 >= h1:
        raise ValueError(f"h1 must be a positive number, got {h1 = } instead.")

    if not isinstance(rad_const, (int, float)) or 0 >= rad_const:
        raise ValueError(f"rad_const must be a positive number, got {rad_const = } instead.")

    if not isinstance(eps_0, (int, float)) or 0 > eps_0:
        raise ValueError(f"eps_0 must be a non-negative number, got {eps_0 = } instead.")

    L_star = np.ceil(k * mu / (1 - k * mu))
    print(f"Running 1d-multiscale interpolation with: {number_of_levels} levels, {L_star = }, "
          f"{mu = }, {nu = }, {k = }, {h1 = }, {rad_const = } and {eps_0 = }.\n"
          f"Displayed results computed over {M} equispaced points.\n")

    # --- Function body -------------------------------------------------------
    # setup lists for the evaluations
    approx_list = [np.zeros(M)]
    error_list = [eval_target]
    # setup of variables for routine
    removed_mask = np.zeros(N, dtype=np.bool)  # cut mask initialization
    local_set_cardinality_ratio = np.zeros(number_of_levels)  # store the percentage of used points at every level
    threshold = eps_0  # define starting threshold
    # iterate over the levels of the approximation
    for current_level in range(number_of_levels):
        # set up the cut_mask (for l < L_star we do not want to count already removed points)
        N_l = nested_size[current_level]
        if current_level < L_star:
            removed_mask = np.zeros(N, dtype=np.bool)
        # select the points to reduce computational cost
        removed_mask, local_mask, N_check = point_array_selection(
            nested_set,
            N_l,
            target_rhs,
            removed_mask,
            threshold,
            k * rad_const * mu ** current_level * h1 * np.abs(current_level * np.log(mu) + np.log(h1))
        )
        # store the percentage of used point at the current level
        local_set_cardinality_ratio[current_level] = N_check / N_l
        # update rhs and point set
        selected_multiset = nested_set[:N_l][local_mask]
        # compute the scale and the kernel matrix
        delta = nu * h1 * mu ** current_level
        kernel_matrix = w31_kernel_matrix(nested_set[:N_l], nested_set[:N_l], delta, dim)
        # SVD decomposition
        U, S, Vt = svds(A=kernel_matrix, k=N_l, solver="propack")
        inv_kernel_matrix = Vt.T @ np.diag(1/S) @ U.T

        # update rhs on the finest grid, computing the cardinals
        # since the set are nested, we do not need to compute the cardinals on points contained at this level.
        # the local approximation on previous iteration points is equal to the rhs, due to the cardinal proprieties
        local_approx = np.copy(target_rhs)
        chi = inv_kernel_matrix @ w31_kernel_matrix(nested_set[:N_l], nested_set[N_l:], delta, dim)
        local_approx[N_l:] = target_rhs[:N_l][local_mask]@chi[local_mask, :]
        # update the threshold
        eps_mask = np.zeros(N, dtype=bool)
        for i in np.arange(N)[removed_mask]:
            eps_mask[~eps_mask] = np.sqrt(np.sum(((nested_set[~eps_mask] - nested_set[i]) ** 2), axis=-1)) < \
                                  rad_const * mu ** current_level * h1 * np.abs(current_level * np.log(mu) + np.log(h1))
        threshold += np.max(local_approx[eps_mask], initial=0)
        # update evaluation of the solution and error
        # copy previous values
        approx_list += [approx_list[-1]]
        error_list += [error_list[-1]]
        # computing cardinals over evaluation point and update approximation and error
        chi = inv_kernel_matrix @ w31_kernel_matrix(nested_set[:N_l], eval_set, delta, dim)
        approx_list[-1] += target_rhs[:N_l][local_mask]@chi[local_mask, :]
        error_list[-1] -= target_rhs[:N_l][local_mask]@chi[local_mask, :]
        # update rhs
        target_rhs -= local_approx

        # plotting the solution
        approx_fig = plt.figure(num=current_level)
        approx_ax = approx_fig.add_subplot(1, 2, 1)
        approx_ax.scatter(nested_set[:N_l][:, 0], nested_set[:N_l][:, 1], color="red")
        approx_ax.scatter(selected_multiset[:, 0], selected_multiset[:, 1], color="blue")
        approx_ax.set_aspect('equal')
        approx_ax.set_xlim(np.min(eval_set) - 0.1, np.max(eval_set) + 0.1)
        approx_ax.set_ylim(np.min(eval_set) - 0.1, np.max(eval_set) + 0.1)
        approx_ax2 = approx_fig.add_subplot(1, 2, 2, projection='3d')
        approx_ax2.plot_surface(eval_set[:, 0].reshape(eval_grid_shape),
                         eval_set[:, 1].reshape(eval_grid_shape),
                         error_list[-2].reshape(eval_grid_shape),
                         linewidth=0,
                         antialiased=False)
        approx_ax2.set_title(f"approximation error at {current_level = }", fontsize="small")
        plt.show()
        if current_level in [1,3]:
            write_array_on_file(selected_multiset, f"adapt_2d_set_{current_level}.csv")
            write_matrix_on_file(
                np.concatenate([eval_points.reshape((-1, 2)), error_list[-2].reshape((-1, 1))], axis=-1),
                f"adapt_2d_error_{current_level}.csv")
        if not N_check:
            break
    point_ratio_fig = plt.figure(num=number_of_levels)
    point_ratio_ax = point_ratio_fig.add_subplot(1, 1, 1)
    point_ratio_ax.plot(np.arange(number_of_levels), local_set_cardinality_ratio, label="#tilda_X over #X")
    plt.title("ratio of spared points w.r.t. level")
    plt.show()


def multiscale_interp_mix1d(nested_size, nested_set, target_rhs, mu, nu, k, eval_set, eval_target, h1, rad_const,
                            adapt_start=-1, eps_0=1e-5):
    """
    Mixed-adapt algorithm from paper.

    Parameters:
        nested_size (np.ndarray of ints):
                List of increasing positive integers. Each entry should be associated with a nested subset of nested_set.
                Indeed, nested_set[:nested_size[i]] should have fill distance h1*mu**i.
        nested_set (np.ndarray of floats):
                Array of nested points. It is assumed that the array stores the finest available set and
                through a nested_size it is possible to take windows of the array to recover a
                nested level. The array is assumed to have dimension 1 and shape (N,) where N is the
                number of points.
        target_rhs (np.ndarray of floats):
                Array of values associated with the nested set with shape (N,).
        mu (float):
                Parameter associated to the nested_set. Must be positive and smaller than 1/k.
        nu (float):
                Parameter related to support of the kernel function. Must be positive.
        k (float):
                Parameter related to the nested set construction. Must be greater than 1.
        eval_set (np.ndarray of floats):
                Set of points where the approximation is evaluated and displayed. The array must have dimension 1
                and shape (M,).
        eval_target (np.ndarray of floats):
                Array of values associated with the evaluation points with shape (M,).
        h1 (float):
                Fill distance of the first set, i.e. nested_set[:nested_size[0]]. Must be positive.
        rad_const (float):
                Parameter involved in the selection process. Must be positive.
        eps_0 (float):
                Starting thresholding parameter. Must be non-negative. Default value: 1e-5.
        adapt_start (int):
                Starting level for the adaptive routine
    """
    # note: target eval is not necessary for the approximation itself, just to have error plots.
    # --- Input validation ----------------------------------------------------
    if not isinstance(nested_size, np.ndarray) or nested_size.dtype != int or not np.array_equal(nested_size,
                                                                                                 np.sort(nested_size)):
        raise ValueError(f"nested_size must be an array of increasing positive integers! "
                         f"instead got {nested_size.tolist() = } with {nested_size.dtype = }.")
    else:
        number_of_levels = len(nested_size)

    if not isinstance(nested_set, np.ndarray) or nested_set.ndim != 1 or nested_size[-1] != len(nested_set):
        raise ValueError(f"nested_set must be a ndarray with ndim 1 and length {nested_size[-1]}! "
                         f"instead got {nested_set.ndim = } and {len(nested_set) = }.")
    else:
        N = nested_size[-1]

    if not isinstance(target_rhs, np.ndarray) or target_rhs.shape != (N,):
        raise ValueError(f"target_rhs must be a ndarray with shape ({N},)! instead got {target_rhs.shape = }.")

    if not isinstance(k, (int, float)) or k <= 1:
        raise ValueError(f"k must be a number greater than 1, got {k = } instead.")

    if not isinstance(mu, (int, float)) or not 0 < mu < 1.0 / k:
        raise ValueError(f"mu must be in (0, 1/k), i.e, (0, {1 / k}), got {mu = } instead.")

    if not isinstance(nu, (int, float)) or 0 >= nu:
        raise ValueError(f"nu must be a positive number, got {nu = } instead.")

    if not isinstance(eval_set, np.ndarray) or eval_set.ndim != 1:
        raise ValueError(f"eval_set must be a ndarray with ndim 1!"
                         f" instead got {eval_set.ndim = }.")
    else:
        M = len(eval_set)
    if not isinstance(eval_target, np.ndarray) or eval_target.shape != (M,):
        raise ValueError(f"eval_target must be a ndarray with shape ({M},)! instead got {eval_target.shape = }.")

    if not isinstance(h1, (int, float)) or 0 >= h1:
        raise ValueError(f"h1 must be a positive number, got {h1 = } instead.")

    if not isinstance(rad_const, (int, float)) or 0 >= rad_const:
        raise ValueError(f"rad_const must be a positive number, got {rad_const = } instead.")

    if not isinstance(eps_0, (int, float)) or 0 > eps_0:
        raise ValueError(f"eps_0 must be a non-negative number, got {eps_0 = } instead.")

    if not isinstance(adapt_start, int):
        raise ValueError(f"adapt_start must be an integer, got {adapt_start = } instead.")

    L_star = np.ceil(k * mu / (1 - k * mu))
    if adapt_start < 0:
        adapt_start = L_star
    print(f"Running 1d-multiscale interpolation with: {number_of_levels} levels, {L_star = }, "
          f"{mu = }, {nu = }, {k = }, {h1 = }, {rad_const = }, {eps_0 = } and {adapt_start = }.\n"
          f"Displayed results computed over {M} equispaced points.\n")

    # --- Function body -------------------------------------------------------
    # setup lists for the evaluations
    approx_list = [np.zeros(M)]
    error_list = [eval_target]
    # setup of variables for routine
    removed_mask = np.zeros(N, dtype=np.bool)  # cut mask initialization
    local_set_cardinality_ratio = np.zeros(number_of_levels)  # store the percentage of used points at every level
    threshold = eps_0  # define starting threshold
    tmp_thresholding_list = np.zeros(number_of_levels)
    # iterate over the levels of the approximation
    for current_level in range(number_of_levels):
        # just for L >= adapt_start we apply the adaptivity steps
        N_l = nested_size[current_level]
        if current_level >= adapt_start:
            # select the points to reduce computational cost
            removed_mask, local_mask, N_check = point_selection_1d(
                nested_set,
                N_l,
                target_rhs,
                removed_mask,
                threshold,
                k * rad_const * mu ** current_level * h1 * np.abs(current_level * np.log(mu) + np.log(h1))
            )
            local_set_cardinality_ratio[current_level] = N_check / N_l
        else:  # no adaptivity, take all points
            N_check = N_l
            local_mask = np.ones(N_l, dtype=np.bool)
            local_set_cardinality_ratio[current_level] = 1
        # update rhs and point set
        selected_multiset = nested_set[:N_l][local_mask]
        # compute the scale and the kernel matrix
        delta = nu * h1 * mu ** current_level
        kernel_matrix = w11_kernel_matrix(nested_set[:N_l], nested_set[:N_l], delta, 1)
        # SVD decomposition
        U, S, Vt = svds(A=kernel_matrix, k=N_l, solver="propack")
        inv_kernel_matrix = Vt.T @ np.diag(1/S) @ U.T

        # update rhs on the finest grid, computing the cardinals
        # since the set are nested, we do not need to compute the cardinals on points contained at this level.
        # the local approximation on previous iteration points is equal to the rhs, due to the cardinal proprieties
        local_approx = np.copy(target_rhs)
        chi = inv_kernel_matrix @ w11_kernel_matrix(nested_set[:N_l], nested_set[N_l:], delta, 1)
        local_approx[N_l:] = target_rhs[:N_l][local_mask]@chi[local_mask, :]

        # update the threshold
        if current_level >= adapt_start:
            eps_mask = np.zeros(N, dtype=bool)
            for i in np.arange(N)[removed_mask]:
                eps_mask[~eps_mask] = np.abs(nested_set[~eps_mask] - nested_set[i]) < \
                                      rad_const * mu ** current_level * h1 * np.abs(current_level * np.log(mu) + np.log(h1))
            threshold += np.max(local_approx[eps_mask], initial=0)
        tmp_thresholding_list[current_level] = threshold
        # update evaluation of the solution and error
        # copy previous values
        approx_list += [approx_list[-1]]
        error_list += [error_list[-1]]
        # computing cardinals over evaluation point and update approximation and error
        chi = inv_kernel_matrix @ w11_kernel_matrix(nested_set[:N_l], eval_set, delta, 1)
        approx_list[-1] += target_rhs[:N_l][local_mask]@chi[local_mask, :]
        error_list[-1] -= target_rhs[:N_l][local_mask]@chi[local_mask, :]

        # update rhs
        target_rhs -= local_approx

        # plotting the solution
        approx_fig = plt.figure(num=current_level)
        approx_ax = approx_fig.add_subplot(1, 1, 1)
        approx_ax.scatter(nested_set[:N_l], np.zeros(N_l), color="red")
        approx_ax.scatter(selected_multiset, np.zeros(N_check), color="blue", label="selected points")
        approx_ax.plot(eval_points, error_list[-2], color="black")
        approx_ax.set_xlim(np.min(eval_set) - 0.1, np.max(eval_set) + 0.1)
        approx_ax.set_title(f"approximation error at {current_level = }", fontsize="small")
        plt.show()
        if current_level in []:
            write_array_on_file(selected_multiset, f"mix_set_{current_level}.csv")
            write_matrix_on_file(
                np.concatenate([eval_points.reshape((-1, 1)), error_list[-2].reshape((-1, 1))], axis=-1),
                f"mix_error_{current_level}.csv", ".9f")
    point_ratio_fig = plt.figure(num=number_of_levels)
    point_ratio_ax = point_ratio_fig.add_subplot(1, 1, 1)
    point_ratio_ax.plot(np.arange(number_of_levels), local_set_cardinality_ratio, label="#tilda_X over #X")
    plt.title("ratio of spared points w.r.t. level")
    plt.show()
    write_array_on_file(local_set_cardinality_ratio, f"point_ratio_mix_{mu}.csv")
    write_array_on_file(tmp_thresholding_list, f"thresholding_mix_{mu}.csv")



def multiscale_interp_adapt1d_local(nested_size, nested_set, target_rhs, mu, nu, k, eval_set, eval_target, h1, rad_const,
                                    eps_0=1e-2, tolerance=1e-5):
    """
    Bonus algorithm. More computational friendly, require more theoretical assessment.

    Parameters:
        nested_size (np.ndarray of ints):
                List of increasing positive integers. Each entry should be associated with a nested subset of nested_set.
                Indeed, nested_set[:nested_size[i]] should have fill distance h1*mu**i.
        nested_set (np.ndarray of floats):
                Array of nested points. It is assumed that the array stores the finest available set and
                through a nested_size it is possible to take windows of the array to recover a
                nested level. The array is assumed to have dimension 1 and shape (N,) where N is the
                number of points.
        target_rhs (np.ndarray of floats):
                Array of values associated with the nested set with shape (N,).
        mu (float):
                Parameter associated to the nested_set. Must be positive and smaller than 1/k.
        nu (float):
                Parameter related to support of the kernel function. Must be positive.
        k (float):
                Parameter related to the nested set construction. Must be greater than 1.
        eval_set (np.ndarray of floats)
                Set of points where the approximation is evaluated and displayed. The array must have dimension 1
                and shape (M,).
        eval_target (np.ndarray of floats)
                Array of values associated with the evaluation points with shape (M,).
        h1 (float):
                Fill distance of the first set, i.e. nested_set[:nested_size[0]]. Must be positive.
        rad_const (float):
                Parameter involved in the selection process. Must be positive.
        eps_0 (float):
                Starting thresholding parameter. Must be non-negative. Default value: 1e-2.
        tolerance (float):
                Relative tolerance in the conjugate gradient. Must be non-negative. Default value: 1e-5.
    """
    # note: target eval is not necessary for the approximation itself, just to have error plots.
    # --- Input validation ----------------------------------------------------
    if not isinstance(nested_size, np.ndarray) or nested_size.dtype != int or not np.array_equal(nested_size,
                                                                                                 np.sort(nested_size)):
        raise ValueError(f"nested_size must be an array of increasing positive integers! "
                         f"instead got {nested_size.tolist() = } with {nested_size.dtype = }.")
    else:
        number_of_levels = len(nested_size)

    if not isinstance(nested_set, np.ndarray) or nested_set.ndim != 1 or nested_size[-1] != len(nested_set):
        raise ValueError(f"nested_set must be a ndarray with ndim 1 and length {nested_size[-1]}! "
                         f"instead got {nested_set.ndim = } and {len(nested_set) = }.")
    else:
        N = nested_size[-1]

    if not isinstance(target_rhs, np.ndarray) or target_rhs.shape != (N,):
        raise ValueError(f"target_rhs must be a ndarray with shape ({N},)! instead got {target_rhs.shape = }.")

    if not isinstance(k, (int, float)) or k <= 1:
        raise ValueError(f"k must be a number greater than 1, got {k = } instead.")

    if not isinstance(mu, (int, float)) or not 0 < mu < 1.0 / k:
        raise ValueError(f"mu must be in (0, 1/k), i.e, (0, {1 / k}), got {mu = } instead.")

    if not isinstance(nu, (int, float)) or 0 >= nu:
        raise ValueError(f"nu must be a positive number, got {nu = } instead.")

    if not isinstance(eval_set, np.ndarray) or eval_set.ndim != 1:
        raise ValueError(f"eval_set must be a ndarray with ndim 1!"
                         f" instead got {eval_set.ndim = }.")
    else:
        M = len(eval_set)
    if not isinstance(eval_target, np.ndarray) or eval_target.shape != (M,):
        raise ValueError(f"eval_target must be a ndarray with shape ({M},)! instead got {eval_target.shape = }.")

    if not isinstance(h1, (int, float)) or 0 >= h1:
        raise ValueError(f"h1 must be a positive number, got {h1 = } instead.")

    if not isinstance(rad_const, (int, float)) or 0 >= rad_const:
        raise ValueError(f"rad_const must be a positive number, got {rad_const = } instead.")

    if not isinstance(eps_0, (int, float)) or 0 > eps_0:
        raise ValueError(f"eps_0 must be a non-negative number, got {eps_0 = } instead.")

    if not isinstance(tolerance, (int, float)) or 0 > tolerance:
        raise ValueError(f"tolerance must be a non-negative number, got {tolerance = } instead.")

    L_star = np.ceil(k * mu / (1 - k * mu))
    print(f"Running 1d-multiscale interpolation with: {number_of_levels} levels, {L_star = }, "
          f"{mu = }, {nu = }, {k = }, {h1 = }, {rad_const = }, {eps_0 = } and {tolerance = }.\n"
          f"Displayed results computed over {M} equispaced points.\n")

    # --- Function body -------------------------------------------------------
    # setup lists for the evaluations
    approx_list = [np.zeros(M)]
    error_list = [eval_target]
    # setup of variables for routine
    removed_mask = np.zeros(N, dtype=bool)  # cut mask initialization
    local_set_cardinality_ratio = np.zeros(number_of_levels)  # store the percentage of used points at every level
    threshold = eps_0  # define starting threshold
    # iterate over the levels of the approximation
    for current_level in range(number_of_levels):
        # set up the cut_mask (for l < L_star we do not want to count already removed points)
        if current_level < L_star:
            removed_mask = np.zeros(nested_size[-1], dtype=bool)
        # select the points to reduce computational cost
        removed_mask, local_mask, N_check = point_selection_1d(
            nested_set,
            nested_size[current_level],
            target_rhs,
            removed_mask,
            threshold,
            k * rad_const * mu ** current_level * h1 * np.abs(current_level * np.log(mu) + np.log(h1))
        )
        # store the percentage of used point at the current level
        local_set_cardinality_ratio[current_level] = N_check / nested_size[current_level]
        # update rhs and point set
        selected_multiset = nested_set[:nested_size[current_level]][local_mask]
        # compute the kernel matrix
        delta = nu * h1 * mu ** current_level
        dist_matrix = np.empty((N_check, N_check), dtype=np.float64)
        for column in range(N_check):
            dist_matrix[:, column] = np.abs(selected_multiset - selected_multiset[column])
        # here phi_l = delta_l(\|.\|/delta_l)
        kernel_matrix = w31(dist_matrix / delta) / delta  # RBF
        # solve the system
        alpha, _ = cg(kernel_matrix, target_rhs[:nested_size[current_level]][local_mask], rtol=tolerance)

        # update rhs on the finest grid
        # build kernel matrix
        dist_matrix = np.empty((N, N_check), dtype=np.float64)
        for column in range(N_check):
            dist_matrix[:, column] = np.abs(nested_set - selected_multiset[column])
        kernel_matrix = w31(dist_matrix / delta) / delta  # RBF
        # update the rhs
        local_approx = kernel_matrix @ alpha
        target_rhs -= local_approx

        # update the threshold
        eps_mask = np.zeros(N, dtype=bool)
        for i in np.arange(N)[removed_mask]:
            eps_mask[~eps_mask] = np.abs(nested_set[~eps_mask] - nested_set[i]) < \
                                  rad_const * mu ** current_level * h1 * np.abs(current_level * np.log(mu) + np.log(h1))
        threshold += np.max(local_approx[eps_mask], initial=0)
        # update evaluation of the solution and error
        dist_matrix = np.empty((M, N_check), dtype=np.float64)
        for column in range(N_check):
            dist_matrix[:, column] = np.abs(eval_set - selected_multiset[column])
        kernel_matrix = w31(dist_matrix / delta) / delta
        approx_list += [approx_list[-1] + kernel_matrix @ alpha]
        error_list += [error_list[-1] - kernel_matrix @ alpha]

        # plotting the solution
        approx_fig = plt.figure(num=current_level)
        approx_ax = approx_fig.add_subplot(1, 1, 1)
        approx_ax.scatter(nested_set[:nested_size[current_level]], np.zeros(nested_size[current_level]), color="red")
        approx_ax.scatter(selected_multiset, np.zeros(N_check), color="blue", label="selected points")
        approx_ax.plot(eval_points, error_list[-2], color="black")
        approx_ax.set_xlim(np.min(eval_set) - 0.1, np.max(eval_set) + 0.1)
        approx_ax.set_title(f"approximation error at {current_level = }", fontsize="small")
        plt.show()
        if current_level in [2, 7]:
            write_array_on_file(selected_multiset, f"adapt_set_{current_level}.csv")
            write_matrix_on_file(
                np.concatenate([eval_points.reshape((-1, 1)), error_list[-2].reshape((-1, 1))], axis=-1),
                f"adapt_error_{current_level}.csv")
        if not N_check:
            break
    point_ratio_fig = plt.figure(num=number_of_levels)
    point_ratio_ax = point_ratio_fig.add_subplot(1, 1, 1)
    point_ratio_ax.plot(np.arange(number_of_levels), local_set_cardinality_ratio, label="#tilda_X over #X")
    plt.title("ratio of spared points w.r.t. level")
    plt.show()


#######################
# experimental settings for the classic version
# define target function and compute approximation
top = 10
bottom = 0
# 9-7-6-5-4-3
L = 7
d = 1
k_ = 2
mu_ = 0.4
nu_ = 2
kappa = 2
supp_rad = 0.09
supp_cent = top / 2

Fr2 = lambda x: 0.75 * np.exp(-(9 * x[:, 0] - 2) ** 2 / 4 - (9 * x[:, 1] - 2) ** 2 / 4) + \
                0.75 * np.exp(-(9 * x[:, 0] + 1) ** 2 / 49 - (9 * x[:, 1] + 1) ** 2 / 49) + \
                0.5 * np.exp(-(9 * x[:, 0] - 7) ** 2 / 4 - (9 * x[:, 1] - 3) ** 2 / 4) - \
                0.2 * np.exp(-(9 * x[:, 0] - 4) ** 2 - (9 * x[:, 1] - 7) ** 2)
Fr1 = lambda x: 0.75 * np.exp(-(9 * x - 2) ** 2 / 4) + \
                0.75 * np.exp(-(9 * x + 1) ** 2 / 49) + \
                0.5 * np.exp(-(9 * x - 7) ** 2 / 4) - \
                0.2 * np.exp(-(9 * x - 4) ** 2)
C2 = lambda x: np.where(np.sqrt(np.sum((x - supp_cent) ** 2, axis=-1)) >= supp_rad, 0,
                        np.exp(1 / supp_rad - 1 / (supp_rad - np.sum((x - supp_cent) ** 2, axis=-1))))
C1 = lambda x: np.where(np.abs(x - supp_cent) ** 2 >= supp_rad, 0,
                        np.exp(1 / supp_rad - 1 / (supp_rad - np.abs(x - supp_cent) ** 2)))

# ---- 2d adaptive interpolation ------------------------- # k = 2, mu = 0.4 good +L=5, top=3
if d == 2:
    nest2d_set, nest2d_size = generate_ngrid_nested_sequence(dim=d, starting_step_size=mu_ / np.sqrt(d), mu=mu_,
                                                             levels=L, end=top)
    fig = plt.figure(num=21)
    ax = fig.add_subplot()
    for level in range(L - 1, -1, -1):
        ax.scatter(nest2d_set[:nest2d_size[level]][:, 0], nest2d_set[:nest2d_size[level]][:, 1], s=2 ** (2 * L - level))
    plt.show()

    eval_N = 30
    eval_meshed = np.meshgrid(*[np.linspace(bottom, top, eval_N) for _ in range(d)])
    eval_points = np.hstack([eval_meshed[i].reshape((-1, 1)) for i in range(d)])
    eval_func = C2(eval_points).reshape(eval_meshed[0].shape)
    fig = plt.figure(num=22)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(eval_meshed[0], eval_meshed[1], eval_func, linewidth=0, antialiased=False)
    ax.set_title("true function", fontsize="small")
    plt.show()

    multiscale_interp_adapt2d(nested_size=nest2d_size,
                              nested_set=nest2d_set,
                              target_rhs=C2(nest2d_set),
                              mu=mu_, nu=nu_, k=k_,
                              eval_set=eval_points,
                              eval_target=C2(eval_points),
                              eval_grid_shape=eval_meshed[0].shape,
                              h1=mu_ / np.sqrt(d),
                              rad_const=kappa,
                              eps_0=1e-8
                              )

# ---------- 1d setting --------------
if d == 1:
    for kappa in range(5, 7):
        nest1d_set, nest1d_size = generate_1d_nested_sequence(starting_step_size=mu_ / np.sqrt(d), mu=mu_, levels=L,
                                                              end=top, start=bottom)
        fig = plt.figure(num=11)
        ax = fig.add_subplot()
        for level in range(L - 1, -1, -1):
            plt.scatter(nest1d_set[:nest1d_size[level]], level * np.ones((nest1d_size[level])))
        plt.show()

        eval_N = 500
        eval_points = np.linspace(bottom, top, eval_N)
        fig = plt.figure(num=12)
        ax = fig.add_subplot()
        ax.plot(eval_points, C1(eval_points), linewidth=1, antialiased=False)
        ax.set_title("true function", fontsize="small")
        plt.show()

        multiscale_interp_adapt1d(nested_size=nest1d_size,
                                  nested_set=nest1d_set,
                                  target_rhs=C1(nest1d_set),
                                  mu=mu_, nu=nu_, k=k_,
                                  eval_set=eval_points,
                                  eval_target=C1(eval_points),
                                  h1=mu_ / np.sqrt(d),
                                  rad_const=kappa,
                                  eps_0=1e-8
                                  )

# ---------- 1d setting -------------- (mixed)
if d == 3:
    nest1d_set, nest1d_size = generate_1d_nested_sequence(starting_step_size=mu_ / np.sqrt(d), mu=mu_, levels=L,
                                                          end=top, start=bottom)
    fig = plt.figure(num=31)
    ax = fig.add_subplot()
    for level in range(L - 1, -1, -1):
        plt.scatter(nest1d_set[:nest1d_size[level]], level * np.ones((nest1d_size[level])))
    plt.show()

    eval_N = 500
    eval_points = np.linspace(bottom, top, eval_N)
    fig = plt.figure(num=32)
    ax = fig.add_subplot()
    ax.plot(eval_points, C1(eval_points), linewidth=1, antialiased=False)
    ax.set_title("true function", fontsize="small")
    plt.show()

    multiscale_interp_mix1d(nested_size=nest1d_size,
                            nested_set=nest1d_set,
                            target_rhs=C1(nest1d_set),
                            mu=mu_, nu=nu_, k=k_,
                            eval_set=eval_points,
                            eval_target=C1(eval_points),
                            h1=mu_ / np.sqrt(d),
                            rad_const=kappa,
                            #adapt_start=1,
                            eps_0=1e-8
                            )