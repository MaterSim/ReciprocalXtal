import torch
from torch import sin, cos, sqrt
import pandas as pd
from ast import literal_eval
import re
import os

class Symmetry:
    """
    A class to handle symmetry operations for space groups and Wyckoff positions.
    It parses symmetry operations from a CSV file and provides methods to apply
    these operations to atomic coordinates, retrieve free coordinates, and generate
    valid atomic structures based on symmetry operations.

    Attributes:
        device (torch.device): The device to run computations on (CPU or GPU).
        max_abc (float): Maximum value for lattice parameters.
        max_angle (float): Maximum value for angles in degrees.
        free_xyz_cache (dict): Mapping from (SPG, WP) to the free_xyz (e.g. [0, 1]).
        sym_strings (dict): Mapping (SPG) to the XYZ strings
        sym_matrices (torch.Tensor): Tensor containing symmetry operation matrices.
        sym_map (dict): Mapping from (SPG, WP) to (op_id, num_of_ops).
        num_wp (dict): Mapping from (SPG) to (num_of_wps).
    """
    def __init__(self, csv_file):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        df = pd.read_csv(csv_file)['0']
        self.max_abc = 35.0
        self.max_angle = 180.0
        self.free_xyz_cache = {}
        self.sym_map = {}  # Maps (spg, wp_id) -> (start_idx, num_ops)
        self.num_wp = {} # Maps (spg) -> num_of_wps
        self.sym_strings = {} # Maps (spg) -> list of Wyckoff position strings

        # Calculate total number of operations across all SPGs and WPs.
        total_ops = 0
        max_ops_per_wp = 0
        for spg in range(1, 231):
            wyc_strings = literal_eval(df[spg])
            self.sym_strings[spg] = wyc_strings
            self.free_xyz_cache[spg] = {}
            for wyc_id, wyc in enumerate(wyc_strings):
                num_ops = len(wyc)
                total_ops += num_ops
                max_ops_per_wp = max(max_ops_per_wp, num_ops)

        # Create mapping dictionary and corresponding tensors
        self.sym_matrices = torch.zeros((total_ops, 4, 4),
                                        dtype=torch.float64,
                                        device=self.device)

        # Fill tensors and map
        op_idx = 0
        for spg in range(1, 231):
            wyc_strings = self.sym_strings[spg]
            self.num_wp[spg] = len(wyc_strings)
            for wyc_id, wyc in enumerate(wyc_strings):
                num_ops = len(wyc)
                self.sym_map[(spg, wyc_id)] = (op_idx, num_ops)

                # Parse and store operations
                for i, op in enumerate(wyc):
                    self.sym_matrices[op_idx + i] = self.parse_operation(op)

                # Cache free coordinates ids
                if wyc_id == 0:
                    self.free_xyz_cache[spg][wyc_id] = [0, 1, 2]
                else:
                    free_xyz = []
                    rotation = self.sym_matrices[op_idx][:3, :3]
                    for j in range(3):
                        if rotation[j, j]**2 == 1: free_xyz.append(j)
                    self.free_xyz_cache[spg][wyc_id] = free_xyz

                # Update index
                op_idx += num_ops

    def parse_operation(self, operation):
        """
        Parses a symmetry operation string into a 4x4 transformation matrix.

        Args:
            operation (str): The symmetry operation string, e.g. "x,y,z+1/2".

        Returns:
            torch.Tensor: A 4x4 transformation matrix representing the operation.
        """
        rotation = torch.zeros((3, 3), dtype=torch.float64, device=self.device)
        translation = torch.zeros(3, dtype=torch.float64, device=self.device)
        components = operation.split(',')
        variables = ['x', 'y', 'z']

        for i, component in enumerate(components):
            for j, var in enumerate(variables):
                match = re.search(r'([+-]?)(\d*/?\d*)?' + var, component.strip())
                if match:
                    sign = -1 if match.group(1) == '-' else 1
                    coefficient = float(eval(match.group(2))) if match.group(2) else 1
                    rotation[i, j] = sign * coefficient

            match = re.search(r'([+-]?\d*/?\d*)$', component.strip())
            if match and match.group(1):
                translation[i] = float(eval(match.group(1)))

        affine_matrix = torch.eye(4, dtype=torch.float64, device=self.device)
        affine_matrix[:3, :3] = rotation
        affine_matrix[:3, 3] = translation
        return affine_matrix

    def apply_operation(self, xyz, operation):
        """
        Apply a symmetry operation to the xyz coordinates.

        Args:
            xyz (torch.Tensor): The input coordinates of shape (3,).
            operation (torch.Tensor): The symmetry operation matrix of shape (4, 4).

        Returns:
            torch.Tensor: The transformed coordinates of shape (3,).
        """
        # Ensure all tensors are on the same device
        operation = operation.to(xyz.device)
        rotation, translation = operation[:3, :3], operation[:3, 3]
        coord = torch.matmul(rotation, xyz) + translation
        coord -= torch.floor(coord)
        return coord

    def get_xyz_from_free_ids(self, spg, wp_id, free_xyz):
        """
        Get the xyz from free_xyz ids
        """
        spg = spg.item() if isinstance(spg, torch.Tensor) else spg

        if wp_id == 0:
            return free_xyz
        else:
            wp_id = wp_id.item() if isinstance(wp_id, torch.Tensor) else wp_id
            ids = self.get_free_xyz_ids(spg, wp_id)
            xyz = torch.zeros(3, dtype=torch.float64)
            xyz[ids] = free_xyz
            #print('debug get_xyz_from_free_ids1', free_xyz, xyz)

            op_idx = self.sym_map[(spg, wp_id)][0]
            operation = self.sym_matrices[op_idx:op_idx]
            xyz = self.apply_operation(xyz, operation)
            #print('debug get_xyz_from_free_ids2', xyz)
            return xyz

    def get_free_xyz_ids(self, spg, wp_id):
        """
        Get the free (x, y, z) for a Wyckoff position.

        Args:
            spg: space group number
            wp_id: Wyckoff position id

        Returns:
            List of indices (0,1,2) indicating which coordinates are free.
        """
        spg = spg.item() if isinstance(spg, torch.Tensor) else spg
        wp_id = wp_id.item() if isinstance(wp_id, torch.Tensor) else wp_id
        #print("debug get_free_xyz_ids", spg, wp_id)
        return self.free_xyz_cache[spg][wp_id]

    def get_coordinates(self, spg, wp_id, xyz, cell, tol=1e-2, check=True):
        """
        Get the valid xyz coordinates by applying the symmetry operations.

        Args:
            spg (int or torch.Tensor): Space group number.
            wp_id (int or torch.Tensor): Wyckoff position id.
            xyz (torch.Tensor): The input coordinates of shape (3,).
            cell (torch.Tensor): The cell matrix of shape (3, 3).
            tol (float): Tolerance for generator matching.
            check (bool): If True, checks the validity of generator.
        """
        spg_int = spg.item() if isinstance(spg, torch.Tensor) else spg
        wp_id_int = wp_id.item() if isinstance(wp_id, torch.Tensor) else wp_id

        xyz = xyz.to(self.device)  # Ensure xyz is on the correct device
        cell = cell.to(self.device)  # Ensure cell_matrix is on the correct device

        # Check if the input coordinates are valid
        if check:
            xyz1 = self.get_generator(spg_int, wp_id_int, xyz, cell, tol)
        else:
            xyz1 = xyz

        if xyz1 is not None:
            start_idx, num_ops = self.sym_map[(spg_int, wp_id_int)]
            ops = self.sym_matrices[start_idx:start_idx + num_ops]
            coordinates = self.apply_operations(xyz1, ops)
            return coordinates
        else:
            print("Warning: invalid coordinates", spg, wp_id, xyz)
            return None


    def get_coordinates_batch_from_mapping(self, generators, mapping):
        """
        Get the xyz coordinates by applying the symmetry operations
        according to the mapping relation (M1, 3).
        """
        pass

    def apply_operations(self, xyz, affine_matrices):
        """
        Apply the affine transformations to the xyz coordinates.

        Args:
            xyz (torch.Tensor): The input coordinates of shape (3,).
            affine_matrices (torch.Tensor): The matrices of shape (N, 4, 4).

        Returns:
            torch.Tensor: The transformed coordinates of shape (N, 3).
        """
        # Convert xyz to homogeneous coordinates according to the batch size
        xyz = xyz.to(affine_matrices.device)
        xyz1 = torch.cat([xyz, torch.ones(1, dtype=torch.float64,device=xyz.device)]).unsqueeze(0) #(1, 4)
        xyz1 = xyz1.expand(affine_matrices.size(0), -1)  # Shape: (N, 4)

        # Apply the affine transformations using batch matrix multiplication
        trans_xyz_homo = torch.bmm(affine_matrices, xyz1.unsqueeze(2)).squeeze(2)

        # Convert back to Cartesian coordinates and wrap within the range [0, 1)
        trans_xyz = trans_xyz_homo[:, :3] % 1.0

        return trans_xyz
     
    def get_generator(self, spg, wp_id, xyz, cell_matrix, tol=1e-2):
        """
        Get the generator for a special Wyckoff position.
        """
        spg = spg.item() if isinstance(spg, torch.Tensor) else spg
        wp_id = wp_id.item() if isinstance(wp_id, torch.Tensor) else wp_id

        if wp_id == 0:
            return xyz

        # Get general position operations (spg, 0)
        gen_start_idx, gen_num_ops = self.sym_map[(spg, 0)]
        ops_gen = self.sym_matrices[gen_start_idx:gen_start_idx + gen_num_ops]

        # Get special position operations (spg, wp_id)
        spec_start_idx, _ = self.sym_map[(spg, wp_id)]
        op_spec = self.sym_matrices[spec_start_idx]  # First operation

        cell_matrix = cell_matrix.to(xyz.device)
        for op in ops_gen:
            xyz1 = self.apply_operation(xyz, op)
            xyz0 = self.apply_operation(xyz1, op_spec)
            
            diff = xyz1 - xyz0
            diff -= torch.round(diff)
            diff = diff.to(cell_matrix.device)
            diff = diff @ cell_matrix
            if torch.linalg.vector_norm(diff) < tol: return xyz0
         
            
        #print("Warning: invalid generator", spg, wp_id, xyz)
        return None

    
    def get_batch_from_rows(self, rows, radian=True, normalize_in=False,
                            normalize_out=True, tol=1e-1, max_rep=30):
        """
        Converts a batch of rows into (spg_batch, wps_batch, rep_batch),
        ensuring batch logic follows the same sequence as the serial function.
        B is the number of rows in the batch.
        B1 is the number of Wyckoff positions in the batch.

        Args:
            rows (torch.Tensor): Shape (B, D), batch of data rows.
            radian (bool): If True, converts angles from radians to degrees.
            normalize_in (bool): If True, assuming the input rows are normalized
            normalize_out (bool): If True, normalizes the rep_cells to (0, 1)
            tol (float): Tolerance for generator matching.
            max_rep (int): Maximum length of the representation vector.

        Returns:
            tuple:
                - spg_batch: Shape (B,) (int)
                - wps_batch: Shape (B1, 2), (int)
                - rep_batch: Shape (B, max_rep_len), padded with -1 (normalized)
        """
        B = rows.shape[0]
        device = rows.device

        # 1. Extract space group from first column
        spg_batch = rows[:, 0].int()

        # Lists to hold 1D tensors of wps and rep values for each sample
        wps_list = []
        rep_list = []

        # 2. Process each row (ensuring exact order)
        for batch_id in range(B):
            spg = int(spg_batch[batch_id].item())  # Space group
            row = rows[batch_id].clone()  # ✅ Clone before modifying
            #print("debug get_batch_from_rows input", row)
            if radian: row[4:7] = torch.rad2deg(row[4:7])
            # Generate cell matrix
            cell_matrix = para2matrix(row[1:7], spg, normalize_in)
            #print("cell_matrix", cell_matrix)

            if not normalize_in and normalize_out:
                row[1:4] /= self.max_abc  # Normalize lattice parameters
                row[4:7] /= self.max_angle # Normalize angles

            # Select rep_ids based on space group, same as in serial version
            if spg <= 2:
                rep_ids = [1, 2, 3, 4, 5, 6]
            elif spg <= 15:
                rep_ids = [1, 2, 3, 5]
            elif spg <= 74:
                rep_ids = [1, 2, 3]
            elif spg <= 142:
                rep_ids = [1, 3]
            elif spg <= 194:
                rep_ids = [1, 3]
            else:
                rep_ids = [1]

            # Process Wyckoff positions
            for rep_id in range(7, len(row), 4):
                wp = int(row[rep_id])
                #print(f"debug get_batch_from_rows wp: {wp}")
                max_op = self.num_wp[spg]
                if wp >= 0 and wp < max_op:
                    ids = self.get_free_xyz_ids(spg, wp)
                    # reset xyz from the generator
                    xyz = row[rep_id+1: rep_id+4]
                    xyz = self.get_generator(spg, wp, xyz, cell_matrix, tol)

                    # Ignore the bad ids
                    if xyz is not None:
                        #print(f" debug get_batch_from_rows valid wp: {wp}, xyz: {xyz}")
                        wps_list.append([wp, batch_id])
                        row[rep_id+1: rep_id+4] = xyz
                        for id in ids:
                            rep_ids.append(id + rep_id + 1)
            rep_list.append(row[rep_ids])

        #print(wps_list)
        wps_batch = torch.tensor(wps_list, dtype=torch.int64, device=device)
        rep_batch = torch.full((B, max_rep), -1.0, dtype=torch.float64, device=device)
        for i, r in enumerate(rep_list):
            rep_batch[i, :r.size(0)] = r

        return spg_batch, wps_batch, rep_batch

    def get_batch_rep_ids_from_rows(self, rows, radian=True, normalize_in=False,
                            normalize_out=True, tol=1e-1, max_rep=30):
        """
        Converts a batch of rows into (spg_batch, wps_batch, rep_batch),
        ensuring batch logic follows the same sequence as the serial function.
        B is the number of rows in the batch.
        B1 is the number of Wyckoff positions in the batch.

        Args:
            rows (torch.Tensor): Shape (B, D), batch of data rows.
            radian (bool): If True, converts angles from radians to degrees.
            normalize_in (bool): If True, assuming the input rows are normalized
            normalize_out (bool): If True, normalizes the rep_cells to (0, 1)
            tol (float): Tolerance for generator matching.
            max_rep (int): Maximum length of the representation vector.

        Returns:
            tuple:
                - spg_batch: Shape (B,) (int)
                - wps_batch: Shape (B1, 2), (int)
                - rep_batch: Shape (B, max_rep_len), padded with -1 (normalized)
        """
        B = rows.shape[0]
        device = rows.device

        # 1. Extract space group from first column
        spg_batch = rows[:, 0].int()

        # Lists to hold 1D tensors of wps and rep values for each sample
        wps_list = []
        rep_ids_list = []

        # 2. Process each row (ensuring exact order)
        for batch_id in range(B):
            spg = int(spg_batch[batch_id].item())  # Space group
            row = rows[batch_id].clone()  # ✅ Clone before modifying
            #print("debug get_batch_from_rows input", row)
            if radian: row[4:7] = torch.rad2deg(row[4:7])
            # Generate cell matrix
            cell_matrix = para2matrix(row[1:7], spg, normalize_in)
            #print("cell_matrix", cell_matrix)

            if not normalize_in and normalize_out:
                row[1:4] /= self.max_abc  # Normalize lattice parameters
                row[4:7] /= self.max_angle # Normalize angles

            # Select rep_ids based on space group, same as in serial version
            if spg <= 2:
                rep_ids = [1, 2, 3, 4, 5, 6]
            elif spg <= 15:
                rep_ids = [1, 2, 3, 5]
            elif spg <= 74:
                rep_ids = [1, 2, 3]
            elif spg <= 142:
                rep_ids = [1, 3]
            elif spg <= 194:
                rep_ids = [1, 3]
            else:
                rep_ids = [1]

            # Process Wyckoff positions
            for rep_id in range(7, len(row), 4):
                wp = int(row[rep_id])
                max_op = self.num_wp[spg]
                if wp >= 0 and wp < max_op:
                    ids = self.get_free_xyz_ids(spg, wp)
                    # reset xyz from the generator
                    xyz = row[rep_id+1: rep_id+4]
                    xyz = self.get_generator(spg, wp, xyz, cell_matrix, tol)

                    # Ignore the bad ids
                    if xyz is not None:
                        wps_list.append([wp, batch_id])
                        row[rep_id+1: rep_id+4] = xyz
                        for id in ids:
                            rep_ids.append(id + rep_id + 1)
            rep_ids_list.append(torch.tensor(rep_ids, dtype=torch.int64, device=device))

        #print(wps_list)
        wps_batch = torch.tensor(wps_list, dtype=torch.int64, device=device)
        # Ensure all tensors are on the target device and same dtype
        rep_ids_batch = pad_and_stack(rep_ids_list, pad_value=-1)
        return spg_batch, wps_batch, rep_ids_batch

    def get_tuple_from_spg_wps_rep(self, spg, wps, rep, normalize, cell=None, count=None):
        """
        Get the (cell, coordinates, numbers, ids, weights) from (spg, wps, xyzs)

        Args:
            spg (int): Space group number.
            wps (list): List of Wyckoff positions.
            rep (torch.Tensor): Representation parameters, shape (N,).
            normalize (bool): If True, normalizes the cell parameters.
            cell (torch.Tensor, optional): Precomputed cell matrix, shape (3, 3).
            count (int, optional): Current count of free coordinates.

        Returns:
            tuple: (cell, coordinates, numbers, ids, weights)
                - cell (torch.Tensor): Shape (3, 3), the cell matrix.
                - coordinates (torch.Tensor): Shape (M, 3), the atomic coordinates.
                - numbers (torch.Tensor): Shape (M,), atomic numbers.
                - ids (torch.Tensor): Shape (N,), indices of atoms in the structure.
                - weights (torch.Tensor): Shape (N,), weights of each atom.
        """
        if cell is None:
            cell, count = get_cell_from_rep(rep, spg, normalize, self.max_angle)
        (coords, numbers, ids, weights) = self._get_coordinates_tuple(spg, wps, rep, cell, count)
        return (cell, coords, numbers, ids, weights)

    def _get_coordinates_tuple(self, spg, wps, rep, lattice, count):
        """
        Get the (coords, numbers, ids, weights) from (spg, wps, rep, lat, count).

        Args:
            spg (int): Space group number.
            wps (list): List of Wyckoff positions.
            rep (torch.Tensor): Representation parameters, shape (N,).
            lattice (torch.Tensor): Shape (3, 3), the lattice matrix.
            count (int): Current count of free coordinates.

        Returns:
            A tuple containing:
                - coords (torch.Tensor): Shape (M, 3), the atomic coordinates.
                - numbers (torch.Tensor): Shape (M,), atomic numbers.
                - ids (torch.Tensor): Shape (N,), indices of atoms.
                - weights (torch.Tensor): Shape (N,), weights of each atom.
        """
        coords = []
        ids = torch.zeros(len(wps), dtype=torch.int64)
        weights = torch.zeros(len(wps), dtype=torch.float64)

        for i, wp in enumerate(wps):
            xyz_ids = self.get_free_xyz_ids(spg, wp)
            free_xyz = rep[count: count + len(xyz_ids)]#; print("debug", free_xyz, spg)
            xyz = self.get_xyz_from_free_ids(spg, wp, free_xyz)
            xyz = self.get_coordinates(spg, wp, xyz, lattice, check_generator=False)
            coords.append(xyz)
            count += len(xyz_ids)
            if i + 1 < len(wps):
                ids[i+1] = ids[i] + len(xyz)
            weights[i] = len(xyz) #print(wps, ids); import sys; sys.exit()
        coords = torch.cat(coords, dim=0) @ lattice

        # Assign atomic numbers to 6 for now
        numbers = torch.tensor([6]*len(coords), dtype=torch.int64)
        weights /= weights.sum() #print('debug get_tuple', lattice, coords)
        return (coords, numbers, ids, weights)

    def get_tuple_from_batch_opt(self, spg_batch, rep_batch, generators,
                                 g_map, xyz_map, normalize=True):
        """
        Get the (cell, coordinates, numbers, ids, weights) from batched inputs
        for the optimization purposes.
        B is the number of samples in the batch,
        B1 is the number of Wyckoff positions in the batch.
        N is the number of representation parameters per sample.
        M is the total number of operations across all Wyckoff positions.

        Args:
            spg_batch (torch.Tensor): Shape (B,), space group numbers.
            rep_batch (torch.Tensor): Shape (B, N), (a, b, c, ..., wp, x, y, z).
            generators (torch.Tensor): The generators of shape (B1, 3).
            g_map (torch.Tensor): Shape (B1, 4), (rep_id1, rep_id2, rep_id3, s_id).
            xyz_map (torch.Tensor): Shape (M, 3), (struc_id, wp_id, op_index).
            normalize (bool): If True, normalizes the cell parameters.
            max_wp (int): Maximum number of Wyckoff positions per sample.

        Returns:
            tuple: (cell, coordinates, numbers, ids, weights)
                - cell (torch.Tensor): Shape (B, 3, 3), cell matrices for each sample.
                - coordinates (torch.Tensor): Shape (B, max_atoms, 3), coordinates.
                - numbers (torch.Tensor): Shape (B, max_atoms), atomic numbers.
                - ids (torch.Tensor): Shape (B, 8), indices of atoms in the structure.
                - weights (torch.Tensor): Shape (B, 8), weights of each atom.
        """
        # Get the cell
        cell, _, _ = get_cell_batch(rep_batch, spg_batch, normalize,
                                    self.max_abc, self.max_angle)

        # ---- build generators in a differentiable way ----
        # g_map: (B1, 4) → last col = structure id, first 3 = rep indices
        s_ids   = g_map[:, -1]          # (B1,)
        rep_idx = g_map[:, :-1]         # (B1, 3)

        # Expand structure ids to match rep_idx for advanced indexing
        batch_idx = s_ids.unsqueeze(1).expand_as(rep_idx)        # (B1, 3)

        # Boolean mask – True where a real generator exists
        valid_mask = rep_idx >= 0                                # (B1, 3)

        # Gather the representation parameters; invalid spots get zero
        rep_vals = torch.zeros_like(rep_idx, dtype=rep_batch.dtype)
        rep_vals[valid_mask] = rep_batch[batch_idx[valid_mask],
                                        rep_idx[valid_mask]]

        # `generators` now comes directly from differentiable ops
        generators = rep_vals.to(rep_batch.dtype)                # (B1, 3)

        # Get the xyz coordinates from the generators and xyz_map
        xyz = self.get_coords_from_generators_and_map(generators, xyz_map)

        # Get the coordinates from the generators and xyz_map
        coords, numbers, ids, weights = self.get_batch_from_xyz(xyz, xyz_map, cell)

        return cell, coords, numbers, ids, weights

    def get_tuple_from_batch(self, spg_batch, wps_batch, rep_batch, normalize=True):
        """
        Get the (cell, coordinates, numbers, ids, weights) from batched inputs.
        B is the number of samples in the batch,
        B1 is the number of Wyckoff positions in the batch.
        N is the number of representation parameters per sample.

        Args:
            spg_batch (torch.Tensor): Shape (B,), space group numbers.
            wps_batch (torch.Tensor): Shape (B1, 2), (wps_id, struc_id).
            rep_batch (torch.Tensor): Shape (B, N), (a, b, c, ..., wp, x, y, z).
            normalize (bool): If True, normalizes the cell parameters.

        Returns:
            - cell (torch.Tensor): Shape (B, 3, 3), cell matrices for each sample.
            - coordinates (torch.Tensor): Shape (B, max_atoms, 3), coordinates.
            - numbers (torch.Tensor): Shape (B, max_atoms), atomic numbers.
            - ids (torch.Tensor): Shape (B, 8), indices of atoms in the structure.
            - weights (torch.Tensor): Shape (B, 8), weights of each atom.
            - fails: List of indices that failed to process.
            - generators: torch.Tensor: Shape (B1, 3), the generators for each WP.
            - g_map: torch.Tensor: Shape (B1, 4), (rep_id1, rep_id2, rep_id3, s_id).
            - xyz_map: torch.Tensor: Shape (M1, 3), (struc_id, wp_id, op_index).
        """
        # Get the lattice batch
        cell, count, _ = get_cell_batch(rep_batch, spg_batch, normalize,
                                        self.max_abc, self.max_angle)

        # generator xyz for each WP: (B1, 3)
        generators, g_map = get_generators(wps_batch, spg_batch, rep_batch,
                                           count, self.free_xyz_cache)
        #print("debug get_tuple_from_batch", generators, g_map)
        #import sys; sys.exit()

        # Get the full xyz (M1, 3) and xyz_map (M1, 3)
        xyz, xyz_map = self.get_coords_from_generators(generators, wps_batch, spg_batch)
        #print("debug get_tuple_from_batch", xyz_batch); import sys; sys.exit()

        # Get the batch of (coordinates, numbers, ids, weights)
        coords, numbers, ids, weights = self.get_batch_from_xyz(xyz, xyz_map, cell)

        return cell, coords, numbers, ids, weights, generators, g_map, xyz_map
    '''    
    def get_batch_from_xyz(self, xyz, xyz_map, cell):
        """
        Get the (coordinates, numbers, ids, weights) from the xyz coordinates
        M1 is the number of operations across all Wyckoff positions.
        B is the number of structures in the batch.

        Args:
            xyz (torch.Tensor): Shape (M1, 3), the coordinates of atoms.
            xyz_map (torch.Tensor): Shape (M1, 3), (struc_id, wp_id, op_index).
            cell (torch.Tensor): Shape (B, 3, 3), the cell matrices for each sample.

        Returns:
            - coordinates (torch.Tensor): Shape (B, max_atoms, 3), coordinates.
            - numbers (torch.Tensor): Shape (B, max_atoms), atomic numbers.
            - ids (torch.Tensor): Shape (B, max_wp), indices of atoms in the structure.
            - weights (torch.Tensor): Shape (B, max_wp), weights of each atom.
        """
        device = xyz.device
        cell = cell.to(device)
        xyz_map = xyz_map.to(device)
        B = len(cell)  # Number of structures
        print("Debug: get_batch_from_xyz", xyz)

        max_atoms = 0
        max_wp = 0
        batch_results = []
        for struc_id in range(B):
            masks = xyz_map[:, 0] == struc_id#; print(masks)
            coord = xyz[masks] @ cell[struc_id]
            all_ids = xyz_map[masks, 1]  # Get the wp_ids
            unique_ids, inv_ids, counts = all_ids.unique(return_inverse=True,
                                                         return_counts=True)
            first_occ = []
            for i in range(len(unique_ids)):
                first_occ.append((inv_ids == i).nonzero()[0].item())
            weights = counts.float() #/ counts.sum()
            first_occ = torch.tensor(first_occ, dtype=torch.int64, device=device)
            max_atoms = max(max_atoms, coord.shape[0])
            max_wp = max(max_wp, len(first_occ))
            batch_results.append((coord, first_occ, weights))

        # Initialize lists to store results
        coords = torch.full((B, max_atoms, 3), -1000, dtype=torch.float64, device=device)
        numbers = torch.full((B, max_atoms), -1, dtype=torch.int64, device=device)
        ids = torch.full((B, max_wp), -1, dtype=torch.int64, device=device)
        weights = torch.full((B, max_wp), -1, dtype=torch.float64, device=device)

        # Store results in padded tensors
        for i, (coord, atom_ids, weight) in enumerate(batch_results):
            coords[i, :coord.shape[0], :] = coord
            numbers[i, :coord.shape[0]] = 6
            ids[i, :len(atom_ids)] = atom_ids
            weights[i, :weight.shape[0]] = weight
        #print("Debug", generators, ids)#; import sys; sys.exit()
        
        # store inputs and outputs in a log csv file
        log_file = 'get_batch_from_xyz_log.csv'
        # prepare the log entry
        log_entry = {
            'xyz': xyz.cpu().numpy().tolist(),
            'xyz_map': xyz_map.cpu().numpy().tolist(),
            'cell': cell.cpu().numpy().tolist(),
            'coords': coords.cpu().numpy().tolist(),
            'numbers': numbers.cpu().numpy().tolist(),
            'ids': ids.cpu().numpy().tolist(),
            'weights': weights.cpu().numpy().tolist(),
        }
        df_log = pd.DataFrame([log_entry])
        # append to CSV, write header only if file does not exist
        write_header = not os.path.exists(log_file)
        df_log.to_csv(log_file, mode='a', index=False, header=write_header)
        
        return coords, numbers, ids, weights
        '''

    def get_batch_from_xyz(self,xyz, xyz_map, cell):
        """
        xyz      : (M1, 3)      fractional coordinates
        xyz_map  : (M1, 3)      (structure_id, wp_id, op_id)
        cell     : (B, 3, 3)    lattice for each structure
        ----------------------------------------------------------------
        returns  (coords, numbers, ids, weights) with shapes
                (B, max_atoms, 3), (B, max_atoms),
                (B, max_wp),      (B, max_wp)
        """
        device   = xyz.device
        xyz_map  = xyz_map.to(device)
        cell     = cell.to(device)

        B   = cell.shape[0]                  # structures
        M1  = xyz.shape[0]                   # total atoms

        # ------------------------------------------------------------------
        # 1.  Per-atom bookkeeping
        # ------------------------------------------------------------------
        struc_ids = xyz_map[:, 0].long()     # (M1,)
        wp_ids    = xyz_map[:, 1].long()     # (M1,)

        # Cartesian coordinates for every atom
        # xyz_row : (M1, 1, 3)
        xyz_row    = xyz.unsqueeze(1)                     # (M1, 1, 3)
        # coords_all : (M1, 3)  =  row-vector ⋅ cell
        coords_all = torch.bmm(xyz_row, cell[struc_ids])  # (M1, 1, 3)
        coords_all = coords_all.squeeze(1)                # (M1, 3)
        # atoms per structure  & padded length
        atom_counts = torch.bincount(struc_ids, minlength=B)        # (B,)
        max_atoms   = int(atom_counts.max())

        # index of each atom within its structure
        offsets = torch.cumsum(atom_counts, 0) - atom_counts    
        idx_in_struc = torch.arange(M1, device=device) - offsets[struc_ids]

        # flat destination index into (B·max_atoms)
        dest_idx = struc_ids * max_atoms + idx_in_struc              # (M1,)

        # ------------------------------------------------------------------
        # 2.  Build padded coords & numbers with ONE scatter
        # ------------------------------------------------------------------
        coords  = torch.full((B, max_atoms, 3), -1000.0,
                            dtype=torch.float64, device=device)
        numbers = torch.full((B, max_atoms), -1,
                            dtype=torch.int64,  device=device)

        coords.view(-1, 3)[dest_idx] = coords_all
        numbers.view(-1)[dest_idx]   = 6                      # e.g. carbon

        # ------------------------------------------------------------------
        # 3.  ids  &  weights from (structure, wp) tables
        # ------------------------------------------------------------------
        W = int(wp_ids.max().item()) + 1                      # max wp slots
        flat        = struc_ids * W + wp_ids                  # (M1,)
        flat_counts = torch.bincount(flat, minlength=B*W).view(B, W).float()

        # first occurrence per (s, wp)
        first_pos = torch.full((B*W,), M1, dtype=torch.int64, device=device)
        first_pos.scatter_reduce_(0, flat,
                                torch.arange(M1, device=device),
                                reduce='amin', include_self=True)
        first_pos = first_pos.view(B, W)
        first_pos_rel = first_pos - offsets.unsqueeze(1)              # (B, W)
        mask = flat_counts > 0   
        # keep only real wps
        ids_full = torch.where(mask, first_pos_rel, -1)
        weights_full = torch.where(flat_counts > 0, flat_counts, -1.0)

        # ------------------------------------------------------------------
        # mask showing which (structure,row-wp) pairs are populated
        # ------------------------------------------------------------------
                                    # (B, W)  bool

        # prefix-sum along columns → for every ‘True’ cell we get its 0-based
        # position among the *present* WPs of that structure

        mask     = flat_counts > 0                        # (B, W)
        prefix   = mask.long().cumsum(1) - 1              # 0,1,2… per row
        max_wp   = int(prefix[mask].max()) + 1

        ids      = torch.full((B, max_wp), -1, dtype=torch.int64,   device=device)
        weights  = torch.full((B, max_wp), -1, dtype=flat_counts.dtype, device=device)

        rows, cols   = mask.nonzero(as_tuple=True)
        dest_cols    = prefix[mask]

        ids[rows, dest_cols]     = first_pos_rel[mask]          #  ◀ relative indices
        weights[rows, dest_cols] = flat_counts[mask]            #  counts
        return coords, numbers, ids, weights

    def get_coords_from_generators_and_map(self, generators, xyz_map):
        """
        Get the xyz coordinates by applying the symmetry operations.

        Replicate the generators from (B1, 3) to xyz_coords (M1, 3)
        according to the xyz_map (:, 0), and then apply the operations
        to get the coordinates.

        Args:
            generators (torch.Tensor): shape (B1, 3).
            xyz_map (torch.Tensor): shape (M1, 3), (struc_id, wp_id, op_id)

        Returns:
            xyz_coords (torch.Tensor): The coordinates of shape (M1, 3).
        """
        device = generators.device
        M1 = xyz_map.shape[0]

        # Ensure xyz_map is on the same device as generators
        xyz_map = xyz_map.to(device)

        # Get the generators for each coordinate based on wp_id
        wp_ids = xyz_map[:, 1]  # Shape: (M1,)
        xyz = generators[wp_ids]  # Shape: (M1, 3)

        # Get the operation matrices for each coordinate
        op_ids = xyz_map[:, 2]  # Shape: (M1,)

        # Ensure sym_matrices are on the correct device
        if self.sym_matrices.device != device:
            self.sym_matrices = self.sym_matrices.to(device)

        ops = self.sym_matrices[op_ids]  # Shape: (M1, 4, 4)

        # Apply operations in batch
        xyz_homo = torch.cat([xyz, torch.ones(M1, 1, dtype=torch.float64, device=device)], dim=1).unsqueeze(2)
        xyz_coords = torch.bmm(ops, xyz_homo).squeeze(2)[:, :3]

        # Apply modulo to ensure coordinates are within [0,1)
        xyz_coords = xyz_coords % 1.0

        return xyz_coords

    def get_coords_from_generators(self, generators, wps_batch, spg_batch):
        """
        Get the xyz coordinates by applying the symmetry operations.

        Replicate the generators (B1, 3) based on WP multiplicity,
        and get the coordinates (M1, 3), where M1 is total operations.
        Then find the affine matrices, and transform the coordinates.
        We also output the map relation (M1, 2) to track the coordinates
        belongs to which (structure, wp, affine matrice)

        Args:
            generators (torch.Tensor): The generators of shape (B1, 3).
            wps_batch (torch.Tensor): The Wyckoffs of shape (B1, 2).
            spg_batch (torch.Tensor): The space group numbers of shape (B,).

        Returns:
            xyz_coords (torch.Tensor): shape (M1, 3), (x, y, z) coordinates.
            xyz_map (torch.Tensor): shape (M1, 3), (struc_id, wp_id, op_id)
        """
        device = generators.device
        B1 = generators.shape[0]

        coord_list = []
        map_list = []
        for wp_id in range(B1):
            wp = wps_batch[wp_id, 0].item()
            struc_id = wps_batch[wp_id, 1].item()
            spg = spg_batch[struc_id].item()

            # Get operations for this WP
            start_id, num_ops = self.sym_map[(spg, wp)]
            op_ids = torch.arange(start_id, start_id + num_ops, device=device)
            ops = self.sym_matrices[op_ids]

            # Get coordinates for this WP
            xyz = generators[wp_id]
            coords = self.apply_operations(xyz, ops)
            coord_list.append(coords)

            # Create map for this WP: (wp_id, structure_id, op_index)
            wp_map = torch.zeros((num_ops, 3), dtype=torch.int64, device=device)
            wp_map[:, 0] = struc_id
            wp_map[:, 1] = wp_id
            wp_map[:, 2] = op_ids
            map_list.append(wp_map)

        xyz_coords = torch.cat(coord_list, dim=0)
        xyz_map = torch.cat(map_list, dim=0)
        #print("debug get_coordinates_batch", coordinates.shape, xyz_map.shape)

        return xyz_coords, xyz_map

    def get_pyxtal_from_spg_wps_rep(self, spg, wps, rep, normalize=True):
        from pyxtal import pyxtal
        xtal = pyxtal()

        # Convert to tensor if it's a list
        if isinstance(rep, list):
            rep = torch.tensor(rep, dtype=torch.float64)

        if normalize:
            if spg <= 2:
                rep[:3] *= self.max_abc
                rep[3:6] *= self.max_angle
            elif spg <= 15:
                rep[:3] *= self.max_abc
                rep[3] *= self.max_angle *57.2958 ###
            elif spg <= 74:
                rep[:3] *= self.max_abc
            elif spg <= 194:
                rep[:2] *= self.max_abc
            else:
                rep[0] *= self.max_abc

        # Convert back to list if needed for pyxtal
        if isinstance(rep, torch.Tensor):
            rep = rep.tolist()

        # Filter wps: must be int, >=0, and < num_wp[spg]
        if isinstance(wps, torch.Tensor):
            wps = wps.cpu().numpy().tolist()
        #max_wp = self.num_wp[spg]
        #wps = [int(w) for w in wps if isinstance(w, (int, float)) and w != -1 and 0 <= int(w) < max_wp]
        #rep = [r for r in rep if r != -1]

        # Only call from_spg_wps_rep if wps is not empty
        if len(wps) > 0:
            xtal.from_spg_wps_rep(spg, wps, rep)
        else:
            print(f"Warning: Empty or invalid wps for spg={spg}, skipping pyxtal generation.")
        return xtal

def pad_and_stack(tensor_list, pad_value=-1):
    """
    Converts a list of 1D tensors of varying lengths into a padded 2D tensor.

    Args:
        tensor_list (list of torch.Tensor): List of tensors to be stacked.
        pad_value (int/float): The value to pad with.

    Returns:
        torch.Tensor: Stacked and padded tensor of shape (B, max_length)
    """
    max_length = max(t.shape[0] for t in tensor_list)  # Find max length in batch
    padded_tensors = [torch.cat([t, torch.full((max_length - t.shape[0],), pad_value, dtype=t.dtype, device=t.device)])
                      for t in tensor_list]
    return torch.stack(padded_tensors)


def para2matrix(para, spg=None, normalize=False, max_abc=35.0, max_angle=180.0):
    """
    Convert lattice parameters to a lattice matrix.

    Args:
        para (torch.Tensor): Shape (6,), [a, b, c, alpha, beta, gamma].
        spg (int, optional): Space group number for symmetrization.
        normalize (bool): If True, normalizes the lattice parameters.
        max_abc (float): Maximum value for lattice parameters (default: 35.0).
        max_angle (float): Maximum value for angles in degrees (default: 180.0).

    Returns:
        torch.Tensor: Shape (3, 3), the lattice matrix.
    """
    device = para.device if isinstance(para, torch.Tensor) else 'cpu'
    if normalize:
        para[:3] *= max_abc
        para[3:] *= max_angle

    # Symmetrize the lattice parameters based on the space group
    if spg is not None:
        if spg > 194:
            a_mean = para[:3].mean()
            para[0] = para[1] = para[2] = a_mean
            para[3] = para[4] = para[5] = torch.tensor(90.0, device=device)

        elif spg > 142:
            a_mean = para[:2].mean()
            para[0] = para[1] = a_mean
            para[3] = para[4] = torch.tensor(90.0, device=device)
            para[5] = torch.tensor(120.0, device=device)

        elif spg > 74:
            a_mean = para[:2].mean()
            para[0] = para[1] = a_mean
            para[3] = para[4] = para[5] = torch.tensor(90.0, device=device)

        elif spg > 15:
            para[3] = para[4] = para[5] = torch.tensor(90.0, device=device)

        elif spg > 2:
            para[3] = para[5] = torch.tensor(90.0, device=device)

    # Convert angles from degrees to radians
    para[3:] = torch.deg2rad(para[3:])

    # Build lattice matrix
    a, b, c = para[0], para[1], para[2]
    alpha, beta, gamma = para[3], para[4], para[5]

    lattice = torch.zeros((3, 3), dtype=torch.float64, device=device)
    a3 = a * cos(beta)
    a2 = (a * (cos(gamma) - cos(beta) * cos(alpha))) / (sin(alpha)+1e-10)
    lattice[0, 1] = a2
    lattice[0, 2] = a3
    lattice[1, 1] = b * sin(alpha)
    lattice[1, 2] = c * cos(alpha)
    lattice[2, 2] = c

    tmp = a**2 - a3**2 - a2**2
    #print("debug para2matrix", para, lattice)
    if tmp > 0:
        lattice[0, 0] = sqrt(tmp)
    else:
        return None

    return lattice

def get_cell_batch(rep_batch, spg_batch, normalize=True, max_abc=35.0, max_angle=180.0):
    """
    Get the cell batch of representation parameters and space groups.

    Args:
        rep_batch (torch.Tensor): Shape (B, N), batch of representation parameters.
        spg_batch (torch.Tensor): Shape (B,), batch of space group numbers.
        normalize (bool): If True, normalizes the cell parameters.
        max_abc (float): Maximum value for lattice parameters (default: 35.0).
        max_angle (float): Maximum value for angles in degrees (default: 180.0).

    Returns:
        cell_batch (torch.Tensor): Shape (B, 3, 3), cell matrice.
        count_batch (torch.Tensor): Shape (B,), count of free coordinates.
        invalid_mask (torch.Tensor): Shape (B,), mask of valid cells.
    """
    B = spg_batch.shape[0]
    device = rep_batch.device

    para_batch = torch.zeros((B, 6), dtype=torch.float64, device=device)
    cell_batch = torch.zeros((B, 3, 3), dtype=torch.float64, device=device)
    invalid_mask = torch.zeros(B, dtype=torch.bool, device=device)
    count_batch = torch.zeros(B, dtype=torch.int64, device=device)

    # Create masks for different space group ranges
    mask_tri = (spg_batch <= 2)
    mask_mono = (spg_batch > 2) & (spg_batch <= 15)
    mask_ortho = (spg_batch > 15) & (spg_batch <= 74)
    mask_tetra = (spg_batch > 74) & (spg_batch <= 142)
    mask_hex = (spg_batch > 142) & (spg_batch <= 194)
    mask_cubic = (spg_batch > 194)

    # Handle triclinic (all 6 parameters from rep)
    if mask_tri.any():
        para_batch[mask_tri] = rep_batch[mask_tri, :6]
        para_batch[mask_tri, 3:6] *= 57.2958
        count_batch[mask_tri] = 6  # Count of parameters for triclinic
        if normalize: para_batch[mask_tri, 3:] *= max_angle

    # Handle monoclinic (alpha=gamma=90°)
    if mask_mono.any():
        para_batch[mask_mono, :3] = rep_batch[mask_mono, :3]
        para_batch[mask_mono, 3] = 90.0
        para_batch[mask_mono, 4] = rep_batch[mask_mono, 3] *57.2958  ####
        para_batch[mask_mono, 5] = 90.0
        count_batch[mask_mono] = 4  # Count of parameters for monoclinic
        if normalize: para_batch[mask_mono, 4] *= max_angle

    # Handle orthorhombic (alpha=beta=gamma=90°)
    if mask_ortho.any():
        para_batch[mask_ortho, :3] = rep_batch[mask_ortho, :3]
        para_batch[mask_ortho, 3:] = 90.0
        count_batch[mask_ortho] = 3  # Count of parameters for orthorhombic

    # Handle tetragonal (a=b, alpha=beta=gamma=90°)
    if mask_tetra.any():
        para_batch[mask_tetra, 0] = rep_batch[mask_tetra, 0]  # a
        para_batch[mask_tetra, 1] = rep_batch[mask_tetra, 0]  # b = a
        para_batch[mask_tetra, 2] = rep_batch[mask_tetra, 1]  # c
        para_batch[mask_tetra, 3:] = 90.0
        count_batch[mask_tetra] = 2  # Count of parameters for tetragonal

    # Handle hexagonal/trigonal (a=b, alpha=beta=90°, gamma=120°)
    if mask_hex.any():
        para_batch[mask_hex, 0] = rep_batch[mask_hex, 0]  # a
        para_batch[mask_hex, 1] = rep_batch[mask_hex, 0]  # b = a
        para_batch[mask_hex, 2] = rep_batch[mask_hex, 1]  # c
        para_batch[mask_hex, 3:5] = 90.0
        para_batch[mask_hex, 5] = 120.0
        count_batch[mask_hex] = 2

    # Handle cubic (a=b=c, alpha=beta=gamma=90°)
    if mask_cubic.any():
        para_batch[mask_cubic, 0] = rep_batch[mask_cubic, 0]  # a
        para_batch[mask_cubic, 1] = rep_batch[mask_cubic, 0]  # b = a
        para_batch[mask_cubic, 2] = rep_batch[mask_cubic, 0]  # c = a
        para_batch[mask_cubic, 3:] = 90.0
        count_batch[mask_cubic] = 1  # Count of parameters for cubic

    if normalize: para_batch[:, :3] *= max_abc  # Normalize lattice parameters

    # Convert angles from degrees to radians
    para_batch[:, 3:] = torch.deg2rad(para_batch[:, 3:])
    eps = 1e-6 
    # Convert to 3x3 matrices
    a, b, c = para_batch[:, 0], para_batch[:, 1], para_batch[:, 2]
    alpha, beta, gamma = para_batch[:, 3], para_batch[:, 4], para_batch[:, 5]

    a3 = a * cos(beta)
    a2 = (a * (cos(gamma) - cos(beta) * cos(alpha))) / (sin(alpha)+1e-10)

    r2 = a**2 - a3**2 - a2**2   
    valid_mask = r2 > eps 
    safe_r2 = torch.where(valid_mask, r2, torch.ones_like(r2))
    a1 = torch.sqrt(safe_r2)
    a1 = a1 * valid_mask.float()
    cell_batch[:, 0, 0] = a1  # Ensure this is valid
    cell_batch[:, 0, 1] = a2
    cell_batch[:, 0, 2] = a3
    cell_batch[:, 1, 1] = b * sin(alpha)
    cell_batch[:, 1, 2] = b * cos(alpha)
    cell_batch[:, 2, 2] = c
    ''' 
    # Handle invalid values (optional safety check)
    invalid_mask = (a**2 - a3**2 - a2**2) <= 0
    if invalid_mask.any():
        # Set invalid lattices to identity or handle as needed
        cell_batch[invalid_mask] = torch.eye(3, dtype=torch.float64, device=device)

    return cell_batch, count_batch, invalid_mask
    '''
    if (~valid_mask).any():
        cell_batch[~valid_mask] = torch.eye(3, dtype=torch.float64, device=device)

    return cell_batch, count_batch, ~valid_mask    # return mask of invalid samples

def get_generators(wps_batch, spg_batch, rep_batch, count_batch, free_xyz_cache):
    """
    Convert the structure-based batch to WP based batch.
    Here B is the number of stuctures, and B1 is the total number of Wyckoffs.

    Args:
        wps_batch (torch.Tersor): Shape (B1, 2) to store each (wp_id, structure_id).
        spg_batch (torch.Tensor): Shape (B,), space group indices.
        rep_batch (torch.Tensor): Shape (B, N), representation parameters.
        count_batch (torch.Tensor): Shape (B,), counts of free coordinates.
        free_xyz_cache (dict): Cache of free coordinate IDs for each Wyckoff.

    Returns:
        generators (torch.Tensor): Shape (B1, 3)
        g_map (torch.Tensor): Shape (B1, 2), map from rep. via (s_id, rep_id).
    """
    device = wps_batch.device
    B1 = len(wps_batch)  # Total number of Wyckoff positions
    g_map = torch.full((B1, 4), -1, dtype=torch.int64, device=device)
    generators = torch.full((B1, 3), -1000.0, dtype=torch.float64, device=device)

    # Iterate through each structure in the batch
    for batch_id in range(B1):
        struc_id  = wps_batch[batch_id, 1]
        spg = spg_batch[struc_id].item()
        wp = wps_batch[batch_id, 0].item()#; print("debug get_generators", spg, wp, struc_id)
        free_ids = free_xyz_cache[spg][wp]
        g_map[batch_id, -1] = struc_id  # Store structure ID
        for id in range(3):
            if id in free_ids:
                #print("debug get_generators", spg, wp, free_ids, batch_id)
                generators[batch_id, id] = rep_batch[struc_id, count_batch[struc_id]]
                g_map[batch_id, id] = count_batch[struc_id]
                count_batch[struc_id] += 1
        #print("debug get_generators", spg, wp, free_ids,
        #      generators[batch_id], g_map[batch_id])

    #import sys; sys.exit()
    return generators, g_map


def get_cell_from_rep(rep, spg, normalize=True, max_angle=180.0):
    """
    Get the cell matrix from representation parameters and space group.

    Args:
        rep (torch.Tensor): Representation parameters, shape (N,).
        spg (int): Space group number.
        normalize (bool): If True, normalizes the cell parameters.
        max_angle (float): Maximum value for angles in degrees (default: 180.0).
    """
    cell = torch.zeros(6, dtype=torch.float64)
    if spg <= 2:
        cell = rep[:6]
        count = 6
    elif spg <= 15:
        cell[:3], cell[3], cell[4], cell[5] = rep[:3], 90.0, rep[3], 90.0
        count = 4
    elif spg <= 74:
        cell[:3], cell[3:] = rep[:3], 90.0
        count = 3
    elif spg <= 142:
        cell[:2], cell[2], cell[3:] = rep[0], rep[1], 90.0
        count = 2
    elif spg <= 194:
        cell[:2], cell[2], cell[3:5], cell[5] = rep[0], rep[1], 90.0, 120.0
        count = 2
    else:
        cell[:3], cell[3:] = rep[0], 90.0
        count = 1

    if normalize: cell[3:] /= max_angle
    cell_matrix = para2matrix(cell, normalize=normalize)
    return cell_matrix, count
