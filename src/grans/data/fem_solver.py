"""Finite Element Method solver for 2D PDEs

Supports:
- Poisson equation: -∇·(κ∇u) = f
- Advection-diffusion equation: -∇·(κ∇u) + v·∇u = f
- Custom random sparse matrices

Refactored from original femsolver.py with simplified interface
"""

import copy
import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
import meshpy.triangle as triangle
from typing import Callable, Tuple


class FEMSolver:
    """FEM solver for 2D Poisson equation

    Solves: -∇·(κ∇u) = f
    with boundary condition: u = g on ∂Ω

    This is a simplified version focusing on the core functionality.
    """

    def __init__(self, target_nodes: int = 250):
        """Initialize FEM solver

        Args:
            target_nodes: Target number of mesh nodes (approximate)
        """
        self.nodes = None
        self.elements = None
        self.is_boundary = None
        self.is_g_boundary = None
        self.target_nodes = target_nodes

        # Basis function gradients for linear triangular elements
        self.dbasis = np.array([
            [-1, 1, 0],  # dφ/dr
            [-1, 0, 1]   # dφ/ds
        ])

    def set_domain(
        self,
        bound_points: list,
        bc_points: list,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Set the computational domain and generate mesh

        Args:
            bound_points: List of domain corner coordinates
                Example: [(-1, -1), (1, -1), (1, 1), (-1, 1)]
            bc_points: List of points where boundary condition g is applied
                Example: [(-9999, -1)] means y=-1 line

        Returns:
            X: x-coordinates of nodes
            Y: y-coordinates of nodes
            E: Element connectivity
        """
        bound = np.array(bound_points)
        self.x_lb = bound[:, 0].min()
        self.x_ub = bound[:, 0].max()
        self.y_lb = bound[:, 1].min()
        self.y_ub = bound[:, 1].max()

        # Generate mesh
        self.nodes, self.elements = self._make_mesh(bound_points)

        # Identify boundary nodes
        X, Y = self.nodes[:, 0], self.nodes[:, 1]
        tol = 1e-12
        self.is_boundary = np.zeros(len(self.nodes), dtype=bool)
        self.is_g_boundary = np.zeros(len(self.nodes), dtype=bool)

        for x, y in bound_points:
            self.is_boundary |= (np.abs(X - x) < tol) | (np.abs(Y - y) < tol)

        for x, y in bc_points:
            self.is_g_boundary |= (np.abs(X - x) < tol) | (np.abs(Y - y) < tol)

        return X, Y, self.elements

    def solve(
        self,
        kappa: Callable,
        f: Callable,
        g: Callable,
    ) -> Tuple[np.ndarray, sparse.coo_matrix, np.ndarray]:
        """Solve the Poisson equation

        Args:
            kappa: Diffusion coefficient function κ(x,y)
            f: Source term function f(x,y)
            g: Boundary condition function g(x,y)

        Returns:
            u: Solution vector (nodal values)
            A: Stiffness matrix (sparse)
            b: Load vector
        """
        n_nodes = len(self.nodes)
        n_elements = len(self.elements)

        # Assemble stiffness matrix A
        a_builder = _MatrixBuilder()

        for ei in range(n_elements):
            vert_indices = self.elements[ei, :]
            el_verts = self.nodes[vert_indices]
            centroid = np.mean(el_verts, axis=0)

            # Jacobian matrix and its inverse
            x0, x1, x2 = el_verts
            J = np.array([x1 - x0, x2 - x0]).T
            invJT = la.inv(J.T)
            detJ = la.det(J)

            # Element stiffness matrix
            dphi = invJT @ self.dbasis  # Physical space gradients
            Aelem = kappa(centroid) * (detJ / 2.0) * dphi.T @ dphi

            a_builder.add(vert_indices, vert_indices, Aelem)

        A = a_builder.coo_matrix().tocsr().tocoo()

        # Assemble load vector b
        b = np.zeros(n_nodes)

        for ei in range(n_elements):
            vert_indices = self.elements[ei, :]
            el_verts = self.nodes[vert_indices]
            centroid = np.mean(el_verts, axis=0)

            x0, x1, x2 = el_verts
            J = np.array([x1 - x0, x2 - x0]).T
            detJ = la.det(J)

            # Element load vector (assuming constant f over element)
            belem = f(centroid) * (detJ / 6.0) * np.ones(3)

            for i, vi in enumerate(vert_indices):
                b[vi] += belem[i]

        # Apply boundary conditions
        u0 = np.zeros(n_nodes)
        u0[self.is_g_boundary] = g(self.nodes[self.is_g_boundary].T)

        rhs = b - A @ u0

        # Enforce zero boundary on other boundaries
        rhs[self.is_boundary] = 0.0
        for k in range(A.nnz):
            i, j = A.row[k], A.col[k]
            if self.is_boundary[i]:
                A.data[k] = 1 if i == j else 0

        # Solve sparse linear system
        uhat = sla.spsolve(A.tocsr(), rhs)
        u = uhat + u0

        return uhat, A, rhs

    def solve_advection_diffusion(
        self,
        kappa: Callable,
        velocity: Callable,
        f: Callable,
        g: Callable,
    ) -> Tuple[np.ndarray, sparse.coo_matrix, np.ndarray]:
        """Solve the advection-diffusion equation

        Equation: -∇·(κ∇u) + v·∇u = f
        where v is the velocity field

        This produces a non-symmetric matrix due to the advection term.

        Args:
            kappa: Diffusion coefficient function κ(x,y)
            velocity: Velocity field function v(x,y) -> (vx, vy)
            f: Source term function f(x,y)
            g: Boundary condition function g(x,y)

        Returns:
            u: Solution vector (nodal values)
            A: System matrix (sparse, non-symmetric)
            b: Load vector
        """
        n_nodes = len(self.nodes)
        n_elements = len(self.elements)

        # Assemble system matrix A = K + C
        # K: stiffness matrix (diffusion)
        # C: advection matrix
        a_builder = _MatrixBuilder()

        for ei in range(n_elements):
            vert_indices = self.elements[ei, :]
            el_verts = self.nodes[vert_indices]
            centroid = np.mean(el_verts, axis=0)

            # Jacobian matrix and its inverse
            x0, x1, x2 = el_verts
            J = np.array([x1 - x0, x2 - x0]).T
            invJT = la.inv(J.T)
            detJ = la.det(J)

            # Physical space gradients
            dphi = invJT @ self.dbasis  # (2, 3) - gradients of 3 basis functions

            # Diffusion term: K_elem = κ * ∫∇φ_i · ∇φ_j dΩ
            K_elem = kappa(centroid) * (detJ / 2.0) * dphi.T @ dphi

            # Advection term: C_elem = ∫φ_i (v·∇φ_j) dΩ
            v = velocity(centroid)  # (vx, vy)
            v = np.array(v).reshape(2)

            # Basis functions in reference coordinates
            # φ_1 = 1-r-s, φ_2 = r, φ_3 = s
            # We use 1-point Gauss quadrature at centroid
            phi_ref = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])  # Evaluated at (1/3, 1/3)

            # v·∇φ_j for each basis function j
            v_grad_phi = v @ dphi  # (3,) - dot product of v with each gradient

            # C_elem[i,j] = φ_i * (v·∇φ_j) * detJ/2
            C_elem = (detJ / 2.0) * np.outer(phi_ref, v_grad_phi)

            # Total element matrix
            A_elem = K_elem + C_elem

            a_builder.add(vert_indices, vert_indices, A_elem)

        A = a_builder.coo_matrix().tocsr().tocoo()

        # Assemble load vector b (same as Poisson)
        b = np.zeros(n_nodes)

        for ei in range(n_elements):
            vert_indices = self.elements[ei, :]
            el_verts = self.nodes[vert_indices]
            centroid = np.mean(el_verts, axis=0)

            x0, x1, x2 = el_verts
            J = np.array([x1 - x0, x2 - x0]).T
            detJ = la.det(J)

            # Element load vector
            belem = f(centroid) * (detJ / 6.0) * np.ones(3)

            for i, vi in enumerate(vert_indices):
                b[vi] += belem[i]

        # Apply boundary conditions
        u0 = np.zeros(n_nodes)
        u0[self.is_g_boundary] = g(self.nodes[self.is_g_boundary].T)

        rhs = b - A @ u0

        # Enforce zero boundary on other boundaries
        rhs[self.is_boundary] = 0.0
        for k in range(A.nnz):
            i, j = A.row[k], A.col[k]
            if self.is_boundary[i]:
                A.data[k] = 1 if i == j else 0

        # Solve sparse linear system (use GMRES for non-symmetric)
        # Note: Still using spsolve which uses appropriate solver internally
        uhat = sla.spsolve(A.tocsr(), rhs)
        u = uhat + u0

        return uhat, A, rhs

    def solve_helmholtz(
        self,
        kappa: Callable,
        reaction_coeff: Callable,
        f: Callable,
        g: Callable,
    ) -> Tuple[np.ndarray, sparse.coo_matrix, np.ndarray]:
        """Solve the Helmholtz equation

        Equation: -∇·(κ∇u) + α·u = f
        where α is the reaction coefficient (often k² in wave problems)

        This adds a reaction term compared to Poisson equation,
        changing the physical behavior significantly.

        Args:
            kappa: Diffusion coefficient function κ(x,y)
            reaction_coeff: Reaction coefficient function α(x,y)
            f: Source term function f(x,y)
            g: Boundary condition function g(x,y)

        Returns:
            u: Solution vector (nodal values)
            A: System matrix (sparse, symmetric)
            b: Load vector
        """
        n_nodes = len(self.nodes)
        n_elements = len(self.elements)

        # Assemble system matrix A = K + M
        # K: stiffness matrix (diffusion term)
        # M: mass matrix scaled by reaction coefficient (reaction term)
        a_builder = _MatrixBuilder()

        for ei in range(n_elements):
            vert_indices = self.elements[ei, :]
            el_verts = self.nodes[vert_indices]
            centroid = np.mean(el_verts, axis=0)

            # Jacobian matrix and its inverse
            x0, x1, x2 = el_verts
            J = np.array([x1 - x0, x2 - x0]).T
            invJT = la.inv(J.T)
            detJ = la.det(J)

            # Get element parameters at centroid
            kappa_val = kappa(centroid)
            alpha_val = reaction_coeff(centroid)

            # Element stiffness matrix (diffusion term)
            # ∫∇φ·∇φ = invJT^T @ [[1, -1, 0], [-1, 2, -1], [0, -1, 1]] @ invJT * |detJ|/2
            grad_ref = np.array([[-1, -1], [1, 0], [0, 1]])
            grad_phys = grad_ref @ invJT

            Kelem = kappa_val * (grad_phys @ grad_phys.T) * (detJ / 2.0)

            # Element mass matrix (reaction term)
            # ∫φ·φ for linear triangles
            # Using lumped mass matrix for simplicity: M_ii = area/3
            # For consistent mass matrix:
            # M = [[2, 1, 1], [1, 2, 1], [1, 1, 2]] * area/12
            Melem = alpha_val * (detJ / 24.0) * np.array([
                [2, 1, 1],
                [1, 2, 1],
                [1, 1, 2]
            ])

            # Combined element matrix
            Aelem = Kelem + Melem

            a_builder.add(vert_indices, vert_indices, Aelem)

        A = a_builder.coo_matrix().tocsr().tocoo()

        # Assemble load vector
        b = np.zeros(n_nodes)

        for ei in range(n_elements):
            vert_indices = self.elements[ei, :]
            el_verts = self.nodes[vert_indices]
            centroid = np.mean(el_verts, axis=0)

            x0, x1, x2 = el_verts
            J = np.array([x1 - x0, x2 - x0]).T
            detJ = la.det(J)

            # Element load vector
            belem = f(centroid) * (detJ / 6.0) * np.ones(3)

            for i, vi in enumerate(vert_indices):
                b[vi] += belem[i]

        # Apply boundary conditions
        u0 = np.zeros(n_nodes)
        u0[self.is_g_boundary] = g(self.nodes[self.is_g_boundary].T)

        rhs = b - A @ u0

        # Enforce zero boundary on other boundaries
        rhs[self.is_boundary] = 0.0
        for k in range(A.nnz):
            i, j = A.row[k], A.col[k]
            if self.is_boundary[i]:
                A.data[k] = 1 if i == j else 0

        # Solve sparse linear system
        uhat = sla.spsolve(A.tocsr(), rhs)
        u = uhat + u0

        return uhat, A, rhs

    def _make_mesh(self, points: list) -> Tuple[np.ndarray, np.ndarray]:
        """Generate triangular mesh using meshpy

        Args:
            points: List of boundary points

        Returns:
            nodes: Node coordinates (n_nodes, 2)
            elements: Element connectivity (n_elements, 3)
        """
        points = copy.deepcopy(points)

        def round_trip_connect(start, end):
            return [(i, i + 1) for i in range(start, end)] + [(end, start)]

        facets = round_trip_connect(0, len(points) - 1)

        # Add circular hole in the center (for more interesting structure)
        circ_start = len(points)
        points.extend(
            (0.25 * np.cos(angle), 0.25 * np.sin(angle))
            for angle in np.linspace(0, 2 * np.pi, 30, endpoint=False)
        )
        facets.extend(round_trip_connect(circ_start, len(points) - 1))

        # Compute refinement factor based on target_nodes
        # Empirically calibrated: exponent=1.1 gives good balance
        # (exp=0.5 too conservative, exp=2.0 too aggressive)
        refinement_factor = (250.0 / self.target_nodes) ** 1.1

        def needs_refinement(vertices, area):
            bary = np.sum(np.array(vertices), axis=0) / 3
            base_area = 0.01 + la.norm(bary, np.inf) * 0.01
            max_area = base_area * refinement_factor
            return bool(area > max_area)

        # Build mesh
        info = triangle.MeshInfo()
        info.set_points(points)
        info.set_facets(facets)

        built_mesh = triangle.build(info, refinement_func=needs_refinement)
        nodes = np.array(built_mesh.points)
        elements = np.array(built_mesh.elements)

        return nodes, elements


class _MatrixBuilder:
    """Helper class for assembling sparse matrices"""

    def __init__(self):
        self.rows = []
        self.cols = []
        self.vals = []

    def add(self, rows, cols, submat):
        """Add local matrix to global matrix

        Args:
            rows: Global row indices
            cols: Global column indices
            submat: Local matrix
        """
        for i, ri in enumerate(rows):
            for j, cj in enumerate(cols):
                self.rows.append(ri)
                self.cols.append(cj)
                self.vals.append(submat[i, j])

    def coo_matrix(self):
        """Build COO sparse matrix"""
        return sparse.coo_matrix((self.vals, (self.rows, self.cols)))
