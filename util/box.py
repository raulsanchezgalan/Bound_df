import numpy as np

class Box:
    def __init__(self, bottom_left_vertex, side_lengths):
        self.bottom_left_vertex = np.array(bottom_left_vertex, dtype=float)
        self.side_lengths = np.array(side_lengths, dtype=float)
        self.grad_one_norm_at_midpoint = 0

        if self.bottom_left_vertex.shape != self.side_lengths.shape:
            raise ValueError("Dimension mismatch: vertex and side lengths must have the same dimension.")

        if np.any(self.side_lengths <= 0):
            raise ValueError("All side lengths must be positive.")

    def dimension(self):
        return len(self.bottom_left_vertex)

    def volume(self):
        return np.prod(self.side_lengths)

    def vertices(self):
        """Return all vertices of the n-dimensional box."""
        dims = self.dimension()
        corners = []
        for i in range(2 ** dims):
            corner = self.bottom_left_vertex.copy()
            for d in range(dims):
                if (i >> d) & 1:
                    corner[d] += self.side_lengths[d]
            corners.append(corner)
        return np.array(corners)

    def midpoint(self):
        """Return the midpoint of the box."""
        return self.bottom_left_vertex + self.side_lengths / 2

    def subdivide(self):
        """Return 2^n sub-boxes obtained by subdividing the box at its midpoint."""
        dims = self.dimension()
        mid = self.midpoint()
        half_side_lengths = self.side_lengths / 2

        sub_boxes = []
        for i in range(2 ** dims):
            new_vertex = self.bottom_left_vertex.copy()
            for d in range(dims):
                if (i >> d) & 1:
                    new_vertex[d] = mid[d]
            sub_boxes.append(Box(new_vertex, half_side_lengths))

        return sub_boxes

    def __repr__(self):
        return f"Box(bottom_left_vertex={self.bottom_left_vertex.tolist()}, side_lengths={self.side_lengths.tolist()})"