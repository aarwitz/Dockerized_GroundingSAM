import math

import cv2
import numpy as np



def min_in_image_area_rect(contour, image_size):
    """
    Like cv2.minAreaRect(), but only considers area within the image region when deciding which rectangle is smaller

    contour: list of points, assumed to all be within the image (i.e. 0 <= x <= width, 0 <= y <= height)
    image_size: tuple (width, height)

    Returns: (center_x, center_y), (width, height), angle_in_degrees

    angle_in_degrees denotes rotation clockwise when in a left-handed coordinate system

    Note: return format mimics cv2.minAreaRect()
    """

    image_width, image_height = image_size

    # This function tries to provide an improvement when a box is cut off by the edge of the image, like so:
    #
    # ----------------------------------
    # |   \      /                     |
    # |    \   /                       |
    # |     \/                         |
    # |                                |
    # |                                |
    # |                                |
    # |                                |
    # ----------------------------------
    #
    # The plain minimal area bounding rectangle tends to find a horizontal rectangle in these cases, which is not
    # really a good representation of the box's orientation.
    #
    # The idea is that we'll instead try the find the bounding rectangle with the minimum "in-image area", where
    # "im-image area" means the area of the intersection of the rectangle with the image rectangle.
    #
    # We'll need to build up some theory to justify the algorithm.
    #
    # First, let us define a "tight" bounding rectangle to be one which is smallest among bounding rectangles with the
    # same orientation. In other words, it is one where all four sides of the image contain a point from the original
    # point set. We can always choose our minimum in-image-area bounding rectangle to be tight, so we will focus on such
    # rectangles.
    #
    # Next, a theorem of Freeman and Shapira ("Determining the minimum-area encasing rectangle for an arbitrary closed
    # curve", Communications of the ACM, Vol 18 Issue 7, 1975) says that the plain minimum area bounding rectangle has a
    # side collinear to an edge of the convex hull (seen as a polygon) of the point set.
    #
    # I *think* (but haven't been able to prove) that this is true for the minimum in-image-area rectangles too. So
    # let's just take that on faith.
    #
    # The algorithm is then:
    #   - Step 1: Find the convex hull
    #   - Step 2: Find all tight rectangles with a side parallel to an edge of the convex hull
    #   - Step 3: Evaluate the in-image-area of all of those, choosing the smallest one.

    # Step 2: First we find the convex hull
    hull = cv2.convexHull(contour, returnPoints=True)

    # For Step 2, we will use a naive algorithm, were we simply:
    #   - Step 2.1: Loop over all edges of the convex hull
    #   - Step 2.2: For each such edge, find the tight rectangle parallel to that edge.
    # This is O(N^2). (N = number of vertices of convex hull). There's a smarter algorithm for step 2, rotated calipers,
    # that's O(N), but it's more complicated. See http://cgm.cs.mcgill.ca/~godfried/publications/calipers.pdf .
    # We should probably use it when we productize though, because O(N^2) can bring surprises.

    # We will now always use a left-handed coordinate system, where x increases to the right and y increases down.
    #
    # Suppose now that we have an edge direction of the convex hull, represented by a direction vector (dx, dy)
    # which we assume to be a unit vector.
    #
    # To more easily talk about computations on the tight rectangle, from now on we're going to use the term "left",
    # "right", "top", "bottom" relative to this direction, with "right" being in the direction (dx, dy). So
    #   - "right":  in direction ( dx,  dy)
    #   - "top":    in direction ( dy, -dx)
    #   - "left":   in direction (-dx, -dy)
    #   - "bottom": in direction (-dy,  dx)
    #

    num_pts = hull.shape[0]
    assert hull.shape == (num_pts, 1, 2)
    # for some weird reason cv2.convexHull points a rank-3 tensor, so we have to remove the middle dimension
    hull = hull[:, 0, :]

    # differences vectors between consecutive corners along convex hull, taken cyclically
    diffs = np.zeros((num_pts, 2))
    diffs[:-1, :] = hull[1:, :] - hull[:-1, :]
    diffs[-1, :] = hull[0, :] - hull[-1, :]

    norms = np.linalg.norm(diffs, axis=1)
    diffs_to_keep = norms > 1e-5  # points too close together can be ignored anyway
    diffs_kept = diffs[diffs_to_keep, :]
    norms_kept = norms[diffs_to_keep]

    dxs = diffs_kept[:, 0] / norms_kept
    dys = diffs_kept[:, 1] / norms_kept

    # To find the tight rectangle for a given direction, we first need to find the rightmost, topmost, leftmost, and
    # bottommost points of the convex hull. We'll define notions of "r", "t", "l", and "b" coordinates, by:
    #
    #   r-coordinate:  r = (x, y) dot (dx, dy)
    #   t-coordinate:  t = (x, y) dot (dy, -dx)
    #   l-coordinate:  l = (x, y) dot (-dx, -dy)
    #   b-coordinate:  b = (x, y) dot (-dy, dx)
    #
    # We'll then find the maximum r, t, l, and b coordinate values attained by the points in the convex hull.
    #
    #   r_max = maximum of (x, y) dot (dx, dy)
    #   t_max = maximum of (x, y) dot (dy, -dx)
    #   l_max = maximum of (x, y) dot (-dx, -dy)
    #   b_max = maximum (x, y) dot (-dy, dx)
    #

    xs = hull[:, 0]
    ys = hull[:, 1]

    # matrices of r-coordinates and t-coordinates for all directions (rows) and convex hull points (cols)
    r_vals = np.outer(dxs, xs) + np.outer(dys, ys)
    t_vals = np.outer(dys, xs) - np.outer(dxs, ys)

    r_maxes = np.amax(r_vals, axis=1)
    l_maxes = -np.amin(r_vals, axis=1)
    t_maxes = np.amax(t_vals, axis=1)
    b_maxes = -np.amin(t_vals, axis=1)

    # We'll then compute the corner of the tight rectangle. We'll denote these by:
    #   - top right corner:    (trx, try)
    #   - top left corner:     (tlx, tly)
    #   - bottom left corner:  (blx, bly)
    #   - bottom right corner: (brx, bry)
    #
    # To compute for example (trx, try), we need to solve the equation system:
    #
    #   trx * dx + try * dy = r_max  (top right corner has r-coordinate r_max)
    #   trx * dy - try * dx = t_max  (top right corner has t-coordinate t_max)
    #
    # Using the fact that dx*dx + dy*dy = 1, we can solve this pretty easily to get:
    #
    #   trx = dx * r_max + dy * t_max
    #   try = dy * r_max - dx * t_max
    #
    # Similar equations will give
    #
    #   tlx = -dx * l_max + dy * t_max
    #   tly = -dy * l_max - dx * t_max
    #
    #   blx = -dx * l_max - dy * b_max
    #   bly = -dy * l_max + dx * b_max
    #
    #   brx = dx * r_max - dy * b_max
    #   bry = dy * r_max + dx * b_max

    trxs = dxs * r_maxes + dys * t_maxes
    trys = dys * r_maxes - dxs * t_maxes

    tlxs = -dxs * l_maxes + dys * t_maxes
    tlys = -dys * l_maxes - dxs * t_maxes

    blxs = -dxs * l_maxes - dys * b_maxes
    blys = -dys * l_maxes + dxs * b_maxes

    brxs = dxs * r_maxes - dys * b_maxes
    brys = dys * r_maxes + dxs * b_maxes

    # Now we come to step 3, where we need to compute the area of the intersection of our tight rectangle with the
    # image. There are algorithms to do this generically, but our case is actually a lot easier.
    #
    # I claim that the only way in which the tight rectangle can "stick out" of the image is by little triangles, like
    # this:
    #
    #           . P
    #          /|\
    #        /  | \
    # -----/----.--\--------------------
    # |  /      B   \                  |
    # |  \           \ Q               |
    # |   \         /                  |
    # |    \      /                    |
    # |     \   /                      |
    # |      \/                        |
    # |                                |
    # |                                |
    # ----------------------------------
    #
    # The reason is that all other ways requires a whole side of the tight rectangle to be outside the image, which
    # (because each side needs to contain a point from the original set) contradicts the assumption that all points
    # lied inside the image.
    #
    # Now we can have multiple triangles sticking out, but at most four (one for each of the four sides of the image).
    # So we can just compute the area inside the image as the area of the rectangle, minus a term for each of the
    # four possible triangles.
    #
    # To compute the area of the triangle, refer to the figure above, and let h denote the vertical height of P above
    # the top side of the image (i.e. the distance PB)
    #
    # Let alpha denote the angle that the side PQ makes with the vertical, i.e. the angle BPQ
    #
    # The the area of the triangle that sticks out is just
    #
    #   Area(triangle) = (1/2) * (h / sin(alpha)) * (h / cos(alpha))
    #                  = h^2 / (2 * sin(alpha) * cos(alpha))
    #
    # A little thought will reveal that sin(alpha) and cos(alpha) are just abs(dx) and abs(dy) in some order, so
    #
    #   sin(alpha)*cos(alpha) = abs(dx * dy)
    #
    # So to compute the area of the triangles, we just need to sum the horizontal/vertical distances by which the
    # corners of the rectangle "stick out" of the image, and divide by 2 * abs(dx * dy)

    def stick_out_dist_sq(xcoords, ycoords):
        return (
            np.abs(xcoords - np.clip(xcoords, 0, image_width)) ** 2
            + np.abs(ycoords - np.clip(ycoords, 0, image_height)) ** 2
        )

    total_stick_out_dist_sq = (
        stick_out_dist_sq(trxs, trys)
        + stick_out_dist_sq(tlxs, tlys)
        + stick_out_dist_sq(blxs, blys)
        + stick_out_dist_sq(brxs, brys)
    )

    triangle_areas = 0.5 * total_stick_out_dist_sq / np.maximum(1e-6, np.abs(dxs * dys))

    rectangle_areas = np.abs((r_maxes + l_maxes) * (t_maxes + b_maxes))

    rectangle_areas_in_image = rectangle_areas - triangle_areas

    # finally, we just need to find the best rectangle
    best_idx = np.argmin(rectangle_areas_in_image)

    # now, convert to center-size-angle format as returned by cv2.minAreaRect
    center_x = 0.5 * (tlxs[best_idx] + brxs[best_idx])
    center_y = 0.5 * (tlys[best_idx] + brys[best_idx])

    width = abs(r_maxes[best_idx] + l_maxes[best_idx])
    height = abs(t_maxes[best_idx] + b_maxes[best_idx])

    angle = math.atan2(dys[best_idx], dxs[best_idx])

    # for some reason, opencv's minAreaRect outputs degrees, so I guess we should too
    angle_in_degrees = angle / math.pi * 180.0

    # fully enclose all points within the box, including the right and bottom
    center_x += 0.5
    center_y += 0.5
    width += 1
    height += 1

    return (center_x, center_y), (width, height), angle_in_degrees
