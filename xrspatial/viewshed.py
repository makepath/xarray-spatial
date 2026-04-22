from collections import OrderedDict
from math import atan, atan2, fabs
from math import pi as PI
from math import sqrt
from typing import Union

import numpy as np
import xarray

from .gpu_rtx import has_rtx
from .utils import (_validate_raster, has_cuda_and_cupy, has_dask_array,
                    is_cupy_array, is_cupy_backed, is_dask_cupy, ngjit)

E_ROW_ID = 0
E_COL_ID = 1
E_TYPE_ID = 2

E_ANG_ID = 3
E_ELEV_0 = 4
E_ELEV_1 = 5
E_ELEV_2 = 6

AE_ANG_ID = 0
AE_ELEV_0 = 1
AE_ELEV_1 = 2
AE_ELEV_2 = 3

TN_KEY_ID = 0
TN_GRAD_0 = 1
TN_GRAD_1 = 2
TN_GRAD_2 = 3
TN_ANG_0 = 4
TN_ANG_1 = 5
TN_ANG_2 = 6
TN_MAX_GRAD_ID = 7

TN_COLOR_ID = 0
TN_LEFT_ID = 1
TN_RIGHT_ID = 2
TN_PARENT_ID = 3

NIL_ID = -1

# view options default values
OBS_ELEV = 0
TARGET_ELEV = 0

# if a cell is invisible, its value is set to -1
INVISIBLE = -1

# color of node in red-black Tree
RB_RED = 0
RB_BLACK = 1

# event type
ENTERING_EVENT = 1
EXITING_EVENT = -1
CENTER_EVENT = 0

# this value is returned by findMaxValueWithinDist() if there is no key within
# that distance
SMALLEST_GRAD = -9999999999999999999999.0


@ngjit
def _compare(a, b):
    if a < b:
        return -1
    if a > b:
        return 1
    return 0


@ngjit
def _find_value_min_value(tree_vals, node_id):
    return min(tree_vals[node_id][TN_GRAD_0],
               tree_vals[node_id][TN_GRAD_1],
               tree_vals[node_id][TN_GRAD_2])


def _print_tree(status_struct):
    for i in range(len(status_struct)):
        print(i, status_struct[i][0])


def _print_tv(tv):
    print('key=', tv[TN_KEY_ID],
          'grad=', tv[TN_GRAD_0], tv[TN_GRAD_1], tv[TN_GRAD_2],
          'ang=', tv[TN_ANG_0], tv[TN_ANG_1], tv[TN_ANG_2],
          'max_grad=', tv[TN_MAX_GRAD_ID])
    return


@ngjit
def _create_tree_nodes(tree_vals, tree_nodes, x, val, color=RB_RED):
    # Create a TreeNode using given TreeValue

    # every node has null nodes as children initially, create one such object
    # for easy management

    tree_vals[x][TN_KEY_ID] = val[TN_KEY_ID]
    tree_vals[x][TN_GRAD_0] = val[TN_GRAD_0]
    tree_vals[x][TN_GRAD_1] = val[TN_GRAD_1]
    tree_vals[x][TN_GRAD_2] = val[TN_GRAD_2]
    tree_vals[x][TN_ANG_0] = val[TN_ANG_0]
    tree_vals[x][TN_ANG_1] = val[TN_ANG_1]
    tree_vals[x][TN_ANG_2] = val[TN_ANG_2]
    tree_vals[x][TN_MAX_GRAD_ID] = SMALLEST_GRAD

    tree_nodes[x][TN_COLOR_ID] = color
    tree_nodes[x][TN_LEFT_ID] = NIL_ID
    tree_nodes[x][TN_RIGHT_ID] = NIL_ID
    tree_nodes[x][TN_PARENT_ID] = NIL_ID
    return


@ngjit
def _tree_minimum(tree_nodes, x):
    while tree_nodes[x][TN_LEFT_ID] != NIL_ID:
        x = tree_nodes[x][TN_LEFT_ID]
    return x


# function used by deletion
@ngjit
def _tree_successor(tree_nodes, x):
    # Find the highest successor of a node in the tree

    if tree_nodes[x][TN_RIGHT_ID] != NIL_ID:
        return _tree_minimum(tree_nodes, tree_nodes[x][TN_RIGHT_ID])

    y = tree_nodes[x][TN_PARENT_ID]
    while y != NIL_ID and x == tree_nodes[y][TN_RIGHT_ID]:
        x = y
        if tree_nodes[y][TN_PARENT_ID] == NIL_ID:
            return y
        y = tree_nodes[y][TN_PARENT_ID]
    return y


@ngjit
def _find_max_value(node_value):
    # Find the max value in the given tree.
    return node_value[TN_MAX_GRAD_ID]


@ngjit
def _left_rotate(tree_vals, tree_nodes, root, x):
    # A utility function to left rotate subtree rooted with a node.

    y = tree_nodes[x][TN_RIGHT_ID]

    # fix x
    x_left = tree_nodes[x][TN_LEFT_ID]
    y_left = tree_nodes[y][TN_LEFT_ID]
    if tree_vals[x_left][TN_MAX_GRAD_ID] > tree_vals[y_left][TN_MAX_GRAD_ID]:
        tmp_max = tree_vals[x_left][TN_MAX_GRAD_ID]
    else:
        tmp_max = tree_vals[y_left][TN_MAX_GRAD_ID]

    min_value = _find_value_min_value(tree_vals, x)
    if tmp_max > min_value:
        tree_vals[x][TN_MAX_GRAD_ID] = tmp_max
    else:
        tree_vals[x][TN_MAX_GRAD_ID] = min_value

    # fix y
    y_right = tree_nodes[y][TN_RIGHT_ID]
    if tree_vals[x][TN_MAX_GRAD_ID] > tree_vals[y_right][TN_MAX_GRAD_ID]:
        tmp_max = tree_vals[x][TN_MAX_GRAD_ID]
    else:
        tmp_max = tree_vals[y_right][TN_MAX_GRAD_ID]

    min_value = _find_value_min_value(tree_vals, y)
    if tmp_max > min_value:
        tree_vals[y][TN_MAX_GRAD_ID] = tmp_max
    else:
        tree_vals[y][TN_MAX_GRAD_ID] = min_value

    # left rotation
    # see pseudo code on page 278 CLRS

    # turn y's left subtree into x's right subtree
    tree_nodes[x][TN_RIGHT_ID] = tree_nodes[y][TN_LEFT_ID]
    y_left = tree_nodes[y][TN_LEFT_ID]
    tree_nodes[y_left][TN_PARENT_ID] = x
    # link x's parent to y
    tree_nodes[y][TN_PARENT_ID] = tree_nodes[x][TN_PARENT_ID]

    if tree_nodes[x][TN_PARENT_ID] == NIL_ID:
        root = y
    else:
        x_parent = tree_nodes[x][TN_PARENT_ID]
        if x == tree_nodes[x_parent][TN_LEFT_ID]:
            tree_nodes[x_parent][TN_LEFT_ID] = y
        else:
            tree_nodes[x_parent][TN_RIGHT_ID] = y

    tree_nodes[y][TN_LEFT_ID] = x
    tree_nodes[x][TN_PARENT_ID] = y
    return root


@ngjit
def _right_rotate(tree_vals, tree_nodes, root, y):
    # A utility function to right rotate subtree rooted with a node.

    x = tree_nodes[y][TN_LEFT_ID]

    # fix y
    x_right = tree_nodes[x][TN_RIGHT_ID]
    y_right = tree_nodes[y][TN_RIGHT_ID]
    if tree_vals[x_right][TN_MAX_GRAD_ID] > tree_vals[y_right][TN_MAX_GRAD_ID]:
        tmp_max = tree_vals[x_right][TN_MAX_GRAD_ID]
    else:
        tmp_max = tree_vals[y_right][TN_MAX_GRAD_ID]

    min_value = _find_value_min_value(tree_vals, y)
    if tmp_max > min_value:
        tree_vals[y][TN_MAX_GRAD_ID] = tmp_max
    else:
        tree_vals[y][TN_MAX_GRAD_ID] = min_value

    # fix x
    x_left = tree_nodes[x][TN_LEFT_ID]
    if tree_vals[x_left][TN_MAX_GRAD_ID] > tree_vals[y][TN_MAX_GRAD_ID]:
        tmp_max = tree_vals[x_left][TN_MAX_GRAD_ID]
    else:
        tmp_max = tree_vals[y][TN_MAX_GRAD_ID]

    min_value = _find_value_min_value(tree_vals, x)
    if tmp_max > min_value:
        tree_vals[x][TN_MAX_GRAD_ID] = tmp_max
    else:
        tree_vals[x][TN_MAX_GRAD_ID] = min_value

    # rotation
    tree_nodes[y][TN_LEFT_ID] = tree_nodes[x][TN_RIGHT_ID]
    x_right = tree_nodes[x][TN_RIGHT_ID]
    tree_nodes[x_right][TN_PARENT_ID] = y

    tree_nodes[x][TN_PARENT_ID] = tree_nodes[y][TN_PARENT_ID]

    if tree_nodes[y][TN_PARENT_ID] == NIL_ID:
        root = x
    else:
        y_parent = tree_nodes[y][TN_PARENT_ID]
        if tree_nodes[y_parent][TN_LEFT_ID] == y:
            tree_nodes[y_parent][TN_LEFT_ID] = x
        else:
            tree_nodes[y_parent][TN_RIGHT_ID] = x

    tree_nodes[x][TN_RIGHT_ID] = y
    tree_nodes[y][TN_PARENT_ID] = x
    return root


@ngjit
def _rb_insert_fixup(tree_vals, tree_nodes, root, z):
    # Fix red-black tree after insertion. This may change the root pointer.

    # see pseudocode on page 281 in CLRS
    z_parent = tree_nodes[z][TN_PARENT_ID]
    while tree_nodes[z_parent][TN_COLOR_ID] == RB_RED:
        z_parent_parent = tree_nodes[z_parent][TN_PARENT_ID]
        n1 = tree_nodes[z][TN_PARENT_ID]
        n2 = tree_nodes[z_parent_parent][TN_LEFT_ID]
        if n1 == n2:
            y = tree_nodes[z_parent_parent][TN_RIGHT_ID]
            if tree_nodes[y][TN_COLOR_ID] == RB_RED:
                # case 1
                tree_nodes[z_parent][TN_COLOR_ID] = RB_BLACK
                tree_nodes[y][TN_COLOR_ID] = RB_BLACK
                tree_nodes[z_parent_parent][TN_COLOR_ID] = RB_RED
                # re assignment for z
                z = z_parent_parent
            else:
                if z == tree_nodes[z_parent][TN_RIGHT_ID]:
                    # case 2
                    z = z_parent
                    # convert case 2 to case 3
                    root = _left_rotate(tree_vals, tree_nodes, root, z)
                # case 3
                z_parent = tree_nodes[z][TN_PARENT_ID]
                z_parent_parent = tree_nodes[z_parent][TN_PARENT_ID]
                tree_nodes[z_parent][TN_COLOR_ID] = RB_BLACK
                tree_nodes[z_parent_parent][TN_COLOR_ID] = RB_RED
                root = _right_rotate(tree_vals, tree_nodes, root,
                                     z_parent_parent)

        else:
            # (z->parent == z->parent->parent->right)
            y = tree_nodes[z_parent_parent][TN_LEFT_ID]
            if tree_nodes[y][TN_COLOR_ID] == RB_RED:
                # case 1
                tree_nodes[z_parent][TN_COLOR_ID] = RB_BLACK
                tree_nodes[y][TN_COLOR_ID] = RB_BLACK
                tree_nodes[z_parent_parent][TN_COLOR_ID] = RB_RED
                z = z_parent_parent
            else:
                if z == tree_nodes[z_parent][TN_LEFT_ID]:
                    # case 2
                    z = z_parent
                    # convert case 2 to case 3
                    root = _right_rotate(tree_vals, tree_nodes, root, z)
                # case 3
                z_parent = tree_nodes[z][TN_PARENT_ID]
                z_parent_parent = tree_nodes[z_parent][TN_PARENT_ID]
                tree_nodes[z_parent][TN_COLOR_ID] = RB_BLACK
                tree_nodes[z_parent_parent][TN_COLOR_ID] = RB_RED
                root = _left_rotate(tree_vals, tree_nodes, root,
                                    z_parent_parent)

        z_parent = tree_nodes[z][TN_PARENT_ID]

    tree_nodes[root][TN_COLOR_ID] = RB_BLACK
    return root


@ngjit
def _insert_into_tree(tree_vals, tree_nodes, root, node_id, value):
    # Create node and insert it into the tree
    cur_node = root

    if _compare(value[TN_KEY_ID], tree_vals[cur_node][TN_KEY_ID]) == -1:
        next_node = tree_nodes[cur_node][TN_LEFT_ID]
    else:
        next_node = tree_nodes[cur_node][TN_RIGHT_ID]

    while next_node != NIL_ID:
        cur_node = next_node
        if _compare(value[TN_KEY_ID], tree_vals[cur_node][TN_KEY_ID]) == -1:
            next_node = tree_nodes[cur_node][TN_LEFT_ID]
        else:
            next_node = tree_nodes[cur_node][TN_RIGHT_ID]

    # create a new node
    #   and place it at the right place
    #   created node is RED by default
    _create_tree_nodes(tree_vals, tree_nodes, node_id, value, color=RB_RED)
    next_node = node_id

    tree_nodes[next_node][TN_PARENT_ID] = cur_node

    if _compare(value[TN_KEY_ID], tree_vals[cur_node][TN_KEY_ID]) == -1:
        tree_nodes[cur_node][TN_LEFT_ID] = next_node
    else:
        tree_nodes[cur_node][TN_RIGHT_ID] = next_node

    inserted = next_node

    # update augmented maxGradient
    tree_vals[next_node][TN_MAX_GRAD_ID] =\
        _find_value_min_value(tree_vals, next_node)

    while tree_nodes[next_node][TN_PARENT_ID] != NIL_ID:
        next_parent = tree_nodes[next_node][TN_PARENT_ID]
        if tree_vals[next_parent][TN_MAX_GRAD_ID] <\
                tree_vals[next_node][TN_MAX_GRAD_ID]:
            tree_vals[next_parent][TN_MAX_GRAD_ID] =\
                tree_vals[next_node][TN_MAX_GRAD_ID]

        if tree_vals[next_parent][TN_MAX_GRAD_ID] >\
                tree_vals[next_node][TN_MAX_GRAD_ID]:
            break

        next_node = next_parent

    # fix rb tree after insertion
    root = _rb_insert_fixup(tree_vals, tree_nodes, root, inserted)
    return root


@ngjit
def _search_for_node(tree_vals, tree_nodes, root, key):
    # Search for a node with a given key.
    cur_node = root
    while cur_node != NIL_ID and \
            _compare(key, tree_vals[cur_node][TN_KEY_ID]) != 0:

        if _compare(key, tree_vals[cur_node][TN_KEY_ID]) == -1:
            cur_node = tree_nodes[cur_node][TN_LEFT_ID]
        else:
            cur_node = tree_nodes[cur_node][TN_RIGHT_ID]

    return cur_node


# The following is designed for viewshed's algorithm
@ngjit
def _find_max_value_within_key(tree_vals, tree_nodes, root,
                               max_key, ang, gradient):
    key_node = _search_for_node(tree_vals, tree_nodes, root, max_key)
    if key_node == NIL_ID:
        # there is no point in the structure with key < maxKey */
        return SMALLEST_GRAD

    cur_node = key_node
    max = SMALLEST_GRAD
    while tree_nodes[cur_node][TN_PARENT_ID] != NIL_ID:
        cur_parent = tree_nodes[cur_node][TN_PARENT_ID]
        if cur_node == tree_nodes[cur_parent][TN_RIGHT_ID]:
            # its the right node of its parent
            cur_parent_left = tree_nodes[cur_parent][TN_LEFT_ID]
            tmp_max = _find_max_value(tree_vals[cur_parent_left])
            if tmp_max > max:
                max = tmp_max

            min_value = _find_value_min_value(tree_vals, cur_parent)
            if min_value > max:
                max = min_value

        cur_node = cur_parent

    if max > gradient:
        return max

    # traverse all nodes with smaller distance
    max = SMALLEST_GRAD
    cur_node = key_node
    while cur_node != NIL_ID:
        check_me = False
        if tree_vals[cur_node][TN_ANG_0] <= ang\
                <= tree_vals[cur_node][TN_ANG_2]:
            check_me = True
        if (not check_me) and tree_vals[cur_node][TN_KEY_ID] > 0:
            print('Angles outside angle')

        if tree_vals[cur_node][TN_KEY_ID] > max_key:
            raise ValueError("current dist too large ")

        if check_me and cur_node != key_node:

            if ang < tree_vals[cur_node][TN_ANG_1]:
                cur_grad = tree_vals[cur_node][TN_GRAD_1] \
                    + (tree_vals[cur_node][TN_GRAD_0]
                       - tree_vals[cur_node][TN_GRAD_1]) \
                    * (tree_vals[cur_node][TN_ANG_1] - ang) \
                    / (tree_vals[cur_node][TN_ANG_1]
                       - tree_vals[cur_node][TN_ANG_0])

            elif ang > tree_vals[cur_node][TN_ANG_1]:
                cur_grad = tree_vals[cur_node][TN_GRAD_1] \
                    + (tree_vals[cur_node][TN_GRAD_2]
                       - tree_vals[cur_node][TN_GRAD_1]) \
                    * (ang - tree_vals[cur_node][TN_ANG_1]) \
                    / (tree_vals[cur_node][TN_ANG_2]
                       - tree_vals[cur_node][TN_ANG_1])
            else:
                cur_grad = tree_vals[cur_node][TN_GRAD_1]

            if cur_grad > max:
                max = cur_grad

            if max > gradient:
                return max

        # get next smaller key
        if tree_nodes[cur_node][TN_LEFT_ID] != NIL_ID:
            cur_node = tree_nodes[cur_node][TN_LEFT_ID]
            while tree_nodes[cur_node][TN_RIGHT_ID] != NIL_ID:
                cur_node = tree_nodes[cur_node][TN_RIGHT_ID]
        else:
            # at smallest item in this branch, go back up
            last_node = cur_node
            cur_node = tree_nodes[cur_node][TN_PARENT_ID]
            while cur_node != NIL_ID and \
                    last_node == tree_nodes[cur_node][TN_LEFT_ID]:
                last_node = cur_node
                cur_node = tree_nodes[cur_node][TN_PARENT_ID]

    return max


@ngjit
def _rb_delete_fixup(tree_vals, tree_nodes, root, x):
    # Fix the red-black tree after deletion.
    # This may change the root pointer.

    while x != root and tree_nodes[x][TN_COLOR_ID] == RB_BLACK:
        x_parent = tree_nodes[x][TN_PARENT_ID]
        if x == tree_nodes[x_parent][TN_LEFT_ID]:
            w = tree_nodes[x_parent][TN_RIGHT_ID]
            if tree_nodes[w][TN_COLOR_ID] == RB_RED:
                tree_nodes[w][TN_COLOR_ID] = RB_BLACK
                tree_nodes[x_parent][TN_COLOR_ID] = RB_RED
                root = _left_rotate(tree_vals, tree_nodes, root, x_parent)
                w = tree_nodes[x_parent][TN_RIGHT_ID]

            if w == NIL_ID:
                x = tree_nodes[x][TN_PARENT_ID]
                continue

            w_left = tree_nodes[w][TN_LEFT_ID]
            w_right = tree_nodes[w][TN_RIGHT_ID]
            if tree_nodes[w_left][TN_COLOR_ID] == RB_BLACK and \
                    tree_nodes[w_right][TN_COLOR_ID] == RB_BLACK:
                tree_nodes[w][TN_COLOR_ID] = RB_RED
                x = tree_nodes[x][TN_PARENT_ID]
            else:
                if tree_nodes[w_right][TN_COLOR_ID] == RB_BLACK:
                    tree_nodes[w_left][TN_COLOR_ID] = RB_BLACK
                    tree_nodes[w][TN_COLOR_ID] = RB_RED
                    root = _right_rotate(tree_vals, tree_nodes, root, w)
                    x_parent = tree_nodes[x][TN_PARENT_ID]
                    w = tree_nodes[x_parent][TN_RIGHT_ID]

                x_parent = tree_nodes[x][TN_PARENT_ID]
                w_right = tree_nodes[w][TN_RIGHT_ID]
                tree_nodes[w][TN_COLOR_ID] = tree_nodes[x_parent][TN_COLOR_ID]
                tree_nodes[x_parent][TN_COLOR_ID] = RB_BLACK
                tree_nodes[w_right][TN_COLOR_ID] = RB_BLACK
                root = _left_rotate(tree_vals, tree_nodes, root, x_parent)
                x = root
        else:
            # x == x.parent.right
            x_parent = tree_nodes[x][TN_PARENT_ID]
            w = tree_nodes[x_parent][TN_LEFT_ID]
            if tree_nodes[w][TN_COLOR_ID] == RB_RED:
                tree_nodes[w][TN_COLOR_ID] = RB_BLACK
                tree_nodes[x_parent][TN_COLOR_ID] = RB_RED
                root = _right_rotate(tree_vals, tree_nodes, root, x_parent)
                w = tree_nodes[x_parent][TN_LEFT_ID]

            if w == NIL_ID:
                x = x_parent
                continue

            w_left = tree_nodes[w][TN_LEFT_ID]
            w_right = tree_nodes[w][TN_RIGHT_ID]
            # do we need re-assignment here? No changes has been made for x?
            x_parent = tree_nodes[x][TN_PARENT_ID]
            if tree_nodes[w_right][TN_COLOR_ID] == RB_BLACK and \
                    tree_nodes[w_left][TN_COLOR_ID] == RB_BLACK:
                tree_nodes[w][TN_COLOR_ID] = RB_RED
                x = x_parent
            else:
                if tree_nodes[w_left][TN_COLOR_ID] == RB_BLACK:
                    tree_nodes[w_right][TN_COLOR_ID] = RB_BLACK
                    tree_nodes[w][TN_COLOR_ID] = RB_RED
                    root = _left_rotate(tree_vals, tree_nodes, root, w)
                    w = tree_nodes[x_parent][TN_LEFT_ID]
                tree_nodes[w][TN_COLOR_ID] = tree_nodes[x_parent][TN_COLOR_ID]
                tree_nodes[x_parent][TN_COLOR_ID] = RB_BLACK
                w_left = tree_nodes[w][TN_LEFT_ID]
                tree_nodes[w_left][TN_COLOR_ID] = RB_BLACK
                root = _right_rotate(tree_vals, tree_nodes, root, x_parent)
                x = root

    tree_nodes[x][TN_COLOR_ID] = RB_BLACK
    return root


@ngjit
def _delete_from_tree(tree_vals, tree_nodes, root, key):
    # Delete the node out of the tree. This may change the root pointer.

    z = _search_for_node(tree_vals, tree_nodes, root, key)

    if z == NIL_ID:
        # node to delete is not found
        raise ValueError("node not found")

    # 1-3
    if tree_nodes[z][TN_LEFT_ID] == NIL_ID or\
            tree_nodes[z][TN_RIGHT_ID] == NIL_ID:
        y = z
    else:
        y = _tree_successor(tree_nodes, z)

    if y == NIL_ID:
        raise ValueError("successor not found")

    deleted = y

    # 4-6
    if tree_nodes[y][TN_LEFT_ID] != NIL_ID:
        x = tree_nodes[y][TN_LEFT_ID]
    else:
        x = tree_nodes[y][TN_RIGHT_ID]

    # 7
    tree_nodes[x][TN_PARENT_ID] = tree_nodes[y][TN_PARENT_ID]

    # 8-12
    if tree_nodes[y][TN_PARENT_ID] == NIL_ID:
        root = x
        # augmentation to be fixed
        to_fix = root
    else:
        y_parent = tree_nodes[y][TN_PARENT_ID]
        if y == tree_nodes[y_parent][TN_LEFT_ID]:
            tree_nodes[y_parent][TN_LEFT_ID] = x
        else:
            tree_nodes[y_parent][TN_RIGHT_ID] = x
        # augmentation to be fixed
        to_fix = y_parent

    # fix augmentation for removing y
    cur_node = y

    while tree_nodes[cur_node][TN_PARENT_ID] != NIL_ID:
        cur_parent = tree_nodes[cur_node][TN_PARENT_ID]
        if tree_vals[cur_parent][TN_MAX_GRAD_ID] == \
                _find_value_min_value(tree_vals, y):
            cur_parent_left = tree_nodes[cur_parent][TN_LEFT_ID]
            cur_parent_right = tree_nodes[cur_parent][TN_RIGHT_ID]
            left = _find_max_value(tree_vals[cur_parent_left])
            right = _find_max_value(tree_vals[cur_parent_right])

            if left > right:
                tree_vals[cur_parent][TN_MAX_GRAD_ID] = left
            else:
                tree_vals[cur_parent][TN_MAX_GRAD_ID] = right

            min_value = _find_value_min_value(tree_vals, cur_parent)
            if min_value > tree_vals[cur_parent][TN_MAX_GRAD_ID]:
                tree_vals[cur_parent][TN_MAX_GRAD_ID] = min_value

        else:
            break

        cur_node = cur_parent

    # fix augmentation for x
    to_fix_left = tree_nodes[to_fix][TN_LEFT_ID]
    to_fix_right = tree_nodes[to_fix][TN_RIGHT_ID]
    if tree_vals[to_fix_left][TN_MAX_GRAD_ID] >\
            tree_vals[to_fix_right][TN_MAX_GRAD_ID]:
        tmp_max = tree_vals[to_fix_left][TN_MAX_GRAD_ID]
    else:
        tmp_max = tree_vals[to_fix_right][TN_MAX_GRAD_ID]

    min_value = _find_value_min_value(tree_vals, to_fix)
    if tmp_max > min_value:
        tree_vals[to_fix][TN_MAX_GRAD_ID] = tmp_max
    else:
        tree_vals[to_fix][TN_MAX_GRAD_ID] = min_value

    # 13-15
    if y != NIL_ID and y != z:
        z_gradient = _find_value_min_value(tree_vals, z)
        tree_vals[z][TN_KEY_ID] = tree_vals[y][TN_KEY_ID]
        tree_vals[z][TN_GRAD_0] = tree_vals[y][TN_GRAD_0]
        tree_vals[z][TN_GRAD_1] = tree_vals[y][TN_GRAD_1]
        tree_vals[z][TN_GRAD_2] = tree_vals[y][TN_GRAD_2]
        tree_vals[z][TN_ANG_0] = tree_vals[y][TN_ANG_0]
        tree_vals[z][TN_ANG_1] = tree_vals[y][TN_ANG_1]
        tree_vals[z][TN_ANG_2] = tree_vals[y][TN_ANG_2]

        to_fix = z
        # fix augmentation
        to_fix_left = tree_nodes[to_fix][TN_LEFT_ID]
        to_fix_right = tree_nodes[to_fix][TN_RIGHT_ID]
        if tree_vals[to_fix_left][TN_MAX_GRAD_ID] > \
                tree_vals[to_fix_right][TN_MAX_GRAD_ID]:
            tmp_max = tree_vals[to_fix_left][TN_MAX_GRAD_ID]
        else:
            tmp_max = tree_vals[to_fix_right][TN_MAX_GRAD_ID]

        min_value = _find_value_min_value(tree_vals, to_fix)
        if tmp_max > min_value:
            tree_vals[to_fix][TN_MAX_GRAD_ID] = tmp_max
        else:
            tree_vals[to_fix][TN_MAX_GRAD_ID] = min_value

        while tree_nodes[z][TN_PARENT_ID] != NIL_ID:
            z_parent = tree_nodes[z][TN_PARENT_ID]
            if tree_vals[z_parent][TN_MAX_GRAD_ID] == z_gradient:
                z_parent_left = tree_nodes[z_parent][TN_LEFT_ID]
                z_parent_right = tree_nodes[z_parent][TN_RIGHT_ID]
                x_parent = tree_nodes[x][TN_PARENT_ID]
                x_parent_right = tree_nodes[x_parent][TN_RIGHT_ID]
                if _find_value_min_value(tree_vals, z_parent) != z_gradient\
                    and \
                    not (tree_vals[z_parent_left][TN_MAX_GRAD_ID] == z_gradient
                         and
                         tree_vals[x_parent_right][TN_MAX_GRAD_ID] ==
                         z_gradient):

                    left = _find_max_value(tree_vals[z_parent_left])
                    right = _find_max_value(tree_vals[z_parent_right])

                    if left > right:
                        tree_vals[z_parent][TN_MAX_GRAD_ID] = left
                    else:
                        tree_vals[z_parent][TN_MAX_GRAD_ID] = right

                    min_value = _find_value_min_value(tree_vals, z_parent)
                    if min_value > tree_vals[z_parent][TN_MAX_GRAD_ID]:
                        tree_vals[z_parent][TN_MAX_GRAD_ID] = min_value

            else:
                if tree_vals[z][TN_MAX_GRAD_ID] >\
                        tree_vals[z_parent][TN_MAX_GRAD_ID]:
                    tree_vals[z_parent][TN_MAX_GRAD_ID] =\
                        tree_vals[z][TN_MAX_GRAD_ID]

            z = z_parent

    # 16-17
    if tree_nodes[y][TN_COLOR_ID] == RB_BLACK and x != NIL_ID:
        root = _rb_delete_fixup(tree_vals, tree_nodes, root, x)

    # 18
    return root, deleted


def _print_status_node(sn, row, col):
    print("row=", row, "col=", col, "dist_to_viewpoint=",
          sn[TN_KEY_ID], "grad=", sn[TN_GRAD_0], sn[TN_GRAD_1], sn[TN_GRAD_2],
          "ang=", sn[TN_ANG_0], sn[TN_ANG_1], sn[TN_ANG_2])
    return


@ngjit
def _max_grad_in_status_struct(tree_vals, tree_nodes, root,
                               distance, angle, gradient):
    # Find the node with max Gradient within the distance (from vp)
    # Note: if there is nothing in the status structure,
    #         it means this cell is VISIBLE

    if root == NIL_ID:
        return SMALLEST_GRAD

    # it is also possible that the status structure is not empty, but
    # there are no events with key < dist ---in this case it returns
    # SMALLEST_GRAD;

    # find max within the max key

    return _find_max_value_within_key(tree_vals, tree_nodes, root,
                                      distance, angle, gradient)


@ngjit
def _col_to_east(col, window_west, window_ew_res):
    # Column to easting.
    # Converts a column relative to a window to an east coordinate.
    return window_west + col * window_ew_res


@ngjit
def _row_to_north(row, window_north, window_ns_res):
    # Row to northing.
    # Converts a row relative to a window to an north coordinate.
    return window_north - row * window_ns_res


@ngjit
def _radian(x):
    # Convert degree into radian.
    return x * PI / 180.0


@ngjit
def _hypot(x, y):
    return sqrt(x * x + y * y)


@ngjit
def _g_distance(e1, n1, e2, n2):
    # Computes the distance, in meters, from (x1, y1) to (x2, y2)

    # assume meter grid
    factor = 1.0
    return factor * _hypot(e1 - e2, n1 - n2)


@ngjit
def _set_visibility(visibility_grid, i, j, value):
    visibility_grid[i][j] = value
    return


@ngjit
def _calculate_event_row_col(event_type, event_row, event_col,
                             viewpoint_row, viewpoint_col):
    # Calculate the neighbouring of the given event.
    x = 0
    y = 0
    if event_type == CENTER_EVENT:
        raise ValueError("_calculate_event_row_col() must not be called for "
                         "CENTER events")

    if event_row < viewpoint_row and event_col < viewpoint_col:
        # first quadrant
        if event_type == ENTERING_EVENT:
            # if it is ENTERING_EVENT
            y = event_row - 1
            x = event_col + 1
        else:
            # if it is EXITING_EVENT
            y = event_row + 1
            x = event_col - 1

    elif event_col == viewpoint_col and event_row < viewpoint_row:
        # between the first and second quadrant
        if event_type == ENTERING_EVENT:
            # if it is ENTERING_EVENT
            y = event_row + 1
            x = event_col + 1
        else:
            # if it is EXITING_EVENT
            y = event_row + 1
            x = event_col - 1

    elif event_col > viewpoint_col and event_row < viewpoint_row:
        # second quadrant
        if event_type == ENTERING_EVENT:
            # if it is ENTERING_EVENT
            y = event_row + 1
            x = event_col + 1
        else:
            # if it is EXITING_EVENT
            y = event_row - 1
            x = event_col - 1

    elif event_col > viewpoint_col and event_row == viewpoint_row:
        # between the second and forth quadrant
        if event_type == ENTERING_EVENT:
            # if it is ENTERING_EVENT
            y = event_row + 1
            x = event_col - 1
        else:
            # if it is EXITING_EVENT
            y = event_row - 1
            x = event_col - 1

    elif event_col > viewpoint_col and event_row > viewpoint_row:
        # forth quadrant
        if event_type == ENTERING_EVENT:
            # if it is ENTERING_EVENT
            y = event_row + 1
            x = event_col - 1
        else:
            # if it is EXITING_EVENT
            y = event_row - 1
            x = event_col + 1

    elif event_col == viewpoint_col and event_row > viewpoint_row:
        # between the third and fourth quadrant
        if event_type == ENTERING_EVENT:
            # if it is ENTERING_EVENT
            y = event_row - 1
            x = event_col - 1
        else:
            # if it is EXITING_EVENT
            y = event_row - 1
            x = event_col + 1

    elif event_col < viewpoint_col and event_row > viewpoint_row:
        # third quadrant
        if event_type == ENTERING_EVENT:
            # if it is ENTERING_EVENT
            y = event_row - 1
            x = event_col - 1
        else:
            # if it is EXITING_EVENT
            y = event_row + 1
            x = event_col + 1

    elif event_col < viewpoint_col and event_row == viewpoint_row:
        # between the first and third quadrant
        if event_type == ENTERING_EVENT:
            # if it is ENTERING_EVENT
            y = event_row - 1
            x = event_col + 1
        else:
            # if it is EXITING_EVENT
            y = event_row + 1
            x = event_col + 1

    else:
        # must be the vp cell itself
        assert event_row == viewpoint_row and event_col == viewpoint_col
        x = event_col
        y = event_row

    if abs(x - event_col > 1) or abs(y - event_row > 1):
        raise ValueError("_calculate_event_row_col()")

    return y, x


@ngjit
def _calc_event_elev(event_type, event_row, event_col, n_rows, n_cols,
                     viewpoint_row, viewpoint_col, inrast):
    # Calculate ENTER and EXIT event elevation (bilinear interpolation)

    row1, col1 = _calculate_event_row_col(event_type, event_row, event_col,
                                          viewpoint_row, viewpoint_col)

    event_elev = inrast[1][event_col]

    if 0 <= row1 < n_rows and 0 <= col1 < n_cols:
        elev1 = inrast[row1 - event_row + 1][col1]
        elev2 = inrast[row1 - event_row + 1][event_col]
        elev3 = inrast[1][col1]
        elev4 = inrast[1][event_col]
        if np.isnan(elev1) or np.isnan(elev2) or np.isnan(elev3) \
                or np.isnan(elev4):
            event_elev = inrast[1][event_col]
        else:
            event_elev = (elev1 + elev2 + elev3 + elev4) / 4.0

    return event_elev


@ngjit
def _calc_event_pos(event_type, event_row, event_col,
                    viewpoint_row, viewpoint_col):
    # Calculate the exact position of the given event,
    # and store them in x and y.

    # Quadrants:  1 2
    #   3 4
    #   ----->x
    #   |
    #   |
    #   |
    #   V y

    x = 0
    y = 0
    if event_type == CENTER_EVENT:
        # FOR CENTER_EVENTS
        y = event_row
        x = event_col
        return y, x

    if event_row < viewpoint_row and event_col < viewpoint_col:
        # first quadrant
        if event_type == ENTERING_EVENT:
            # if it is ENTERING_EVENT
            y = event_row - 0.5
            x = event_col + 0.5
        else:
            # if it is EXITING_EVENT
            y = event_row + 0.5
            x = event_col - 0.5

    elif event_row < viewpoint_row and event_col == viewpoint_col:
        # between the first and second quadrant
        if event_type == ENTERING_EVENT:
            # if it is ENTERING_EVENT
            y = event_row + 0.5
            x = event_col + 0.5
        else:
            # if it is EXITING_EVENT
            y = event_row + 0.5
            x = event_col - 0.5

    elif event_row < viewpoint_row and event_col > viewpoint_col:
        # second quadrant
        if event_type == ENTERING_EVENT:
            # if it is ENTERING_EVENT
            y = event_row + 0.5
            x = event_col + 0.5
        else:
            # if it is EXITING_EVENT
            y = event_row - 0.5
            x = event_col - 0.5

    elif event_row == viewpoint_row and event_col > viewpoint_col:
        # between the second and the fourth quadrant
        if event_type == ENTERING_EVENT:
            # if it is ENTERING_EVENT
            y = event_row + 0.5
            x = event_col - 0.5
        else:
            # if it is EXITING_EVENT
            y = event_row - 0.5
            x = event_col - 0.5

    elif event_row > viewpoint_row and event_col > viewpoint_col:
        # fourth quadrant
        if event_type == ENTERING_EVENT:
            # if it is ENTERING_EVENT
            y = event_row + 0.5
            x = event_col - 0.5
        else:
            # if it is EXITING_EVENT
            y = event_row - 0.5
            x = event_col + 0.5

    elif event_row > viewpoint_row and event_col == viewpoint_col:
        # between the third and fourth quadrant
        if event_type == ENTERING_EVENT:
            # if it is ENTERING_EVENT
            y = event_row - 0.5
            x = event_col - 0.5
        else:
            # if it is EXITING_EVENT
            y = event_row - 0.5
            x = event_col + 0.5

    elif event_row > viewpoint_row and event_col < viewpoint_col:
        # third quadrant
        if event_type == ENTERING_EVENT:
            # if it is ENTERING_EVENT
            y = event_row - 0.5
            x = event_col - 0.5
        else:
            # if it is EXITING_EVENT
            y = event_row + 0.5
            x = event_col + 0.5

    elif event_row == viewpoint_row and event_col < viewpoint_col:
        # between first and third quadrant
        if event_type == ENTERING_EVENT:
            # if it is ENTERING_EVENT
            y = event_row - 0.5
            x = event_col + 0.5
        else:
            # if it is EXITING_EVENT
            y = event_row + 0.5
            x = event_col + 0.5

    else:
        # must be the vp cell itself
        assert event_row == viewpoint_row and event_col == viewpoint_col
        x = event_col
        y = event_row

    assert abs(event_col - x) < 1 and abs(event_row - y) < 1

    return y, x


@ngjit
def _calculate_angle(event_x, event_y, viewpoint_x, viewpoint_y):
    if viewpoint_x == event_x and viewpoint_y > event_y:
        # between 1st and 2nd quadrant
        return PI / 2

    if viewpoint_x == event_x and viewpoint_y < event_y:
        # between 3rd and 4th quadrant
        return PI * 3.0 / 2.0

    if event_x == viewpoint_x and event_y == viewpoint_y:
        return 0

    if viewpoint_y == event_y and event_x > viewpoint_x:
        # between 1st and 4th quadrant
        return 0

    if viewpoint_x > event_x and viewpoint_y == event_y:
        # between 1st and 3rd quadrant
        return PI

    # Calculate angle between (x1, y1) and (x2, y2)
    ang = atan(fabs(event_y - viewpoint_y) / fabs(event_x - viewpoint_x))

    if event_x > viewpoint_x and event_y < viewpoint_y:
        # first quadrant
        return ang

    if viewpoint_x > event_x and viewpoint_y > event_y:
        # 2nd quadrant
        return PI - ang

    if viewpoint_x > event_x and viewpoint_y < event_y:
        # 3rd quadrant
        return PI + ang

    if viewpoint_x < event_x and viewpoint_y < event_y:
        # 4th quadrant
        return PI * 2.0 - ang

    return 0


@ngjit
def _calc_event_grad(row, col, elev, viewpoint_row, viewpoint_col,
                     viewpoint_elev, ew_res, ns_res):
    # Calculate event gradient

    diff_elev = elev - viewpoint_elev

    dx = (col - viewpoint_col) * ew_res
    dy = (row - viewpoint_row) * ns_res
    distance_to_viewpoint = (dx * dx) + (dy * dy)

    # PI / 2 above, - PI / 2 below
    if distance_to_viewpoint == 0:
        if diff_elev > 0:
            gradient = PI / 2
        elif diff_elev < 0:
            gradient = - PI / 2
        else:
            gradient = 0
    else:
        gradient = atan(diff_elev / sqrt(distance_to_viewpoint))
    return gradient


# given a StatusNode, fill in its dist2vp and gradient
@ngjit
def _calc_dist_n_grad(status_node_row, status_node_col, elev, viewpoint_row,
                      viewpoint_col, viewpoint_elev, ew_res, ns_res):
    diff_elev = elev - viewpoint_elev

    dx = (status_node_col - viewpoint_col) * ew_res
    dy = (status_node_row - viewpoint_row) * ns_res
    distance_to_viewpoint = (dx * dx) + (dy * dy)

    # PI / 2 above, - PI / 2 below
    if distance_to_viewpoint == 0:
        if diff_elev > 0:
            gradient = PI / 2
        elif diff_elev < 0:
            gradient = - PI / 2
        else:
            gradient = 0
    else:
        gradient = atan(diff_elev / sqrt(distance_to_viewpoint))
    return distance_to_viewpoint, gradient


# ported https://github.com/OSGeo/grass/blob/master/raster/r.viewshed/grass.cpp
# function _init_event_list_in_memory()
@ngjit
def _init_event_list(event_list, raster, vp_row, vp_col,
                     data, visibility_grid):
    # Initialize and fill all the events for the map into event_list

    n_rows, n_cols = raster.shape
    inrast = np.empty(shape=(3, n_cols), dtype=np.float64)
    inrast.fill(np.nan)

    # scan through the raster data
    # read first row
    inrast[2] = raster[0]

    # index of the event array: row, col, elev_0, elev_1, elev_2, ang, type
    e = np.zeros((7,), dtype=np.float64)

    count_event = 0
    for i in range(n_rows):
        # read in the raster row
        tmprast = inrast[0]
        inrast[0] = inrast[1]
        inrast[1] = inrast[2]
        inrast[2] = tmprast

        if i < n_rows - 1:
            inrast[2] = raster[i + 1]
        else:
            for j in range(n_cols):
                # when assign to None, it is forced to np.nan
                inrast[2][j] = np.nan

        # fill event list with events from this row
        for j in range(n_cols):
            # integer
            e_row = i
            e_col = j

            # float
            e[E_ROW_ID] = i
            e[E_COL_ID] = j

            # read the elevation value into the event
            e[E_ELEV_1] = inrast[1][j]

            # write it into the row of data going through the vp
            if i == vp_row:
                data[0][j] = e[E_ELEV_1]
                data[1][j] = e[E_ELEV_1]
                data[2][j] = e[E_ELEV_1]

            # set the vp, and don't insert it into eventlist
            if i == vp_row and j == vp_col:
                _set_visibility(visibility_grid, i, j, 180)
                continue

            # if it got here it is not the vp, not NODATA, and
            # within max distance from vp generate its 3 events
            # and insert them

            # get ENTER elevation
            e[E_TYPE_ID] = ENTERING_EVENT
            e[E_ELEV_0] = _calc_event_elev(e[E_TYPE_ID], e_row, e_col,
                                           n_rows, n_cols,
                                           vp_row, vp_col, inrast)

            # get EXIT event
            e[E_TYPE_ID] = EXITING_EVENT
            e[E_ELEV_2] = _calc_event_elev(e[E_TYPE_ID], e_row, e_col,
                                           n_rows, n_cols,
                                           vp_row, vp_col, inrast)

            # write adjusted elevation into the row of data
            # going through the vp
            if i == vp_row:
                data[0][j] = e[E_ELEV_0]
                data[1][j] = e[E_ELEV_1]
                data[2][j] = e[E_ELEV_2]

            # put event into event list
            e[E_TYPE_ID] = ENTERING_EVENT
            ay, ax = _calc_event_pos(e[E_TYPE_ID], e_row, e_col,
                                     vp_row, vp_col)
            e[E_ANG_ID] = _calculate_angle(ax, ay, vp_col, vp_row)
            event_list[count_event] = e
            count_event += 1

            e[E_TYPE_ID] = CENTER_EVENT
            ay, ax = _calc_event_pos(e[E_TYPE_ID], e_row, e_col,
                                     vp_row, vp_col)
            e[E_ANG_ID] = _calculate_angle(ax, ay, vp_col, vp_row)
            event_list[count_event] = e
            count_event += 1

            e[E_TYPE_ID] = EXITING_EVENT
            ay, ax = _calc_event_pos(e[E_TYPE_ID], e_row, e_col,
                                     vp_row, vp_col)
            e[E_ANG_ID] = _calculate_angle(ax, ay, vp_col, vp_row)
            event_list[count_event] = e
            count_event += 1

    return


@ngjit
def _create_status_struct(tree_vals, tree_nodes):
    # Create and initialize the status struct.
    # return a Tree object with a dummy root.

    # dummy status node
    dummy_node_value = np.array([0.0, -1, -1, SMALLEST_GRAD, SMALLEST_GRAD,
                                SMALLEST_GRAD, 0.0, 0.0, 0.0, SMALLEST_GRAD])

    # node 0 is root
    root = 0
    _create_tree_nodes(tree_vals, tree_nodes, root, dummy_node_value, RB_BLACK)

    # last row is NIL
    _create_tree_nodes(tree_vals, tree_nodes, NIL_ID,
                       dummy_node_value, RB_BLACK)

    num_nodes = tree_vals.shape[0]
    tree_nodes[NIL_ID][TN_LEFT_ID] = num_nodes
    tree_nodes[NIL_ID][TN_RIGHT_ID] = num_nodes
    tree_nodes[NIL_ID][TN_PARENT_ID] = num_nodes

    return root


# /*find the vertical ang in degrees between the vp and the
#    point represented by the StatusNode.  Assumes all values (except
#    gradient) in sn have been filled. The value returned is in [0,
#    180]. A value of 0 is directly below the specified viewing position,
#    90 is due horizontal, and 180 is directly above the observer.
#    If doCurv is set we need to consider the curvature of the
#    earth */
@ngjit
def _get_vertical_ang(viewpoint_elev, distance_to_viewpoint, elev):
    # Find the vertical angle in degrees between the vp
    # and the point represented by the StatusNode

    # determine the difference in elevation, based on the curvature
    diff_elev = viewpoint_elev - elev

    # calculate and return the ang in degrees
    assert abs(distance_to_viewpoint) > 0.0

    # 0 above, 180 below
    if diff_elev == 0.0:
        return 90
    elif diff_elev > 0:
        return atan(sqrt(distance_to_viewpoint) / diff_elev) * 180 / PI

    return atan(abs(diff_elev) / sqrt(distance_to_viewpoint)) * 180 / PI + 90


@ngjit
def _init_status_node(status_node):
    status_node[TN_KEY_ID] = -1

    status_node[TN_GRAD_0] = np.nan
    status_node[TN_GRAD_1] = np.nan
    status_node[TN_GRAD_2] = np.nan

    status_node[TN_ANG_0] = np.nan
    status_node[TN_ANG_1] = np.nan
    status_node[TN_ANG_2] = np.nan

    return


def _print_event(event):
    if event[E_TYPE_ID] == 1:
        t = "ENTERING   "
    elif event[E_TYPE_ID] == -1:
        t = "EXITING    "
    else:
        t = "CENTER     "

    print('row = ', event[E_ROW_ID],
          'col = ', event[E_COL_ID],
          'event_type = ', t,
          'elevation = ', event[E_ELEV_0], event[E_ELEV_1], event[E_ELEV_2],
          'ang = ', event[E_ANG_ID])
    return


@ngjit
def _push(stack, item):
    stack[0] += 1
    stack[stack[0]] = item
    return


@ngjit
def _pop(stack):
    item = stack[stack[0]]
    stack[0] -= 1
    return item


# Viewshed's sweep algorithm on the grid stored in the given file, and
# with the given vp.  Create a visibility grid and return
# it. The computation runs in memory, which means the input grid, the
# status structure and the output grid are stored in arrays in
# memory.
#
# The output: A cell x in the visibility grid is recorded as follows:
#
# if it is NODATA, then x  is set to NODATA
# if it is invisible, then x is set to INVISIBLE
# if it is visible,  then x is set to the vertical ang wrt to vp

# https://github.com/OSGeo/grass/blob/master/raster/r.viewshed/viewshed.cpp
# function viewshed_in_memory()

@ngjit
def _viewshed_cpu_sweep(raster, vp_row, vp_col, vp_elev, vp_target, ew_res,
                        ns_res, event_rcts, event_aes, data, visibility_grid):
    n_rows, n_cols = raster.shape

    # for e in event_list:
    #     _print_event(e)

    # create the status structure
    # create 2d array of the RB-tree
    num_nodes = n_cols - vp_col + n_cols * n_rows + 10

    status_values = np.zeros((num_nodes, 8), dtype=np.float64)
    status_struct = np.zeros((num_nodes, 4), dtype=np.int64)

    root = _create_status_struct(status_values, status_struct)

    # idle row idx in the 2d data array of status_struct tree
    idle = np.zeros((num_nodes,), dtype=np.int64)
    for i in range(0, num_nodes - 1):
        idle[i] = num_nodes - i
    idle[0] = num_nodes - 2

    # Put cells that are initially on the sweepline into status structure
    status_node = np.zeros((7,), dtype=np.float64)
    for i in range(vp_col + 1, n_cols):
        _init_status_node(status_node)
        status_row = vp_row
        status_col = i

        # event properties
        e_row = vp_row
        e_col = i
        e_elev_0 = data[0][i]
        e_elev_1 = data[1][i]
        e_elev_2 = data[2][i]

        if (not np.isnan(data[1][i])):
            # calculate Distance to VP and Gradient,
            # store them into status_node
            # need either 3 elevation values or
            # 3 gradients calculated from 3 elevation values
            # need also 3 angs

            e_type = ENTERING_EVENT
            ay, ax = _calc_event_pos(e_type, e_row, e_col, vp_row, vp_col)
            status_node[TN_ANG_0] = _calculate_angle(ax, ay, vp_col, vp_row)
            status_node[TN_GRAD_0] = _calc_event_grad(ay, ax, e_elev_0,
                                                      vp_row, vp_col, vp_elev,
                                                      ew_res, ns_res)

            e_type = CENTER_EVENT
            ay, ax = _calc_event_pos(e_type, e_row, e_col, vp_row, vp_col)
            status_node[TN_ANG_1] = _calculate_angle(ax, ay, vp_col, vp_row)
            status_node[TN_KEY_ID], status_node[TN_GRAD_1] = \
                _calc_dist_n_grad(status_row, status_col, e_elev_1,
                                  vp_row, vp_col, vp_elev, ew_res, ns_res)

            e_type = EXITING_EVENT
            ay, ax = _calc_event_pos(e_type, e_row, e_col, vp_row, vp_col)
            status_node[TN_ANG_2] = _calculate_angle(ax, ay, vp_col, vp_row)
            status_node[TN_GRAD_2] = _calc_event_grad(ay, ax, e_elev_2,
                                                      vp_row, vp_col, vp_elev,
                                                      ew_res, ns_res)

            assert status_node[TN_ANG_1] == 0

            if status_node[TN_ANG_0] > status_node[TN_ANG_1]:
                status_node[TN_ANG_0] -= 2 * PI

            # insert sn into the status structure
            id = _pop(idle)
            root = _insert_into_tree(status_values, status_struct, root,
                                     id, status_node)

    # sweep the event_list

    nevents = len(event_rcts)

    for i in range(nevents):
        # get out one event at a time and process it according to its type
        e_rct = event_rcts[i]
        e_ae = event_aes[i]
        # e = event_list[i]

        # status_node = StatusNode(row=e[E_ROW_ID], col=e[E_COL_ID])
        _init_status_node(status_node)
        status_row = e_rct[E_ROW_ID]
        status_col = e_rct[E_COL_ID]

        # calculate Distance to VP and Gradient
        status_node[TN_KEY_ID], status_node[TN_GRAD_1] = \
            _calc_dist_n_grad(status_row, status_col,
                              e_ae[AE_ELEV_1] + vp_target,
                              vp_row, vp_col, vp_elev, ew_res, ns_res,)

        etype = e_rct[E_TYPE_ID]
        if etype == ENTERING_EVENT:
            # insert node into structure

            #  need either 3 elevation values or
            # 	     * 3 gradients calculated from 3 elevation values */
            # 	    /* need also 3 angs */
            ay, ax = _calc_event_pos(e_rct[E_TYPE_ID], e_rct[E_ROW_ID],
                                     e_rct[E_COL_ID], vp_row, vp_col)
            status_node[TN_ANG_0] = e_ae[AE_ANG_ID]
            status_node[TN_GRAD_0] = _calc_event_grad(ay, ax, e_ae[AE_ELEV_0],
                                                      vp_row, vp_col, vp_elev,
                                                      ew_res, ns_res)

            e_rct[E_TYPE_ID] = CENTER_EVENT
            ay, ax = _calc_event_pos(e_rct[E_TYPE_ID], e_rct[E_ROW_ID],
                                     e_rct[E_COL_ID], vp_row, vp_col)
            status_node[TN_ANG_1] = _calculate_angle(ax, ay, vp_col, vp_row)
            status_node[TN_KEY_ID], status_node[TN_GRAD_1] = \
                _calc_dist_n_grad(status_row, status_col, e_ae[AE_ELEV_1],
                                  vp_row, vp_col, vp_elev, ew_res, ns_res)

            e_rct[E_TYPE_ID] = EXITING_EVENT
            ay, ax = _calc_event_pos(e_rct[E_TYPE_ID], e_rct[E_ROW_ID],
                                     e_rct[E_COL_ID], vp_row, vp_col)
            status_node[TN_ANG_2] = _calculate_angle(ax, ay, vp_col, vp_row)
            status_node[TN_GRAD_2] = _calc_event_grad(ay, ax, e_ae[AE_ELEV_2],
                                                      vp_row, vp_col, vp_elev,
                                                      ew_res, ns_res)

            e_rct[E_TYPE_ID] = ENTERING_EVENT

            if e_ae[AE_ANG_ID] < PI:
                if status_node[TN_ANG_0] > status_node[TN_ANG_1]:
                    status_node[TN_ANG_0] -= 2 * PI
            else:
                if status_node[TN_ANG_0] > status_node[TN_ANG_1]:
                    status_node[TN_ANG_1] += 2 * PI
                    status_node[TN_ANG_2] += 2 * PI

            id = _pop(idle)
            root = _insert_into_tree(status_values, status_struct, root,
                                     id, status_node)

        elif etype == EXITING_EVENT:
            # delete node out of status structure
            root, deleted = _delete_from_tree(status_values, status_struct,
                                              root, status_node[TN_KEY_ID])
            _push(idle, deleted)

        elif etype == CENTER_EVENT:
            # calculate visibility
            # consider current ang and gradient
            max = _max_grad_in_status_struct(status_values, status_struct,
                                             root, status_node[TN_KEY_ID],
                                             e_ae[AE_ANG_ID],
                                             status_node[TN_GRAD_1])

            # the point is visible: store its vertical ang
            if max <= status_node[TN_GRAD_1]:
                vert_ang = _get_vertical_ang(vp_elev, status_node[TN_KEY_ID],
                                             e_ae[AE_ELEV_1] + vp_target)

                _set_visibility(visibility_grid, status_row,
                                status_col, vert_ang)

                assert vert_ang >= 0
                # when you write the visibility grid you assume that
                # 		   visible values are positive

    return visibility_grid


def _viewshed_cpu(
    raster: xarray.DataArray,  # contains numpy array
    x: Union[int, float],
    y: Union[int, float],
    observer_elev: float = OBS_ELEV,
    target_elev: float = TARGET_ELEV,
) -> xarray.DataArray:

    height, width = raster.shape

    # Peak-memory guard.  The sweep algorithm allocates an event_list of
    # 3*H*W rows (168 bytes/pixel), plus the red-black status structure
    # (status_values + status_struct + idle ~= 104 bytes/pixel),
    # visibility_grid and raster (~16 bytes/pixel), and a lexsort
    # temporary that roughly doubles event_list during sort.
    # Round to ~500 bytes/pixel and refuse if it would eat most of RAM.
    peak_bytes = 500 * height * width
    avail = _available_memory_bytes()
    if peak_bytes > 0.5 * avail:
        raise MemoryError(
            f"viewshed CPU sweep would need ~{peak_bytes / 1e9:.1f} GB of "
            f"working memory for a {height}x{width} raster, which exceeds "
            f"50% of available RAM ({avail / 1e9:.1f} GB). "
            f"Pass max_distance= to limit the analysis area."
        )

    y_coords = raster.indexes.get('y').values
    x_coords = raster.indexes.get('x').values

    # validate x arg
    if not (x_coords.min() <= x <= x_coords.max()):
        raise ValueError("x argument outside of raster x_range")

    # validate y arg
    if not (y_coords.min() <= y <= y_coords.max()):
        raise ValueError("y argument outside of raster y_range")

    selection = raster.sel(x=[x], y=[y], method='nearest')
    x = selection.x.values[0]
    y = selection.y.values[0]

    y_view = np.where(y_coords == y)[0][0]
    y_range = (y_coords[0], y_coords[-1])

    x_view = np.where(x_coords == x)[0][0]
    x_range = (x_coords[0], x_coords[-1])

    # viewpoint properties
    viewpoint_row = y_view
    viewpoint_col = x_view
    viewpoint_elev = raster.data[y_view, x_view] + observer_elev
    viewpoint_target = 0.0
    if abs(target_elev) > 0:
        viewpoint_target = target_elev

    # int getgrdhead(FILE * fd, struct Cell_head *cellhd)
    ew_res = (x_range[1] - x_range[0]) / (width - 1)
    ns_res = (y_range[1] - y_range[0]) / (height - 1)

    # create the visibility grid of the sizes specified in the header
    visibility_grid = np.empty(shape=raster.shape, dtype=np.float64)
    # set everything initially invisible
    visibility_grid.fill(INVISIBLE)
    n_rows, n_cols = raster.shape

    data = np.zeros(shape=(3, n_cols), dtype=np.float64)

    # construct the event list corresponding to the given input file and vp;
    # this creates an array of all the cells on the same row as the vp
    num_events = 3 * (n_rows * n_cols - 1)
    event_list = np.zeros((num_events, 7), dtype=np.float64)

    raster.data = raster.data.astype(np.float64, copy=False)

    _init_event_list(event_list=event_list, raster=raster.data,
                     vp_row=viewpoint_row, vp_col=viewpoint_col,
                     data=data, visibility_grid=visibility_grid)

    # sort the events radially by ang
    event_list = event_list[np.lexsort((event_list[:, E_TYPE_ID],
                                        event_list[:, E_ANG_ID]))]

    # event indices: row, col, type, ang, enter elev, center elev, exit elev
    # split event into 2 arrays: one of 3 integer elements: row, col, type;
    #                          and one of 4 float elements: angle, elevations.
    event_rcts = np.array(event_list[:, :3], dtype=np.int64)
    event_aes = event_list[:, 3:].copy()

    viewshed_img = _viewshed_cpu_sweep(
        raster.data, viewpoint_row, viewpoint_col, viewpoint_elev,
        viewpoint_target, ew_res, ns_res, event_rcts, event_aes, data,
        visibility_grid)

    visibility = xarray.DataArray(viewshed_img,
                                  coords=raster.coords,
                                  attrs=raster.attrs,
                                  dims=raster.dims)
    return visibility


def viewshed(raster: xarray.DataArray,
             x: Union[int, float],
             y: Union[int, float],
             observer_elev: float = OBS_ELEV,
             target_elev: float = TARGET_ELEV,
             max_distance: float = None) -> xarray.DataArray:
    """
    Calculate viewshed of a raster (the visible cells in the raster)
    for the given viewpoint (observer) location.

    Parameters
    ----------
    raster : xr.DataArray
        Input raster image.
    x : int, float
        x-coordinate in data space of observer location.
    y : int, float
        y-coordinate in data space of observer location.
    observer_elev : float
        Observer elevation above the terrain.
    target_elev : float
        Target elevation offset above the terrain, which is the height
        in surface units to be added to the z-value of each pixel
        when it is being analyzed for visibility.
    max_distance : float, optional
        Maximum analysis distance from the observer in surface units.
        Cells beyond this distance are marked INVISIBLE without being
        evaluated. When set and the raster is dask-backed, only the
        chunks within the distance window are loaded — this is the most
        efficient way to run viewshed on very large dask rasters.

    Returns
    -------
    viewshed: xarray.DataArray
        A cell x in the visibility grid is recorded as follows:
        If it is invisible, then x is set to INVISIBLE.
        If it is visible,  then x is set to the vertical angle w.r.t
        the viewpoint. The value returned is in [0, 180]. A value of 0 is
        directly below the specified viewing position,
        90 is due horizontal, and 180 is directly above the observer.

    Notes
    -----
    The CPU (numpy), GPU (cupy with RTX), and dask backends use
    different algorithms and may produce slightly different results for
    the same input.

    - **CPU**: Angular sweep algorithm ported from GRASS GIS
      ``r.viewshed``. Operates directly on the grid and produces exact
      visibility angles.
    - **GPU**: Ray tracing via NVIDIA OptiX RTX against a triangulated
      mesh of the terrain. The mesh discretisation can introduce small
      angular errors (typically < 0.03 degrees for visible cells).
    - **Dask**: When ``max_distance`` is set or the grid fits in memory,
      the exact CPU algorithm is used on the relevant window. For very
      large grids that exceed memory, a horizon-profile distance-sweep
      algorithm is used instead. This algorithm discretises angles and
      may produce minor visibility differences near the boundary of
      occluded regions.

    Both backends agree on which cells are visible vs invisible in the
    vast majority of cases, but the reported vertical angles may differ
    by a small amount near cell boundaries.

    Examples
    --------
    .. sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial import viewshed
        >>> data = np.array([
        ...     [0, 0, 1, 0, 0],
        ...     [1, 3, 0, 0, 0],
        ...     [10, 2, 5, 2, -1],
        ...     [11, 1, 2, 9, 0]])
        >>> terrain = xr.DataArray(data, dims=['y', 'x'])
        >>> h, w = data.shape
        >>> terrain['y'] = np.linspace(1, h, h)
        >>> terrain['x'] = np.linspace(1, w, w)
        >>> terrain
        <xarray.DataArray (y: 4, x: 5)>
        array([[ 0,  0,  1,  0,  0],
               [ 1,  3,  0,  0,  0],
               [10,  2,  5,  2, -1],
               [11,  1,  2,  9,  0]])
        Coordinates:
          * y        (y) float64 1.0 2.0 3.0 4.0
          * x        (x) float64 1.0 2.0 3.0 4.0 5.0
        >>> viewshed(terrain, x=3, y=2)
        <xarray.DataArray (y: 4, x: 5)>
        array([[ -1.        ,  90.        , 135.        ,  90.        , -1.],
               [ -1.        , 161.56505118, 180.        ,  90.        , 90.],
               [167.39561735, 144.73561032, 168.69006753, 144.73561032, -1.],
               [165.57993189,  -1.        ,  -1.        , 166.0472636 , -1.]])
        Coordinates:
          * x        (x) float64 1.0 2.0 3.0 4.0 5.0
          * y        (y) float64 1.0 2.0 3.0 4.0

    """
    _validate_raster(raster, func_name='viewshed', name='raster')

    # --- max_distance: extract spatial window for any backend ---
    if max_distance is not None:
        return _viewshed_windowed(raster, x, y, observer_elev, target_elev,
                                  max_distance)

    if isinstance(raster.data, np.ndarray):
        return _viewshed_cpu(raster, x, y, observer_elev, target_elev)

    elif has_cuda_and_cupy() and is_cupy_array(raster.data):
        if has_rtx():
            # Run on gpu
            from .gpu_rtx.viewshed import viewshed_gpu
            return viewshed_gpu(raster, x, y, observer_elev, target_elev)
        else:
            # Convert to numpy and run on cpu
            import cupy as cp
            raster.data = cp.asnumpy(raster.data)
            return _viewshed_cpu(raster, x, y, observer_elev, target_elev)

    elif has_dask_array():
        import dask.array as da
        if isinstance(raster.data, da.Array):
            return _viewshed_dask(raster, x, y, observer_elev, target_elev)

    raise TypeError(f"Unsupported raster array type: {type(raster.data)}")


# ---------------------------------------------------------------------------
# Dask backend helpers
# ---------------------------------------------------------------------------

def _dask_embed_window(window_np, H, W, r_lo, r_hi, c_lo, c_hi, chunks):
    """Embed a small numpy result into a full-size lazy dask INVISIBLE array.

    Builds the output chunk-by-chunk: chunks that overlap the window get a
    numpy array with the window values pasted in; all other chunks are
    created via ``dask.array.full`` so they consume no memory until
    materialised.
    """
    import dask.array as da

    y_offsets = _chunk_offsets(chunks[0])
    x_offsets = _chunk_offsets(chunks[1])
    n_yc = len(chunks[0])
    n_xc = len(chunks[1])

    rows = []
    for yi in range(n_yc):
        row_blocks = []
        cy0, cy1 = int(y_offsets[yi]), int(y_offsets[yi + 1])
        cy_size = cy1 - cy0
        for xi in range(n_xc):
            cx0, cx1 = int(x_offsets[xi]), int(x_offsets[xi + 1])
            cx_size = cx1 - cx0

            # Does this chunk overlap the result window?
            ov_r0 = max(cy0, r_lo)
            ov_r1 = min(cy1, r_hi)
            ov_c0 = max(cx0, c_lo)
            ov_c1 = min(cx1, c_hi)

            if ov_r0 < ov_r1 and ov_c0 < ov_c1:
                # This chunk overlaps — build a concrete numpy block
                block = np.full((cy_size, cx_size), INVISIBLE,
                                dtype=np.float64)
                # Local indices within the block and within window_np
                block[ov_r0 - cy0:ov_r1 - cy0,
                      ov_c0 - cx0:ov_c1 - cx0] = \
                    window_np[ov_r0 - r_lo:ov_r1 - r_lo,
                              ov_c0 - c_lo:ov_c1 - c_lo]
                row_blocks.append(da.from_delayed(
                    _identity_delayed(block),
                    shape=(cy_size, cx_size), dtype=np.float64))
            else:
                # No overlap — lazy INVISIBLE block (zero memory)
                row_blocks.append(
                    da.full((cy_size, cx_size), INVISIBLE,
                            dtype=np.float64, chunks=(cy_size, cx_size)))
        rows.append(da.concatenate(row_blocks, axis=1))
    return da.concatenate(rows, axis=0)


def _identity_delayed(x):
    """Wrap a concrete value in a dask delayed for da.from_delayed."""
    import dask
    return dask.delayed(lambda v: v)(x)


def _available_memory_bytes():
    """Best-effort estimate of available memory in bytes."""
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        import psutil
        return psutil.virtual_memory().available
    except (ImportError, AttributeError):
        pass
    return 2 * 1024 ** 3


def _chunk_offsets(chunk_sizes):
    """Convert a tuple of chunk sizes to cumulative offset boundaries.

    Returns an array of length len(chunk_sizes)+1 where offsets[i] is the
    start index of chunk i and offsets[-1] is the total length.
    """
    offsets = np.empty(len(chunk_sizes) + 1, dtype=np.int64)
    offsets[0] = 0
    for i, s in enumerate(chunk_sizes):
        offsets[i + 1] = offsets[i] + s
    return offsets


def _chunk_index_for(offsets, idx):
    """Return the chunk index that contains global index *idx*."""
    # Binary search
    lo, hi = 0, len(offsets) - 2
    while lo < hi:
        mid = (lo + hi) // 2
        if offsets[mid + 1] <= idx:
            lo = mid + 1
        else:
            hi = mid
    return lo


class _ChunkCache:
    """LRU cache for computed dask chunks with a byte budget."""

    def __init__(self, budget_bytes):
        self._cache = OrderedDict()   # key → numpy array
        self._bytes = 0
        self._budget = budget_bytes

    def get(self, key, dask_data):
        """Return the numpy chunk for *key* = (ty, tx), loading if needed."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        arr = dask_data.blocks[key].compute()
        nbytes = arr.nbytes
        # Evict LRU entries until we have room
        while self._bytes + nbytes > self._budget and self._cache:
            _, evicted = self._cache.popitem(last=False)
            self._bytes -= evicted.nbytes
        self._cache[key] = arr
        self._bytes += nbytes
        return arr


@ngjit
def _sweep_ring(rows, cols, elevs, n_cells,
                obs_r, obs_c, obs_elev, target_elev,
                ew_res, ns_res, horizon, visibility, n_angles):
    """Process all cells on one Chebyshev ring through the horizon profile.

    Uses a two-pass approach: first check visibility of all ring cells
    against the *pre-ring* horizon, then update the horizon. This prevents
    cells within the same ring from incorrectly occluding each other.
    """
    bin_width = 2.0 * PI / n_angles
    INF = 1e30

    # Pre-compute per-cell values.
    # Two gradient arrays: one WITH target_elev for visibility testing, one
    # WITHOUT for the horizon (occlusion) profile.  The terrain itself blocks
    # the view, not imaginary targets on every cell.
    gradients = np.empty(n_cells, dtype=np.float64)
    terrain_gradients = np.empty(n_cells, dtype=np.float64)
    center_bins = np.empty(n_cells, dtype=np.int64)
    half_bins_arr = np.empty(n_cells, dtype=np.int64)
    dist_sqs = np.empty(n_cells, dtype=np.float64)
    eff_elevs = np.empty(n_cells, dtype=np.float64)
    valid = np.empty(n_cells, dtype=np.int64)

    cell_size = max(fabs(ew_res), fabs(ns_res))

    for i in range(n_cells):
        elev = elevs[i]
        if elev != elev:  # NaN check (numba-safe)
            valid[i] = 0
            continue
        r = rows[i]
        c = cols[i]
        dx = (c - obs_c) * ew_res
        dy = (r - obs_r) * ns_res
        dist_sq = dx * dx + dy * dy
        if dist_sq == 0.0:
            valid[i] = 0
            continue
        valid[i] = 1
        dist = sqrt(dist_sq)
        eff_elev = elev + target_elev
        eff_elevs[i] = eff_elev
        dist_sqs[i] = dist_sq
        gradients[i] = atan((eff_elev - obs_elev) / dist)
        terrain_gradients[i] = atan((elev - obs_elev) / dist)

        angle = atan2(dy, dx)
        ang_width = cell_size / dist
        hb = int(ang_width / bin_width / 2.0)
        if hb < 0:
            hb = 0
        half_bins_arr[i] = hb
        center_bins[i] = int((angle + PI) / bin_width) % n_angles

    # Pass 1: check visibility against current (pre-ring) horizon
    for i in range(n_cells):
        if valid[i] == 0:
            continue
        max_h = -INF
        cb = center_bins[i]
        hb = half_bins_arr[i]
        for b in range(-hb, hb + 1):
            idx = (cb + b) % n_angles
            if horizon[idx] > max_h:
                max_h = horizon[idx]
        if gradients[i] > max_h:
            r = rows[i]
            c = cols[i]
            visibility[r, c] = _get_vertical_ang(
                obs_elev, dist_sqs[i], eff_elevs[i])

    # Pass 2: update horizon with raw terrain gradients (no target_elev)
    for i in range(n_cells):
        if valid[i] == 0:
            continue
        cb = center_bins[i]
        hb = half_bins_arr[i]
        grad = terrain_gradients[i]
        for b in range(-hb, hb + 1):
            idx = (cb + b) % n_angles
            if grad > horizon[idx]:
                horizon[idx] = grad


def _ring_cells(d, obs_r, obs_c, H, W):
    """Generate row/col arrays for cells on the Chebyshev ring at distance d.

    Traverses the four edges of the square ring in order:
    top (left→right), right (top→bottom), bottom (right→left),
    left (bottom→top), excluding corners already counted.
    """
    r_min = max(0, obs_r - d)
    r_max = min(H - 1, obs_r + d)
    c_min = max(0, obs_c - d)
    c_max = min(W - 1, obs_c + d)

    rows_list = []
    cols_list = []

    # Top edge: row = obs_r - d, cols from c_min to c_max
    if obs_r - d >= 0:
        cols_top = np.arange(c_min, c_max + 1, dtype=np.int64)
        rows_list.append(np.full(len(cols_top), obs_r - d, dtype=np.int64))
        cols_list.append(cols_top)

    # Bottom edge: row = obs_r + d, cols from c_min to c_max
    if obs_r + d <= H - 1:
        cols_bot = np.arange(c_min, c_max + 1, dtype=np.int64)
        rows_list.append(np.full(len(cols_bot), obs_r + d, dtype=np.int64))
        cols_list.append(cols_bot)

    # Left edge: col = obs_c - d, rows from r_min+1 to r_max-1
    # (exclude corners already in top/bottom)
    inner_r_min = r_min + (1 if obs_r - d >= 0 else 0)
    inner_r_max = r_max - (1 if obs_r + d <= H - 1 else 0)
    if obs_c - d >= 0 and inner_r_min <= inner_r_max:
        rows_left = np.arange(inner_r_min, inner_r_max + 1, dtype=np.int64)
        rows_list.append(rows_left)
        cols_list.append(np.full(len(rows_left), obs_c - d, dtype=np.int64))

    # Right edge: col = obs_c + d, rows from r_min+1 to r_max-1
    if obs_c + d <= W - 1 and inner_r_min <= inner_r_max:
        rows_right = np.arange(inner_r_min, inner_r_max + 1, dtype=np.int64)
        rows_list.append(rows_right)
        cols_list.append(np.full(len(rows_right), obs_c + d, dtype=np.int64))

    if rows_list:
        return np.concatenate(rows_list), np.concatenate(cols_list)
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)


def _extract_elevations(rows, cols, dask_data, cache, y_offsets, x_offsets):
    """Look up elevations for given (row, col) pairs from cached chunks."""
    n = len(rows)
    elevs = np.empty(n, dtype=np.float64)
    for i in range(n):
        r, c = int(rows[i]), int(cols[i])
        ty = _chunk_index_for(y_offsets, r)
        tx = _chunk_index_for(x_offsets, c)
        chunk = cache.get((ty, tx), dask_data)
        local_r = r - int(y_offsets[ty])
        local_c = c - int(x_offsets[tx])
        elevs[i] = chunk[local_r, local_c]
    return elevs


def _viewshed_distance_sweep(dask_data, H, W, obs_r, obs_c,
                              obs_elev, target_elev, ew_res, ns_res,
                              chunks_y, chunks_x, max_distance):
    """Out-of-core horizon-profile distance sweep viewshed."""
    # Maximum Chebyshev distance in cells
    max_d_cells = max(obs_r, H - 1 - obs_r, obs_c, W - 1 - obs_c)
    if max_distance is not None:
        cell_size = max(abs(ew_res), abs(ns_res))
        max_d_dist = int(np.ceil(max_distance / cell_size))
        max_d_cells = min(max_d_cells, max_d_dist)

    n_angles = 16 * int(np.ceil(np.sqrt(2) * max_d_cells)) + 16
    # Clamp to reasonable bounds
    n_angles = max(n_angles, 360)
    n_angles = min(n_angles, 1_000_000)

    horizon = np.full(n_angles, -1e30, dtype=np.float64)
    visibility = np.full((H, W), INVISIBLE, dtype=np.float64)
    visibility[obs_r, obs_c] = 180.0

    y_offsets = _chunk_offsets(chunks_y)
    x_offsets = _chunk_offsets(chunks_x)

    budget = max(int(0.25 * _available_memory_bytes()), 64 * 1024 * 1024)
    cache = _ChunkCache(budget)

    for d in range(1, max_d_cells + 1):
        rows, cols = _ring_cells(d, obs_r, obs_c, H, W)
        if len(rows) == 0:
            continue
        elevs = _extract_elevations(rows, cols, dask_data, cache,
                                    y_offsets, x_offsets)
        _sweep_ring(rows, cols, elevs, len(rows),
                    obs_r, obs_c, obs_elev, target_elev,
                    ew_res, ns_res, horizon, visibility, n_angles)

    return visibility


def _viewshed_windowed(raster, x, y, observer_elev, target_elev,
                       max_distance):
    """Run viewshed on a spatial window around the observer.

    Works for any backend: numpy, cupy, dask+numpy, dask+cupy.  The window
    is extracted via xarray slicing, computed to an in-memory array, then
    dispatched to the appropriate single-array backend.  The result is
    embedded in a full-size INVISIBLE output.
    """
    height, width = raster.shape
    y_coords = raster.indexes.get('y').values
    x_coords = raster.indexes.get('x').values

    if not (x_coords.min() <= x <= x_coords.max()):
        raise ValueError("x argument outside of raster x_range")
    if not (y_coords.min() <= y <= y_coords.max()):
        raise ValueError("y argument outside of raster y_range")

    selection = raster.sel(x=[x], y=[y], method='nearest')
    x = selection.x.values[0]
    y = selection.y.values[0]

    obs_r = int(np.where(y_coords == y)[0][0])
    obs_c = int(np.where(x_coords == x)[0][0])

    y_range = (y_coords[0], y_coords[-1])
    x_range = (x_coords[0], x_coords[-1])
    ew_res = (x_range[1] - x_range[0]) / (width - 1) if width > 1 else 1.0
    ns_res = (y_range[1] - y_range[0]) / (height - 1) if height > 1 else 1.0
    cell_size = max(abs(ew_res), abs(ns_res))
    radius_cells = int(np.ceil(max_distance / cell_size))

    r_lo = max(0, obs_r - radius_cells)
    r_hi = min(height, obs_r + radius_cells + 1)
    c_lo = max(0, obs_c - radius_cells)
    c_hi = min(width, obs_c + radius_cells + 1)

    window = raster.isel(y=slice(r_lo, r_hi), x=slice(c_lo, c_hi))

    # Materialise to in-memory array (numpy or cupy)
    is_cupy = has_cuda_and_cupy() and (
        is_cupy_array(raster.data) or is_cupy_backed(raster))
    if has_dask_array():
        import dask.array as da
        if isinstance(window.data, da.Array):
            window = window.copy()
            window.data = window.data.compute()

    if is_cupy and has_rtx():
        import cupy as cp
        if not is_cupy_array(window.data):
            window.data = cp.asarray(window.data)
        from .gpu_rtx.viewshed import viewshed_gpu
        local_result = viewshed_gpu(
            window, x, y, observer_elev, target_elev)
    else:
        if is_cupy:
            import cupy as cp
            window.data = cp.asnumpy(window.data)
        elif not isinstance(window.data, np.ndarray):
            window.data = np.asarray(window.data)
        local_result = _viewshed_cpu(
            window, x, y, observer_elev, target_elev)

    # Mask cells beyond max_distance (the window is a square, not a circle)
    win_y = local_result.coords['y'].values
    win_x = local_result.coords['x'].values
    wx, wy = np.meshgrid(win_x, win_y)
    dist_sq = (wx - x) ** 2 + (wy - y) ** 2
    outside = dist_sq > max_distance ** 2
    if isinstance(local_result.data, np.ndarray):
        local_result.values[outside] = INVISIBLE
    else:
        # cupy path
        import cupy as cp
        local_result.data[cp.asarray(outside)] = INVISIBLE

    # Embed in full-size INVISIBLE output, preserving array type
    is_dask = has_dask_array() and isinstance(raster.data, da.Array)

    if is_dask:
        # Build output lazily to avoid allocating the full grid in memory.
        # The window result is a small numpy array; the surrounding region
        # is filled with INVISIBLE via dask.array.full.
        local_vals = local_result.values if isinstance(
            local_result.data, np.ndarray) else local_result.data.get()
        full_vis = _dask_embed_window(
            local_vals, height, width, r_lo, r_hi, c_lo, c_hi,
            raster.data.chunks)
    elif is_cupy and has_rtx():
        import cupy as cp
        full_vis = cp.full((height, width), INVISIBLE, dtype=np.float64)
        full_vis[r_lo:r_hi, c_lo:c_hi] = local_result.data
    else:
        local_vals = local_result.values
        full_vis = np.full((height, width), INVISIBLE, dtype=np.float64)
        full_vis[r_lo:r_hi, c_lo:c_hi] = local_vals

    return xarray.DataArray(full_vis, coords=raster.coords,
                            dims=raster.dims, attrs=raster.attrs)


def _viewshed_dask(raster, x, y, observer_elev, target_elev):
    """Dask-backed viewshed (no max_distance — handled by caller).

    Two-tier strategy:
    - Tier B: grid fits in memory → compute and run exact R2 (CPU or GPU).
    - Tier C: out-of-core horizon-profile distance sweep.
    """
    import dask.array as da

    height, width = raster.shape
    y_coords = raster.indexes.get('y').values
    x_coords = raster.indexes.get('x').values

    if not (x_coords.min() <= x <= x_coords.max()):
        raise ValueError("x argument outside of raster x_range")
    if not (y_coords.min() <= y <= y_coords.max()):
        raise ValueError("y argument outside of raster y_range")

    selection = raster.sel(x=[x], y=[y], method='nearest')
    x = selection.x.values[0]
    y = selection.y.values[0]

    obs_r = int(np.where(y_coords == y)[0][0])
    obs_c = int(np.where(x_coords == x)[0][0])

    y_range = (y_coords[0], y_coords[-1])
    x_range = (x_coords[0], x_coords[-1])
    ew_res = (x_range[1] - x_range[0]) / (width - 1) if width > 1 else 1.0
    ns_res = (y_range[1] - y_range[0]) / (height - 1) if height > 1 else 1.0

    cupy_backed = is_dask_cupy(raster)

    # --- Tier B: full grid fits in memory → compute and run exact algo ---
    # Peak memory: event_list sort needs 2x 168*H*W + raster 8*H*W +
    # visibility_grid 8*H*W ≈ 360 bytes/pixel, plus the computed raster.
    r2_bytes = 360 * height * width + 8 * height * width  # working + raster
    avail = _available_memory_bytes()
    if r2_bytes < 0.5 * avail:
        raster_mem = raster.copy()
        raster_mem.data = raster.data.compute()
        if cupy_backed and has_rtx():
            from .gpu_rtx.viewshed import viewshed_gpu
            result = viewshed_gpu(raster_mem, x, y,
                                  observer_elev, target_elev)
        else:
            if cupy_backed:
                import cupy as cp
                raster_mem.data = cp.asnumpy(raster_mem.data)
            result = _viewshed_cpu(raster_mem, x, y,
                                   observer_elev, target_elev)
        result_np = result.data if isinstance(result.data, np.ndarray) \
            else result.data.get()
        vis_da = da.from_array(result_np, chunks=raster.data.chunks)
        return xarray.DataArray(vis_da, coords=raster.coords,
                                dims=raster.dims, attrs=raster.attrs)

    # --- Tier C: out-of-core distance sweep (CPU only) ---
    output_bytes = height * width * 8
    if output_bytes > 0.8 * avail:
        raise MemoryError(
            f"Output grid ({output_bytes / 1e9:.1f} GB) exceeds 80% of "
            f"available RAM ({avail / 1e9:.1f} GB). "
            f"Use max_distance to limit the analysis area."
        )

    # For dask+cupy, chunks compute to cupy arrays — cache needs numpy
    dask_data = raster.data
    if cupy_backed:
        dask_data = dask_data.map_blocks(
            lambda block: block.get(), dtype=np.float64,
            meta=np.array(()))

    obs_elev_val = dask_data.blocks[
        _chunk_index_for(_chunk_offsets(dask_data.chunks[0]), obs_r),
        _chunk_index_for(_chunk_offsets(dask_data.chunks[1]), obs_c),
    ].compute()
    local_r = obs_r - int(_chunk_offsets(dask_data.chunks[0])[
        _chunk_index_for(_chunk_offsets(dask_data.chunks[0]), obs_r)])
    local_c = obs_c - int(_chunk_offsets(dask_data.chunks[1])[
        _chunk_index_for(_chunk_offsets(dask_data.chunks[1]), obs_c)])
    terrain_elev = float(obs_elev_val[local_r, local_c])
    vp_elev = terrain_elev + observer_elev

    visibility = _viewshed_distance_sweep(
        dask_data, height, width, obs_r, obs_c,
        vp_elev, target_elev, ew_res, ns_res,
        dask_data.chunks[0], dask_data.chunks[1],
        None,
    )

    vis_da = da.from_array(visibility, chunks=raster.data.chunks)
    return xarray.DataArray(vis_da, coords=raster.coords,
                            dims=raster.dims, attrs=raster.attrs)
