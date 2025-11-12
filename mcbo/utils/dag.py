from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Union


def _normalise_node_name(name: str) -> str:
    """Return a stripped node name and validate that it is not empty."""

    if name is None:
        raise ValueError("Node names in the edge list must not be None")
    normalised = name.strip()
    if normalised == "":
        raise ValueError("Node names in the edge list must not be empty")
    return normalised


def _validate_no_cycles(parents: Dict[str, Set[str]], children: Dict[str, Set[str]]):
    """Validate that the adjacency description defines a DAG."""

    indegree = {node: len(parent_set) for node, parent_set in parents.items()}
    queue = [node for node, deg in indegree.items() if deg == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if visited != len(parents):
        raise ValueError("Edge list defines a cyclic graph and cannot be converted to a DAG")


def _validate_column_order(
    columns: Sequence[str],
    parents: Dict[str, Set[str]],
    *,
    source_path: Union[str, Path],
):
    """Ensure that the provided dataset column order respects DAG parents."""

    if len(columns) == 0:
        raise ValueError("columns must contain at least one column name")
    if len(set(columns)) != len(columns):
        raise ValueError("columns must not contain duplicated names")

    position = {name: idx for idx, name in enumerate(columns)}
    missing = set(parents).difference(position)
    if missing:
        raise ValueError(
            "Column order does not include all nodes from the edge list"
            f" {source_path!s}: missing {sorted(missing)}"
        )
    for node, parent_set in parents.items():
        node_position = position[node]
        for parent in parent_set:
            parent_position = position[parent]
            if parent_position > node_position:
                raise ValueError(
                    "Column order is not topological with respect to the edge list"
                    f" {source_path!s}: parent '{parent}' appears after child '{node}'"
                )


@dataclass(frozen=True)
class DAGSpecification:
    """Container describing the DAG associated with a dataset."""

    dag: "DAG"
    node_order: List[str]
    node_to_index: Dict[str, int]
    node_categories: Dict[str, str]
    parent_indices: List[List[int]]
    active_input_indices: List[List[int]]


def load_dag_specification(
    edge_csv_path: Union[str, Path],
    dataset_columns: Sequence[str],
) -> DAGSpecification:
    """Build a :class:`DAG` and metadata from an edge list and dataset columns.

    Parameters
    ----------
    edge_csv_path:
        Path to a CSV file describing edges. The file must contain ``source`` and
        ``target`` columns.
    dataset_columns:
        Sequence describing the column order of the dataset associated with the
        DAG. The order must be topological with respect to the edges.

    Returns
    -------
    DAGSpecification
        An immutable container with the instantiated :class:`DAG`, mappings from
        node names to indices, the node category (``"exogenous"``, ``"endogenous"``
        or ``"target"``) and the action-input mapping compatible with
        ``env_profile["active_input_indices"]``.
    """

    edge_csv_path = Path(edge_csv_path)
    dataset_columns = [column.strip() for column in dataset_columns]

    nodes: Set[str] = set(dataset_columns)
    parents: Dict[str, Set[str]] = {name: set() for name in nodes}
    children: Dict[str, Set[str]] = {name: set() for name in nodes}

    with edge_csv_path.open("r", encoding="utf8") as handle:
        header = handle.readline()
        if not header:
            raise ValueError(f"Edge list file {edge_csv_path!s} is empty")
        columns = [column.strip() for column in header.split(",")]
        if "source" not in columns or "target" not in columns:
            raise ValueError(
                f"Edge list {edge_csv_path!s} must contain 'source' and 'target' columns"
            )
        source_idx = columns.index("source")
        target_idx = columns.index("target")
        for line_number, line in enumerate(handle, start=2):
            values = [value.strip() for value in line.rstrip("\n").split(",")]
            if len(values) <= max(source_idx, target_idx):
                continue
            raw_source = values[source_idx]
            raw_target = values[target_idx]
            if not raw_source or not raw_target:
                continue
            source = _normalise_node_name(raw_source)
            target = _normalise_node_name(raw_target)
            if source not in nodes:
                raise ValueError(
                    f"Edge list references source node '{source}' on line {line_number} "
                    f"that is missing from dataset columns"
                )
            if target not in nodes:
                raise ValueError(
                    f"Edge list references target node '{target}' on line {line_number} "
                    f"that is missing from dataset columns"
                )
            parents[target].add(source)
            children[source].add(target)

    _validate_no_cycles(parents, children)
    _validate_column_order(dataset_columns, parents, source_path=edge_csv_path)

    node_order = list(dataset_columns)
    node_to_index = {node: idx for idx, node in enumerate(node_order)}

    parent_indices: List[List[int]] = []
    node_categories: Dict[str, str] = {}

    exogenous_nodes = [node for node in node_order if len(parents[node]) == 0]
    exogenous_index_map = {node: idx for idx, node in enumerate(exogenous_nodes)}

    active_input_indices: List[List[int]] = []
    for node in node_order:
        node_parents = sorted(parents[node], key=node_to_index.get)
        parent_indices.append([node_to_index[parent] for parent in node_parents])

        if len(children[node]) == 0:
            node_categories[node] = "target"
        elif len(parents[node]) == 0:
            node_categories[node] = "exogenous"
        else:
            node_categories[node] = "endogenous"

        if node in exogenous_index_map:
            active_input_indices.append([exogenous_index_map[node]])
        else:
            active_input_indices.append([])

    dag = DAG(parent_indices)

    return DAGSpecification(
        dag=dag,
        node_order=node_order,
        node_to_index=node_to_index,
        node_categories=node_categories,
        parent_indices=parent_indices,
        active_input_indices=active_input_indices,
    )


class DAG(object):
    """
    Defines a DAG based upon a list of the parents of each node.
    """

    def __init__(self, parent_nodes: List[List[Optional[int]]]):
        self.parent_nodes = parent_nodes
        self.n_nodes = len(parent_nodes)
        self.root_nodes = []
        for k in range(self.n_nodes):
            if len(parent_nodes[k]) == 0:
                self.root_nodes.append(k)

    def get_n_nodes(self):
        return self.n_nodes

    def get_parent_nodes(self, k):
        return self.parent_nodes[k]

    def get_root_nodes(self):
        return self.root_nodes


class FNActionInput(object):
    """
    Defines the action targets based upon a list of the action indices effecting
    each graph node. Used only for function networks.
    """

    def __init__(self, active_input_indices: List[List[Optional[int]]]):
        self.active_input_indices = active_input_indices
        # input dim is the highest action index
        self.input_dim = (
            max([0 if len(x) == 0 else max(x) for x in active_input_indices]) + 1
        )

    def get_input_dim(self):
        return self.input_dim

    def get_active_input_indices(self):
        return self.active_input_indices
