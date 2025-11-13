from pathlib import Path

import pytest

from mcbo.utils.dag import DAGSpecification, load_dag_specification


@pytest.fixture(scope="module")
def output_dag_spec():
    columns = [
        "BranchPredictor",
        "decodeWidth",
        "fetchWidth",
        "intIssueWidth",
        "maxBrCount",
        "memIssueWidth",
        "nDCacheMSHRs",
        "nDCacheWays",
        "nICacheWays",
        "nL2TLBEntries",
        "nL2TLBWays",
        "numFetchBufferEntries",
        "numIntPhysRegisters",
        "numLdqEntries",
        "numRobEntries",
        "area_cell_area",
        "area_cell_count",
        "area_net_area",
        "power_clock",
        "power_internal",
        "power_leakage",
        "power_logic",
        "power_memory",
        "power_pad",
        "power_register",
        "power_switching",
        "area",
        "power",
    ]

    return load_dag_specification(Path("output_edges.csv"), columns)


def test_dag_specification_counts(output_dag_spec: DAGSpecification):
    assert output_dag_spec.dag.get_n_nodes() == 28
    assert len(output_dag_spec.node_order) == 28
    assert set(output_dag_spec.node_categories.values()) == {
        "exogenous",
        "endogenous",
        "target",
    }


def test_parent_indices_for_power(output_dag_spec: DAGSpecification):
    power_index = output_dag_spec.node_to_index["power"]
    parent_indices = output_dag_spec.parent_indices[power_index]
    expected_parents = {
        output_dag_spec.node_to_index[name]
        for name in [
            "power_leakage",
            "power_internal",
            "power_switching",
            "power_memory",
            "power_register",
            "power_logic",
            "power_clock",
            "power_pad",
        ]
    }
    assert set(parent_indices) == expected_parents


def test_active_input_indices(output_dag_spec: DAGSpecification):
    active_indices = output_dag_spec.active_input_indices
    exogenous_nodes = [
        output_dag_spec.node_to_index[name]
        for name in output_dag_spec.node_order
        if output_dag_spec.node_categories[name] == "exogenous"
    ]

    for offset, node_index in enumerate(exogenous_nodes):
        assert active_indices[node_index] == [offset]

    target_index = output_dag_spec.node_to_index["power"]
    assert active_indices[target_index] == []


def test_invalid_column_order_raises(tmp_path: Path):
    # Copy a minimal edge file containing a simple chain A -> B.
    edge_file = tmp_path / "edges.csv"
    edge_file.write_text("source,target\nA,B\n")

    with pytest.raises(ValueError):
        load_dag_specification(edge_file, ["B", "A"])
