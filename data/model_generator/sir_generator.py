from threading import current_thread
from typing import List, Type, Dict, Tuple, Optional
from queue import SimpleQueue

import numpy as np


class SirNode(object):
    def __init__(self, s: int, i: int, r: int):
        self.s = s
        self.i = i
        self.r = r
        self.node_id = "_".join([str(self.s), str(self.i), str(self.r)])

    def get_values(self):
        return (self.s, self.i, self.r)

    def __str__(self) -> str:
        return self.node_id


class SirGenerator(object):
    def __init__(self, s0, i0, r0):
        self.s0 = s0
        self.i0 = i0
        self.r0 = r0
        self.param_label_a = "alpha"
        self.param_label_b = "beta"
        self.model_id = f"sir_{s0}_{i0}_{r0}"
        self.adj_list: Dict[str, List[Tuple[str, Tuple[int, int]]]] = {}
        self.node_list: Dict[str, SirNode] = {}
        self.checked_node: List[str] = []
        self.bscc_nodes: List[str] = []
        self.rates: List[Tuple[int, int]] = []
        self.uniform_rate: Tuple[int, int] = (0, 0)
        self.ctmc_prog: str = ""
        self.dtmc_prog: str = ""

    def _get_adj_nodes(self, node: SirNode):
        if node.node_id not in self.adj_list:
            return KeyError(f"Node {node.node_id} not found.")
        return self.adj_list[node]

    def _is_r1_applicable(self, node: SirNode):
        if node.s > 0 and node.i > 0:
            return True
        return False

    def _is_r2_applicable(self, node: SirNode):
        if node.i > 0:
            return True
        return False

    def _init(self):
        node = SirNode(self.s0, self.i0, self.r0)
        print(f"Init({node})")
        self.adj_list[node.node_id] = []
        self.node_list[node.node_id] = node
        return node

    def _expand(self, node: SirNode):
        if not (self._is_r1_applicable(node) or self._is_r2_applicable(node)):
            self.bscc_nodes.append(node.node_id)
            return
        if node.node_id in self.checked_node:
            return
        self.adj_list[node.node_id] = []
        if self._is_r1_applicable(node):
            new_node = SirNode(node.s - 1, node.i + 1, node.r)
            print(f"Append({new_node}) weight {node.s}a")
            self.node_list[new_node.node_id] = new_node
            self.adj_list[node.node_id].append((new_node.node_id, (node.s, 0)))
            self.rates.append((node.s, 0))
            self._expand(new_node)
        if self._is_r2_applicable(node):
            new_node = SirNode(node.s, node.i - 1, node.r + 1)
            print(f"Append({new_node}) weight {node.i}b")
            self.node_list[new_node.node_id] = new_node
            self.adj_list[node.node_id].append((new_node.node_id, (0, node.i)))
            self.rates.append((0, node.i))
            self._expand(new_node)
        self.checked_node.append(node.node_id)

    def _get_uniform_rate(self):
        coef_a = np.max(np.array([rate[0] for rate in self.rates]))
        coef_b = np.max(np.array([rate[1] for rate in self.rates]))
        self.uniform_rate = (coef_a, coef_b)

    def _uniformize(self):
        self._get_uniform_rate()
        for k, v in self.adj_list.items():
            outgoing_rates = [rate for _, rate in v]
            sum_coef_a = np.sum(np.array([rate[0] for rate in outgoing_rates]))
            sum_coef_b = np.sum(np.array([rate[1] for rate in outgoing_rates]))
            coef_a = self.uniform_rate[0] - sum_coef_a
            coef_b = self.uniform_rate[1] - sum_coef_b
            if coef_a < 0 or coef_b < 0:
                raise ValueError(f"Fuck you {sum_coef_a} and {sum_coef_b}")
            if coef_a == 0 and coef_b == 0:
                continue
            self.adj_list[k].append((k, (coef_a, coef_b)))

    def run(self):
        start_node = self._init()
        self._expand(start_node)
        print(f"GRAPH CTMC")
        for k, v in self.adj_list.items():
            edges = " + ".join([(str(rate) + ":" + str(node)) for node, rate in v])
            print(f"({k}): {edges}")
        print(f"\t BSCC: {self.bscc_nodes}")
        print(f"GRAPH UNIFORMIZED DTMC, Uniformization rate={self.uniform_rate}")
        self._uniformize()
        for k, v in self.adj_list.items():
            edges = " + ".join([(str(rate) + ":" + str(node)) for node, rate in v])
            print(f"({k}): {edges}")
        print(f"\t BSCC: {self.bscc_nodes}")
        self._compose_udtmc_model()
        print(f"COMPOSED PROGRAM DTMC\n{self.dtmc_prog}")
        self._compose_udtmc_pctl()
        print(f"COMPOSED PCTL PROPS\n{self.pctl_props}")

    def _gcmd_node_lhs(self, node: SirNode):
        return f"s={node.s} & i={node.i} & r={node.r}"

    def _gcmd_node_rhs(self, node: SirNode):
        return f"(s'={node.s}) & (i'={node.i}) & (r'={node.r})"

    def _gcmd_rate_rhs(self, rate: Tuple[int, int]):
        terms: List[str] = []
        if rate[0] != 0:
            ca = f"{rate[0]}*{self.param_label_a}"
            terms.append(ca)
        if rate[1] != 0:
            cb = f"{rate[1]}*{self.param_label_b}"
            terms.append(cb)
        expr: str = "+".join(terms)
        return f"({expr})"

    def _gcmd_uniform_rate_rhs(self, rate: Tuple[int, int]):
        terms: List[str] = []
        if rate[0] != 0:
            ca = f"{rate[0]}*{self.param_label_a}"
            terms.append(ca)
        if rate[1] != 0:
            cb = f"{rate[1]}*{self.param_label_b}"
            terms.append(cb)
        rate_expr: str = "+".join(terms)
        uniform_expr = self._gcmd_rate_rhs(self.uniform_rate)
        expr = f"(({rate_expr})/{uniform_expr})"
        return expr

    def _compose_udtmc_model(self):
        trans_expr_lst: List[str] = []
        for node_id, edges in self.adj_list.items():
            lhs_trans_expr: str = self._gcmd_node_lhs(self.node_list[node_id])
            next_terms = []
            for adj_id, rate in edges:
                expr = ":".join(
                    [
                        self._gcmd_uniform_rate_rhs(rate),
                        self._gcmd_node_rhs(self.node_list[adj_id]),
                    ]
                )
                next_terms.append(expr)
            trans_expr_lst.append(
                "[] " + " -> ".join([lhs_trans_expr, " + ".join(next_terms)]) + ";"
            )
        for bscc_id in self.bscc_nodes:
            trans_expr_lst.append(
                "[] "
                + " -> ".join(
                    [
                        self._gcmd_node_lhs(self.node_list[bscc_id]),
                        self._gcmd_node_rhs(self.node_list[bscc_id]),
                    ]
                )
                + ";"
            )
        trans_str = "\n\t".join(trans_expr_lst)
        var_str = f"\n\ts : [0..{self.s0}] init {self.s0};\n\ti : [0..{self.s0}] init {self.i0};\n\tr : [0..{self.s0}] init {self.r0};"
        prog_body: str = f"module {self.model_id}\n {var_str} {trans_str} \nendmodule"
        prog_header: str = f"dtmc\n  const double {self.param_label_a};\n  const double {self.param_label_b};\n"
        prog_foot: str = "\n".join(
            [
                f'label "bscc_{bscc}" = {self._gcmd_node_lhs(self.node_list[bscc])} ;'
                for bscc in self.bscc_nodes
            ]
        )
        prog_foot += f"\n// Number of states: {len(self.node_list)}"
        prog_foot += f"\n// Number of BSCCs: {len(self.bscc_nodes)}"
        self.dtmc_prog = "\n".join([prog_header, prog_body, prog_foot])

    def _compose_udtmc_pctl(self):
        self.pctl_props = "\n".join(
            [f'P=? [F "bscc_{bscc}"]' for bscc in self.bscc_nodes]
        )

    def save(self, model_path: str, props_path: str):
        self._save_model(model_path)
        self._save_props(props_path)

    def _save_model(self, model_path: str):
        with open(model_path, "w") as fptr:
            fptr.write(self.dtmc_prog)

    def _save_props(self, props_path: str):
        with open(props_path, "w") as fptr:
            fptr.write(self.pctl_props)
