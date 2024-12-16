from __future__ import annotations
import numpy as np
import graphviz
from IPython.display import display
from typing import Literal


class Value:
    def __init__(
        self, value: float, ancestors: tuple[Value, ...] = (), name="", operand=""
    ):
        self.value = value
        self.ancestors = ancestors
        self.name = name
        self.grad = 0.0
        self._backward = lambda: None
        self.operand = operand

    # make values printable
    def __repr__(self) -> str:
        return f"{self.name}, value={self.value}, grad={self.grad}"

    # Addition
    def __add__(self, other: Value) -> Value:
        if not isinstance(other, Value):
            other = Value(other)
        result = Value(self.value + other.value, (self, other), name="add", operand="+")

        def _backward():
            self.grad += result.grad
            other.grad += result.grad

        result._backward = _backward
        return result

    def __iadd__(self, other: Value) -> Value:
        if not isinstance(other, Value):
            other = Value(other)
        result = Value(
            self.value + other.value, (self, other), name="iadd", operand="+="
        )

        def _backward():
            self.grad += result.grad
            other.grad += result.grad

        result._backward = _backward
        return result

    def __radd__(self, other):
        return self + other

    # Subtraktion
    def __sub__(self, other: Value) -> Value:
        if not isinstance(other, Value):
            other = Value(other)
        result = Value(self.value - other.value, (self, other), name="sub", operand="-")

        def _backward():
            self.grad += 1.0 * result.grad
            other.grad += -1.0 * result.grad

        result._backward = _backward
        return result

    def __rsub__(self, other) -> Value:
        if not isinstance(other, (int, float)):
            raise NotImplementedError()
        return Value(other) - self

    # Multiplikation
    def __mul__(self, other: Value) -> Value:
        if not isinstance(other, Value):
            other = Value(other)
        result = Value(self.value * other.value, (self, other), name="mul", operand="*")

        def _backward():
            self.grad += other.value * result.grad
            other.grad += self.value * result.grad

        result._backward = _backward
        return result

    def __rmul__(self, other) -> Value:
        return self * other

    # Floatingpointdivision
    def __truediv__(self, other: Value) -> Value:
        if not isinstance(other, Value):
            other = Value(other)
        result = Value(self.value / other.value, (self, other), name="div", operand="/")

        def _backward():
            self.grad += 1 / other.value * result.grad
            other.grad += -self.value / other.value**2 * result.grad

        result._backward = _backward
        return result

    def __rtruediv__(self, other) -> Value:
        if not isinstance(other, (int, float)):
            raise NotImplementedError()
        return Value(other) / self

    # Potenzierung (x**n)
    def __pow__(self, other: Value) -> Value:
        if not isinstance(other, Value):
            other = Value(other)
        result = Value(self.value**other.value, (self, other), name="pow", operand="^")

        def _backward():
            self.grad += other.value * self.value ** (other.value - 1.0) * result.grad
            # assert self.value >= 0, "cannot compute log with negative base
            other.grad += self.value**other.value * np.log(self.value) * result.grad
            # print(self.grad, other.grad)

        result._backward = _backward
        return result

    # Exponentierung (e**x)
    def exp(self) -> Value:
        result = Value(np.exp(self.value), (self,), name="exp", operand="e^")

        def _backward():
            self.grad += result.value * result.grad

        result._backward = _backward
        return result

    def log(self) -> Value:
        result_value = np.log(self.value)
        result = Value(result_value, (self,), name="log")

        def _backward():
            if self.value > 0:
                self.grad += (1 / self.value) * result.grad
            else:
                self.grad += 0.0  # Gradient is zero for non-positive input

        result._backward = _backward
        return result

    # Negation
    def __neg__(self) -> Value:
        result = Value(-self.value, (self,), name="neg", operand="-")

        def _backward():
            self.grad += -result.grad

        result._backward = _backward
        return result

    def sigmoid(self) -> Value:
        sigmoid_value = 1 / (1 + np.exp(-self.value))
        result = Value(sigmoid_value, (self,), name="sigmoid")

        def _backward():
            self.grad += sigmoid_value * (1 - sigmoid_value) * result.grad

        result._backward = _backward
        return result

    # how to fix backward with ne values
    def relu(self) -> Value:
        result_value = self.value if self.value > 0 else 0.0
        result = Value(result_value, (self,), name="ReLU")

        def _backward():
            self.grad += self.value * result.grad if self.value > 0 else 0.0

        result._backward = _backward
        return result

    # Vergleichsoperatoren <, >, >=, <=
    def __lt__(self, other: Value) -> bool:
        if not isinstance(other, Value):
            other = Value(other)
        return self.value < other.value

    def __gt__(self, other: Value) -> bool:
        if not isinstance(other, Value):
            other = Value(other)
        return self.value > other.value

    def __le__(self, other: Value) -> bool:
        if not isinstance(other, Value):
            other = Value(other)
        return self.value <= other.value

    def __ge__(self, other: Value) -> bool:
        if not isinstance(other, Value):
            other = Value(other)
        return self.value >= other.value

    def cross_entropy_loss(y_pred: Value, y_gt) -> Value:
        eps = 1e-15

        if y_gt == 0:
            return -((1 - y_pred + eps).log())
        else:
            return -((y_pred + eps).log())

    def backward(self) -> None:
        # iterate through the graph, calculate gradients and update nodes
        topo_sorted_nodes = []
        visited = set()

        # topological sort of the nodes
        def build_topo(node: Value):
            if node not in visited:
                visited.add(node)
                for ancestor in node.ancestors:
                    build_topo(ancestor)
                topo_sorted_nodes.append(node)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo_sorted_nodes):
            node._backward()

    def plot_graph(self):
        # "graph visualization python", graphviz
        dot = graphviz.Digraph(format="svg", graph_attr={"rankdir": "LR"})

        def add_nodes(dot: graphviz.Digraph, node: Value):
            label = f"{node.name}|value={node.value}|grad={node.grad}"
            unique_node_name = str(id(node))

            # add value nodes to graph
            dot.node(
                name=unique_node_name,
                label=label,
                shape="record",
                color=(
                    "lightgreen" if node.ancestors == () and node.name != "" else None
                ),  # check if input
                style="filled",
            )

            if node.operand:  # check if there is an operand to display
                op_name = unique_node_name + node.operand
                # add operation node
                dot.node(
                    name=op_name,
                    label=node.operand,
                )
                # draw edge from operand to result
                dot.edge(op_name, unique_node_name)

            # iterate through the ancestors to build the whole graph
            for ancestor in node.ancestors:
                ancestor_name = add_nodes(dot, ancestor)
                if node.operand:
                    # ensure ancestor edge goes to operand node if it exists
                    dot.edge(ancestor_name, op_name)
                else:
                    dot.edge(ancestor_name, unique_node_name)

            return unique_node_name

        add_nodes(dot, self)
        display(dot)


np.random.seed(0xDEADBEEF)


class Neuron:
    def __init__(self, num_inputs: int) -> None:
        self.weights = [Value(np.random.randn()) for _ in range(num_inputs)]
        self.bias = Value(0.0, name="bias")

    def __call__(self, x: np.ndarray) -> Value:
        # implement f(x) = activation (bias + sum(weights * values))
        if isinstance(x, np.ndarray):
            x = x.flatten()
        res = sum(w_i * x_i for w_i, x_i in zip(self.weights, x)) + self.bias
        return res

    def parameters(self) -> list[Value]:
        return self.weights + [self.bias]

    def param_count(self) -> int:
        return len(self.weights + [self.bias])


class Layer:
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        use_activation: Literal["relu", "sigmoid"],
    ) -> None:
        self.neurons = [Neuron(num_inputs) for _ in range(num_outputs)]
        self.use_activation = use_activation

    def __call__(self, x: np.ndarray) -> list[Value]:
        outputs = [n(x) for n in self.neurons]
        if self.use_activation == "relu":
            return [o.relu() for o in outputs]
        return [o.sigmoid() for o in outputs]

    def parameters(self) -> list:
        params = [p for n in self.neurons for p in n.parameters()]
        return params


class MLP:
    def __init__(self, num_inputs: int, num_hidden: list[int], num_out: int) -> None:
        size = [num_inputs] + num_hidden
        self.layers = [
            Layer(size[i], size[i + 1], "relu") for i in range(len(num_hidden))
        ] + [Layer(num_hidden[-1], num_out, "sigmoid")]

    def __call__(self, x: np.ndarray) -> Value:
        for layer in self.layers:
            x = layer(x)
        return x[0]

    def parameters(self) -> list:
        params = [p for l in self.layers for p in l.parameters()]
        return params

    def epoch_loss_and_accuracy(
        self, images: np.ndarray, labels: np.ndarray
    ) -> tuple[float, float]:
        loss = 0
        correct_pred = 0
        for image, label in zip(images, labels):
            pred = self(image)

            loss += Value.cross_entropy_loss(pred, label)

            if np.fabs(pred.value - label.item()) < 0.5:
                correct_pred += 1.0

        return loss / len(images), correct_pred / len(images)
