from abc import ABC, abstractmethod
from functools import reduce
from operator import mul
from typing import Iterable, Optional, Union

import torch


class Nodes(torch.nn.Module):
    # language=rst
    """
    Abstract base class for groups of neurons.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        learning: bool = True,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract base class constructor.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record decaying spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param learning: Whether to be in learning or testing.
        """
        super().__init__()

        assert (
            n is not None or shape is not None
        ), "Must provide either no. of neurons or shape of layer"

        if n is None:
            self.n = reduce(mul, shape)  # No. of neurons product of shape.
        else:
            self.n = n  # No. of neurons provided.

        if shape is None:
            self.shape = [self.n]  # Shape is equal to the size of the layer.
        else:
            self.shape = shape  # Shape is passed in as an argument.

        assert self.n == reduce(
            mul, self.shape
        ), "No. of neurons and shape do not match"

        self.traces = traces  # Whether to record synaptic traces.
        self.traces_additive = (
            traces_additive  # Whether to record spike traces additively.
        )
        self.register_buffer("s", torch.ByteTensor())  # Spike occurrences.

        self.sum_input = sum_input  # Whether to sum all inputs.

        if self.traces:
            self.register_buffer("x", torch.Tensor())  # Firing traces.
            self.register_buffer(
                "tc_trace", torch.tensor(tc_trace)
            )  # Time constant of spike trace decay.
            self.register_buffer(
                "trace_scale", torch.tensor(trace_scale)
            )  # Scaling factor for spike trace.
            self.register_buffer(
                "trace_decay", torch.empty_like(self.tc_trace)
            )  # Set in compute_decays.

        if self.sum_input:
            self.register_buffer("summed", torch.FloatTensor())  # Summed inputs.

        self.dt = None
        self.batch_size = None
        self.trace_decay = None
        self.learning = learning

    @abstractmethod
    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Abstract base class method for a single simulation step.

        :param x: Inputs to the layer.
        """
        if self.traces:
            # Decay and set spike traces.
            self.x *= self.trace_decay

            if self.traces_additive:
                self.x += self.trace_scale * self.s.float()
            else:
                self.x.masked_fill_(self.s.bool(), self.trace_scale)

        if self.sum_input:
            # Add current input to running sum.
            self.summed += x.float()

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Abstract base class method for resetting state variables.
        """
        self.s.zero_()

        if self.traces:
            self.x.zero_()  # Spike traces.

        if self.sum_input:
            self.summed.zero_()  # Summed inputs.

    def compute_decays(self, dt) -> None:
        # language=rst
        """
        Abstract base class method for setting decays.
        """
        self.dt = torch.tensor(dt)
        if self.traces:
            self.trace_decay = torch.exp(
                -self.dt / self.tc_trace
            )  # Spike trace decay (per timestep).

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        self.batch_size = batch_size
        self.s = torch.zeros(
            batch_size, *self.shape, device=self.s.device, dtype=torch.bool
        )

        if self.traces:
            self.x = torch.zeros(batch_size, *self.shape, device=self.x.device)

        if self.sum_input:
            self.summed = torch.zeros(
                batch_size, *self.shape, device=self.summed.device
            )

    def train(self, mode: bool = True) -> "Nodes":
        # language=rst
        """
        Sets the layer in training mode.

        :param bool mode: Turn training on or off
        :return: self as specified in `torch.nn.Module`
        """
        self.learning = mode
        return super().train(mode)


class AbstractInput(ABC):
    # language=rst
    """
    Abstract base class for groups of input neurons.
    """


class Input(Nodes, AbstractInput):
    # language=rst
    """
    Layer of nodes with user-specified spiking behavior.sadadsfdsf
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of input neurons.
        dsadấdâdsada
        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record decaying spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        """
        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input,
        )

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        On each simulation step, set the spikes of the population equal to the inputs.

        :param x: Inputs to the layer.
        """
        # Set spike occurrences to input values.
        self.s = x

        super().forward(x)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()


class IzhikevichNodes(Nodes):
    # language=rst
    """
    Layer of `Izhikevich neurons<https://www.izhikevich.org/publications/spikes.htm>`_.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        excitatory: float = 1,
        thresh: Union[float, torch.Tensor] = 45.0,
        rest: Union[float, torch.Tensor] = -65.0,
        lbound: float = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of Izhikevich neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param excitatory: Percent of excitatory (vs. inhibitory) neurons in the layer; in range ``[0, 1]``.
        :param thresh: Spike threshold voltage.
        :param rest: Resting membrane voltage.
        :param lbound: Lower bound of the voltage.
        """
        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input,
        )

        self.register_buffer("rest", torch.tensor(rest))  # Rest voltage.
        self.register_buffer("thresh", torch.tensor(thresh))  # Spike threshold voltage.
        self.lbound = lbound

        self.register_buffer("r", None)
        self.register_buffer("a", None)
        self.register_buffer("b", None)
        self.register_buffer("c", None)
        self.register_buffer("d", None)
        self.register_buffer("S", None)
        self.register_buffer("excitatory", None)

        if excitatory > 1:
            excitatory = 1
        elif excitatory < 0:
            excitatory = 0

        if excitatory == 1:
            self.r = torch.rand(n)
            self.a = 0.02 * torch.ones(n)
            self.b = 0.2 * torch.ones(n)
            self.c = -65.0 + 15 * (self.r**2)
            self.d = 8 - 6 * (self.r**2)
            self.S = 0.5 * torch.rand(n, n)
            self.excitatory = torch.ones(n).byte()

        elif excitatory == 0:
            self.r = torch.rand(n)
            self.a = 0.02 + 0.08 * self.r
            self.b = 0.25 - 0.05 * self.r
            self.c = -65.0 * torch.ones(n)
            self.d = 2 * torch.ones(n)
            self.S = -torch.rand(n, n)

            self.excitatory = torch.zeros(n).byte()

        else:
            self.excitatory = torch.zeros(n).byte()

            ex = int(n * excitatory)
            inh = n - ex

            # init
            self.r = torch.zeros(n)
            self.a = torch.zeros(n)
            self.b = torch.zeros(n)
            self.c = torch.zeros(n)
            self.d = torch.zeros(n)
            self.S = torch.zeros(n, n)

            # excitatory
            self.r[:ex] = torch.rand(ex)
            self.a[:ex] = 0.02 * torch.ones(ex)
            self.b[:ex] = 0.2 * torch.ones(ex)
            self.c[:ex] = -65.0 + 15 * self.r[:ex] ** 2
            self.d[:ex] = 8 - 6 * self.r[:ex] ** 2
            self.S[:, :ex] = 0.5 * torch.rand(n, ex)
            self.excitatory[:ex] = 1

            # inhibitory
            self.r[ex:] = torch.rand(inh)
            self.a[ex:] = 0.02 + 0.08 * self.r[ex:]
            self.b[ex:] = 0.25 - 0.05 * self.r[ex:]
            self.c[ex:] = -65.0 * torch.ones(inh)
            self.d[ex:] = 2 * torch.ones(inh)
            self.S[:, ex:] = -torch.rand(n, inh)
            self.excitatory[ex:] = 0

        self.register_buffer("v", self.rest * torch.ones(n))  # Neuron voltages.
        self.register_buffer("u", self.b * self.v)  # Neuron recovery.

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """

        # Voltage and recovery reset.
        self.v = torch.where(self.s, self.c, self.v)
        self.u = torch.where(self.s, self.u + self.d, self.u)

        # Add inter-columnar input.
        if self.s.any():
            x += torch.cat(
                [self.S[:, self.s[i]].sum(dim=1)[None] for i in range(self.s.shape[0])],
                dim=0,
            )

        # Apply v and u updates.
        self.v += self.dt * 0.5 * (0.04 * self.v**2 + 5 * self.v + 140 - self.u + x)
        self.v += self.dt * 0.5 * (0.04 * self.v**2 + 5 * self.v + 140 - self.u + x)
        self.u += self.dt * self.a * (self.b * self.v - self.u)

        # Voltage clipping to lower bound.
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        # Check for spiking neurons.
        self.s = self.v >= self.thresh

        super().forward(x)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.rest)  # Neuron voltages.
        self.u = self.b * self.v  # Neuron recovery.

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.rest * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.u = self.b * self.v


class IzhikevichNodes(Nodes):
    # language=rst
    """
    Layer of `Izhikevich neurons<https://www.izhikevich.org/publications/spikes.htm>`_.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        excitatory: float = 1,
        thresh: Union[float, torch.Tensor] = 45.0,
        rest: Union[float, torch.Tensor] = -65.0,
        lbound: float = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of Izhikevich neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param excitatory: Percent of excitatory (vs. inhibitory) neurons in the layer; in range ``[0, 1]``.
        :param thresh: Spike threshold voltage.
        :param rest: Resting membrane voltage.
        :param lbound: Lower bound of the voltage.
        """
        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input,
        )

        self.register_buffer("rest", torch.tensor(rest))  # Rest voltage.
        self.register_buffer("thresh", torch.tensor(thresh))  # Spike threshold voltage.
        self.lbound = lbound

        self.register_buffer("r", None)
        self.register_buffer("a", None)
        self.register_buffer("b", None)
        self.register_buffer("c", None)
        self.register_buffer("d", None)
        self.register_buffer("S", None)
        self.register_buffer("excitatory", None)

        if excitatory > 1:
            excitatory = 1
        elif excitatory < 0:
            excitatory = 0

        if excitatory == 1:
            self.r = torch.rand(n)
            self.a = 0.02 * torch.ones(n)
            self.b = 0.2 * torch.ones(n)
            self.c = -65.0 + 15 * (self.r**2)
            self.d = 8 - 6 * (self.r**2)
            self.S = 0.5 * torch.rand(n, n)
            self.excitatory = torch.ones(n).byte()

        elif excitatory == 0:
            self.r = torch.rand(n)
            self.a = 0.02 + 0.08 * self.r
            self.b = 0.25 - 0.05 * self.r
            self.c = -65.0 * torch.ones(n)
            self.d = 2 * torch.ones(n)
            self.S = -torch.rand(n, n)

            self.excitatory = torch.zeros(n).byte()

        else:
            self.excitatory = torch.zeros(n).byte()

            ex = int(n * excitatory)
            inh = n - ex

            # init
            self.r = torch.zeros(n)
            self.a = torch.zeros(n)
            self.b = torch.zeros(n)
            self.c = torch.zeros(n)
            self.d = torch.zeros(n)
            self.S = torch.zeros(n, n)

            # excitatory
            self.r[:ex] = torch.rand(ex)
            self.a[:ex] = 0.02 * torch.ones(ex)
            self.b[:ex] = 0.2 * torch.ones(ex)
            self.c[:ex] = -65.0 + 15 * self.r[:ex] ** 2
            self.d[:ex] = 8 - 6 * self.r[:ex] ** 2
            self.S[:, :ex] = 0.5 * torch.rand(n, ex)
            self.excitatory[:ex] = 1

            # inhibitory
            self.r[ex:] = torch.rand(inh)
            self.a[ex:] = 0.02 + 0.08 * self.r[ex:]
            self.b[ex:] = 0.25 - 0.05 * self.r[ex:]
            self.c[ex:] = -65.0 * torch.ones(inh)
            self.d[ex:] = 2 * torch.ones(inh)
            self.S[:, ex:] = -torch.rand(n, inh)
            self.excitatory[ex:] = 0

        self.register_buffer("v", self.rest * torch.ones(n))  # Neuron voltages.
        self.register_buffer("u", self.b * self.v)  # Neuron recovery.

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """

        # Voltage and recovery reset.
        self.v = torch.where(self.s, self.c, self.v)
        self.u = torch.where(self.s, self.u + self.d, self.u)

        # Add inter-columnar input.
        if self.s.any():
            x += torch.cat(
                [self.S[:, self.s[i]].sum(dim=1)[None] for i in range(self.s.shape[0])],
                dim=0,
            )

        # Apply v and u updates.
        self.v += self.dt * 0.5 * (0.04 * self.v**2 + 5 * self.v + 140 - self.u + x)
        self.v += self.dt * 0.5 * (0.04 * self.v**2 + 5 * self.v + 140 - self.u + x)
        self.u += self.dt * self.a * (self.b * self.v - self.u)

        # Voltage clipping to lower bound.
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        # Check for spiking neurons.
        self.s = self.v >= self.thresh

        super().forward(x)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.rest)  # Neuron voltages.
        self.u = self.b * self.v  # Neuron recovery.

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.rest * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.u = self.b * self.v
