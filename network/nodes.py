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
    Layer of nodes with user-specified spiking behavior.
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


class McCullochPitts(Nodes):
    # language=rst
    """
    Layer of `McCulloch-Pitts neurons
    <http://wwwold.ece.utep.edu/research/webfuzzy/docs/kk-thesis/kk-thesis-html/node12.html>`_.
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
        thresh: Union[float, torch.Tensor] = 1.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a McCulloch-Pitts layer of neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
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

        self.register_buffer(
            "thresh", torch.tensor(thresh, dtype=torch.float)
        )  # Spike threshold voltage.
        self.register_buffer("v", torch.FloatTensor())  # Neuron voltages.

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        self.v = x  # Voltages are equal to the inputs.
        self.s = self.v >= self.thresh  # Check for spiking neurons.

        super().forward(x)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = torch.zeros(batch_size, *self.shape, device=self.v.device)


class IFNodes(Nodes):
    # language=rst
    """
    Layer of `integrate-and-fire (IF) neurons <http://neuronaldynamics.epfl.ch/online/Ch1.S3.html>`_.
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
        thresh: Union[float, torch.Tensor] = -52.0,
        reset: Union[float, torch.Tensor] = -65.0,
        refrac: Union[int, torch.Tensor] = 5,
        lbound: float = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of IF neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        :param reset: Post-spike reset voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
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

        self.register_buffer(
            "reset", torch.tensor(reset, dtype=torch.float)
        )  # Post-spike reset voltage.
        self.register_buffer(
            "thresh", torch.tensor(thresh, dtype=torch.float)
        )  # Spike threshold voltage.
        self.register_buffer(
            "refrac", torch.tensor(refrac)
        )  # Post-spike refractory period.
        self.register_buffer("v", torch.FloatTensor())  # Neuron voltages.
        self.register_buffer(
            "refrac_count", torch.FloatTensor()
        )  # Refractory period counters.

        self.lbound = lbound  # Lower bound of voltage.

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        # Integrate input voltages.
        self.v += (self.refrac_count <= 0).float() * x

        # Decrement refractory counters.
        self.refrac_count -= self.dt

        # Check for spiking neurons.
        self.s = self.v >= self.thresh

        # Refractoriness and voltage reset.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)

        # Voltage clipping to lower bound.
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        super().forward(x)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.reset)  # Neuron voltages.
        self.refrac_count.zero_()  # Refractory period counters.

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.reset * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)


class IzhikevichHominNodes(Nodes):
    # language=rst
    """
    Layer of `Izhikevich neurons<https://www.izhikevich.org/publications/spikes.htm>`_.
    """
    TYPE_PRESETS = {
        "RS": {"d": 6},
        "IB": {"d": 4},
        "CH": {"d": 0.5},
        "LTS": {"d": 0.375},
    }

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        excitatory: float = 1.0,
        thresh: Union[float, torch.Tensor] = 4.5,  # scaled ~ 45mV
        rest: Union[float, torch.Tensor] = -6.5,  # scaled ~ -65mV
        type_ex: str = "RS",
        type_in: str = "RS",
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

        # --- Setup c and d ---
        tkey_ex = str(type_ex).upper()
        if tkey_ex not in self.TYPE_PRESETS:
            tkey_ex = "RS"

        tkey_in = str(type_in).upper()
        if tkey_in not in self.TYPE_PRESETS:
            tkey_in = "RS"

        d_preset_ex = self.TYPE_PRESETS[tkey_ex]["d"]
        d_preset_in = self.TYPE_PRESETS[tkey_in]["d"]

        # self.register_buffer("r", None)
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

        self.a = 2**-6 * torch.ones(n)
        self.b = 2**-2 * torch.ones(n)
        self.c = -6.5 * torch.ones(n)

        if excitatory == 1:
            # self.r = torch.rand(n)
            self.d = d_preset_ex * torch.ones(n)
            self.S = 0.5 * torch.rand(n, n)
            self.excitatory = torch.ones(n).byte()

        elif excitatory == 0:
            # self.r = torch.rand(n)
            self.d = d_preset_in * torch.ones(n)
            self.S = -torch.rand(n, n)
            self.excitatory = torch.zeros(n).byte()

        else:
            self.excitatory = torch.zeros(n).byte()

            ex = int(n * excitatory)
            inh = n - ex

            # init
            self.d = torch.zeros(n)
            self.S = torch.zeros(n, n)

            # excitatory
            self.d[:ex] = d_preset_ex * torch.ones(ex)
            self.S[:, :ex] = 0.5 * torch.rand(n, ex)
            self.excitatory[:ex] = 1

            # inhibitory
            self.d[ex:] = d_preset_in * torch.ones(inh)
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
        self.v += self.dt * 0.5 * (2**-2 * self.v**2 + 2**2 * self.v + 14 - self.u + x)
        self.v += self.dt * 0.5 * (2**-2 * self.v**2 + 2**2 * self.v + 14 - self.u + x)
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


class LIFNodes(Nodes):
    # language=rst
    """
    Layer of `leaky integrate-and-fire (LIF) neurons
    <http://web.archive.org/web/20190318204706/http://icwww.epfl.ch/~gerstner/SPNM/node26.html#SECTION02311000000000000000>`_.
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
        thresh: Union[float, torch.Tensor] = -52.0,
        rest: Union[float, torch.Tensor] = -65.0,
        reset: Union[float, torch.Tensor] = -65.0,
        refrac: Union[int, torch.Tensor] = 5,
        tc_decay: Union[float, torch.Tensor] = 100.0,
        lbound: float = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of LIF neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        :param rest: Resting membrane voltage.
        :param reset: Post-spike reset voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param tc_decay: Time constant of neuron voltage decay.
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

        self.register_buffer(
            "rest", torch.tensor(rest, dtype=torch.float)
        )  # Rest voltage.
        self.register_buffer(
            "reset", torch.tensor(reset, dtype=torch.float)
        )  # Post-spike reset voltage.
        self.register_buffer(
            "thresh", torch.tensor(thresh, dtype=torch.float)
        )  # Spike threshold voltage.
        self.register_buffer(
            "refrac", torch.tensor(refrac)
        )  # Post-spike refractory period.
        self.register_buffer(
            "tc_decay", torch.tensor(tc_decay, dtype=torch.float)
        )  # Time constant of neuron voltage decay.
        self.register_buffer(
            "decay", torch.zeros(*self.shape)
        )  # Set in compute_decays.
        self.register_buffer("v", torch.FloatTensor())  # Neuron voltages.
        self.register_buffer(
            "refrac_count", torch.FloatTensor()
        )  # Refractory period counters.

        if lbound is None:
            self.lbound = None  # Lower bound of voltage.
        else:
            self.lbound = torch.tensor(
                lbound, dtype=torch.float
            )  # Lower bound of voltage.

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        # Decay voltages.
        self.v = self.decay * (self.v - self.rest) + self.rest

        # Integrate inputs.
        x.masked_fill_(self.refrac_count > 0, 0.0)

        # Decrement refractory counters.
        self.refrac_count -= self.dt

        self.v += x  # interlaced

        # Check for spiking neurons.
        self.s = self.v >= self.thresh

        # Refractoriness and voltage reset.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)

        # Voltage clipping to lower bound.
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        super().forward(x)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.rest)  # Neuron voltages.
        self.refrac_count.zero_()  # Refractory period counters.

    def compute_decays(self, dt) -> None:
        # language=rst
        """
        Sets the relevant decays.
        """
        super().compute_decays(dt=dt)
        self.decay = torch.exp(
            -self.dt / self.tc_decay
        )  # Neuron voltage decay (per timestep).

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.rest * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)


class BoostedLIFNodes(Nodes):
    # Same as LIFNodes, faster: no rest, no reset, no lbound
    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        thresh: Union[float, torch.Tensor] = 13.0,
        refrac: Union[int, torch.Tensor] = 5,
        tc_decay: Union[float, torch.Tensor] = 100.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of LIF neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        :param reset: Post-spike reset voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param tc_decay: Time constant of neuron voltage decay.
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

        self.register_buffer(
            "thresh", torch.tensor(thresh, dtype=torch.float)
        )  # Spike threshold voltage.
        self.register_buffer(
            "refrac", torch.tensor(refrac)
        )  # Post-spike refractory period.
        self.register_buffer(
            "tc_decay", torch.tensor(tc_decay, dtype=torch.float)
        )  # Time constant of neuron voltage decay.
        self.register_buffer(
            "decay", torch.zeros(*self.shape)
        )  # Set in compute_decays.
        self.register_buffer("v", torch.FloatTensor())  # Neuron voltages.
        self.register_buffer(
            "refrac_count", torch.tensor(0)
        )  # Refractory period counters.

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        # Decay voltages.
        self.v *= self.decay

        # Integrate inputs.
        if x is not None:
            x.masked_fill_(self.refrac_count > 0, 0.0)

        # Decrement refractory counters.
        self.refrac_count -= self.dt

        if x is not None:
            self.v += x

        # Check for spiking neurons.
        self.s = self.v >= self.thresh

        # Refractoriness and voltage reset.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, 0)

        super().forward(x)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(0)  # Neuron voltages.
        self.refrac_count.zero_()  # Refractory period counters.

    def compute_decays(self, dt) -> None:
        # language=rst
        """
        Sets the relevant decays.
        """
        super().compute_decays(dt=dt)
        self.decay = torch.exp(
            -self.dt / self.tc_decay
        )  # Neuron voltage decay (per timestep).

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = torch.zeros(batch_size, *self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)


class CurrentLIFNodes(Nodes):
    # language=rst
    """
    Layer of `current-based leaky integrate-and-fire (LIF) neurons
    <http://web.archive.org/web/20190318204706/http://icwww.epfl.ch/~gerstner/SPNM/node26.html#SECTION02313000000000000000>`_.
    Total synaptic input current is modeled as a decaying memory of input spikes multiplied by synaptic strengths.
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
        thresh: Union[float, torch.Tensor] = -52.0,
        rest: Union[float, torch.Tensor] = -65.0,
        reset: Union[float, torch.Tensor] = -65.0,
        refrac: Union[int, torch.Tensor] = 5,
        tc_decay: Union[float, torch.Tensor] = 100.0,
        tc_i_decay: Union[float, torch.Tensor] = 2.0,
        lbound: float = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of synaptic input current-based LIF neurons.
        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        :param rest: Resting membrane voltage.
        :param reset: Post-spike reset voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param tc_decay: Time constant of neuron voltage decay.
        :param tc_i_decay: Time constant of synaptic input current decay.
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
        self.register_buffer("reset", torch.tensor(reset))  # Post-spike reset voltage.
        self.register_buffer("thresh", torch.tensor(thresh))  # Spike threshold voltage.
        self.register_buffer(
            "refrac", torch.tensor(refrac)
        )  # Post-spike refractory period.
        self.register_buffer(
            "tc_decay", torch.tensor(tc_decay)
        )  # Time constant of neuron voltage decay.
        self.register_buffer(
            "decay", torch.empty_like(self.tc_decay)
        )  # Set in compute_decays.
        self.register_buffer(
            "tc_i_decay", torch.tensor(tc_i_decay)
        )  # Time constant of synaptic input current decay.
        self.register_buffer(
            "i_decay", torch.empty_like(self.tc_i_decay)
        )  # Set in compute_decays.

        self.register_buffer("v", torch.FloatTensor())  # Neuron voltages.
        self.register_buffer("i", torch.FloatTensor())  # Synaptic input currents.
        self.register_buffer(
            "refrac_count", torch.FloatTensor()
        )  # Refractory period counters.

        self.lbound = lbound  # Lower bound of voltage.

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        # Decay voltages and current.
        self.v = self.decay * (self.v - self.rest) + self.rest
        self.i *= self.i_decay

        # Decrement refractory counters.
        self.refrac_count -= self.dt

        # Integrate inputs.
        self.i += x
        self.v += (self.refrac_count <= 0).float() * self.i

        # Check for spiking neurons.
        self.s = self.v >= self.thresh

        # Refractoriness and voltage reset.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)

        # Voltage clipping to lower bound.
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        super().forward(x)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.rest)  # Neuron voltages.
        self.i.zero_()  # Synaptic input currents.
        self.refrac_count.zero_()  # Refractory period counters.

    def compute_decays(self, dt) -> None:
        # language=rst
        """
        Sets the relevant decays.
        """
        super().compute_decays(dt=dt)
        self.decay = torch.exp(
            -self.dt / self.tc_decay
        )  # Neuron voltage decay (per timestep).
        self.i_decay = torch.exp(
            -self.dt / self.tc_i_decay
        )  # Synaptic input current decay (per timestep).

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.rest * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.i = torch.zeros_like(self.v, device=self.i.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)


class AdaptiveLIFNodes(Nodes):
    # language=rst
    """
    Layer of leaky integrate-and-fire (LIF) neurons with adaptive thresholds. A neuron's voltage threshold is increased
    by some constant each time it spikes; otherwise, it is decaying back to its default value.
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
        rest: Union[float, torch.Tensor] = -65.0,
        reset: Union[float, torch.Tensor] = -65.0,
        thresh: Union[float, torch.Tensor] = -52.0,
        refrac: Union[int, torch.Tensor] = 5,
        tc_decay: Union[float, torch.Tensor] = 100.0,
        theta_plus: Union[float, torch.Tensor] = 0.05,
        tc_theta_decay: Union[float, torch.Tensor] = 1e7,
        lbound: float = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of LIF neurons with adaptive firing thresholds.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param rest: Resting membrane voltage.
        :param reset: Post-spike reset voltage.
        :param thresh: Spike threshold voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param tc_decay: Time constant of neuron voltage decay.
        :param theta_plus: Voltage increase of threshold after spiking.
        :param tc_theta_decay: Time constant of adaptive threshold decay.
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
        self.register_buffer("reset", torch.tensor(reset))  # Post-spike reset voltage.
        self.register_buffer("thresh", torch.tensor(thresh))  # Spike threshold voltage.
        self.register_buffer(
            "refrac", torch.tensor(refrac)
        )  # Post-spike refractory period.
        self.register_buffer(
            "tc_decay", torch.tensor(tc_decay)
        )  # Time constant of neuron voltage decay.
        self.register_buffer(
            "decay", torch.empty_like(self.tc_decay, dtype=torch.float32)
        )  # Set in compute_decays.
        self.register_buffer(
            "theta_plus", torch.tensor(theta_plus)
        )  # Constant threshold increase on spike.
        self.register_buffer(
            "tc_theta_decay", torch.tensor(tc_theta_decay)
        )  # Time constant of adaptive threshold decay.
        self.register_buffer(
            "theta_decay", torch.empty_like(self.tc_theta_decay)
        )  # Set in compute_decays.

        self.register_buffer("v", torch.FloatTensor())  # Neuron voltages.
        self.register_buffer("theta", torch.zeros(*self.shape))  # Adaptive thresholds.
        self.register_buffer(
            "refrac_count", torch.FloatTensor()
        )  # Refractory period counters.
        self.lbound = lbound  # Lower bound of voltage.

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        # Decay voltages and adaptive thresholds.
        self.v = self.decay * (self.v - self.rest) + self.rest
        if self.learning:
            self.theta *= self.theta_decay

        # Integrate inputs.
        self.v += (self.refrac_count <= 0).float() * x

        # Decrement refractory counters.
        self.refrac_count -= self.dt

        # Check for spiking neurons.
        self.s = self.v >= self.thresh + self.theta

        # Refractoriness, voltage reset, and adaptive thresholds.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)
        if self.learning:
            self.theta += self.theta_plus * self.s.float().sum(0)

        # voltage clipping to lowerbound
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        super().forward(x)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.rest)  # Neuron voltages.
        self.refrac_count.zero_()  # Refractory period counters.

    def compute_decays(self, dt) -> None:
        # language=rst
        """
        Sets the relevant decays.
        """
        super().compute_decays(dt=dt)
        self.decay = torch.exp(
            -self.dt / self.tc_decay
        )  # Neuron voltage decay (per timestep).
        self.theta_decay = torch.exp(
            -self.dt / self.tc_theta_decay
        )  # Adaptive threshold decay (per timestep).

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.rest * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)


class DiehlAndCookNodes(Nodes):
    # language=rst
    """
    Layer of leaky integrate-and-fire (LIF) neurons with adaptive thresholds (modified for Diehl & Cook 2015
    replication).
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
        thresh: Union[float, torch.Tensor] = -52.0,
        rest: Union[float, torch.Tensor] = -65.0,
        reset: Union[float, torch.Tensor] = -65.0,
        refrac: Union[int, torch.Tensor] = 5,
        tc_decay: Union[float, torch.Tensor] = 100.0,
        theta_plus: Union[float, torch.Tensor] = 0.05,
        tc_theta_decay: Union[float, torch.Tensor] = 1e7,
        lbound: float = None,
        one_spike: bool = True,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of Diehl & Cook 2015 neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        :param rest: Resting membrane voltage.
        :param reset: Post-spike reset voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param tc_decay: Time constant of neuron voltage decay.
        :param theta_plus: Voltage increase of threshold after spiking.
        :param tc_theta_decay: Time constant of adaptive threshold decay.
        :param lbound: Lower bound of the voltage.
        :param one_spike: Whether to allow only one spike per timestep.
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
        self.register_buffer("reset", torch.tensor(reset))  # Post-spike reset voltage.
        self.register_buffer("thresh", torch.tensor(thresh))  # Spike threshold voltage.
        self.register_buffer(
            "refrac", torch.tensor(refrac)
        )  # Post-spike refractory period.
        self.register_buffer(
            "tc_decay", torch.tensor(tc_decay)
        )  # Time constant of neuron voltage decay.
        self.register_buffer(
            "decay", torch.empty_like(self.tc_decay)
        )  # Set in compute_decays.
        self.register_buffer(
            "theta_plus", torch.tensor(theta_plus)
        )  # Constant threshold increase on spike.
        self.register_buffer(
            "tc_theta_decay", torch.tensor(tc_theta_decay)
        )  # Time constant of adaptive threshold decay.
        self.register_buffer(
            "theta_decay", torch.empty_like(self.tc_theta_decay)
        )  # Set in compute_decays.
        self.register_buffer("v", torch.FloatTensor())  # Neuron voltages.
        self.register_buffer("theta", torch.zeros(*self.shape))  # Adaptive thresholds.
        self.register_buffer(
            "refrac_count", torch.FloatTensor()
        )  # Refractory period counters.

        self.lbound = lbound  # Lower bound of voltage.
        self.one_spike = one_spike  # One spike per timestep.

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        # Decay voltages and adaptive thresholds.
        self.v = self.decay * (self.v - self.rest) + self.rest
        if self.learning:
            self.theta *= self.theta_decay

        # Integrate inputs.
        self.v += (self.refrac_count <= 0).float() * x

        # Decrement refractory counters.
        self.refrac_count -= self.dt

        # Check for spiking neurons.
        self.s = self.v >= self.thresh + self.theta

        # Refractoriness, voltage reset, and adaptive thresholds.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)
        if self.learning:
            self.theta += self.theta_plus * self.s.float().sum(0)

        # Choose only a single neuron to spike.
        if self.one_spike:
            if self.s.any():
                _any = self.s.view(self.batch_size, -1).any(1)
                ind = torch.multinomial(
                    self.s.float().view(self.batch_size, -1)[_any], 1
                )
                _any = _any.nonzero()
                self.s.zero_()
                self.s.view(self.batch_size, -1)[_any, ind] = 1

        # Voltage clipping to lower bound.
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        super().forward(x)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.rest)  # Neuron voltages.
        self.refrac_count.zero_()  # Refractory period counters.

    def compute_decays(self, dt) -> None:
        # language=rst
        """
        Sets the relevant decays.
        """
        super().compute_decays(dt=dt)
        self.decay = torch.exp(
            -self.dt / self.tc_decay
        )  # Neuron voltage decay (per timestep).
        self.theta_decay = torch.exp(
            -self.dt / self.tc_theta_decay
        )  # Adaptive threshold decay (per timestep).

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.rest * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)


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
            self.c = -65.0 * torch.ones(n)
            self.d = 8 * torch.ones(n)
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


class CSRMNodes(Nodes):
    """
    A layer of Cumulative Spike Response Model (Gerstner and van Hemmen 1992, Gerstner et al. 1996) nodes.
    It accounts for a model where refractoriness and adaptation were modeled by the combined effects
    of the spike after potentials of several previous spikes, rather than only the most recent spike.
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
        rest: Union[float, torch.Tensor] = -65.0,
        thresh: Union[float, torch.Tensor] = -52.0,
        responseKernel: str = "ExponentialKernel",
        refractoryKernel: str = "EtaKernel",
        tau: Union[float, torch.Tensor] = 1,
        res_window_size: Union[float, torch.Tensor] = 20,
        ref_window_size: Union[float, torch.Tensor] = 10,
        reset_const: Union[float, torch.Tensor] = 50,
        tc_decay: Union[float, torch.Tensor] = 100.0,
        theta_plus: Union[float, torch.Tensor] = 0.05,
        tc_theta_decay: Union[float, torch.Tensor] = 1e7,
        lbound: float = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of Cumulative Spike Response Model nodes.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param rest: Resting membrane voltage.
        :param thresh: Spike threshold voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param tc_decay: Time constant of neuron voltage decay.
        :param theta_plus: Voltage increase of threshold after spiking.
        :param tc_theta_decay: Time constant of adaptive threshold decay.
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
        self.register_buffer(
            "tau", torch.tensor(tau)
        )  # Time constant of Spike Response Kernel
        self.register_buffer(
            "reset_const", torch.tensor(reset_const)
        )  # Reset constant of refractory kernel
        self.register_buffer(
            "res_window_size", torch.tensor(res_window_size)
        )  # Window size for sampling incoming input current
        self.register_buffer(
            "ref_window_size", torch.tensor(ref_window_size)
        )  # Window size for sampling previous spikes
        self.register_buffer(
            "tc_decay", torch.tensor(tc_decay)
        )  # Time constant of neuron voltage decay.
        self.register_buffer(
            "decay", torch.empty_like(self.tc_decay, dtype=torch.float32)
        )  # Set in compute_decays.
        self.register_buffer(
            "theta_plus", torch.tensor(theta_plus)
        )  # Constant threshold increase on spike.
        self.register_buffer(
            "tc_theta_decay", torch.tensor(tc_theta_decay)
        )  # Time constant of adaptive threshold decay.
        self.register_buffer(
            "theta_decay", torch.empty_like(self.tc_theta_decay)
        )  # Set in compute_decays.

        self.register_buffer("v", torch.FloatTensor())  # Neuron voltages.
        self.register_buffer(
            "last_spikes", torch.ByteTensor()
        )  # Previous spikes occurrences in time window
        self.register_buffer("theta", torch.zeros(*self.shape))  # Adaptive thresholds.
        self.lbound = lbound  # Lower bound of voltage.

        self.responseKernel = responseKernel  # Type of spike response kernel used

        self.refractoryKernel = refractoryKernel  # Type of refractory kernel used

        self.register_buffer(
            "resKernel", torch.FloatTensor()
        )  # Vector of synaptic response kernel values over a window of time

        self.register_buffer(
            "refKernel", torch.FloatTensor()
        )  # Vector of refractory kernel values over a window of time

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        # Decay voltages.
        self.v *= self.decay

        if self.learning:
            self.theta *= self.theta_decay

        # Integrate inputs.
        v = torch.einsum(
            "i,kij->kj", self.resKernel, x
        )  # Response due to incoming current
        v += torch.einsum(
            "i,kij->kj", self.refKernel, self.last_spikes
        )  # Refractoriness due to previous spikes
        self.v += v.view(x.size(0), *self.shape)

        # Check for spiking neurons.
        self.s = self.v >= self.thresh + self.theta

        if self.learning:
            self.theta += self.theta_plus * self.s.float().sum(0)

        # Add the spike vector into the first in first out matrix of windowed (ref) spike trains
        self.last_spikes = torch.cat(
            (self.last_spikes[:, 1:, :], self.s[:, None, :]), 1
        )

        # Voltage clipping to lower bound.
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        super().forward(x)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.rest)  # Neuron voltages.

    def compute_decays(self, dt) -> None:
        # language=rst
        """
        Sets the relevant decays.
        """
        super().compute_decays(dt=dt)
        self.decay = torch.exp(
            -self.dt / self.tc_decay
        )  # Neuron voltage decay (per timestep).
        self.theta_decay = torch.exp(
            -self.dt / self.tc_theta_decay
        )  # Adaptive threshold decay (per timestep).

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.rest * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.last_spikes = torch.zeros(batch_size, self.ref_window_size, *self.shape)

        resKernels = {
            "AlphaKernel": self.AlphaKernel,
            "AlphaKernelSLAYER": self.AlphaKernelSLAYER,
            "LaplacianKernel": self.LaplacianKernel,
            "ExponentialKernel": self.ExponentialKernel,
            "RectangularKernel": self.RectangularKernel,
            "TriangularKernel": self.TriangularKernel,
        }

        if self.responseKernel not in resKernels.keys():
            raise Exception(" The given response Kernel is not implemented")

        self.resKernel = resKernels[self.responseKernel](self.dt)

        refKernels = {"EtaKernel": self.EtaKernel}

        if self.refractoryKernel not in refKernels.keys():
            raise Exception(" The given refractory Kernel is not implemented")

        self.refKernel = refKernels[self.refractoryKernel](self.dt)

    def AlphaKernel(self, dt):
        t = torch.arange(0, self.res_window_size, dt)
        kernelVec = (1 / (self.tau**2)) * t * torch.exp(-t / self.tau)
        return torch.flip(kernelVec, [0])

    def AlphaKernelSLAYER(self, dt):
        t = torch.arange(0, self.res_window_size, dt)
        kernelVec = (1 / self.tau) * t * torch.exp(1 - t / self.tau)
        return torch.flip(kernelVec, [0])

    def LaplacianKernel(self, dt):
        t = torch.arange(0, self.res_window_size, dt)
        kernelVec = (1 / (self.tau * 2)) * torch.exp(-1 * torch.abs(t / self.tau))
        return torch.flip(kernelVec, [0])

    def ExponentialKernel(self, dt):
        t = torch.arange(0, self.res_window_size, dt)
        kernelVec = (1 / self.tau) * torch.exp(-t / self.tau)
        return torch.flip(kernelVec, [0])

    def RectangularKernel(self, dt):
        t = torch.arange(0, self.res_window_size, dt)
        kernelVec = 1 / (self.tau * 2)
        return torch.flip(kernelVec, [0])

    def TriangularKernel(self, dt):
        t = torch.arange(0, self.res_window_size, dt)
        kernelVec = (1 / self.tau) * (1 - (t / self.tau))
        return torch.flip(kernelVec, [0])

    def EtaKernel(self, dt):
        t = torch.arange(0, self.ref_window_size, dt)
        kernelVec = -self.reset_const * torch.exp(-t / self.tau)
        return torch.flip(kernelVec, [0])


class SRM0Nodes(Nodes):
    # language=rst
    """
    Layer of simplified spike response model (SRM0) neurons with stochastic threshold (escape noise). Adapted from
    `(Vasilaki et al., 2009) <https://intranet.physio.unibe.ch/Publikationen/Dokumente/Vasilaki2009PloSComputBio_1.pdf>`_.
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
        thresh: Union[float, torch.Tensor] = -50.0,
        rest: Union[float, torch.Tensor] = -70.0,
        reset: Union[float, torch.Tensor] = -70.0,
        refrac: Union[int, torch.Tensor] = 5,
        tc_decay: Union[float, torch.Tensor] = 10.0,
        lbound: float = None,
        eps_0: Union[float, torch.Tensor] = 1.0,
        rho_0: Union[float, torch.Tensor] = 1.0,
        d_thresh: Union[float, torch.Tensor] = 5.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of SRM0 neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        :param rest: Resting membrane voltage.
        :param reset: Post-spike reset voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param tc_decay: Time constant of neuron voltage decay.
        :param lbound: Lower bound of the voltage.
        :param eps_0: Scaling factor for pre-synaptic spike contributions.
        :param rho_0: Stochastic intensity at threshold.
        :param d_thresh: Width of the threshold region.
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
        self.register_buffer("reset", torch.tensor(reset))  # Post-spike reset voltage.
        self.register_buffer("thresh", torch.tensor(thresh))  # Spike threshold voltage.
        self.register_buffer(
            "refrac", torch.tensor(refrac)
        )  # Post-spike refractory period.
        self.register_buffer(
            "tc_decay", torch.tensor(tc_decay)
        )  # Time constant of neuron voltage decay.
        self.register_buffer("decay", torch.tensor(tc_decay))  # Set in compute_decays.
        self.register_buffer(
            "eps_0", torch.tensor(eps_0)
        )  # Scaling factor for pre-synaptic spike contributions.
        self.register_buffer(
            "rho_0", torch.tensor(rho_0)
        )  # Stochastic intensity at threshold.
        self.register_buffer(
            "d_thresh", torch.tensor(d_thresh)
        )  # Width of the threshold region.
        self.register_buffer("v", torch.FloatTensor())  # Neuron voltages.
        self.register_buffer(
            "refrac_count", torch.FloatTensor()
        )  # Refractory period counters.

        self.lbound = lbound  # Lower bound of voltage.

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        # Decay voltages.
        self.v = self.decay * (self.v - self.rest) + self.rest

        # Integrate inputs.
        self.v += (self.refrac_count <= 0).float() * self.eps_0 * x

        # Compute (instantaneous) probabilities of spiking, clamp between 0 and 1 using exponentials.
        # Also known as 'escape noise', this simulates nearby neurons.
        self.rho = self.rho_0 * torch.exp((self.v - self.thresh) / self.d_thresh)
        self.s_prob = 1.0 - torch.exp(-self.rho * self.dt)

        # Decrement refractory counters.
        self.refrac_count -= self.dt

        # Check for spiking neurons (spike when probability > some random number).
        self.s = torch.rand_like(self.s_prob) < self.s_prob

        # Refractoriness and voltage reset.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)

        # Voltage clipping to lower bound.
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        super().forward(x)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.rest)  # Neuron voltages.
        self.refrac_count.zero_()  # Refractory period counters.

    def compute_decays(self, dt) -> None:
        # language=rst
        """
        Sets the relevant decays.
        """
        super().compute_decays(dt=dt)
        self.decay = torch.exp(
            -self.dt / self.tc_decay
        )  # Neuron voltage decay (per timestep).

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.rest * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)


"""
Custom IzhikevichNodes với approximate multiplier từ evoapproxlib
"""


import torch
import numpy as np
from typing import Optional, Iterable, Union
from bindsnet.network.nodes import Nodes

# Import approximate multiplier
try:
    import pyximport

    pyximport.install()
    import mul12s_2PT

    def u2s(v):  # 24b unsigned to 24b signed
        if v & 8388608:  # Bit 23 (MSB)
            return v - 16777216  # 2^24
        return v

    def approx_mul(a, b):
        """
        Approximate multiplication cho 2 số nguyên 12-bit signed
        Input: a, b trong range [-2048, 2047]
        Output: approximate result của a * b
        """
        # Clamp về range 12-bit signed
        a = max(-2048, min(2047, int(a)))
        b = max(-2048, min(2047, int(b)))

        # Sử dụng approximate multiplier
        result = u2s(mul12s_2PT.mul(a, b))
        return result

    APPROX_MUL_AVAILABLE = True
except ImportError:
    print("Warning: mul12s_2PT not available, using standard multiplication")
    APPROX_MUL_AVAILABLE = False

    def approx_mul(a, b):
        return int(a) * int(b)


class IzhikevichNodesApprox(Nodes):
    """
    Izhikevich neurons với approximate multiplier.
    Lưu ý: Approximate multiplier chỉ làm việc với số nguyên, nên chúng ta cần
    scale các giá trị float về integer range trước khi nhân.
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
        scale_factor: int = 100,  # Scale factor để chuyển float -> int
        **kwargs,
    ) -> None:
        """
        Instantiates a layer of Izhikevich neurons với approximate multiplier.

        :param scale_factor: Factor để scale float values về integer range cho approximate multiplier
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

        self.register_buffer("rest", torch.tensor(rest))
        self.register_buffer("thresh", torch.tensor(thresh))
        self.lbound = lbound
        self.scale_factor = scale_factor  # Để scale float -> int

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
            self.c = -65.0 * torch.ones(n)
            self.d = 8 * torch.ones(n)
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

        self.register_buffer("v", self.rest * torch.ones(n))
        self.register_buffer("u", self.b * self.v)

    def _approx_multiply_tensor(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Approximate multiplication cho tensors.
        Scale về integer, nhân approximate, scale lại về float.
        """
        if not APPROX_MUL_AVAILABLE:
            return a * b

        # Scale về integer range (12-bit signed: -2048 to 2047)
        a_int = (a * self.scale_factor).clamp(-2048, 2047).int()
        b_int = (b * self.scale_factor).clamp(-2048, 2047).int()

        # Vectorized approximate multiplication
        # Sử dụng list comprehension và stack để vectorize
        flat_a = a_int.flatten().cpu().numpy()
        flat_b = b_int.flatten().cpu().numpy()

        # Apply approximate multiplication element-wise
        result_list = [
            approx_mul(int(flat_a[i]), int(flat_b[i])) for i in range(len(flat_a))
        ]

        result = torch.tensor(result_list, device=a.device, dtype=torch.float32)
        result = result.reshape(a_int.shape)

        # Scale lại về float và chia cho scale_factor^2 (vì đã scale cả 2 số)
        return result / (self.scale_factor**2)

    def forward(self, x: torch.Tensor) -> None:
        """
        Runs a single simulation step với approximate multiplier.
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

        # Apply v and u updates với approximate multiplication
        # Thay thế các phép nhân quan trọng bằng approximate multiplier

        # v^2 term với approximate multiplier
        v_squared_approx = self._approx_multiply_tensor(self.v, self.v)

        # 0.04 * v^2
        term1 = 0.04 * v_squared_approx  # Giữ phép nhân này vì là constant * tensor

        # 5 * v
        term2 = 5 * self.v  # Giữ phép nhân này vì là constant * tensor

        # b * v (trong recovery update)
        b_v_approx = self._approx_multiply_tensor(self.b, self.v)

        # Update v với approximate terms
        self.v += self.dt * 0.5 * (term1 + term2 + 140 - self.u + x)
        self.v += self.dt * 0.5 * (term1 + term2 + 140 - self.u + x)

        # Update u với approximate multiplication
        self.u += self.dt * self.a * (b_v_approx - self.u)

        # Voltage clipping to lower bound.
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        # Check for spiking neurons.
        self.s = self.v >= self.thresh

        super().forward(x)

    def reset_state_variables(self) -> None:
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.rest)
        self.u = self.b * self.v

    def set_batch_size(self, batch_size) -> None:
        """
        Sets mini-batch size. Called when layer is added to a network.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.rest * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.u = self.b * self.v


class IzhikevichHominNodesFixed(Nodes):
    """
    Layer of Izhikevich neurons implemented using pure Fixed-Point Arithmetic (Q1.6.9).
    This mimics hardware behavior (FPGA/ASIC) directly in Python.
    """
    
    # Cấu hình Fixed-Point Q1.6.9
    FRAC_BITS = 9
    SCALE = 1 << FRAC_BITS  # 512

    # Các hằng số phần cứng (HOMIN)
    # 14.0 * 512 = 7168
    CONST_14_FIXED = 14 * SCALE 
    
    TYPE_PRESETS = {
        "RS": {"d": 6},
        "IB": {"d": 4},
        "CH": {"d": 0.5},
        "LTS": {"d": 0.375},
    }

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        excitatory: float = 1.0,
        thresh: float = 30.0,   # Ngưỡng chuẩn HOMIN là 30 (User có thể để 4.5 nếu muốn)
        rest: float = -65.0,    # Reset chuẩn HOMIN là -65 (User có thể để -6.5 nếu muốn)
        type_ex: str = "RS",
        type_in: str = "RS",
        lbound: float = None,
        **kwargs,
    ) -> None:
        
        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input,
        )

        # --- 1. Chuyển đổi tham số sang Fixed-Point (Integer) ---
        # thresh_fixed = thresh * 512
        self.register_buffer("thresh", torch.tensor(int(thresh * self.SCALE), dtype=torch.int32))
        self.register_buffer("rest", torch.tensor(int(rest * self.SCALE), dtype=torch.int32))
        
        if lbound is not None:
            self.lbound = int(lbound * self.SCALE)
        else:
            self.lbound = None

        # --- 2. Xử lý Presets (d) ---
        tkey_ex = type_ex.upper() if type_ex.upper() in self.TYPE_PRESETS else "RS"
        tkey_in = type_in.upper() if type_in.upper() in self.TYPE_PRESETS else "RS"

        d_val_ex = self.TYPE_PRESETS[tkey_ex]["d"]
        d_val_in = self.TYPE_PRESETS[tkey_in]["d"]

        # --- 3. Khởi tạo Trọng số S và tham số d (Dạng Integer) ---
        self.register_buffer("d", torch.zeros(n, dtype=torch.int32))
        self.register_buffer("S", torch.zeros((n, n), dtype=torch.int32))
        self.register_buffer("excitatory", torch.zeros(n, dtype=torch.uint8)) # Byte tensor

        if excitatory > 1: excitatory = 1
        elif excitatory < 0: excitatory = 0

        # Xác định số lượng nơ-ron hưng phấn/ức chế
        ex = int(n * excitatory)
        inh = n - ex

        # Thiết lập d và S (đã scale)
        # Excitatory part
        if ex > 0:
            self.d[:ex] = int(d_val_ex * self.SCALE)
            self.excitatory[:ex] = 1
            # Trọng số ngẫu nhiên từ 0 đến 0.5 -> Scale thành 0 đến 256
            self.S[:, :ex] = (0.5 * torch.rand(n, ex) * self.SCALE).int()

        # Inhibitory part
        if inh > 0:
            self.d[ex:] = int(d_val_in * self.SCALE)
            self.excitatory[ex:] = 0
            # Trọng số âm -> Scale thành số âm
            self.S[:, ex:] = (-1.0 * torch.rand(n, inh) * self.SCALE).int()

        # --- 4. Khởi tạo biến trạng thái (Integer) ---
        # v bắt đầu tại rest
        self.register_buffer("v", self.rest * torch.ones(n, dtype=torch.int32))
        # u = b * v = 0.25 * v -> v >> 2
        self.register_buffer("u", self.v >> 2) 

    def forward(self, x: torch.Tensor) -> None:
        """
        Runs a single simulation step using BITWISE OPERATIONS.
        :param x: Input current. Must be integer (scaled by 512) or will be cast.
        """
        # Đảm bảo input x là số nguyên (int32)
        if x.dtype != torch.int32 and x.dtype != torch.long:
            x = (x * self.SCALE).int()
        
        # 1. Logic Reset (So sánh với threshold từ bước trước)
        # Nếu spike: v = c (rest), u = u + d
        self.v = torch.where(self.s, self.rest, self.v)
        self.u = torch.where(self.s, self.u + self.d, self.u)

        # 2. Xử lý Synaptic Input (Cộng trọng số S)
        # Nếu có spike từ các nơ-ron khác, cộng trọng số S vào input x
        if self.s.any():
            # Lấy các cột trọng số tương ứng với nơ-ron phát xung và cộng dồn
            # S đã là int, x là int -> Phép cộng integer
            x += torch.cat(
                [self.S[:, self.s[i]].sum(dim=1)[None] for i in range(self.s.shape[0])],
                dim=0,
            )

        # 3. Tính toán Dynamics v (HOMIN Equation 7)
        # Công thức: v_next = v + ( (v^2/4) + 5v + 140 - u + I ) * dt
        
        # A. Thành phần bậc 2: 0.25 * v^2
        # v (Q9) * v (Q9) = v^2 (Q18)
        # Cần đưa về Q9 (>>9) và nhân 0.25 (>>2) => Tổng dịch phải 11
        # Sử dụng .long() để tránh tràn số khi bình phương
        v_sq = (self.v.long() * self.v.long()) >> 11  
        
        # B. Thành phần tuyến tính: 5v = 4v + v
        v_linear = (self.v << 2) + self.v
        
        # C. Tổng hợp (dùng long để cộng trừ an toàn)
        # 14 ở đây là self.CONST_14_FIXED (đã scale)
        term_sum = v_sq + v_linear + self.CONST_14_FIXED - self.u + x
        
        # D. Nhân bước thời gian dt = 1/32 (>>> 5)
        # Kết quả trả về int32
        delta_v = (term_sum >> 5).int()
        
        # Cập nhật v
        self.v += delta_v

        # 4. Tính toán Dynamics u
        # Công thức: u_next = u + (0.25v - u) * a
        # a * dt tương đương phép dịch >>> 11
        # 0.25v tương đương v >>> 2
        u_diff = (self.v >> 2) - self.u
        delta_u = u_diff >> 11 # Dịch 11 bit (dt + a)
        
        self.u += delta_u

        # 5. Voltage Clipping (Giới hạn dưới)
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        # 6. Kiểm tra phát xung (Comparator)
        self.s = self.v >= self.thresh

        # BindsNet hook (ghi lại spike trace nếu cần)
        super().forward(x)

    def reset_state_variables(self) -> None:
        """
        Resets state variables to initial values (Integer).
        """
        super().reset_state_variables()
        self.v.fill_(self.rest)
        self.u = self.v >> 2 # u = 0.25 * v

    def set_batch_size(self, batch_size) -> None:
        super().set_batch_size(batch_size=batch_size)
        # Khởi tạo lại với đúng kích thước batch
        self.v = self.rest * torch.ones(batch_size, *self.shape, device=self.v.device, dtype=torch.int32)
        self.u = self.v >> 2



# class IzhikevichRSNodesFixed(Nodes):
#     """
#     Phiên bản Fixed Point của Izhikevich RS Nodes sử dụng PyTorch.
#     """

#     def __init__(
#         self,
#         n: Optional[int] = None,
#         shape: Optional[Iterable[int]] = None,
#         traces: bool = False,
#         traces_additive: bool = False,
#         tc_trace: Union[float, torch.Tensor] = 20.0,
#         trace_scale: Union[float, torch.Tensor] = 1.0,
#         sum_input: bool = False,
#         excitatory: float = 1,
#         thresh: Union[float, torch.Tensor] = 30.0, # Thường là 30mV
#         rest: Union[float, torch.Tensor] = -65.0,
#         lbound: float = None,
#         fractional_bits: int = 16, # Số bit cho phần thập phân
#         **kwargs,
#     ) -> None:
#         super().__init__(
#             n=n,
#             shape=shape,
#             traces=traces,
#             traces_additive=traces_additive,
#             tc_trace=tc_trace,
#             trace_scale=trace_scale,
#             sum_input=sum_input,
#         )

#         self.f_bits = fractional_bits
#         self.scale = 1 << self.f_bits

#         # Chuyển đổi các ngưỡng và hằng số sang Fixed Point
#         self.register_buffer("rest", self._to_fixed(torch.tensor(rest)))
#         self.register_buffer("thresh", self._to_fixed(torch.tensor(thresh)))
        
#         # Các hằng số trong phương trình Izhikevich
#         self.register_buffer("c004", self._to_fixed(torch.tensor(0.04)))
#         self.register_buffer("c5", self._to_fixed(torch.tensor(5.0)))
#         self.register_buffer("c140", self._to_fixed(torch.tensor(140.0)))

#         # Khởi tạo tham số a, b, c, d
#         self._init_parameters(n, excitatory)
        
#         # Khởi tạo trạng thái v, u ở dạng Fixed Point
#         self.register_buffer("v", self.rest.clone().repeat(n))
#         self.register_buffer("u", self._mul_fixed(self.b, self.v))

#     def _to_fixed(self, x):
#         return (x * self.scale).round()

#     def _from_fixed(self, x):
#         return x.float() / self.scale

#     def _mul_fixed(self, x, y):
#         # (x * y) / scale để giữ nguyên vị trí dấu phẩy tĩnh
#         return (x * y) >> self.f_bits

#     def _init_parameters(self, n, excitatory):
#         # Logic khởi tạo tham số a, b, c, d tương tự bản gốc nhưng chuyển sang Fixed Point
#         r = torch.rand(n)
#         a_f = torch.zeros(n)
#         b_f = torch.zeros(n)
#         c_f = torch.zeros(n)
#         d_f = torch.zeros(n)

#         ex = int(n * excitatory)
#         # Excitatory
#         a_f[:ex] = 0.02
#         b_f[:ex] = 0.2
#         c_f[:ex] = -65.0 + 15 * (r[:ex] ** 2)
#         d_f[:ex] = 8 - 6 * (r[:ex] ** 2)
        
#         # Inhibitory
#         if ex < n:
#             a_f[ex:] = 0.02 + 0.08 * r[ex:]
#             b_f[ex:] = 0.25 - 0.05 * r[ex:]
#             c_f[ex:] = -65.0
#             d_f[ex:] = 2.0

#         self.register_buffer("a", self._to_fixed(a_f))
#         self.register_buffer("b", self._to_fixed(b_f))
#         self.register_buffer("c", self._to_fixed(c_f))
#         self.register_buffer("d", self._to_fixed(d_f))

#     def forward(self, x: torch.Tensor) -> None:
#         # 1. Chuyển đổi đầu vào x sang Fixed Point
#         x_fixed = self._to_fixed(x)
#         dt_fixed = self._to_fixed(torch.tensor(self.dt))

#         # 2. Reset điện áp và biến hồi phục nếu có Spike
#         # self.s là tensor boolean đánh dấu các neuron đã spike ở bước trước
#         self.v = torch.where(self.s, self.c, self.v)
#         self.u = torch.where(self.s, self.u + self.d, self.u)

#         # 3. Cập nhật phương trình v (Thực hiện 2 bước 0.5*dt để ổn định như bản gốc)
#         # dv = 0.04*v^2 + 5*v + 140 - u + I
#         for _ in range(2):
#             v_sq = self._mul_fixed(self.v, self.v)
#             term1 = self._mul_fixed(self.c004, v_sq)
#             term2 = self._mul_fixed(self.c5, self.v)
            
#             dv = term1 + term2 + self.c140 - self.u + x_fixed
            
#             # self.v += 0.5 * dv * dt
#             # Vì ta chạy 2 lần nên mỗi lần là 0.5 * dt
#             step_v = self._mul_fixed(dv, dt_fixed >> 1) 
#             self.v += step_v

#         # 4. Cập nhật phương trình u
#         # du = a * (b * v - u)
#         b_v = self._mul_fixed(self.b, self.v)
#         du = self._mul_fixed(self.a, b_v - self.u)
#         self.u += self._mul_fixed(du, dt_fixed)

#         # 5. Kiểm tra Spike
#         self.s = self.v >= self.thresh

#         # Gọi super forward để cập nhật traces nếu cần
#         super(Nodes, self).forward(x)

#     def reset_state_variables(self) -> None:
#         super().reset_state_variables()
#         self.v.fill_(self.rest)
#         self.u = self._mul_fixed(self.b, self.v)

class IzhikevichNodes(Nodes):
    # language=rst
    """
    Consolidated layer of `Izhikevich neurons<https://www.izhikevich.org/publications/spikes.htm>`_.
    Supports different neuron types: RS (Regular Spiking), IB (Intrinsically Bursting), CH (Chattering), FS (Fast Spiking).
    """

    TYPE_PRESETS = {
        "RS": {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0},
        "IB": {"a": 0.02, "b": 0.2, "c": -55.0, "d": 4.0},
        "CH": {"a": 0.02, "b": 0.2, "c": -50.0, "d": 2.0},
        "FS": {"a": 0.1, "b": 0.2, "c": -65.0, "d": 2.0},
    }

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
        neuron_type: str = "RS",
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
        :param neuron_type: Model type: "RS", "IB", "CH", "FS", defaults to "RS".
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

        self.register_buffer("rest", torch.tensor(rest))
        self.register_buffer("thresh", torch.tensor(thresh))
        self.lbound = lbound
        self.neuron_type = neuron_type.upper()

        preset = self.TYPE_PRESETS.get(self.neuron_type, self.TYPE_PRESETS["RS"])
        a_val, b_val, c_val, d_val = preset["a"], preset["b"], preset["c"], preset["d"]

        self.register_buffer("r", torch.rand(n))
        self.register_buffer("a", a_val * torch.ones(n))
        self.register_buffer("b", b_val * torch.ones(n))
        self.register_buffer("c", c_val * torch.ones(n))
        self.register_buffer("d", d_val * torch.ones(n))
        self.register_buffer("S", torch.zeros(n, n))
        self.register_buffer("excitatory", torch.zeros(n).byte())

        if excitatory > 1:
            excitatory = 1
        elif excitatory < 0:
            excitatory = 0

        ex = int(n * excitatory)
        inh = n - ex

        if ex > 0:
            self.S[:, :ex] = 0.5 * torch.rand(n, ex)
            self.excitatory[:ex] = 1

        if inh > 0:
            self.S[:, ex:] = -torch.rand(n, inh)
            self.excitatory[ex:] = 0

        self.register_buffer("v", self.rest * torch.ones(n))
        self.register_buffer("u", self.b * self.v)

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        self.v = torch.where(self.s, self.c, self.v)
        self.u = torch.where(self.s, self.u + self.d, self.u)

        if self.s.any():
            x += torch.cat(
                [self.S[:, self.s[i]].sum(dim=1)[None] for i in range(self.s.shape[0])],
                dim=0,
            )

        self.v += self.dt * 0.5 * (0.04 * self.v**2 + 5 * self.v + 140 - self.u + x)
        self.v += self.dt * 0.5 * (0.04 * self.v**2 + 5 * self.v + 140 - self.u + x)
        self.u += self.dt * self.a * (self.b * self.v - self.u)

        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        self.s = self.v >= self.thresh
        super().forward(x)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.rest)
        self.u = self.b * self.v

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.rest * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.u = self.b * self.v


class IzhikevichRSNodes(IzhikevichNodes):
    def __init__(self, **kwargs):
        super().__init__(neuron_type="RS", **kwargs)

class IzhikevichRSFixedNodes(Nodes):
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
        fractional_bits: int = 16,
        **kwargs,
    ) -> None:
        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input,
        )
        self.f_bits = fractional_bits
        self.scale = 1 << self.f_bits


        dt_value = self.dt if self.dt is not None else 1.0
        self.register_buffer("dt_fixed", torch.tensor(dt_value * self.scale).round().long())

        self.register_buffer("rest", torch.tensor(rest * self.scale).round().long())
        self.register_buffer("thresh", torch.tensor(thresh * self.scale).round().long())
        
        if lbound is not None: 
            self.lbound = (lbound * self.scale).round().long()
        else:
            self.lbound = None
        
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
            self.a = (0.02 * torch.ones(n) * self.scale).round().long()
            self.b = (0.2 * torch.ones(n) * self.scale).round().long()
            self.c = (-65.0 * torch.ones(n) * self.scale).round().long()
            self.d = (8 * torch.ones(n) * self.scale).round().long()
            self.S = 0.5 * torch.rand(n, n)
            self.excitatory = torch.ones(n).byte()

        elif excitatory == 0:
            self.r = torch.rand(n)
            self.a = (0.02 * torch.ones(n) * self.scale).round().long()
            self.b = (0.2 * torch.ones(n) * self.scale).round().long()
            self.c = (-65.0 * torch.ones(n) * self.scale).round().long()
            self.d = (8 * torch.ones(n) * self.scale).long()
            self.S = -torch.rand(n, n)
            self.excitatory = torch.zeros(n).byte()

        else:
            self.excitatory = torch.zeros(n).byte()
            ex = int(n * excitatory)
            inh = n - ex

            self.r = torch.zeros(n)
            self.a = torch.zeros(n).long()
            self.b = torch.zeros(n).long()
            self.c = torch.zeros(n).long()
            self.d = torch.zeros(n).long()
            self.S = torch.zeros(n, n)

            self.r[:ex] = torch.rand(ex)
            self.a[:ex] = (0.02 * torch.ones(ex) * self.scale).round().long()
            self.b[:ex] = (0.2 * torch.ones(ex) * self.scale).round().long()
            self.c[:ex] = (-65.0 * torch.ones(ex) * self.scale).round().long()
            self.d[:ex] = (8 * torch.ones(ex) * self.scale).round().long()
            self.S[:, :ex] = 0.5 * torch.rand(n, ex)
            self.excitatory[:ex] = 1

            self.r[ex:] = torch.rand(inh)
            self.a[ex:] = (0.02 * torch.ones(inh) * self.scale).round().long()
            self.b[ex:] = (0.2 * torch.ones(inh) * self.scale).round().long()
            self.c[ex:] = (-65.0 * torch.ones(inh) * self.scale).round().long()
            self.d[ex:] = (8 * torch.ones(inh) * self.scale).round().long()
            self.S[:, ex:] = -torch.rand(n, inh)
            self.excitatory[ex:] = 0

        self.register_buffer("v", (self.rest * torch.ones(n)).long())
        self.register_buffer("u", self.mul_fixed(self.b, self.v))

        self.register_buffer("c004", torch.tensor(0.04 * self.scale).round().long())
        self.register_buffer("c5", torch.tensor(5.0 * self.scale).round().long())
        self.register_buffer("c140", torch.tensor(140.0 * self.scale).round().long())

    def mul_fixed(self, a, b):
        return (a * b).round().long() >> self.f_bits

    def forward(self, x: torch.Tensor) -> None:
        x_fixed = (x * self.scale).round().long()

        self.v = torch.where(self.s, self.c, self.v)
        self.u = torch.where(self.s, self.u + self.d, self.u)

        if self.s.any():
            input_spikes = (torch.cat(
                [self.S[:, self.s[i]].sum(dim=1)[None] for i in range(self.s.shape[0])],
                dim=0,
            ) * self.scale).round().long()
            x_fixed += input_spikes


        for _ in range(2):
            v_sq = self.mul_fixed(self.v, self.v)
            v5 = self.mul_fixed(self.c5, self.v)
            v_sq004 = self.mul_fixed(self.c004, v_sq)
            
            dv = (v_sq004 + v5 + self.c140 - self.u + x_fixed) >> 1
            self.v += self.mul_fixed(self.dt_fixed, dv)

        du = self.mul_fixed(self.a, self.mul_fixed(self.b, self.v) - self.u)
        self.u += self.mul_fixed(self.dt_fixed, du)

        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        self.s = self.v >= self.thresh
        super().forward(x)
    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.rest)  # Neuron voltages.
        self.u = self.mul_fixed(self.b, self.v)  # Neuron recovery.

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        # Sử dụng self.v.device để đảm bảo tensor mới nằm cùng thiết bị (CPU/GPU)
        self.v = (self.rest * torch.ones(batch_size, *self.shape, device=self.v.device)).long()
        self.u = self.mul_fixed(self.b, self.v)

class IzhikevichIBNodes(IzhikevichNodes):
    def __init__(self, **kwargs):
        super().__init__(neuron_type="IB", **kwargs)


# toideptraivlcsadsađâ
