# High-Speed CAN Transceiver Basics

## Role of the transceiver

A CAN transceiver is the analog interface between a CAN controller (inside an MCU) and the
physical two-wire bus. On the controller side it exposes two logic pins: TXD, which the
controller drives to request dominant or recessive bits, and RXD, on which the transceiver
reports the current bus state back to the controller. On the bus side it drives and senses the
differential CAN_H and CAN_L lines. This description follows the generic ISO 11898-2 high-speed
CAN transceiver behavior rather than any single vendor part.

## Bus levels

The transceiver converts logic levels to bus levels:

- Recessive state (TXD high, logic 1): the transceiver does not drive the bus. CAN_H and CAN_L
  are pulled toward a common bias near 2.5 V, so the differential voltage is approximately 0 V
  (typically below 0.5 V).
- Dominant state (TXD low, logic 0): the transceiver actively drives CAN_H high (about 3.5 V)
  and CAN_L low (about 1.5 V), producing a differential voltage of roughly 2 V (typically at
  least 1.5 V).

A receiving transceiver reports a bus differential above about 0.9 V as dominant and below
about 0.5 V as recessive, with the region between treated as a threshold band. Because the
dominant state is actively driven and the recessive state is passive, a dominant bit from any
node overrides recessive bits from all others — the wired-AND behavior that makes CAN
arbitration work.

## Termination

The bus must be terminated with a 120-ohm resistor at each physical end of the trunk, matching
the characteristic impedance of the twisted pair. Correct termination absorbs reflections that
would otherwise corrupt the fast dominant-to-recessive edges. Split termination, where each
120-ohm resistor is divided into two 60-ohm halves with a capacitor to ground at the midpoint,
improves common-mode noise behavior.

## Common-mode range and protection

High-speed CAN transceivers tolerate a wide common-mode input range, commonly at least -2 V to
+7 V, so nodes at different ground potentials can still communicate. Many devices add bus fault
protection to higher voltages and integrate ESD protection on the bus pins. A common-mode choke
is sometimes fitted to reduce emissions.

## Operating modes

Typical transceivers offer at least two modes selected by a control pin (often labeled S, STB,
or EN):

- Normal mode: full-speed transmit and receive.
- Standby or low-power mode: the transmitter is disabled to save power, while a low-power
  receiver can still monitor the bus and wake the node on a valid wake-up pattern.

Some transceivers also provide a slope-control or silent mode. Slope control limits the
dominant-edge slew rate to reduce electromagnetic emissions at lower bit rates. A dedicated
TXD-dominant timeout protects the bus: if TXD is stuck low, the transmitter is disabled after a
timeout so a single faulty node cannot hold the bus dominant forever.
