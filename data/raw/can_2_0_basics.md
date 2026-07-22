# CAN 2.0 Basics

## Overview

The Controller Area Network (CAN) is a multi-master, message-based serial bus standardized
in ISO 11898. Nodes broadcast frames onto a shared two-wire differential bus; every node
receives every frame and decides in software whether the message is relevant. There is no
addressing of nodes — messages carry an identifier that describes the content, not the
destination.

## Physical layer and signaling

High-speed CAN (ISO 11898-2) uses a twisted pair, CAN_H and CAN_L, terminated by a 120-ohm
resistor at each end of the bus (240 ohm total in parallel). The bus is differential: a
recessive bit leaves both lines near 2.5 V (a differential voltage near 0 V), while a
dominant bit drives CAN_H high (about 3.5 V) and CAN_L low (about 1.5 V), giving a
differential voltage of roughly 2 V. A dominant bit (logical 0) always overrides a recessive
bit (logical 1) when two nodes transmit at once, which is the basis of bus arbitration.

## Bus length versus bit rate

Because arbitration requires a bit to propagate to the far end of the bus and back within one
bit time, the maximum bus length falls as the bit rate rises. Common rule-of-thumb maximums
for high-speed CAN are:

- 1 Mbit/s: about 40 meters
- 500 kbit/s: about 100 meters
- 250 kbit/s: about 250 meters
- 125 kbit/s: about 500 meters
- 50 kbit/s: about 1000 meters
- 20 kbit/s: about 2500 meters
- 10 kbit/s: about 5000 meters

So at 500 kbps the maximum practical CAN bus length is roughly 100 meters. A high-speed CAN
segment typically supports up to about 110 nodes, limited by transceiver drive capability.

## Standard vs extended identifiers

CAN 2.0 defines two frame formats that differ in identifier length:

- CAN 2.0A uses an 11-bit "standard" identifier, allowing up to 2048 distinct identifiers.
- CAN 2.0B uses a 29-bit "extended" identifier (an 11-bit base ID plus an 18-bit extension),
  allowing more than 500 million identifiers.

In the frame, the Identifier Extension (IDE) bit selects the format: dominant IDE means an
11-bit standard identifier, recessive IDE means a 29-bit extended identifier. Standard and
extended frames can coexist on the same bus. The identifier also sets message priority.

## Arbitration and priority

When multiple nodes start transmitting simultaneously, they arbitrate bit by bit on the
identifier field. Each transmitter monitors the bus while sending; a node that sends a
recessive bit but reads back a dominant bit has lost arbitration and immediately backs off,
becoming a receiver. The message with the numerically lowest identifier (the most dominant
bits) wins and continues without interruption or data loss. Lower identifier value therefore
means higher priority. When a standard and an extended frame share the same 11-bit base, the
standard frame wins because its dominant RTR/IDE bit comes before the extension.

## Data frame fields

A classic CAN data frame contains a Start-of-Frame bit, the arbitration field (identifier +
RTR bit), a control field (including the IDE bit and a 4-bit Data Length Code), 0-8 data
bytes, a 15-bit CRC with a CRC delimiter, an ACK slot and delimiter, and a 7-bit
End-of-Frame. The Data Length Code (DLC) encodes a payload of 0 to 8 bytes for classic CAN.

## Bit stuffing

To keep receivers synchronized, the transmitter inserts a complementary stuff bit after five
consecutive bits of the same value in the frame fields up to the CRC. Receivers remove these
stuff bits. A violation of the stuffing rule is detected as a stuff error.
