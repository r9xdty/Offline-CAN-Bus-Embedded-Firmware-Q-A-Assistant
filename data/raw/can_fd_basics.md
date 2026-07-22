# CAN FD Basics

## What CAN FD adds

CAN FD (CAN with Flexible Data-rate) is an extension of classic CAN, standardized as part of
ISO 11898-1:2015. It keeps the same arbitration and physical-layer principles but relaxes two
classic limits: it allows a larger payload and a faster data-phase bit rate. CAN FD is
designed to coexist with the classic frame format on controllers that support it.

## Larger payload

Classic CAN carries at most 8 data bytes per frame. CAN FD extends the payload to as many as
64 data bytes. The Data Length Code is reinterpreted for FD frames so that DLC values above 8
map to the discrete payload sizes 12, 16, 20, 24, 32, 48, and 64 bytes. Larger payloads mean
fewer frames and less per-frame overhead for block transfers.

## Dual bit rate and the BRS bit

A CAN FD frame can be transmitted at two different bit rates. The arbitration phase — where
nodes contend for the bus — always runs at the slower nominal bit rate so that arbitration
still works across the whole bus length. If the Bit Rate Switch (BRS) bit is recessive, the
controller switches to a faster data bit rate for the data and CRC fields, then switches back
to the nominal rate for the acknowledgment and end of frame. The Extended Data Length (EDL,
also called FDF) bit marks a frame as FD rather than classic.

Because only the data phase is sped up, the bus length is still governed by the nominal
arbitration rate, while throughput benefits from the faster data phase. Typical data-phase
rates are 2 Mbit/s and 5 Mbit/s; some transceivers (CAN FD / SIC or dedicated 8 Mbit/s parts)
support up to about 8 Mbit/s over short, well-terminated buses.

## CRC and reliability changes

CAN FD strengthens error detection to match the larger frames. It uses a longer CRC — a
17-bit CRC for payloads up to 16 bytes and a 21-bit CRC for larger payloads — instead of the
classic 15-bit CRC. Stuff bits in the CRC field are handled with a fixed-stuff-bit scheme,
and a stuff-bit counter is included so that the receiver can validate the number of dynamic
stuff bits. There is no remote-frame (RTR) request in CAN FD; the RTR position is replaced by
the RRS bit, which is always dominant.

## Compatibility notes

A classic CAN controller cannot correctly receive an FD frame and will flag an error, so a
mixed bus requires all nodes to tolerate or support FD framing. Many modern MCUs implement an
FDCAN peripheral that can be configured for classic CAN, CAN FD without bit-rate switching, or
CAN FD with bit-rate switching. When bit-rate switching is enabled, both a nominal bit-timing
configuration and a separate data bit-timing configuration must be programmed.
