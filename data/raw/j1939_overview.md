# SAE J1939 Overview

## What J1939 is

SAE J1939 is a higher-layer protocol built on CAN, used in heavy-duty vehicles such as trucks,
buses, and agricultural and construction equipment. It defines how nodes (called Electronic
Control Units, or ECUs) exchange standardized parameters over the CAN bus. J1939 runs on
CAN 2.0B and therefore always uses the 29-bit extended identifier. The classic network speed
is 250 kbit/s; the newer J1939-14 physical layer defines a 500 kbit/s bus.

## The 29-bit identifier layout

J1939 gives structure to the 29-bit CAN identifier. From most significant to least
significant, it is divided into:

- Priority (3 bits): 0 is highest priority, 7 is lowest. Priority only affects arbitration, not
  the meaning of the message.
- Parameter Group Number fields: a reserved/data-page bit, the PDU Format (PF) byte, and the
  PDU Specific (PS) byte.
- Source Address (8 bits): identifies the transmitting ECU.

## Parameter Group Numbers and Suspect Parameter Numbers

A Parameter Group Number (PGN) identifies a group of related signals carried in a message. The
PGN is derived from the data page, PDU Format, and PDU Specific fields of the identifier. Each
individual signal within a message is a Suspect Parameter Number (SPN), which specifies the
data's position, length, scaling, offset, and engineering units. For example, engine speed and
coolant temperature are SPNs carried inside particular PGNs.

## PDU1 versus PDU2

The PDU Format byte selects one of two addressing modes:

- PDU1 format (PF from 0 to 239): destination-specific. The PDU Specific byte is the
  Destination Address, so the message is directed at one ECU.
- PDU2 format (PF from 240 to 255): broadcast. The PDU Specific byte is a Group Extension that
  further identifies the PGN, and the message is broadcast to all nodes.

## Transport protocol for large messages

A single CAN frame carries at most 8 data bytes, but many J1939 parameter groups are longer.
The J1939 Transport Protocol (TP) fragments messages of 9 to 1785 bytes across multiple
frames. It uses connection-management frames (TP.CM) to open and manage the transfer and data
-transfer frames (TP.DT) to carry the numbered segments. There are two variants: a
destination-specific mode using Request-To-Send / Clear-To-Send handshaking (RTS/CTS), and a
Broadcast Announce Message (BAM) mode for one-to-many transfers with timed pacing.

## Address claiming

J1939 ECUs obtain their source address through an address-claim procedure. Each ECU has a
64-bit NAME that encodes its function and manufacturer. On startup an ECU broadcasts an
Address Claimed message for its desired address; if two ECUs claim the same address, the one
with the numerically lower NAME wins and the other must claim a different address or go silent.
This lets nodes negotiate unique addresses without central configuration.
