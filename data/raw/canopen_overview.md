# CANopen Overview

## What CANopen is

CANopen is a higher-layer protocol built on CAN, standardized as CiA 301 by CAN in Automation.
It is widely used in industrial automation, motion control, and embedded machinery. CANopen
uses the classic 11-bit standard identifier and organizes each device around an Object
Dictionary. A CANopen network supports up to 127 nodes, each with a node ID from 1 to 127.

## The Object Dictionary

The Object Dictionary is the heart of a CANopen device: a standardized table of all the
device's accessible parameters and process data. Each entry is addressed by a 16-bit index and
an 8-bit sub-index. Ranges of the dictionary are reserved for communication parameters, the
standardized device profile, and manufacturer-specific data. Configuration tools and other
nodes read and write these entries to set up and operate the device.

## Communication objects and COB-ID

Messages in CANopen are called communication objects, each identified by a COB-ID
(Communication Object Identifier) that maps onto the 11-bit CAN identifier. In the default
"pre-defined connection set," the COB-ID is formed from a 4-bit function code and the 7-bit
node ID, so the function code sets the base priority and the node ID distinguishes devices.

## PDOs and SDOs

CANopen distinguishes two main ways to move data:

- Process Data Objects (PDOs) carry real-time process data with no protocol overhead — the
  payload is raw application data mapped from the Object Dictionary. PDOs are broadcast and are
  used for fast, cyclic or event-driven exchange of measured values and commands. Transmit
  PDOs (TPDOs) are sent by the device; Receive PDOs (RPDOs) are consumed by it.
- Service Data Objects (SDOs) provide confirmed, addressed access to any Object Dictionary
  entry using a request/response handshake. SDOs are used for configuration and diagnostics
  where reliability matters more than speed.

## Network management (NMT) states

A CANopen device is controlled by the Network Management (NMT) state machine, which has these
states: Initialization, Pre-operational, Operational, and Stopped. After power-up the device
initializes and then enters Pre-operational, where SDO communication is possible but PDOs are
not exchanged. An NMT master sends a Start command to move nodes to Operational, where PDOs are
active. In the Stopped state only NMT and heartbeat messages are allowed.

## Heartbeat and node monitoring

To detect failed nodes, CANopen uses error control. In the heartbeat mechanism each device
periodically transmits a heartbeat message carrying its current NMT state; a heartbeat consumer
that misses the expected message within the configured time flags the producer as lost. An
older node-guarding mechanism polls nodes instead. Heartbeat is the preferred method in modern
networks because it needs no polling traffic.
