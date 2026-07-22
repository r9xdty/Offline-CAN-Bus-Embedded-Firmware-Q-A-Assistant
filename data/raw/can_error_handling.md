# CAN Error Handling and Fault Confinement

## Error detection mechanisms

CAN defines five error-detection mechanisms that every node applies to every frame. Three are
message-level checks and two are bit-level checks:

- Bit error: a transmitter that sends one bit level but monitors the opposite level on the bus
  (outside the arbitration and ACK-slot fields) detects a bit error.
- Stuff error: six consecutive bits of the same polarity where a stuff bit was required
  violates the bit-stuffing rule.
- CRC error: the receiver's computed CRC does not match the transmitted CRC field.
- Form error: a fixed-form field (such as a delimiter or the end-of-frame) contains an
  illegal value.
- Acknowledgment error: no receiver drives the ACK slot dominant, so the transmitter sees its
  own recessive ACK bit.

When any node detects an error it transmits an error frame, which deliberately violates the
bit-stuffing rule to force all other nodes to discard the corrupt message and to trigger an
automatic retransmission.

## Error counters

Fault confinement keeps a persistently faulty node from monopolizing the bus. Each node
maintains two counters:

- Transmit Error Counter (TEC)
- Receive Error Counter (REC)

A detected transmit error increases the TEC (typically by 8), and a detected receive error
increases the REC (typically by 1 or by 8 depending on the situation). Successful
transmission or reception decreases the corresponding counter. The counters therefore track
how error-prone a node currently is.

## Error states: error-active, error-passive, bus-off

Every node is always in one of three fault-confinement states, selected by the counter
values:

- Error-active: the normal state. Both TEC and REC are at or below 127. An error-active node
  signals detected errors by sending an active error flag of six dominant bits, which is
  visible to the whole bus and forces retransmission.
- Error-passive: entered when the TEC or the REC exceeds 127. An error-passive node still
  participates in communication but must signal errors with a passive error flag of six
  recessive bits, which cannot disturb other nodes' traffic. A passive-error state therefore
  means the node has accumulated enough errors that the network no longer lets it assert
  dominant error flags; it keeps operating but with reduced influence, and after transmitting
  it must wait an additional suspend-transmission time before sending again. This is the
  network's way of throttling a node that is becoming unreliable without cutting it off
  entirely.
- Bus-off: entered when the TEC exceeds 255. A bus-off node is disconnected from the bus — it
  stops transmitting and receiving entirely. It may only rejoin (return to error-active with
  cleared counters) after a recovery sequence, typically observing 128 occurrences of 11
  consecutive recessive bits on the bus.

## Recovery

Bus-off recovery is often left to firmware policy: the application decides whether to restart
the CAN controller automatically or wait for operator intervention. Because the transition
error-active -> error-passive -> bus-off is driven purely by the error counters, monitoring
the TEC and REC is a practical way for firmware to observe bus health before a node drops off.
