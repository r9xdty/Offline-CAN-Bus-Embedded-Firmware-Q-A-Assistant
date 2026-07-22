# STM32 bxCAN Bit Timing Configuration

## The nominal bit time

On an STM32 with the bxCAN peripheral, one CAN bit is divided into time quanta (tq). The
nominal bit time is the sum of three segments:

- Synchronization Segment (SYNC_SEG): always exactly 1 time quantum. Edges are expected here.
- Bit Segment 1 (BS1, also called TSEG1): covers the propagation-delay compensation and the
  first phase buffer. It is programmable from 1 to 16 time quanta.
- Bit Segment 2 (BS2, also called TSEG2): the second phase buffer, after the sample point. It
  is programmable from 1 to 8 time quanta.

The sample point — the instant the bus level is read for the bit value — sits at the boundary
between BS1 and BS2. So:

    nominal bit time = SYNC_SEG + BS1 + BS2   (in time quanta)
    sample point (%) = (SYNC_SEG + BS1) / (SYNC_SEG + BS1 + BS2) x 100

A sample point around 87.5% is a common target for robust operation.

## The time quantum and the baud-rate prescaler

The length of one time quantum is derived from the APB clock (PCLK) feeding the CAN
peripheral through the Baud Rate Prescaler (BRP):

    t_q = (BRP + 1) / f_PCLK

Putting it together, the CAN bit rate is:

    bit rate = f_PCLK / ( (BRP + 1) x (1 + (TS1 + 1) + (TS2 + 1)) )

## Register encoding in CAN_BTR

The bit timing is programmed in the CAN_BTR register. Each field stores its value minus one:

- BRP[9:0] holds (prescaler - 1)
- TS1[3:0] holds (BS1 in tq - 1)
- TS2[2:0] holds (BS2 in tq - 1)
- SJW[1:0] holds (SJW in tq - 1)

So a field value of 0 means one time quantum. CAN_BTR also carries the loopback (LBKM) and
silent (SILM) test-mode bits, which are set while the peripheral is in initialization mode.

## Synchronization Jump Width (SJW)

The Synchronization Jump Width (SJW) sets how many time quanta the controller may lengthen BS1
or shorten BS2 during resynchronization to stay locked to the transmitter's edges. It is
programmable from 1 to 4 time quanta and must not be larger than BS2. A larger SJW tolerates
more oscillator drift but reduces timing margin.

## Worked example: 500 kbps

Suppose PCLK for the CAN peripheral is 42 MHz and the target bit rate is 500 kbit/s. That
requires 84 time-quanta-clocks per bit. Choosing a prescaler of 6 gives
t_q = 6 / 42 MHz = 142.9 ns, and 14 time quanta per bit (14 x 142.9 ns = 2 us = 500 kbps).
Splitting 14 tq as SYNC_SEG = 1, BS1 = 11, BS2 = 2 yields a sample point of
(1 + 11) / 14 = 85.7%. In CAN_BTR this is programmed as BRP = 5, TS1 = 10, TS2 = 1, and a
typical SJW = 1 (SJW field = 0). Always configure bit timing only while the peripheral is in
initialization mode, then leave initialization mode to begin normal communication.

## FDCAN note

On STM32 parts with the FDCAN peripheral instead of bxCAN, the same segment concepts apply
but there are separate nominal and data bit-timing registers (NBTP and DBTP), each with its
own prescaler and TSEG1/TSEG2/SJW fields, so the arbitration phase and the faster data phase
are timed independently.
