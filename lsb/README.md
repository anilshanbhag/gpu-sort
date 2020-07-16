LSB Radix Sort
=============

The LSB radix sort processes the radix sort starting from the least significant bits to the most significant bits.
It can process upto 7 bits at a time. As a result, sorting 32-bit keys takes 5 radix partitioning passes, each pass processing 6,6,6,7,7 bits respectively.

To Run
---

```
make setup
make sort
```
