MSB Radix Sort
=============

The MSB radix sort processes the radix sort in a hierarchical manner starting from the most significant bits to the least significant bits.
It can process 8 bits at a time. For details checkout the paper: A memory bandwidth-efficient hybrid radix sort on gpusi by Stehle and Jacobsen (citation below).

```
@inproceedings{stehle2017memory,
  title={A memory bandwidth-efficient hybrid radix sort on gpus},
  author={Stehle, Elias and Jacobsen, Hans-Arno},
  booktitle={Proceedings of the 2017 ACM International Conference on Management of Data},
  pages={417--432},
  year={2017}
}
```

Note: I am not the author of this code, code provided by paper authors.

To Run
----

```
cd  into the build dir
cmake  -DCMAKE_CUDA_FLAGS="-arch=sm_52” ../
make
./cuda-lib  --gtest_list_tests
./cuda-lib --gtest_filter=Sort_Keys.Entropy_UINT -s1000
```
