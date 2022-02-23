# End to end deep neural network based semantic communication system for image

Final paper for Advanced Topics on Communications (IIT8003).

Notable codes:
- `utils/qam_modem_tf.py`: Tensorflow implementation of QAM modulation (Supports up to 256QAM, but easily extendable by modifying `QAMDemodulator.call()` method).
- `utils/qam_modem_naive.py`: Pure python implementation of QAM modulation.

## System architecture
![sysarch](https://user-images.githubusercontent.com/23615360/147174574-dc0b4883-3e33-47c1-9737-5449c1609aaa.png)

## Layer-wise image results
![layers](https://user-images.githubusercontent.com/23615360/147174579-e91de734-089a-4b88-b3cd-9ebe691091ab.png)

## Numerical results (SSIM)
![ssim](https://user-images.githubusercontent.com/23615360/147174581-ac3dbec7-d199-4981-a223-669ca6bdfd1d.png)
