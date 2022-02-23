import math
import random
import time

def to_gray_code(i):
    return i^(i>>1)


def split_bit(num, n_bit):
    lower_half_bit_mask = 2 ** (n_bit//2) - 1
    upper_half_bit_mask = lower_half_bit_mask << (n_bit//2)
    lower_half_bit = num & lower_half_bit_mask
    upper_half_bit = (num & upper_half_bit_mask) >> (n_bit//2)

    return lower_half_bit, upper_half_bit


def num_to_constellation(num, n_bit):
    m = 2 ** n_bit
    avg_power = math.sqrt((m-1)/3*2)

    l, h = split_bit(num, n_bit)
    d_q = to_gray_code(l) + 1
    d_i = to_gray_code(h) + 1

    a_q = 2 * d_q - 1 - n_bit
    a_i = 2 * d_i - 1 - n_bit

    return (a_i / avg_power, a_q / avg_power)


def to_binary(num, n_bit):
    binary_num = num & (1 << (n_bit-1))
    last_bit = binary_num
    for i in range(n_bit-2, -1, -1):
        bitmask = 1 << i                       # 10000
        last_bit = bitmask & (binary_num >> 1) # ?0000
        bit = (num & bitmask)                  # !0000
        xorbit = bit ^ last_bit                # *1111
        binary_num += xorbit & bitmask         # *0000
    return binary_num


def qam_detect(num, n_bit):
    '''
    input: power-averaged single input (quadrature or in-phase)
    '''
    m = 2 ** n_bit
    avg_power = math.sqrt((m-1)/3*2)

    yhat = math.floor(num * avg_power / 2) * 2 + 1
    max_val = 2 * 2 ** (n_bit // 2) - 1 - n_bit
    min_val = 1 - n_bit

    yhat = min(max_val, max(yhat, min_val))
    # print(f'{num * avg_power:.2f} => {yhat}')
    return yhat


def concat_bit(upper_bit, lower_bit, n_bit):
    return (upper_bit << (n_bit//2)) | lower_bit


def noised_constellation_to_num(i, q, n_bit):
    ihat = qam_detect(i, n_bit)
    qhat = qam_detect(q, n_bit)

    d_i = (ihat + n_bit + 1) // 2
    d_q = (qhat + n_bit + 1) // 2

    return concat_bit(to_binary(d_i - 1, n_bit),
                      to_binary(d_q - 1, n_bit), n_bit)


def qam_modem_awgn(target, n_bit, snrdB):
    # AWGN
    snr = 10 ** (snrdB / 10) # in dB
    sigma = 1 / math.sqrt(snr*2)

    i, q = num_to_constellation(target, n_bit)
    ihat = i + random.gauss(0, sigma)
    qhat = q + random.gauss(0, sigma)
    num = noised_constellation_to_num(ihat, qhat, n_bit)

    return num


def qam_modem_rayleigh(target, n_bit, snrdB):
    # Rayleigh
    snr = 10 ** (snrdB / 10) # in dB
    sigma = 1 / math.sqrt(snr*2)

    i, q = num_to_constellation(target, n_bit)
    hi = random.gauss(0, 1/math.sqrt(2))
    hq = random.gauss(0, 1/math.sqrt(2))
    ihat = (hi * i + random.gauss(0, sigma)) / hi
    qhat = (hq * q + random.gauss(0, sigma)) / hq
    num = noised_constellation_to_num(ihat, qhat, n_bit)

    return num


if __name__ == '__main__':
    '''
    For test purposes.
    Calculates average BER at specific SNR
    '''
    start = time.time()

    m = 256
    n_bit = int(math.log2(m))
    snrdB = 15
    num_repeat = int(1e+6)

    snr = 10 ** (snrdB / 10) # in dB
    sigma = 1 / math.sqrt(snr*2)
    power = 0
    biterror = 0
    for _ in range(num_repeat):
        target = random.randint(0, m-1)
        i, q = num_to_constellation(target, n_bit)
        power += i ** 2 + q ** 2
        # print(f'Modulate: {target:b}, {i} {q}')

        avg_power = math.sqrt((m-1)/3*2)

        sigma1 = random.gauss(0, sigma)
        sigma2 = random.gauss(0, sigma)

        ihat = i + sigma1
        qhat = q + sigma2

        num = noised_constellation_to_num(ihat, qhat, n_bit)

        # print(f'{i * avg_power:4.2f}=>{ihat * avg_power:4.2f}, {q * avg_power:4.2f}=>{qhat * avg_power:4.2f}')
        '''
        if target != num:
            # print(f'Bit error: {target:b} => {num:b}')
            biterror += bin(target ^ num).count("1")
        '''

    end = time.time() - start
    print(f'N: {num_repeat}, Elapsed: {end:.4f}s')

    print(f'SNR: {1/(2 * sigma**2):.4f} / Target: {snr:.4f}')
    print(f'AVG Power: {power / num_repeat:.4f}')
    print(f'Eb/N0: {(power / num_repeat)/(2 * sigma**2):.4f}')
    # print(f'BER: {biterror / num_repeat / n_bit}')
