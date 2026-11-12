import numpy as np


# ============================================================
# 0. УТИЛИТЫ
# ============================================================

def _get_rng(rng):
    """Возвращает генератор; если None — создаёт новый."""
    return rng if rng is not None else np.random.default_rng()


# ============================================================
# 1. БЕЛЫЙ ГАУССОВ ШУМ (тепловой шум электроники)
# ============================================================

def white_noise(n, sigma=1.0, rng=None):
    """Белый гауссов шум с нулевым средним и заданным СКО."""
    rng = _get_rng(rng)
    return sigma * rng.standard_normal(n)


# ============================================================
# 2. ОКРАШЕННЫЙ ШУМ (1/f, 1/f²) — фоновое излучение плазмы
# ============================================================

def colored_noise(n, fs, exponent=1.0, rng=None):
    """Гауссов окрашенный шум через спектральный синтез. PSD ~ 1/f^exponent."""
    rng = _get_rng(rng)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    f = freqs.copy()
    f[0] = f[1] if len(f) > 1 else 1.0

    amplitude = f ** (-exponent / 2)
    amplitude[0] = 0.0

    phase = rng.uniform(0, 2 * np.pi, len(f))
    spectrum = amplitude * np.exp(1j * phase)

    x = np.fft.irfft(spectrum, n=n)
    x -= x.mean()
    std = x.std()
    if std > 0:
        x /= std
    return x


# ============================================================
# 3. УЗКОПОЛОСНЫЙ ШУМ (сильная линия примеси в фоне плазмы)
# ============================================================

def bandpass_noise(n, fs, f_center, bandwidth, rng=None):
    """Узкополосный гауссов шум в полосе [f_center - B/2, f_center + B/2]."""
    rng = _get_rng(rng)
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1 / fs)

    mask = (freqs >= f_center - bandwidth / 2) & (freqs <= f_center + bandwidth / 2)
    X[~mask] = 0.0
    xb = np.fft.irfft(X, n=n)
    std = xb.std()
    if std > 0:
        xb /= std
    return xb


# ============================================================
# 4. ДРОБОВОЙ ШУМ ФОТОДЕТЕКТОРА
# ============================================================

def shot_noise(mean_photocurrent, n, scale=1.0, rng=None):
    """Дробовой шум фотодетектора. Дисперсия пропорциональна среднему фототоку."""
    rng = _get_rng(rng)
    if np.isscalar(mean_photocurrent):
        sigma = scale * np.sqrt(max(mean_photocurrent, 0.0))
        return sigma * rng.standard_normal(n)
    else:
        sigma = scale * np.sqrt(np.abs(mean_photocurrent))
        return sigma * rng.standard_normal(n)


# ============================================================
# 5. ФОНОВОЕ ИЗЛУЧЕНИЕ ПЛАЗМЫ
# ============================================================

def plasma_background(n, fs, level=1.0, components=None, rng=None):
    """Фоновое излучение плазмы: сумма окрашенного шума и узкополосных линий."""
    rng = _get_rng(rng)

    if components is None:
        components = [{'type': 'pink', 'power': 1.0}]

    total = np.zeros(n)
    for c in components:
        t = c['type']
        p = c.get('power', 1.0)
        if t == 'white':
            x = rng.standard_normal(n)
        elif t == 'pink':
            x = colored_noise(n, fs, exponent=1.0, rng=rng)
        elif t == 'brown':
            x = colored_noise(n, fs, exponent=2.0, rng=rng)
        elif t == 'bandpass':
            x = bandpass_noise(n, fs, c['f_center'], c['bandwidth'], rng=rng)
        else:
            raise ValueError(f"unknown component type: {t}")
        total += np.sqrt(p) * x

    std = total.std()
    if std > 0:
        total /= std
    return level * total


# ============================================================
# 6. НАВОДКИ ЭЛЕКТРОПИТАНИЯ (50/60 Гц и гармоники)
# ============================================================

def power_line_interference(n, fs, f_line=50.0, harmonics=(1, 2, 3),
                            amplitudes=None, rng=None):
    """Наводки от сети электропитания и её гармоник."""
    rng = _get_rng(rng)
    t = np.arange(n) / fs
    if amplitudes is None:
        amplitudes = [1.0 / k for k in harmonics]

    x = np.zeros(n)
    for k, a in zip(harmonics, amplitudes):
        phase = rng.uniform(0, 2 * np.pi)
        x += a * np.sin(2 * np.pi * f_line * k * t + phase)
    return x


# ============================================================
# 7. ЛАЗЕРНЫЙ СИГНАЛ: П-образный, заданная частота и скважность
# ============================================================

def laser_signal(n, fs, f_rep=1e6, duty=0.5, amplitude=1.0,
                 phase=0.0, rng=None):
    """П-образный (прямоугольный) лазерный сигнал с частотой f_rep и duty."""
    t = np.arange(n) / fs
    T = 1.0 / f_rep
    u = ((t / T) + phase) % 1.0
    return amplitude * (u < duty).astype(float)


# ============================================================
# 8. ПОЛНЫЙ СИГНАЛ ДЕТЕКТОРА: лазер + все шумы (ТР и рассеяние исключены)
# ============================================================

def thomson_signal(n, fs, t_laser,
                   laser_amplitude=0.0, laser_f_rep=1e6, laser_duty=0.5,
                   laser_phase=0.0,
                   bg_level=0.0, bg_components=None,
                   shot_scale=0.0, thermal_sigma=0.0,
                   line_interf_amp=0.0, f_line=50.0,
                   rng=None):
    """
    Полный сигнал фотодетектора БЕЗ полезного томсоновского сигнала
    и БЕЗ паразитного рассеяния:
      лазерный П-образный сигнал + фон плазмы
      + дробовой шум + тепловой шум + наводки сети.
    """
    rng = _get_rng(rng)
    t = np.arange(n) / fs
    components = {}

    # --- Лазерный сигнал ---
    if laser_amplitude > 0:
        laser = laser_signal(n, fs, f_rep=laser_f_rep, duty=laser_duty,
                             amplitude=laser_amplitude, phase=laser_phase,
                             rng=rng)
    else:
        laser = np.zeros(n)
    components['laser'] = laser

    # --- Фон плазмы ---
    if bg_level > 0:
        bg = plasma_background(n, fs, bg_level, bg_components, rng=rng)
    else:
        bg = np.zeros(n)
    components['background'] = bg

    # --- Дробовой шум ---
    mean_level = np.abs(laser) + np.abs(bg)
    if shot_scale > 0:
        shot = shot_noise(mean_level, n, scale=shot_scale, rng=rng)
    else:
        shot = np.zeros(n)
    components['shot'] = shot

    # --- Тепловой шум ---
    thermal = white_noise(n, sigma=thermal_sigma, rng=rng)
    components['thermal'] = thermal

    # --- Наводки сети ---
    if line_interf_amp > 0:
        line = line_interf_amp * power_line_interference(n, fs, f_line, rng=rng)
    else:
        line = np.zeros(n)
    components['line'] = line

    total = laser + bg + shot + thermal + line
    return total, components


# ============================================================
# 9. СИНХРОННЫЙ ДЕТЕКТОР (lock-in)
# ============================================================

def synchronous_detector(signal, fs, f_ref, duty=0.5, phase=0.0,
                         lowpass_window=None):
    """
    Синхронное детектирование (lock-in) с опорным П-образным сигналом.

    Применяется к готовому массиву signal (лазер + шум, наложены).
    Умножает вход на опорный меандр частоты f_ref, затем усредняет
    (скользящее среднее) — НЧ-фильтр.

    Возвращает
    ----------
    demod : ndarray или float
        Демодулированный сигнал той же длины, что вход (если задано окно);
        иначе одно число — среднее.
    ref : ndarray
        Опорный меандр.
    """
    n = len(signal)
    t = np.arange(n) / fs
    T = 1.0 / f_ref
    u = ((t / T) + phase) % 1.0
    ref = (u < duty).astype(float)

    mixed = signal * ref

    if lowpass_window is None:
        return mixed.mean(), ref

    w = max(1, min(int(lowpass_window), n))
    kernel = np.ones(w) / w
    demod = np.convolve(mixed, kernel, mode='same')

    # гарантируем длину n
    if len(demod) > n:
        demod = demod[:n]
    elif len(demod) < n:
        demod = np.pad(demod, (0, n - len(demod)), mode='edge')

    return demod, ref


def estimate_snr(signal, laser, fs, f_ref, duty=0.5):
    """
    Оценка отношения сигнал/шум на входе синхронного детектора.
    SNR = P(laser) / P(signal - laser)
    """
    noise = signal - laser
    p_sig = np.mean(laser ** 2)
    p_noise = np.mean(noise ** 2)
    if p_noise <= 0:
        return np.inf
    return p_sig / p_noise


# ============================================================
# ПРИМЕР: генерация смеси «лазер + шум», затем lock-in
# ============================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from numpy.fft import rfft, rfftfreq

    fs = 1e9                      # 1 ГГц
    n = int(fs * 5e-6)            # окно 5 мкс — 5 периодов при 1 МГц
    t_laser = 50e-9
    f_laser = 1e6                 # 1 МГц
    duty = 0.5
    rng = np.random.default_rng(42)

    # ------------------------------------------------------------
    # 1. Готовим смесь «лазер + шум» (наложены друг на друга)
    # ------------------------------------------------------------
    signal, comp = thomson_signal(
        n=n, fs=fs, t_laser=t_laser,
        laser_amplitude=1.0,
        laser_f_rep=f_laser,
        laser_duty=duty,
        laser_phase=0.0,
        bg_level=0.5,
        bg_components=[
            {'type': 'pink', 'power': 1.0},
            {'type': 'bandpass', 'power': 2.0,
             'f_center': 200e6, 'bandwidth': 5e6},
        ],
        shot_scale=0.05,
        thermal_sigma=0.01,
        line_interf_amp=0.02,
        f_line=50.0,
        rng=rng
    )

    print("Полный сигнал: min=%.3f, max=%.3f, mean=%.3f"
          % (signal.min(), signal.max(), signal.mean()))
    print("Компоненты:", list(comp.keys()))
    for k, v in comp.items():
        print(f"  {k:12s}: std={v.std():.4f}, max|.|={np.abs(v).max():.4f}")

    # SNR на входе (если есть «чистый» лазер для сравнения)
    snr_in = estimate_snr(signal, comp['laser'], fs, f_laser, duty)
    print(f"\nSNR на входе (по мощности): {snr_in:.4f} "
          f"({10*np.log10(snr_in):.2f} дБ)")

    # ------------------------------------------------------------
    # 2. Синхронное детектирование готовой смеси
    # ------------------------------------------------------------
    # окно усреднения: 2 периода опоры, но не больше n//4
    w = min(int(round(2 / f_laser * fs)), max(1, n // 4))
    print(f"\nОкно усреднения lock-in: w = {w} отсчётов "
          f"({w/fs*1e9:.1f} нс, {w/fs*f_laser:.2f} периода опоры)")

    demod, ref = synchronous_detector(signal, fs, f_laser, duty=duty,
                                      phase=0.0, lowpass_window=w)

    print(f"len(demod) = {len(demod)}, len(signal) = {n}")

    dc_out = float(np.mean(demod))
    std_out = float(np.std(demod))
    n_eff = max(1.0, n / w)
    stderr = std_out / np.sqrt(n_eff)
    detected = abs(dc_out) > 3 * stderr
    print(f"Выход синхронного детектора:")
    print(f"  среднее = {dc_out:.4f}")
    print(f"  std     = {std_out:.4f}")
    print(f"  stderr  = {stderr:.4f}  (N_eff = {n_eff:.0f})")
    print(f"  обнаружен (|mean| > 3*stderr): {detected}")

    # ------------------------------------------------------------
    # 3. Визуализация
    # ------------------------------------------------------------
    t_ns = np.arange(n) / fs * 1e9

    # --- 3.1. Полный сигнал ---
    plt.figure(figsize=(11, 4))
    plt.plot(t_ns, signal, lw=0.8, color='C0', label='вход (лазер + шум)')
    plt.plot(t_ns, comp['laser'], lw=1.2, color='C3', alpha=0.7,
             label='лазер (без шума)')
    plt.xlabel('t, нс'); plt.ylabel('сигнал, усл. ед.')
    plt.title('Вход синхронного детектора: лазер + шум')
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()

    # --- 3.2. Компоненты ---
    n_comp = len(comp)
    ncols = 2
    nrows = int(np.ceil(n_comp / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, x) in zip(axes, comp.items()):
        ax.plot(t_ns, x, lw=0.8)
        ax.set_title(f"{name}  (std={x.std():.3g}, max|.|={np.abs(x).max():.3g})")
        ax.grid(alpha=0.3)
    for ax in axes[n_comp:]:
        ax.set_visible(False)
    for ax in axes[-ncols:]:
        ax.set_xlabel('t, нс')
    fig.suptitle('Компоненты сигнала', y=1.02)
    fig.tight_layout()

    # --- 3.3. Спектр ---
    F = rfft(signal)
    freqs = rfftfreq(n, d=1 / fs)
    psd = np.abs(F) ** 2
    plt.figure(figsize=(11, 4))
    plt.semilogy(freqs / 1e6, psd, lw=0.8)
    plt.xlim(0, fs / 2 / 1e6)
    plt.xlabel('частота, МГц'); plt.ylabel('|X(f)|²')
    plt.title('Спектр входного сигнала')
    plt.grid(alpha=0.3, which='both')
    plt.tight_layout()

    # --- 3.4. Все компоненты на одном графике ---
    plt.figure(figsize=(12, 6))
    plt.plot(t_ns, signal, color='k', lw=1.6, alpha=0.85,
             label='полный сигнал', zorder=10)
    for name, x in comp.items():
        plt.plot(t_ns, x, lw=0.9, alpha=0.8, label=name)
    plt.xlabel('t, нс')
    plt.ylabel('сигнал, усл. ед.')
    plt.title('Все компоненты сигнала на одном графике')
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    # --- 3.5. Синхронное детектирование: вход, опора, выход ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(t_ns, signal, lw=0.6, color='C0', label='вход')
    axes[0].plot(t_ns, comp['laser'], lw=1.0, color='C3', alpha=0.7,
                 label='лазер')
    axes[0].set_ylabel('вход'); axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(t_ns, ref, lw=0.8, color='C1', label='опора (меандр)')
    axes[1].set_ylabel('опора'); axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(t_ns, demod, lw=0.8, color='C2', label='выход lock-in')
    axes[2].axhline(dc_out, color='k', ls='--', lw=1,
                    label=f'среднее = {dc_out:.3f}')
    axes[2].set_xlabel('t, нс'); axes[2].set_ylabel('выход')
    axes[2].legend(loc='upper right', fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.suptitle('Синхронное детектирование: выделение лазерного сигнала',
                 y=1.02)
    fig.tight_layout()

    plt.show()