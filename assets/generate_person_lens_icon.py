#!/usr/bin/env python3

"""Generate the desktop icon from a person-shaped gravitational lens."""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import (
    gaussian_filter,
    gaussian_gradient_magnitude,
    map_coordinates,
)


SIZE = 1024
CENTER = SIZE / 2
OUTPUT = Path(__file__).with_name("GravyLensingIcon.png")


def source_field() -> np.ndarray:
    """Build the same dark star field and cyan grid used by the iOS icon."""
    y, x = np.mgrid[:SIZE, :SIZE]
    radius = np.hypot(x - CENTER, y - CENTER) / CENTER
    warm = np.exp(-((radius / 0.82) ** 2))

    image = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
    image[..., 0] = 3 + 16 * warm
    image[..., 1] = 4 + 6 * warm
    image[..., 2] = 17 + 14 * warm

    spacing = 205
    dx = np.abs((x - CENTER) % spacing - spacing / 2)
    dy = np.abs((y - CENTER) % spacing - spacing / 2)
    grid = np.maximum(np.exp(-((dx / 3.1) ** 2)), np.exp(-((dy / 3.1) ** 2)))
    grid_glow = np.maximum(
        np.exp(-((dx / 8.5) ** 2)), np.exp(-((dy / 8.5) ** 2))
    )
    image += grid_glow[..., None] * np.array([0, 18, 29], dtype=np.float32)
    image += grid[..., None] * np.array([0, 105, 143], dtype=np.float32)

    rng = np.random.default_rng(1979)
    stars = np.zeros((SIZE, SIZE), dtype=np.float32)
    for _ in range(205):
        sx, sy = rng.integers(18, SIZE - 18, size=2)
        length = int(rng.integers(1, 4))
        stars[sy, sx : sx + length] += float(rng.uniform(95, 235))
    stars = gaussian_filter(stars, sigma=(0.45, 0.8))
    image += stars[..., None]

    return np.clip(image, 0, 255)


def person_mass() -> tuple[np.ndarray, np.ndarray]:
    """Return a smooth waist-up mass map and its hard silhouette."""
    y, x = np.mgrid[:SIZE, :SIZE]

    head = ((x - CENTER) / 148) ** 2 + ((y - 407) / 168) ** 2 <= 1

    shoulder_t = np.clip((y - 545) / 155, 0, 1)
    shoulder_curve = shoulder_t * shoulder_t * (3 - 2 * shoulder_t)
    shoulder_width = 82 + (360 - 82) * shoulder_curve
    chest_t = np.clip((y - 700) / (SIZE - 700), 0, 1)
    chest_curve = chest_t * chest_t * (3 - 2 * chest_t)
    chest_width = 360 + (282 - 360) * chest_curve
    half_width = np.where(y <= 700, shoulder_width, chest_width)
    torso = (y >= 545) & (np.abs(x - CENTER) <= half_width)

    hard = head | torso
    mass = gaussian_filter(hard.astype(np.float32), sigma=19)
    return mass / mass.max(), hard.astype(np.float32)


def deflection(mass: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve the thin-lens Poisson equation and return pixel deflections."""
    padded_size = SIZE * 2
    offset = SIZE // 2
    padded = np.zeros((padded_size, padded_size), dtype=np.float32)
    padded[offset : offset + SIZE, offset : offset + SIZE] = mass

    fy = np.fft.fftfreq(padded_size)[:, None]
    fx = np.fft.fftfreq(padded_size)[None, :]
    wave_squared = fx * fx + fy * fy
    wave_squared[0, 0] = 1

    density = padded - padded.mean()
    density_hat = np.fft.fft2(density)
    potential_hat = -density_hat / (4 * np.pi**2 * wave_squared)
    potential_hat[0, 0] = 0

    alpha_x = np.fft.ifft2(2j * np.pi * fx * potential_hat).real
    alpha_y = np.fft.ifft2(2j * np.pi * fy * potential_hat).real
    alpha_x = alpha_x[offset : offset + SIZE, offset : offset + SIZE]
    alpha_y = alpha_y[offset : offset + SIZE, offset : offset + SIZE]
    magnitude = np.hypot(alpha_x, alpha_y)
    scale = 270 / np.percentile(magnitude, 99.5)
    return alpha_x * scale, alpha_y * scale


def render() -> Image.Image:
    source = source_field()
    mass, hard = person_mass()
    alpha_x, alpha_y = deflection(mass)
    y, x = np.mgrid[:SIZE, :SIZE]

    source_x = x - alpha_x
    source_y = y - alpha_y
    warped = np.stack(
        [
            map_coordinates(
                source[..., channel], [source_y, source_x], order=1, mode="reflect"
            )
            for channel in range(3)
        ],
        axis=-1,
    )

    mass_edge = gaussian_gradient_magnitude(mass, sigma=3)
    mass_edge /= mass_edge.max()
    halo = gaussian_filter(mass_edge, 62)
    halo /= halo.max()
    inner_glow = gaussian_filter(mass_edge, 19)
    inner_glow /= inner_glow.max()

    outline_edge = gaussian_gradient_magnitude(gaussian_filter(hard, 1.2), sigma=0.9)
    outline_edge /= outline_edge.max()

    orange = np.array([255, 137, 65], dtype=np.float32)
    cream = np.array([225, 255, 218], dtype=np.float32)
    cyan = np.array([0, 137, 184], dtype=np.float32)

    warped += halo[..., None] * orange * 0.54
    warped = warped * (1 - inner_glow[..., None] * 0.22) + (
        orange * inner_glow[..., None] * 0.46
    )

    # A two-tone edge echoes the cream/cyan iOS lens curves without painting
    # over the warped field that proves the person is causing the deflection.
    edge_line = np.clip(outline_edge * 1.12, 0, 1)
    line_glow = gaussian_filter(edge_line, 4)
    vertical_mix = np.clip((np.mgrid[:SIZE, :SIZE][0] - 220) / 800, 0, 1)
    outline = cream * (1 - vertical_mix[..., None]) + cyan * vertical_mix[..., None]
    warped = warped * (1 - line_glow[..., None] * 0.20) + (
        outline * line_glow[..., None] * 0.20
    )
    warped = warped * (1 - edge_line[..., None] * 0.88) + outline * edge_line[..., None]

    return Image.fromarray(np.uint8(np.clip(warped, 0, 255)))


if __name__ == "__main__":
    render().save(OUTPUT, optimize=True)
    print(OUTPUT)
