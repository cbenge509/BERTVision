/**
 * Color extraction and palette generation utilities.
 *
 * Provides functions to extract dominant colors from images and generate
 * complementary color palettes for theming.
 */

/**
 * RGB color representation.
 */
export interface RGBColor {
  r: number;
  g: number;
  b: number;
}

/**
 * HSL color representation.
 */
export interface HSLColor {
  h: number;
  s: number;
  l: number;
}

/**
 * Complete color palette derived from a primary color.
 */
export interface ColorPalette {
  primary: string;
  primaryLight: string;
  primaryDark: string;
  primaryRgb: string;
}

/**
 * Converts an RGB color to a hex string.
 */
export function rgbToHex(color: RGBColor): string {
  const toHex = (n: number): string => {
    const clamped = Math.max(0, Math.min(255, Math.round(n)));
    return clamped.toString(16).padStart(2, '0');
  };
  return `#${toHex(color.r)}${toHex(color.g)}${toHex(color.b)}`;
}

/**
 * Converts a hex color string to RGB.
 */
export function hexToRgb(hex: string): RGBColor {
  const cleanHex = hex.replace(/^#/, '');
  const fullHex =
    cleanHex.length === 3
      ? cleanHex
          .split('')
          .map(c => c + c)
          .join('')
      : cleanHex;

  const num = parseInt(fullHex, 16);
  return {
    r: (num >> 16) & 255,
    g: (num >> 8) & 255,
    b: num & 255,
  };
}

/**
 * Converts RGB to HSL color space.
 */
export function rgbToHsl(color: RGBColor): HSLColor {
  const r = color.r / 255;
  const g = color.g / 255;
  const b = color.b / 255;

  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;

  if (max === min) {
    return {h: 0, s: 0, l};
  }

  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);

  let h: number;
  switch (max) {
    case r:
      h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
      break;
    case g:
      h = ((b - r) / d + 2) / 6;
      break;
    default:
      h = ((r - g) / d + 4) / 6;
      break;
  }

  return {h, s, l};
}

/**
 * Converts HSL to RGB color space.
 */
export function hslToRgb(color: HSLColor): RGBColor {
  const {h, s, l} = color;

  if (s === 0) {
    const gray = Math.round(l * 255);
    return {r: gray, g: gray, b: gray};
  }

  const hueToRgb = (p: number, q: number, t: number): number => {
    let tNorm = t;
    if (tNorm < 0) tNorm += 1;
    if (tNorm > 1) tNorm -= 1;
    if (tNorm < 1 / 6) return p + (q - p) * 6 * tNorm;
    if (tNorm < 1 / 2) return q;
    if (tNorm < 2 / 3) return p + (q - p) * (2 / 3 - tNorm) * 6;
    return p;
  };

  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;

  return {
    r: Math.round(hueToRgb(p, q, h + 1 / 3) * 255),
    g: Math.round(hueToRgb(p, q, h) * 255),
    b: Math.round(hueToRgb(p, q, h - 1 / 3) * 255),
  };
}

/**
 * Adjusts the lightness of a color.
 *
 * @param hex - The hex color to adjust
 * @param amount - Amount to adjust lightness (-1 to 1)
 * @returns New hex color with adjusted lightness
 */
export function adjustLightness(hex: string, amount: number): string {
  const rgb = hexToRgb(hex);
  const hsl = rgbToHsl(rgb);
  hsl.l = Math.max(0, Math.min(1, hsl.l + amount));
  return rgbToHex(hslToRgb(hsl));
}

/**
 * Generates a complete color palette from a primary color.
 *
 * @param primaryHex - The primary color in hex format
 * @returns Complete color palette with light and dark variants
 */
export function generatePalette(primaryHex: string): ColorPalette {
  const rgb = hexToRgb(primaryHex);

  return {
    primary: primaryHex,
    primaryLight: adjustLightness(primaryHex, 0.15),
    primaryDark: adjustLightness(primaryHex, -0.15),
    primaryRgb: `${rgb.r} ${rgb.g} ${rgb.b}`,
  };
}

/**
 * Extracts the dominant color from image pixel data.
 *
 * Uses a simple color quantization approach to find the most
 * prominent color in an image, filtering out near-white and
 * near-black colors for better accent color extraction.
 *
 * @param imageData - Canvas ImageData from an image
 * @param sampleSize - Number of pixels to sample (default 10000)
 * @returns Dominant color in hex format
 */
export function extractDominantColor(
  imageData: ImageData,
  sampleSize = 10000
): string {
  const {data, width, height} = imageData;
  const totalPixels = width * height;
  const step = Math.max(1, Math.floor(totalPixels / sampleSize));

  // Color buckets for quantization (32 buckets per channel = 32768 colors)
  const buckets = new Map<string, {count: number; r: number; g: number; b: number}>();

  for (let i = 0; i < totalPixels; i += step) {
    const idx = i * 4;
    const r = data[idx] ?? 0;
    const g = data[idx + 1] ?? 0;
    const b = data[idx + 2] ?? 0;
    const a = data[idx + 3] ?? 255;

    // Skip transparent pixels
    if (a < 128) continue;

    // Skip near-white and near-black (not good accent colors)
    const brightness = (r + g + b) / 3;
    if (brightness < 30 || brightness > 225) continue;

    // Skip very desaturated colors
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const saturation = max === 0 ? 0 : (max - min) / max;
    if (saturation < 0.15) continue;

    // Quantize to reduce color space
    const qr = Math.floor(r / 8) * 8;
    const qg = Math.floor(g / 8) * 8;
    const qb = Math.floor(b / 8) * 8;
    const key = `${qr},${qg},${qb}`;

    const existing = buckets.get(key);
    if (existing) {
      existing.count++;
      existing.r += r;
      existing.g += g;
      existing.b += b;
    } else {
      buckets.set(key, {count: 1, r, g, b});
    }
  }

  // Find the most frequent bucket
  let maxCount = 0;
  let dominantColor: RGBColor = {r: 59, g: 130, b: 246}; // Fallback: blue

  for (const bucket of buckets.values()) {
    if (bucket.count > maxCount) {
      maxCount = bucket.count;
      dominantColor = {
        r: Math.round(bucket.r / bucket.count),
        g: Math.round(bucket.g / bucket.count),
        b: Math.round(bucket.b / bucket.count),
      };
    }
  }

  return rgbToHex(dominantColor);
}

/**
 * Default accent color palette for the BERTVision site.
 *
 * This palette is derived from the hero image's dominant colors
 * (deep blue/teal tones from the neural network visualization).
 */
export const defaultPalette: ColorPalette = generatePalette('#3b82f6');

/**
 * Generates CSS custom properties from a color palette.
 *
 * @param palette - The color palette to convert
 * @returns CSS custom properties string
 */
export function paletteToCSS(palette: ColorPalette): string {
  return `
  --color-primary: ${palette.primary};
  --color-primary-light: ${palette.primaryLight};
  --color-primary-dark: ${palette.primaryDark};
  --color-primary-rgb: ${palette.primaryRgb};
`.trim();
}
