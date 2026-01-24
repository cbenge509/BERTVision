import {describe, it, expect} from 'vitest';
import {
  rgbToHex,
  hexToRgb,
  rgbToHsl,
  hslToRgb,
  adjustLightness,
  generatePalette,
  paletteToCSS,
  defaultPalette,
  type RGBColor,
} from '../colors';

describe('colors', () => {
  describe('rgbToHex', () => {
    it('converts RGB to hex correctly', () => {
      expect(rgbToHex({r: 255, g: 255, b: 255})).toBe('#ffffff');
      expect(rgbToHex({r: 0, g: 0, b: 0})).toBe('#000000');
      expect(rgbToHex({r: 59, g: 130, b: 246})).toBe('#3b82f6');
    });

    it('clamps values outside 0-255 range', () => {
      expect(rgbToHex({r: -10, g: 300, b: 128})).toBe('#00ff80');
    });

    it('rounds non-integer values', () => {
      expect(rgbToHex({r: 59.4, g: 130.6, b: 245.9})).toBe('#3b83f6');
    });
  });

  describe('hexToRgb', () => {
    it('converts 6-digit hex to RGB', () => {
      expect(hexToRgb('#ffffff')).toEqual({r: 255, g: 255, b: 255});
      expect(hexToRgb('#000000')).toEqual({r: 0, g: 0, b: 0});
      expect(hexToRgb('#3b82f6')).toEqual({r: 59, g: 130, b: 246});
    });

    it('handles hex without hash prefix', () => {
      expect(hexToRgb('3b82f6')).toEqual({r: 59, g: 130, b: 246});
    });

    it('expands 3-digit hex to 6-digit', () => {
      expect(hexToRgb('#fff')).toEqual({r: 255, g: 255, b: 255});
      expect(hexToRgb('#000')).toEqual({r: 0, g: 0, b: 0});
      expect(hexToRgb('#abc')).toEqual({r: 170, g: 187, b: 204});
    });
  });

  describe('rgbToHsl', () => {
    it('converts pure colors correctly', () => {
      const red = rgbToHsl({r: 255, g: 0, b: 0});
      expect(red.h).toBeCloseTo(0, 2);
      expect(red.s).toBeCloseTo(1, 2);
      expect(red.l).toBeCloseTo(0.5, 2);

      const green = rgbToHsl({r: 0, g: 255, b: 0});
      expect(green.h).toBeCloseTo(1 / 3, 2);

      const blue = rgbToHsl({r: 0, g: 0, b: 255});
      expect(blue.h).toBeCloseTo(2 / 3, 2);
    });

    it('handles grayscale colors', () => {
      const gray = rgbToHsl({r: 128, g: 128, b: 128});
      expect(gray.h).toBe(0);
      expect(gray.s).toBe(0);
      expect(gray.l).toBeCloseTo(0.5, 1);
    });

    it('handles white and black', () => {
      const white = rgbToHsl({r: 255, g: 255, b: 255});
      expect(white.l).toBeCloseTo(1, 2);

      const black = rgbToHsl({r: 0, g: 0, b: 0});
      expect(black.l).toBeCloseTo(0, 2);
    });
  });

  describe('hslToRgb', () => {
    it('converts pure colors correctly', () => {
      expect(hslToRgb({h: 0, s: 1, l: 0.5})).toEqual({r: 255, g: 0, b: 0});
      expect(hslToRgb({h: 1 / 3, s: 1, l: 0.5})).toEqual({r: 0, g: 255, b: 0});
      expect(hslToRgb({h: 2 / 3, s: 1, l: 0.5})).toEqual({r: 0, g: 0, b: 255});
    });

    it('handles grayscale colors', () => {
      const gray = hslToRgb({h: 0, s: 0, l: 0.5});
      expect(gray.r).toBeCloseTo(128, 0);
      expect(gray.g).toBeCloseTo(128, 0);
      expect(gray.b).toBeCloseTo(128, 0);
    });

    it('handles white and black', () => {
      expect(hslToRgb({h: 0, s: 0, l: 1})).toEqual({r: 255, g: 255, b: 255});
      expect(hslToRgb({h: 0, s: 0, l: 0})).toEqual({r: 0, g: 0, b: 0});
    });
  });

  describe('RGB to HSL round-trip', () => {
    const testColors: RGBColor[] = [
      {r: 59, g: 130, b: 246},
      {r: 255, g: 100, b: 50},
      {r: 50, g: 200, b: 150},
      {r: 128, g: 64, b: 192},
    ];

    testColors.forEach(color => {
      it(`round-trips ${JSON.stringify(color)} correctly`, () => {
        const hsl = rgbToHsl(color);
        const backToRgb = hslToRgb(hsl);
        expect(backToRgb.r).toBeCloseTo(color.r, 0);
        expect(backToRgb.g).toBeCloseTo(color.g, 0);
        expect(backToRgb.b).toBeCloseTo(color.b, 0);
      });
    });
  });

  describe('adjustLightness', () => {
    it('lightens colors when amount is positive', () => {
      const original = '#3b82f6';
      const lighter = adjustLightness(original, 0.2);
      const originalHsl = rgbToHsl(hexToRgb(original));
      const lighterHsl = rgbToHsl(hexToRgb(lighter));
      expect(lighterHsl.l).toBeGreaterThan(originalHsl.l);
    });

    it('darkens colors when amount is negative', () => {
      const original = '#3b82f6';
      const darker = adjustLightness(original, -0.2);
      const originalHsl = rgbToHsl(hexToRgb(original));
      const darkerHsl = rgbToHsl(hexToRgb(darker));
      expect(darkerHsl.l).toBeLessThan(originalHsl.l);
    });

    it('clamps lightness to valid range', () => {
      const white = adjustLightness('#ffffff', 0.5);
      const whiteHsl = rgbToHsl(hexToRgb(white));
      expect(whiteHsl.l).toBeLessThanOrEqual(1);

      const black = adjustLightness('#000000', -0.5);
      const blackHsl = rgbToHsl(hexToRgb(black));
      expect(blackHsl.l).toBeGreaterThanOrEqual(0);
    });
  });

  describe('generatePalette', () => {
    it('generates all palette colors', () => {
      const palette = generatePalette('#3b82f6');
      expect(palette.primary).toBe('#3b82f6');
      expect(palette.primaryLight).toBeDefined();
      expect(palette.primaryDark).toBeDefined();
      expect(palette.primaryRgb).toBe('59 130 246');
    });

    it('generates lighter and darker variants', () => {
      const palette = generatePalette('#3b82f6');
      const primaryHsl = rgbToHsl(hexToRgb(palette.primary));
      const lightHsl = rgbToHsl(hexToRgb(palette.primaryLight));
      const darkHsl = rgbToHsl(hexToRgb(palette.primaryDark));

      expect(lightHsl.l).toBeGreaterThan(primaryHsl.l);
      expect(darkHsl.l).toBeLessThan(primaryHsl.l);
    });
  });

  describe('defaultPalette', () => {
    it('has valid structure', () => {
      expect(defaultPalette.primary).toMatch(/^#[0-9a-f]{6}$/i);
      expect(defaultPalette.primaryLight).toMatch(/^#[0-9a-f]{6}$/i);
      expect(defaultPalette.primaryDark).toMatch(/^#[0-9a-f]{6}$/i);
      expect(defaultPalette.primaryRgb).toMatch(/^\d+ \d+ \d+$/);
    });
  });

  describe('paletteToCSS', () => {
    it('generates valid CSS custom properties', () => {
      const palette = generatePalette('#3b82f6');
      const css = paletteToCSS(palette);

      expect(css).toContain('--color-primary: #3b82f6');
      expect(css).toContain('--color-primary-light:');
      expect(css).toContain('--color-primary-dark:');
      expect(css).toContain('--color-primary-rgb: 59 130 246');
    });
  });
});
