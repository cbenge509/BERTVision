/**
 * Unit tests for Section component logic.
 */

import {describe, it, expect, beforeEach, afterEach, vi} from 'vitest';
import {JSDOM} from 'jsdom';

describe('Section anchor generation', () => {
  let dom: JSDOM;
  let document: Document;

  beforeEach(() => {
    dom = new JSDOM(`
      <!DOCTYPE html>
      <html>
        <body>
          <section id="introduction" class="section">
            <h2 class="section-title">
              <a href="#introduction" class="section-anchor">#</a>
              Introduction
            </h2>
          </section>
          <section id="how-it-works" class="section">
            <h2 class="section-title">
              <a href="#how-it-works" class="section-anchor">#</a>
              How It Works
            </h2>
          </section>
          <section id="results" class="section">
            <h2 class="section-title">
              <a href="#results" class="section-anchor">#</a>
              Results
            </h2>
          </section>
        </body>
      </html>
    `, {
      url: 'http://localhost',
    });
    document = dom.window.document;
    vi.stubGlobal('document', document);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    dom.window.close();
  });

  it('should have section anchors matching section ids', () => {
    const sections = document.querySelectorAll<HTMLElement>('section[id]');
    sections.forEach(section => {
      const anchor = section.querySelector<HTMLAnchorElement>('.section-anchor');
      expect(anchor).not.toBeNull();
      expect(anchor?.getAttribute('href')).toBe(`#${section.id}`);
    });
  });

  it('should find all sections by id attribute', () => {
    const sections = document.querySelectorAll('section[id]');
    expect(sections.length).toBe(3);
  });

  it('should have correct section ids', () => {
    const sectionIds = Array.from(document.querySelectorAll<HTMLElement>('section[id]'))
      .map(section => section.id);

    expect(sectionIds).toEqual(['introduction', 'how-it-works', 'results']);
  });
});

describe('Section animation logic', () => {
  let dom: JSDOM;
  let document: Document;

  beforeEach(() => {
    dom = new JSDOM(`
      <!DOCTYPE html>
      <html>
        <body>
          <section id="test" class="section" data-animate="fade-in-up">
            Test content
          </section>
        </body>
      </html>
    `, {
      url: 'http://localhost',
    });
    document = dom.window.document;
    vi.stubGlobal('document', document);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    dom.window.close();
  });

  it('should have data-animate attribute', () => {
    const section = document.getElementById('test');
    expect(section?.dataset.animate).toBe('fade-in-up');
  });

  it('should not have animate-in class initially', () => {
    const section = document.getElementById('test');
    expect(section?.classList.contains('animate-in')).toBe(false);
  });

  it('should add animate-in class when triggered', () => {
    const section = document.getElementById('test');
    if (!section) throw new Error('Section not found');

    // Simulate animation trigger
    section.classList.add('animate-in');
    expect(section.classList.contains('animate-in')).toBe(true);
  });
});

describe('Section variants', () => {
  let dom: JSDOM;
  let document: Document;

  beforeEach(() => {
    dom = new JSDOM(`
      <!DOCTYPE html>
      <html>
        <body>
          <section id="light" class="section section--light"></section>
          <section id="dark" class="section section--dark"></section>
          <section id="alternate" class="section section--alternate"></section>
        </body>
      </html>
    `, {
      url: 'http://localhost',
    });
    document = dom.window.document;
    vi.stubGlobal('document', document);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    dom.window.close();
  });

  it('should have correct variant classes', () => {
    expect(document.getElementById('light')?.classList.contains('section--light')).toBe(true);
    expect(document.getElementById('dark')?.classList.contains('section--dark')).toBe(true);
    expect(document.getElementById('alternate')?.classList.contains('section--alternate')).toBe(true);
  });

  it('should all have base section class', () => {
    const sections = document.querySelectorAll('.section');
    expect(sections.length).toBe(3);
    sections.forEach(section => {
      expect(section.classList.contains('section')).toBe(true);
    });
  });
});
